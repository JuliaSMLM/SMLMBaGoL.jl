# Prior distributions for BaGoL

"""
Uniform spatial prior over rectangular region.
"""
struct UniformSpatialPrior
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64
end

function UniformSpatialPrior(locs::Vector{<:SMLMData.AbstractEmitter}; padding::Float64=0.05)
    xs = [loc.x for loc in locs]
    ys = [loc.y for loc in locs]
    x_min, x_max = extrema(xs)
    y_min, y_max = extrema(ys)

    # Percentage-based padding
    dx = (x_max - x_min) * padding
    dy = (y_max - y_min) * padding

    # Minimum padding based on localization precision (3σ ensures proposals stay in bounds)
    σ_x_mean = mean(loc.σ_x for loc in locs)
    σ_y_mean = mean(loc.σ_y for loc in locs)
    dx = max(dx, 3 * σ_x_mean)
    dy = max(dy, 3 * σ_y_mean)

    x_lo = x_min - dx
    x_hi = x_max + dx
    y_lo = y_min - dy
    y_hi = y_max + dy

    # Minimum area floor: prevents per-cluster spatial bonus from dominating
    # when data is compact. Each localization gets one resolution circle of area.
    N = length(locs)
    σ_med = median([(loc.σ_x + loc.σ_y) / 2 for loc in locs])
    min_area = N * π * (3 * σ_med)^2
    current_area = (x_hi - x_lo) * (y_hi - y_lo)

    if current_area < min_area
        scale = sqrt(min_area / current_area)
        cx = (x_lo + x_hi) / 2
        cy = (y_lo + y_hi) / 2
        hw = (x_hi - x_lo) / 2 * scale
        hh = (y_hi - y_lo) / 2 * scale
        x_lo, x_hi = cx - hw, cx + hw
        y_lo, y_hi = cy - hh, cy + hh
    end

    UniformSpatialPrior(x_lo, x_hi, y_lo, y_hi)
end

area(prior::UniformSpatialPrior) = (prior.x_max - prior.x_min) * (prior.y_max - prior.y_min)

function log_spatial_prior(x::Float64, y::Float64, prior::UniformSpatialPrior)
    if prior.x_min <= x <= prior.x_max && prior.y_min <= y <= prior.y_max
        return -log(area(prior))
    else
        return -Inf
    end
end

"""
Sample a position from the spatial prior.
"""
function sample_position(prior::UniformSpatialPrior)
    x = prior.x_min + rand() * (prior.x_max - prior.x_min)
    y = prior.y_min + rand() * (prior.y_max - prior.y_min)
    return x, y
end

"""
Log prior on number of emitters K - independent Poisson.
"""
function log_prior_k(k::Int, λ_K::Float64)
    if k < 0
        return -Inf
    end
    return k * log(λ_K) - λ_K - logfactorial(k)
end

"""
Log NegBin PMF for a single emitter's count.

Model: n_k ~ NegBin(α, p) where p = α/(α+μ)
  - E[n_k] = μ
  - Var[n_k] = μ(1 + μ/α)
  - α=1: geometric (dSTORM)
  - α>1: peaked (DNA-PAINT)
  - α→∞: Poisson(μ)

Using Distributions.jl NegativeBinomial(r, p) parameterization:
  r = α, p = α/(α+μ)
"""
function log_negbin_pmf(n::Int, μ::Float64, α::Float64)
    if n < 0
        return -Inf
    end
    p = α / (α + μ)
    return logpdf(NegativeBinomial(α, p), n)
end

"""
Log prior on TOTAL count N given K emitters, using marginal distribution.

From Fazel et al. (2022): P(K|ξ) ∝ Gamma(N; K*shape, μ/shape)

The sum of K independent Gamma(shape, scale) variables is Gamma(K*shape, scale).
This is a continuous approximation to the NegBin(K*α, p) total count.

Model: N ~ Gamma(K*shape, μ/shape)
  - E[N] = K*μ
  - Var[N] = K*μ²/shape
"""
function log_prior_total_count(N::Int, K::Int, μ::Float64, shape::Float64)
    if N < K  # Need at least 1 loc per emitter
        return -Inf
    end
    if K <= 0
        return -Inf
    end
    total_shape = K * shape
    scale = μ / shape
    dist = Gamma(total_shape, scale)
    return logpdf(dist, Float64(N))
end

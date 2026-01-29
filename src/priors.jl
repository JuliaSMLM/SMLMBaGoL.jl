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

    UniformSpatialPrior(x_min - dx, x_max + dx, y_min - dy, y_max + dy)
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
Log prior on count for a single emitter using Gamma(shape, scale=μ/shape).

Model: n_j ~ Gamma(shape, μ/shape)
  - E[n_j] = μ
  - Var[n_j] = μ²/shape
  - CV[n_j] = 1/√shape

Physical interpretation:
  - shape=1: Exponential (dSTORM - blink until bleach)
  - shape>1: Peaked distribution (DNA-PAINT-like)
  - shape→∞: Delta function at μ

Note: Treating integer counts as continuous. Valid approximation for n > 5.
"""
function log_prior_count(n::Int, μ::Float64, shape::Float64)
    if n < 1  # Require at least 1 localization per emitter
        return -Inf
    end
    # Gamma(shape, scale) where scale = μ/shape
    scale = μ / shape
    dist = Gamma(shape, scale)
    return logpdf(dist, Float64(n))
end

"""
Log prior on TOTAL count N given K emitters, using marginal distribution.

From Fazel et al. (2022): P(K|ξ) ∝ Gamma(N; K*shape, μ/shape)

The sum of K independent Gamma(shape, scale) variables is Gamma(K*shape, scale).
This is the CORRECT prior that avoids the normalization bias from using
individual count priors.

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
    # Sum of K Gamma(shape, scale) is Gamma(K*shape, scale)
    total_shape = K * shape
    scale = μ / shape
    dist = Gamma(total_shape, scale)
    return logpdf(dist, Float64(N))
end

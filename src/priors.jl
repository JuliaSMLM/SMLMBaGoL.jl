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
    dx = (x_max - x_min) * padding
    dy = (y_max - y_min) * padding
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

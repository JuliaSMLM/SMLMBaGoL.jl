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
Log prior on count for a single emitter given μ and α (NegBinomial marginal).
The per-emitter rate λ_j ~ Gamma(α, α/μ), then n_j ~ Poisson(λ_j).
Marginally: n_j ~ NegBinomial(α, α/(α+μ)).
"""
function log_prior_count(n::Int, μ::Float64, α::Float64)
    if n < 0
        return -Inf
    end
    # NegBinomial(r, p) where r=α (failures), p=success prob
    # Using Distributions.jl parameterization: NegativeBinomial(r, p)
    # where p = α/(α+μ) is the success probability
    p = α / (α + μ)
    dist = NegativeBinomial(α, p)
    return logpdf(dist, n)
end

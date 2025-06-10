struct UniformSpatialPrior{T<:Real} <: AbstractPrior
    x_min::T
    x_max::T
    y_min::T
    y_max::T
end

struct GammaPrior{T<:Real} <: AbstractPrior
    α::T  # shape parameter
    β::T  # scale parameter
end

struct HierarchicalGammaPrior{T<:Real} <: AbstractPrior
    α::T  # shape parameter
    β::T  # scale parameter
    α_prior::Tuple{T, T}  # (a₀, b₀) for α
    β_prior::Tuple{T, T}  # (c₀, d₀) for β
end

struct CompoundPrior{T<:Real} <: AbstractPrior
    spatial_prior::UniformSpatialPrior{T}
    K_prior::GammaPrior{T}  # prior on number of emitters
end

function log_prior_spatial(emitter::AbstractEmitter, prior::UniformSpatialPrior)
    if prior.x_min ≤ emitter.x ≤ prior.x_max && prior.y_min ≤ emitter.y ≤ prior.y_max
        area = (prior.x_max - prior.x_min) * (prior.y_max - prior.y_min)
        return -log(area)
    else
        return -Inf
    end
end

function log_prior_K(K::Int, prior::GammaPrior)
    K < 0 && return -Inf
    return (prior.α - 1) * log(K) - K / prior.β - loggamma(prior.α) - prior.α * log(prior.β)
end

function log_prior(state::BaGoLState)
    K = length(state.emitters)
    
    if isa(state.prior, CompoundPrior)
        ll = log_prior_K(K, state.prior.K_prior)
        for emitter in state.emitters
            ll += log_prior_spatial(emitter, state.prior.spatial_prior)
        end
        return ll
    else
        error("Unsupported prior type: $(typeof(state.prior))")
    end
end

function sample_spatial_prior(prior::UniformSpatialPrior{T}, rng=Random.GLOBAL_RNG) where T
    x = prior.x_min + (prior.x_max - prior.x_min) * rand(rng, T)
    y = prior.y_min + (prior.y_max - prior.y_min) * rand(rng, T)
    return (x, y)
end

function create_spatial_prior_from_localizations(localizations::Vector{<:AbstractLocalization}, margin::Real=0.5)
    x_coords = [loc.x for loc in localizations]
    y_coords = [loc.y for loc in localizations]
    
    x_min, x_max = extrema(x_coords)
    y_min, y_max = extrema(y_coords)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    return UniformSpatialPrior(
        x_min - margin * x_range,
        x_max + margin * x_range,
        y_min - margin * y_range,
        y_max + margin * y_range
    )
end
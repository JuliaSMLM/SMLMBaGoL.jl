# Add new abstract types for clarity
abstract type AbstractSpatialPrior <: AbstractPrior end
abstract type AbstractCountPrior <: AbstractPrior end

# Spatial prior remains the same but inherits from new abstract type
struct UniformSpatialPrior{T<:Real} <: AbstractSpatialPrior
    x_min::T
    x_max::T
    y_min::T
    y_max::T
end

# Rename and clarify hierarchical prior to match math spec
struct HierarchicalNegBinomialPrior{T<:Real} <: AbstractCountPrior
    μ::T  # mean number of localizations per emitter (NOT shape/scale!)
    κ::T  # overdispersion parameter (concentration)
    τ²::T # additional localization variance (current value)
    μ_hyperprior::Tuple{T, T}  # (a₀, b₀) for Gamma prior on μ
    κ_hyperprior::Tuple{T, T}  # (c₀, d₀) for Gamma prior on κ
    τ²_hyperprior::Tuple{T, T} # (a_τ, b_τ) for InverseGamma prior on τ²
end


function log_prior_spatial(emitter::AbstractEmitter, prior::UniformSpatialPrior)
    if prior.x_min ≤ emitter.x ≤ prior.x_max && prior.y_min ≤ emitter.y ≤ prior.y_max
        area = (prior.x_max - prior.x_min) * (prior.y_max - prior.y_min)
        return -log(area)
    else
        return -Inf
    end
end

# Helper functions to extract κ from count priors
function get_concentration_parameter(prior::HierarchicalNegBinomialPrior)
    return prior.κ
end


# Function to get spatial prior density (needed for birth/death moves)
function log_spatial_prior_density(emitter::AbstractEmitter, prior::UniformSpatialPrior)
    if prior.x_min ≤ emitter.x ≤ prior.x_max && prior.y_min ≤ emitter.y ≤ prior.y_max
        area = (prior.x_max - prior.x_min) * (prior.y_max - prior.y_min)
        return -log(area)
    else
        return -Inf
    end
end

# log_prior function moved to state.jl since it depends on BaGoLState

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
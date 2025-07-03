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
    
    # Inner constructor with validation
    function HierarchicalNegBinomialPrior{T}(μ::T, κ::T, τ²::T, μ_hyperprior, κ_hyperprior, τ²_hyperprior) where T
        μ > 0 || error("μ must be positive")
        κ > 0 || error("κ must be positive")  
        τ² > 0 || error("τ² must be positive")
        all(x -> x > 0, μ_hyperprior) || error("μ hyperprior parameters must be positive")
        all(x -> x > 0, κ_hyperprior) || error("κ hyperprior parameters must be positive")
        all(x -> x > 0, τ²_hyperprior) || error("τ² hyperprior parameters must be positive")
        
        new{T}(μ, κ, τ², μ_hyperprior, κ_hyperprior, τ²_hyperprior)
    end
end

# Outer constructor for type inference
function HierarchicalNegBinomialPrior(μ::T, κ::T, τ²::T, μ_hyperprior, κ_hyperprior, τ²_hyperprior) where T
    HierarchicalNegBinomialPrior{T}(μ, κ, τ², μ_hyperprior, κ_hyperprior, τ²_hyperprior)
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

"""
    log_prior_k_given_N(k::Int, N::Int, μ::T, κ::T) where T<:Real

Log prior probability of having k emitters given N total localizations.
This prior is centered at k ≈ N/μ and uses a Negative Binomial distribution
to allow overdispersion controlled by κ.

This prior is crucial for preventing the pathological behavior where the model
prefers many emitters (k → ∞) with no systematic noise (τ² → 0).
"""
function log_prior_k_given_N(k::Int, N::Int, μ::T, κ::T) where T<:Real
    # Prior on k centered at N/μ with overdispersion controlled by κ
    # Using Negative Binomial to allow more flexibility than Poisson
    
    # Expected number of emitters
    k_expected = N / μ
    
    # Negative Binomial parameterization:
    # Mean = k_expected, Variance = k_expected + k_expected²/κ
    # This gives r = κ, p = κ/(κ + k_expected)
    r = κ
    p = κ / (κ + k_expected)
    
    # Return log probability
    return logpdf(NegativeBinomial(r, p), k)
end
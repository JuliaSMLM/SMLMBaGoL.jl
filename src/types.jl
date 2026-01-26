# Core types for BaGoL RJMCMC

"""
Emitter position with associated localizations.
Parametric on coordinate type T (typically Float32 or Float64).
"""
mutable struct Emitter{T<:AbstractFloat}
    x::T
    y::T
    allocated::Vector{Int}  # Indices of localizations assigned to this emitter
end

Emitter(x::T, y::T) where T<:AbstractFloat = Emitter{T}(x, y, Int[])

"""
Current state of the RJMCMC chain.
"""
mutable struct BaGoLState{T<:AbstractFloat}
    emitters::Vector{Emitter{T}}
    log_posterior::Float64
end

"""
Recorded sample from the RJMCMC chain.
"""
struct BaGoLSample{T<:AbstractFloat}
    emitters::Vector{Emitter{T}}
    log_posterior::Float64
    μ::Float64  # Current hierarchical mean
    α::Float64  # Current shape parameter
end

"""
Configuration for RJMCMC chain.
"""
Base.@kwdef struct RJMCMCConfig
    α::Float64 = 2.0    # Shape parameter for count distribution (fixed)
    λ_K::Float64 = 10.0 # Poisson prior mean for K (independent)
    μ_prior_a::Float64 = 2.0   # Gamma hyperprior shape for μ
    μ_prior_b::Float64 = 0.2   # Gamma hyperprior rate for μ (mean = a/b = 10)
    n_iterations::Int = 10000
    burn_in::Int = 2000
    hierarchical_interval::Int = 100
    move_σ::Float64 = 0.010  # Proposal std for move step
end

"""
Complete RJMCMC chain with samples and diagnostics.
"""
mutable struct RJMCMCChain{T<:AbstractFloat}
    config::RJMCMCConfig
    samples::Vector{BaGoLSample{T}}
    μ::Float64  # Current hierarchical mean
    α::Float64  # Current shape parameter (mutable for learning)
    learn_α::Bool  # Whether to update α during MCMC
    current_state::BaGoLState{T}
    iteration::Int
    acceptance::Dict{Symbol, Tuple{Int, Int}}  # (accepted, total) per move type
end

function RJMCMCChain(config::RJMCMCConfig, initial_state::BaGoLState{T}; α_init::Float64=config.α, learn_α::Bool=false) where T
    μ_init = config.μ_prior_a / config.μ_prior_b
    RJMCMCChain{T}(
        config,
        BaGoLSample{T}[],
        μ_init,
        α_init,
        learn_α,
        initial_state,
        0,
        Dict(:birth => (0, 0), :death => (0, 0), :move => (0, 0), :allocate => (0, 0))
    )
end

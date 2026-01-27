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
    μ::Float64      # Mean locs per emitter
    shape::Float64  # Gamma shape (1=exponential, higher=more peaked)
end

"""
Configuration for RJMCMC chain.

Count model: n_j ~ Gamma(shape, scale=μ/shape)
  - E[n_j] = μ
  - CV[n_j] = 1/√shape
  - shape=1: exponential (dSTORM)
  - shape>1: peaked (DNA-PAINT-like)
"""
Base.@kwdef struct RJMCMCConfig
    shape::Float64 = 2.0        # Gamma shape for count distribution
    λ_K::Float64 = 10.0         # Poisson prior mean for K
    μ_prior_shape::Float64 = 2.0    # Gamma hyperprior shape for μ
    μ_prior_scale::Float64 = 5.0    # Gamma hyperprior scale for μ (mean = shape*scale = 10)
    shape_prior_shape::Float64 = 2.0   # Gamma hyperprior shape for shape
    shape_prior_scale::Float64 = 1.0   # Gamma hyperprior scale for shape (mean = 2)
    n_iterations::Int = 10000
    burn_in::Int = 2000
    hierarchical_interval::Int = 100
end

"""
Complete RJMCMC chain with samples and diagnostics.
"""
mutable struct RJMCMCChain{T<:AbstractFloat}
    config::RJMCMCConfig
    samples::Vector{BaGoLSample{T}}
    μ::Float64      # Current mean locs per emitter
    shape::Float64  # Current Gamma shape (mutable for learning)
    learn_shape::Bool  # Whether to update shape during MCMC
    current_state::BaGoLState{T}
    iteration::Int
    acceptance::Dict{Symbol, Tuple{Int, Int}}  # (accepted, total) per move type
end

function RJMCMCChain(config::RJMCMCConfig, initial_state::BaGoLState{T};
                     shape_init::Float64=config.shape, learn_shape::Bool=false) where T
    μ_init = config.μ_prior_shape * config.μ_prior_scale
    RJMCMCChain{T}(
        config,
        BaGoLSample{T}[],
        μ_init,
        shape_init,
        learn_shape,
        initial_state,
        0,
        Dict(:birth => (0, 0), :death => (0, 0), :move => (0, 0), :allocate => (0, 0))
    )
end

"""
Diagnostics from BaGoL analysis for QC and visualization.
"""
struct BaGoLDiagnostics
    n_emitters::Int
    posterior_k::Vector{Int}
    acceptance_rates::Dict{Symbol, Float64}
    final_μ::Float64
    final_shape::Float64
    n_partitions::Int  # 1 for non-partitioned runs
end

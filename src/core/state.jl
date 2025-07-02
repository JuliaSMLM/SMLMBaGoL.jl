struct BaGoLState{E<:AbstractEmitter, L<:AbstractLocalization, T<:Real} <: AbstractChainState
    emitters::Vector{E}
    localizations::Vector{L}
    allocations::Vector{Int}  # maps localization i to emitter allocations[i]
    spatial_prior::AbstractSpatialPrior  # Changed from generic prior
    count_prior::AbstractCountPrior      # NEW: explicit count prior
    log_likelihood::T
end

mutable struct RJMCMCChain{T<:Real, E<:AbstractEmitter, L<:AbstractLocalization}
    localizations::Vector{L}
    current_state::BaGoLState{E,L,T}
    spatial_prior::AbstractSpatialPrior  # Separated priors
    count_prior::AbstractCountPrior      # Separated priors
    move_weights::Dict{Type{<:AbstractRJMCMCMove}, Float64}
    samples::Vector{BaGoLState{E,L,T}}
    burn_in::Int
    thin::Int
    rng::AbstractRNG
    hierarchical_history::Vector{HierarchicalUpdate{T}}  # Better typing
    birth_proposal::BirthProposalDistribution{T}  # Cached birth proposal distribution
end

# Log prior function (moved here from priors.jl due to dependency on BaGoLState)
function log_prior(state::BaGoLState)
    # Only spatial prior contributes to log prior
    # K emerges from RJMCMC process, no explicit prior
    ll = 0.0
    for emitter in state.emitters
        ll += log_prior_spatial(emitter, state.spatial_prior)
    end
    return ll
end
# RJMCMC split and merge emitter moves
using Random
using SMLMData: Emitter2D

"""
Split an emitter into two emitters.
Currently simplified - not fully implemented.
"""
function split_emitter!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    # Not implemented in simplified version
    return false, log_prob_current
end

"""
Merge two emitters into one.
Currently simplified - not fully implemented.
"""
function merge_emitter!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    # Not implemented in simplified version
    return false, log_prob_current
end
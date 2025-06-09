# RJMCMC reallocation moves - observation assignment
using Random
using StatsBase: sample, Weights
using SMLMData: Emitter2D

"""
Reallocate observations to emitters.
"""
function reallocate!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_emitters = length(state)
    n_emitters == 0 && return false, log_prob_current
    
    # Store old allocations
    old_allocations = copy(allocations)
    
    # Reallocate using categorical sampling
    for i in eachindex(observations)
        obs = observations[i]
        
        # Compute weights for each emitter
        log_weights = Vector{T}(undef, n_emitters)
        
        @inbounds for j in 1:n_emitters
            log_weights[j] = log_likelihood(obs, state[j])
        end
        
        # Normalize and sample
        log_weights .-= maximum(log_weights)
        weights = exp.(log_weights)
        weights ./= sum(weights)
        
        allocations[i] = sample(rng, 1:n_emitters, Weights(weights))
    end
    
    # Always accept reallocation moves (Gibbs sampling)
    log_prob_new = compute_log_posterior(state, allocations, observations, prior)
    
    return true, log_prob_new
end

"""
Reallocate all observations (helper function).
"""
function reallocate_all!(
    allocations::Vector{Int},
    state::Vector{Emitter2D{T}},
    observations::Vector{O},
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_emitters = length(state)
    
    if n_emitters == 0
        fill!(allocations, 0)
        return
    end
    
    for i in eachindex(observations)
        obs = observations[i]
        
        # Find nearest emitter
        min_dist = Inf
        best_j = 1
        
        @inbounds for j in 1:n_emitters
            em = state[j]
            # Handle observations with or without uncertainty
            if hasproperty(obs, :σ_x)
                dist = ((obs.x - em.x) / obs.σ_x)^2 + ((obs.y - em.y) / obs.σ_y)^2
            else
                dist = (obs.x - em.x)^2 + (obs.y - em.y)^2
            end
            if dist < min_dist
                min_dist = dist
                best_j = j
            end
        end
        
        allocations[i] = best_j
    end
end
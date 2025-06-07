# Core RJMCMC implementation
using Distributions
using Random
using LinearAlgebra
using StatsBase: sample, Weights

"""
Run a single RJMCMC chain for a subregion.
"""
function run_single_chain(
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    steps::Int,
    burnin::Int,
    move_probs::MoveProbs{T},
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    # Initialize state
    state, allocations = initialize_state(observations, prior, rng)
    
    # Pre-allocate output storage
    n_samples = steps - burnin
    states = Vector{Vector{Emitter2D{T}}}()
    log_probs = Vector{T}()
    allocation_history = Vector{Vector{Int}}()
    
    sizehint!(states, n_samples)
    sizehint!(log_probs, n_samples)
    sizehint!(allocation_history, n_samples)
    
    # Current log probability
    log_prob = compute_log_posterior(state, allocations, observations, prior)
    
    # MCMC loop
    n_accepted = 0
    for step in 1:steps
        # Propose move
        accepted, log_prob_new = rjmcmc_step!(
            state, allocations, observations, prior, move_probs, log_prob, rng
        )
        
        if accepted
            log_prob = log_prob_new
            n_accepted += 1
        end
        
        # Store samples after burn-in
        if step > burnin
            push!(states, copy(state))
            push!(log_probs, log_prob)
            push!(allocation_history, copy(allocations))
        end
    end
    
    return BaGoLChain(states, log_probs, allocation_history, observations)
end

"""
Initialize MCMC state with smart starting positions.
"""
function initialize_state(
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_obs = length(observations)
    
    # Start with expected number of emitters based on prior
    expected_λ = prior.α / prior.β
    n_emitters = max(1, round(Int, n_obs / expected_λ))
    
    # Initialize emitters near observations
    state = Vector{Emitter2D{T}}()
    
    if n_emitters >= n_obs
        # One emitter per observation
        for obs in observations
            push!(state, Emitter2D(obs.x, obs.y, obs.photons))
        end
    else
        # K-means style initialization
        selected = sample(rng, 1:n_obs, n_emitters, replace=false)
        for idx in selected
            obs = observations[idx]
            push!(state, Emitter2D(obs.x, obs.y, obs.photons))
        end
    end
    
    # Initialize allocations
    allocations = initialize_allocations(observations, state, rng)
    
    return state, allocations
end

"""
Initialize allocations using nearest neighbor assignment.
"""
function initialize_allocations(
    observations::Vector{O},
    state::Vector{Emitter2D{T}},
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_obs = length(observations)
    n_emitters = length(state)
    allocations = Vector{Int}(undef, n_obs)
    
    if n_emitters == 0
        fill!(allocations, 0)
    else
        # Assign each observation to nearest emitter
        @inbounds for i in 1:n_obs
            obs = observations[i]
            min_dist = Inf
            best_j = 1
            
            for j in 1:n_emitters
                em = state[j]
                dist = (obs.x - em.x)^2 + (obs.y - em.y)^2
                if dist < min_dist
                    min_dist = dist
                    best_j = j
                end
            end
            
            allocations[i] = best_j
        end
    end
    
    return allocations
end

"""
Single RJMCMC step with all move types.
"""
function rjmcmc_step!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    move_probs::MoveProbs{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    # Sample move type
    move_type = sample_move_type(length(state), move_probs, rng)
    
    if move_type == MOVE_EMITTER
        return move_emitter!(state, allocations, observations, prior, log_prob_current, rng)
    elseif move_type == ADD_EMITTER
        return add_emitter!(state, allocations, observations, prior, log_prob_current, rng)
    elseif move_type == REMOVE_EMITTER
        return remove_emitter!(state, allocations, observations, prior, log_prob_current, rng)
    elseif move_type == SPLIT_EMITTER
        return split_emitter!(state, allocations, observations, prior, log_prob_current, rng)
    elseif move_type == MERGE_EMITTER
        return merge_emitter!(state, allocations, observations, prior, log_prob_current, rng)
    else # REALLOCATE
        return reallocate!(state, allocations, observations, prior, log_prob_current, rng)
    end
end

"""
Sample move type based on current state and probabilities.
"""
function sample_move_type(
    n_emitters::Int,
    probs::MoveProbs{T},
    rng::AbstractRNG
) where T
    
    # Adjust probabilities based on state
    p = zeros(T, 6)
    
    p[1] = n_emitters > 0 ? probs.move : zero(T)
    p[2] = probs.add
    p[3] = n_emitters > 1 ? probs.remove : zero(T)
    p[4] = n_emitters > 0 ? probs.split : zero(T)
    p[5] = n_emitters > 1 ? probs.merge : zero(T)
    p[6] = n_emitters > 0 ? probs.reallocate : zero(T)
    
    # Normalize
    p ./= sum(p)
    
    # Sample
    r = rand(rng)
    cumsum = zero(T)
    for i in 1:6
        cumsum += p[i]
        if r <= cumsum
            return MoveType(i)
        end
    end
    
    return REALLOCATE  # fallback
end

# Move implementations

"""
Move an emitter to a new position.
"""
function move_emitter!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_emitters = length(state)
    n_emitters == 0 && return false, log_prob_current
    
    # Select emitter to move
    j = rand(rng, 1:n_emitters)
    
    # Get observations allocated to this emitter
    obs_indices = findall(==(j), allocations)
    isempty(obs_indices) && return false, log_prob_current
    
    # Propose new position (weighted average with noise)
    x_new = zero(T)
    y_new = zero(T)
    weight_sum = zero(T)
    
    for i in obs_indices
        obs = observations[i]
        # Handle uncertainty if available
        if hasproperty(obs, :σ_x)
            weight = one(T) / (obs.σ_x * obs.σ_y)
        else
            weight = one(T)  # Equal weighting if no uncertainty
        end
        x_new += weight * obs.x
        y_new += weight * obs.y
        weight_sum += weight
    end
    
    x_new /= weight_sum
    y_new /= weight_sum
    
    # Add noise
    σ_move = T(0.1)
    x_new += σ_move * randn(rng, T)
    y_new += σ_move * randn(rng, T)
    
    # Store old position
    old_emitter = state[j]
    
    # Update state
    state[j] = Emitter2D(x_new, y_new, old_emitter.photons)
    
    # Compute new log probability
    log_prob_new = compute_log_posterior(state, allocations, observations, prior)
    
    # Metropolis-Hastings accept/reject
    log_ratio = log_prob_new - log_prob_current
    
    if log(rand(rng)) < log_ratio
        return true, log_prob_new
    else
        # Revert
        state[j] = old_emitter
        return false, log_prob_current
    end
end

"""
Add a new emitter.
"""
function add_emitter!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_obs = length(observations)
    n_obs == 0 && return false, log_prob_current
    
    # Sample position from prior (mixture of Gaussians at observations)
    i = rand(rng, 1:n_obs)
    obs = observations[i]
    
    # Handle uncertainty if available
    σ_x = hasproperty(obs, :σ_x) ? obs.σ_x : T(0.1)
    σ_y = hasproperty(obs, :σ_y) ? obs.σ_y : T(0.1)
    
    x_new = obs.x + σ_x * randn(rng, T)
    y_new = obs.y + σ_y * randn(rng, T)
    
    # Add new emitter
    new_emitter = Emitter2D(x_new, y_new, obs.photons)
    push!(state, new_emitter)
    
    # Reallocate observations
    reallocate_all!(allocations, state, observations, rng)
    
    # Compute new log probability
    log_prob_new = compute_log_posterior(state, allocations, observations, prior)
    
    # Compute acceptance ratio (includes dimension matching term)
    n_emitters_old = length(state) - 1
    n_emitters_new = length(state)
    
    log_ratio = log_prob_new - log_prob_current
    log_ratio += log(n_emitters_new) - log(n_obs)  # Proposal ratio
    
    if log(rand(rng)) < log_ratio
        return true, log_prob_new
    else
        # Revert
        pop!(state)
        reallocate_all!(allocations, state, observations, rng)
        return false, log_prob_current
    end
end

"""
Remove an emitter.
"""
function remove_emitter!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_emitters = length(state)
    n_emitters <= 1 && return false, log_prob_current
    
    # Select emitter to remove
    j = rand(rng, 1:n_emitters)
    
    # Store removed emitter
    removed = state[j]
    
    # Remove emitter
    deleteat!(state, j)
    
    # Update allocations
    for i in eachindex(allocations)
        if allocations[i] == j
            allocations[i] = 0  # Unallocated
        elseif allocations[i] > j
            allocations[i] -= 1  # Shift indices
        end
    end
    
    # Reallocate observations
    reallocate_all!(allocations, state, observations, rng)
    
    # Compute new log probability
    log_prob_new = compute_log_posterior(state, allocations, observations, prior)
    
    # Compute acceptance ratio
    n_emitters_old = n_emitters
    n_emitters_new = n_emitters - 1
    n_obs = length(observations)
    
    log_ratio = log_prob_new - log_prob_current
    log_ratio += log(n_obs) - log(n_emitters_old)  # Proposal ratio
    
    if log(rand(rng)) < log_ratio
        return true, log_prob_new
    else
        # Revert
        insert!(state, j, removed)
        reallocate_all!(allocations, state, observations, rng)
        return false, log_prob_current
    end
end

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

# Split and merge moves (simplified versions)

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

# Likelihood and posterior calculations

"""
Compute log likelihood of observation given emitter.
"""
@inline function log_likelihood(
    obs::O,
    emitter::Emitter2D{T}
) where {T<:AbstractFloat, O}
    
    # Handle different observation types
    if hasproperty(obs, :σ_x)
        # Full likelihood with uncertainties
        return -T(0.5) * (
            ((obs.x - emitter.x) / obs.σ_x)^2 + 
            ((obs.y - emitter.y) / obs.σ_y)^2 +
            log(2π * obs.σ_x * obs.σ_y)
        )
    else
        # Simple distance-based likelihood (assume σ = 0.1)
        σ = T(0.1)
        return -T(0.5) * (
            ((obs.x - emitter.x) / σ)^2 + 
            ((obs.y - emitter.y) / σ)^2 +
            log(2π * σ^2)
        )
    end
end

"""
Compute full log posterior probability.
"""
function compute_log_posterior(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T}
) where {T<:AbstractFloat, O}
    
    log_prob = zero(T)
    
    # Likelihood term
    @inbounds for i in eachindex(observations)
        j = allocations[i]
        if j > 0
            log_prob += log_likelihood(observations[i], state[j])
        else
            log_prob += -T(10.0)  # Penalty for unallocated
        end
    end
    
    # Prior on number of emitters (Poisson)
    n_emitters = length(state)
    n_obs = length(observations)
    λ_mean = prior.α / prior.β
    expected_emitters = n_obs / λ_mean
    
    log_prob += logpdf(Poisson(expected_emitters), n_emitters)
    
    # Prior on positions (improper uniform for now)
    # Could add proper spatial prior if needed
    
    return log_prob
end
# Core RJMCMC infrastructure - chains, initialization, and main loop
using Distributions
using Random
using StatsBase: sample, Weights
using SMLMData: Emitter2D

# Build spatial prior as mixture of Gaussians
function build_spatial_prior(observations::Vector{O}) where O
    n_obs = length(observations)
    means = Vector{Vector{Float64}}()
    covs = Vector{Matrix{Float64}}()
    
    for obs in observations
        σ_x = hasproperty(obs, :σ_x) ? obs.σ_x : 0.1
        σ_y = hasproperty(obs, :σ_y) ? obs.σ_y : 0.1
        
        push!(means, [obs.y, obs.x])  # Note: y, x order for MvNormal
        push!(covs, [σ_y^2 0.0; 0.0 σ_x^2])
    end
    
    components = [MvNormal(mean, cov) for (mean, cov) in zip(means, covs)]
    weights = fill(1.0/n_obs, n_obs)
    
    return MixtureModel(components, weights)
end

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
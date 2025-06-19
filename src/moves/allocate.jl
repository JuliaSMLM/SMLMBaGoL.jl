function propose_move(::Type{Allocate}, state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    length(state.emitters) == 0 && return nothing
    length(state.localizations) == 0 && return nothing
    
    new_state = deepcopy(state)
    
    # Select random localization to reallocate
    loc_idx = rand(rng, 1:length(state.localizations))
    current_allocation = state.allocations[loc_idx]
    
    # Compute likelihood-based probabilities for reallocation
    loc = state.localizations[loc_idx]
    log_probs = Vector{Float64}(undef, length(state.emitters))
    
    for (i, emitter) in enumerate(state.emitters)
        log_probs[i] = log_likelihood(emitter, loc)
    end
    
    # Convert to probabilities (subtract max for numerical stability)
    max_log_prob = maximum(log_probs)
    probs = exp.(log_probs .- max_log_prob)
    probs ./= sum(probs)
    
    # Sample new allocation
    cumulative = cumsum(probs)
    r = rand(rng)
    new_allocation = 1
    for i in eachindex(cumulative)
        if r <= cumulative[i]
            new_allocation = i
            break
        end
    end
    
    # If same allocation, still return valid state but likelihood won't change
    if new_allocation == current_allocation
        # No change needed, but still valid proposal
        new_state = BaGoLState(new_state.emitters, new_state.localizations, 
                              new_state.allocations, new_state.prior, 
                              state.log_likelihood)
        return new_state
    end
    
    # Update allocation
    new_state.allocations[loc_idx] = new_allocation
    
    # Recompute likelihood
    new_state = BaGoLState(new_state.emitters, new_state.localizations, 
                          new_state.allocations, new_state.prior, 
                          log_likelihood(new_state))
    
    return new_state
end

function log_acceptance_ratio(::Type{Allocate}, current::BaGoLState, proposed::BaGoLState)
    # Find which localization was reallocated
    reallocated_idx = 0
    for i in eachindex(current.allocations)
        if current.allocations[i] != proposed.allocations[i]
            reallocated_idx = i
            break
        end
    end
    
    reallocated_idx == 0 && return 0.0  # No reallocation
    
    loc = current.localizations[reallocated_idx]
    old_emitter_idx = current.allocations[reallocated_idx]
    new_emitter_idx = proposed.allocations[reallocated_idx]
    
    # Handle unallocated localizations (allocation = 0)
    # This shouldn't happen in normal operation, but check for safety
    if old_emitter_idx == 0 || new_emitter_idx == 0
        # Can't compute proper acceptance ratio for unallocated localizations
        # Accept the move with standard likelihood ratio
        return proposed.log_likelihood - current.log_likelihood
    end
    
    # Likelihood ratio (already computed in states)
    log_likelihood_ratio = proposed.log_likelihood - current.log_likelihood
    
    # Proposal ratio: q(old|new) / q(new|old)
    # Need to compute the probability of proposing the reverse move
    
    # Compute likelihood-based probabilities for current state
    log_probs_current = Vector{Float64}(undef, length(current.emitters))
    for (i, emitter) in enumerate(current.emitters)
        log_probs_current[i] = log_likelihood(emitter, loc)
    end
    max_log_prob_current = maximum(log_probs_current)
    probs_current = exp.(log_probs_current .- max_log_prob_current)
    probs_current ./= sum(probs_current)
    
    # Compute likelihood-based probabilities for proposed state  
    log_probs_proposed = Vector{Float64}(undef, length(proposed.emitters))
    for (i, emitter) in enumerate(proposed.emitters)
        log_probs_proposed[i] = log_likelihood(emitter, loc)
    end
    max_log_prob_proposed = maximum(log_probs_proposed)
    probs_proposed = exp.(log_probs_proposed .- max_log_prob_proposed)
    probs_proposed ./= sum(probs_proposed)
    
    # Proposal ratio: P(old_allocation | proposed_state) / P(new_allocation | current_state)
    q_old_given_proposed = probs_proposed[old_emitter_idx]
    q_new_given_current = probs_current[new_emitter_idx]
    
    log_proposal_ratio = log(q_old_given_proposed) - log(q_new_given_current)
    
    # No prior ratio since we're not changing the number of emitters
    return log_likelihood_ratio + log_proposal_ratio
end
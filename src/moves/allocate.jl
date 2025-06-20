function propose_move(::Type{Allocate}, state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    length(state.emitters) == 0 && return nothing
    length(state.localizations) == 0 && return nothing
    
    new_state = deepcopy(state)
    
    # Perform full Gibbs sweep: reallocate ALL localizations
    # This follows the approach in the refactor-rjmcmc branch
    log_probs = Vector{Float64}(undef, length(state.emitters))
    
    for loc_idx in 1:length(state.localizations)
        loc = state.localizations[loc_idx]
        
        # Compute likelihood-based probabilities for this localization
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
        
        # Update allocation for this localization
        new_state.allocations[loc_idx] = new_allocation
    end
    
    # Recompute likelihood after all allocations updated
    new_state = BaGoLState(new_state.emitters, new_state.localizations, 
                          new_state.allocations, new_state.prior, 
                          log_likelihood(new_state))
    
    return new_state
end

function log_acceptance_ratio(::Type{Allocate}, current::BaGoLState, proposed::BaGoLState)
    # For full Gibbs sweep of allocations, the move is its own inverse
    # The proposal probabilities cancel out in the acceptance ratio
    # This is because we're sampling from the full conditional distribution
    
    # Only the likelihood ratio matters for Gibbs sampling of allocations
    log_likelihood_ratio = proposed.log_likelihood - current.log_likelihood
    
    # For Gibbs sampling, the proposal is from the full conditional P(z|y,θ)
    # Since we sample all allocations from their conditionals, the forward and 
    # reverse proposal probabilities are equal, so they cancel in the ratio
    
    return log_likelihood_ratio
end
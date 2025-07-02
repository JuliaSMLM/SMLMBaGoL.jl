function propose_move(::Type{Allocate}, state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    length(state.emitters) == 0 && return nothing
    length(state.localizations) == 0 && return nothing
    
    new_state = deepcopy(state)
    
    # Count current allocations for each emitter (n_j in math spec)
    n_j = zeros(Int, length(state.emitters))
    for alloc in state.allocations
        if 1 ≤ alloc ≤ length(state.emitters)
            n_j[alloc] += 1
        end
    end
    
    # Extract κ from count prior
    κ = get_concentration_parameter(state.count_prior)
    
    # Perform full Gibbs sweep with Pólya weights
    log_probs = Vector{Float64}(undef, length(state.emitters))
    
    for loc_idx in 1:length(state.localizations)
        loc = state.localizations[loc_idx]
        
        # CRITICAL: Include Pólya weight (n_j + κ) as per math spec Section 3
        for (i, emitter) in enumerate(state.emitters)
            # w_ij ∝ (n_j + κ) × L_ij
            log_probs[i] = log(n_j[i] + κ) + log_likelihood(emitter, loc)
        end
        
        # Convert to probabilities (subtract max for numerical stability)
        max_log_prob = maximum(log_probs)
        probs = exp.(log_probs .- max_log_prob)
        probs ./= sum(probs)
        
        # Update n_j counts for real-time tracking
        old_alloc = state.allocations[loc_idx]
        if 1 ≤ old_alloc ≤ length(state.emitters)
            n_j[old_alloc] -= 1
        end
        
        # Sample new allocation using StatsBase.Weights for robust sampling
        new_allocation = sample(rng, 1:length(state.emitters), Weights(probs))
        new_state.allocations[loc_idx] = new_allocation
        n_j[new_allocation] += 1
    end
    
    # Recompute likelihood
    new_state = BaGoLState(new_state.emitters, new_state.localizations, 
                          new_state.allocations, new_state.spatial_prior,
                          new_state.count_prior, log_likelihood(new_state))
    
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
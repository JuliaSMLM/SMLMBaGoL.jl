function propose_move(::Type{Allocate}, state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    length(state.emitters) == 0 && return nothing
    length(state.localizations) == 0 && return nothing
    
    new_state = BaGoLState(
        copy(state.emitters),
        state.localizations,
        copy(state.allocations),
        copy(state.latent_positions),
        state.spatial_prior,
        state.count_prior,
        state.τ²,
        state.log_likelihood
    )
    
    # Pre-compute values for efficiency
    n_emitters = length(state.emitters)
    n_localizations = length(state.localizations)
    κ = get_concentration_parameter(state.count_prior)
    
    # Count current allocations for each emitter (n_j in math spec)
    n_j = zeros(Int, n_emitters)
    for alloc in state.allocations
        if 1 ≤ alloc ≤ n_emitters
            n_j[alloc] += 1
        end
    end
    
    # Pre-allocate arrays for efficiency
    log_probs = Vector{Float64}(undef, n_emitters)
    probs = Vector{Float64}(undef, n_emitters)
    
    # Perform full Gibbs sweep with Pólya weights
    for loc_idx in 1:n_localizations
        loc = state.localizations[loc_idx]
        
        # Update n_j counts for real-time tracking
        old_alloc = state.allocations[loc_idx]
        if 1 ≤ old_alloc ≤ n_emitters
            n_j[old_alloc] -= 1
        end
        
        # CRITICAL: Include Pólya weight (n_j + κ) as per math spec Section 3
        for (i, emitter) in enumerate(state.emitters)
            # w_ij ∝ (n_j + κ) × L_ij
            log_probs[i] = log(n_j[i] + κ) + log_likelihood(emitter, loc, state.τ²)
        end
        
        # Convert to probabilities (subtract max for numerical stability)
        max_log_prob = maximum(log_probs)
        @inbounds for i in 1:n_emitters
            probs[i] = exp(log_probs[i] - max_log_prob)
        end
        prob_sum = sum(probs)
        @inbounds for i in 1:n_emitters
            probs[i] /= prob_sum
        end
        
        # Sample new allocation - use simple cumulative sampling for speed
        r = rand(rng)
        cumulative = 0.0
        new_allocation = n_emitters  # fallback
        for i in 1:n_emitters
            cumulative += probs[i]
            if r ≤ cumulative
                new_allocation = i
                break
            end
        end
        
        new_state.allocations[loc_idx] = new_allocation
        n_j[new_allocation] += 1
    end
    
    # Recompute likelihood
    new_state = BaGoLState(new_state.emitters, new_state.localizations, 
                          new_state.allocations, new_state.latent_positions,
                          new_state.spatial_prior, new_state.count_prior, 
                          state.τ², log_likelihood(new_state))
    
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

# Allocate function with chain access (for interface consistency)
function propose_move(::Type{Allocate}, state::BaGoLState{E,L,T}, chain::RJMCMCChain, rng=Random.GLOBAL_RNG) where {E,L,T}
    # Allocate doesn't need the birth proposal, so just call the original method
    return propose_move(Allocate, state, rng)
end
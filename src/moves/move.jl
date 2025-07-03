function propose_move(::Type{Move}, state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    length(state.emitters) == 0 && return nothing
    
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
    
    # Select random emitter to move
    emitter_idx = rand(rng, 1:length(state.emitters))
    emitter = state.emitters[emitter_idx]
    
    # Find localizations allocated to this emitter
    allocated_locs = [state.localizations[i] for i in eachindex(state.localizations) 
                     if state.allocations[i] == emitter_idx]
    
    if isempty(allocated_locs)
        # No localizations allocated - propose random position from spatial prior
        x_new, y_new = sample_spatial_prior(state.spatial_prior, rng)
        new_emitter = E(x_new, y_new, emitter.photons)
    else
        # Get latent positions for this emitter
        latent_positions_for_emitter = [state.latent_positions[i] for i in eachindex(state.localizations) 
                                        if state.allocations[i] == emitter_idx]
        
        # Calculate mean of latent positions (Section 4: r̄_j)
        x_mean = mean(pos[1] for pos in latent_positions_for_emitter)
        y_mean = mean(pos[2] for pos in latent_positions_for_emitter)
        
        # Sample from posterior N(r̄_j, τ²/n_j I)
        n_j = length(latent_positions_for_emitter)
        x_new = x_mean + randn(rng) * sqrt(state.τ² / n_j)
        y_new = y_mean + randn(rng) * sqrt(state.τ² / n_j)
        
        new_emitter = E(x_new, y_new, emitter.photons)
    end
    
    # Update emitter in new state
    new_state.emitters[emitter_idx] = new_emitter
    
    # Update latent positions for localizations assigned to moved emitter
    for loc_idx in eachindex(state.localizations)
        if new_state.allocations[loc_idx] == emitter_idx
            loc = state.localizations[loc_idx]
            
            # Sample from posterior given new emitter position
            prec_x = 1/state.τ² + 1/loc.σx^2
            prec_y = 1/state.τ² + 1/loc.σy^2
            post_mean_x = (new_emitter.x/state.τ² + loc.x/loc.σx^2) / prec_x
            post_mean_y = (new_emitter.y/state.τ² + loc.y/loc.σy^2) / prec_y
            
            latent_x = post_mean_x + randn(rng) / sqrt(prec_x)
            latent_y = post_mean_y + randn(rng) / sqrt(prec_y)
            
            new_state.latent_positions[loc_idx] = (latent_x, latent_y)
        end
    end
    
    # Recompute likelihood
    new_state = BaGoLState(new_state.emitters, new_state.localizations, 
                          new_state.allocations, new_state.latent_positions,
                          new_state.spatial_prior, new_state.count_prior, 
                          state.τ², log_likelihood(new_state))
    
    return new_state
end

function log_acceptance_ratio(::Type{Move}, current::BaGoLState, proposed::BaGoLState)
    # Find which emitter was moved
    moved_idx = 0
    for i in eachindex(current.emitters)
        if current.emitters[i].x != proposed.emitters[i].x ||
           current.emitters[i].y != proposed.emitters[i].y
            moved_idx = i
            break
        end
    end
    
    moved_idx == 0 && return 0.0  # No emitter moved
    
    current_emitter = current.emitters[moved_idx]
    proposed_emitter = proposed.emitters[moved_idx]
    
    # Prior ratio for the moved emitter
    log_prior_ratio = log_prior_spatial(proposed_emitter, current.spatial_prior) -
                     log_prior_spatial(current_emitter, current.spatial_prior)
    
    # Likelihood ratio 
    log_likelihood_ratio = proposed.log_likelihood - current.log_likelihood
    
    # For symmetric proposals (Gaussian perturbations), proposal ratio = 1
    # For moves to/from prior, we need to account for proposal probabilities
    allocated_locs_current = [current.localizations[i] for i in eachindex(current.localizations) 
                             if current.allocations[i] == moved_idx]
    allocated_locs_proposed = [proposed.localizations[i] for i in eachindex(proposed.localizations) 
                              if proposed.allocations[i] == moved_idx]
    
    # If allocation didn't change, proposal is symmetric
    if length(allocated_locs_current) == length(allocated_locs_proposed) &&
       all(allocated_locs_current .== allocated_locs_proposed)
        log_proposal_ratio = 0.0
    else
        # Complex case - would need to compute actual proposal probabilities
        # For now, assume symmetric
        log_proposal_ratio = 0.0
    end
    
    return log_prior_ratio + log_likelihood_ratio + log_proposal_ratio
end

# Move function with chain access (for interface consistency)
function propose_move(::Type{Move}, state::BaGoLState{E,L,T}, chain::RJMCMCChain, rng=Random.GLOBAL_RNG) where {E,L,T}
    # Move doesn't need the birth proposal, so just call the original method
    return propose_move(Move, state, rng)
end
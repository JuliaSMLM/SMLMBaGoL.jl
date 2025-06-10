function propose_move(::Type{Move}, state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    length(state.emitters) == 0 && return nothing
    
    new_state = deepcopy(state)
    
    # Select random emitter to move
    emitter_idx = rand(rng, 1:length(state.emitters))
    emitter = state.emitters[emitter_idx]
    
    # Find localizations allocated to this emitter
    allocated_locs = [state.localizations[i] for i in eachindex(state.localizations) 
                     if state.allocations[i] == emitter_idx]
    
    if isempty(allocated_locs)
        # No localizations allocated - propose random position from spatial prior
        spatial_prior = isa(state.prior, CompoundPrior) ? state.prior.spatial_prior :
                       create_spatial_prior_from_localizations(state.localizations)
        x_new, y_new = sample_spatial_prior(spatial_prior, rng)
        new_emitter = E(x_new, y_new, emitter.σx, emitter.σy, emitter.id)
    else
        # Propose new position based on allocated localizations with some noise
        # Use weighted average of allocated localizations as center
        x_center = sum(loc.x / (loc.σx^2) for loc in allocated_locs) / 
                  sum(1 / (loc.σx^2) for loc in allocated_locs)
        y_center = sum(loc.y / (loc.σy^2) for loc in allocated_locs) / 
                  sum(1 / (loc.σy^2) for loc in allocated_locs)
        
        # Add small random perturbation
        avg_sigma_x = sqrt(sum(loc.σx^2 for loc in allocated_locs) / length(allocated_locs))
        avg_sigma_y = sqrt(sum(loc.σy^2 for loc in allocated_locs) / length(allocated_locs))
        
        x_new = x_center + randn(rng) * avg_sigma_x * 0.5
        y_new = y_center + randn(rng) * avg_sigma_y * 0.5
        
        new_emitter = E(x_new, y_new, emitter.σx, emitter.σy, emitter.id)
    end
    
    # Update emitter in new state
    new_state.emitters[emitter_idx] = new_emitter
    
    # Recompute likelihood
    new_state = BaGoLState(new_state.emitters, new_state.localizations, 
                          new_state.allocations, new_state.prior, 
                          log_likelihood(new_state))
    
    return new_state
end

function log_acceptance_ratio(::Type{Move}, current::BaGoLState, proposed::BaGoLState)
    # Find which emitter was moved
    moved_idx = 0
    for i in eachindex(current.emitters)
        if current.emitters[i].id != proposed.emitters[i].id || 
           current.emitters[i].x != proposed.emitters[i].x ||
           current.emitters[i].y != proposed.emitters[i].y
            moved_idx = i
            break
        end
    end
    
    moved_idx == 0 && return 0.0  # No emitter moved
    
    current_emitter = current.emitters[moved_idx]
    proposed_emitter = proposed.emitters[moved_idx]
    
    # Prior ratio for the moved emitter
    log_prior_ratio = log_prior_spatial(proposed_emitter, 
                                       isa(current.prior, CompoundPrior) ? 
                                       current.prior.spatial_prior :
                                       create_spatial_prior_from_localizations(current.localizations)) -
                     log_prior_spatial(current_emitter,
                                      isa(current.prior, CompoundPrior) ? 
                                      current.prior.spatial_prior :
                                      create_spatial_prior_from_localizations(current.localizations))
    
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
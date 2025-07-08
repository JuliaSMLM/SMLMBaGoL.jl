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
    
    # Move all emitters
    for emitter_idx in 1:length(state.emitters)
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
            
            # Safety check to prevent NaN from division by zero or negative variance
            if n_j == 0 || state.τ² <= 0
                # Fallback to spatial prior if we can't compute posterior properly
                x_new, y_new = sample_spatial_prior(state.spatial_prior, rng)
            else
                variance = state.τ² / n_j
                std_dev = sqrt(variance)
                x_new = x_mean + randn(rng) * std_dev
                y_new = y_mean + randn(rng) * std_dev
            end
            
            new_emitter = E(x_new, y_new, emitter.photons)
        end
        
        # Final safety check for emitter position
        if !isfinite(new_emitter.x) || !isfinite(new_emitter.y)
            # Fallback to original position if new position is invalid
            new_emitter = emitter
        end
        
        # Update emitter in new state
        new_state.emitters[emitter_idx] = new_emitter
    end
    
    # Update all latent positions given new emitter positions
    for loc_idx in eachindex(state.localizations)
        alloc_idx = new_state.allocations[loc_idx]
        if alloc_idx > 0  # If localization is allocated to an emitter
            loc = state.localizations[loc_idx]
            emitter = new_state.emitters[alloc_idx]
            
            # Sample from posterior given new emitter position
            # Safety checks to prevent division by zero and NaN
            if state.τ² <= 0 || loc.σx <= 0 || loc.σy <= 0
                # Fallback: use localization position if uncertainties are invalid
                latent_x = loc.x
                latent_y = loc.y
            else
                prec_x = 1/state.τ² + 1/loc.σx^2
                prec_y = 1/state.τ² + 1/loc.σy^2
                
                # Additional safety check for precision values
                if prec_x <= 0 || prec_y <= 0 || !isfinite(prec_x) || !isfinite(prec_y)
                    latent_x = loc.x
                    latent_y = loc.y
                else
                    post_mean_x = (emitter.x/state.τ² + loc.x/loc.σx^2) / prec_x
                    post_mean_y = (emitter.y/state.τ² + loc.y/loc.σy^2) / prec_y
                    
                    latent_x = post_mean_x + randn(rng) / sqrt(prec_x)
                    latent_y = post_mean_y + randn(rng) / sqrt(prec_y)
                end
            end
            
            # Final safety check for latent positions
            if !isfinite(latent_x) || !isfinite(latent_y)
                # Fallback to original latent position
                latent_x, latent_y = state.latent_positions[loc_idx]
            end
            
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
    # For Gibbs sampling, we always accept (acceptance ratio = 1, log = 0)
    # This is because we're sampling from the full conditionals
    return 0.0
end

# Move function with chain access (for interface consistency)
function propose_move(::Type{Move}, state::BaGoLState{E,L,T}, chain::RJMCMCChain, rng=Random.GLOBAL_RNG) where {E,L,T}
    # Move doesn't need the birth proposal, so just call the original method
    return propose_move(Move, state, rng)
end
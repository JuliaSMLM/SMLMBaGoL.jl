"""
Gibbs update for latent positions.
This should be called once per RJMCMC iteration, not after every other move.
"""
function propose_move(::Type{UpdateLatent}, state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    # Create new latent positions array
    new_latent_positions = similar(state.latent_positions)
    
    for i in eachindex(state.localizations)
        loc = state.localizations[i]
        emitter_idx = state.allocations[i]
        
        if 1 ≤ emitter_idx ≤ length(state.emitters)
            emitter = state.emitters[emitter_idx]
            
            # Posterior precision (inverse variance)
            prec_x = 1/state.τ² + 1/loc.σx^2
            prec_y = 1/state.τ² + 1/loc.σy^2
            
            # Posterior mean (precision-weighted average)
            post_mean_x = (emitter.x/state.τ² + loc.x/loc.σx^2) / prec_x
            post_mean_y = (emitter.y/state.τ² + loc.y/loc.σy^2) / prec_y
            
            # Sample from posterior
            latent_x = post_mean_x + randn(rng) / sqrt(prec_x)
            latent_y = post_mean_y + randn(rng) / sqrt(prec_y)
            
            new_latent_positions[i] = (latent_x, latent_y)
        else
            # Unallocated: keep at observed position
            new_latent_positions[i] = (loc.x, loc.y)
        end
    end
    
    # Return new state with updated latent positions
    return BaGoLState(
        state.emitters,
        state.localizations,
        state.allocations,
        new_latent_positions,  # Only this changes
        state.spatial_prior,
        state.count_prior,
        state.τ²,
        state.log_likelihood  # Likelihood unchanged by latent update
    )
end

# Latent updates are always accepted (Gibbs sampling)
function log_acceptance_ratio(::Type{UpdateLatent}, current::BaGoLState, proposed::BaGoLState)
    return 0.0  # Always accept (log(1) = 0)
end

# UpdateLatent function with chain access (for interface consistency)
function propose_move(::Type{UpdateLatent}, state::BaGoLState{E,L,T}, chain::RJMCMCChain, rng=Random.GLOBAL_RNG) where {E,L,T}
    # UpdateLatent doesn't need the birth proposal, so just call the original method
    return propose_move(UpdateLatent, state, rng)
end
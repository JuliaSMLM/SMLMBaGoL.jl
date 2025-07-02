# Log-likelihood for single emitter-localization pair with τ²
function log_likelihood(emitter::AbstractEmitter, loc::AbstractLocalization, τ²::Real)
    # Total variance is observed variance plus additional systematic variance
    σx_total² = loc.σx^2 + τ²
    σy_total² = loc.σy^2 + τ²
    
    # Normalized squared distances
    dx² = (loc.x - emitter.x)^2 / σx_total²
    dy² = (loc.y - emitter.y)^2 / σy_total²
    
    # Log of normalizing constant
    log_norm = 0.5 * (log(2π) + log(σx_total²) + log(2π) + log(σy_total²))
    
    return -0.5 * (dx² + dy²) - log_norm
end

# Log-likelihood for entire state (with τ²)
function log_likelihood(state::BaGoLState)
    ll = 0.0
    for (i, loc) in enumerate(state.localizations)
        emitter_idx = state.allocations[i]
        if 1 ≤ emitter_idx ≤ length(state.emitters)
            ll += log_likelihood(state.emitters[emitter_idx], loc, state.τ²)
        end
    end
    return ll
end
function log_likelihood(state::BaGoLState)
    ll = 0.0
    for (i, loc) in enumerate(state.localizations)
        emitter_idx = state.allocations[i]
        if 1 ≤ emitter_idx ≤ length(state.emitters)
            ll += log_likelihood(state.emitters[emitter_idx], loc)
        end
    end
    return ll
end
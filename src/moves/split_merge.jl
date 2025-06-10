function propose_move(::Type{Split}, state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    length(state.emitters) == 0 && return nothing
    
    # Randomly select one emitter to split
    split_idx = rand(rng, 1:length(state.emitters))
    emitter_to_split = state.emitters[split_idx]
    
    # Create two new emitters at the same position as the original
    # They will be separated during reallocation
    # Inherit uncertainty from original emitter
    new_emitter1 = E(emitter_to_split.x, emitter_to_split.y, emitter_to_split.σx, emitter_to_split.σy, rand(rng, 1:1000000))
    new_emitter2 = E(emitter_to_split.x, emitter_to_split.y, emitter_to_split.σx, emitter_to_split.σy, rand(rng, 1:1000000))
    
    # Create new emitter list by replacing split emitter with two new ones
    new_emitters = Vector{E}(undef, length(state.emitters) + 1)
    new_emitters[1:split_idx-1] = state.emitters[1:split_idx-1]
    new_emitters[split_idx] = new_emitter1
    new_emitters[split_idx+1] = new_emitter2
    new_emitters[split_idx+2:end] = state.emitters[split_idx+1:end]
    
    # Create temporary state for reallocation
    temp_state = BaGoLState(new_emitters, state.localizations, state.allocations, state.prior, state.log_likelihood)
    
    # Reallocate all localizations optimally to new emitter set
    new_allocations = propose_reallocation(temp_state, rng)
    
    # Create final state with updated likelihood
    new_state = BaGoLState(new_emitters, state.localizations, new_allocations, state.prior, T(0.0))
    new_likelihood = log_likelihood(new_state)
    
    return BaGoLState(new_emitters, state.localizations, new_allocations, state.prior, new_likelihood)
end

function propose_move(::Type{Merge}, state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    length(state.emitters) < 2 && return nothing
    
    # Randomly select first emitter
    first_idx = rand(rng, 1:length(state.emitters))
    first_emitter = state.emitters[first_idx]
    
    # Find nearest neighbor using Euclidean distance
    min_distance = Inf
    second_idx = -1
    for i in 1:length(state.emitters)
        if i != first_idx
            distance = sqrt((state.emitters[i].x - first_emitter.x)^2 + 
                          (state.emitters[i].y - first_emitter.y)^2)
            if distance < min_distance
                min_distance = distance
                second_idx = i
            end
        end
    end
    
    second_emitter = state.emitters[second_idx]
    
    # Create merged emitter at average position with average uncertainty
    merged_x = (first_emitter.x + second_emitter.x) / 2
    merged_y = (first_emitter.y + second_emitter.y) / 2
    merged_σx = (first_emitter.σx + second_emitter.σx) / 2
    merged_σy = (first_emitter.σy + second_emitter.σy) / 2
    merged_emitter = E(merged_x, merged_y, merged_σx, merged_σy, rand(rng, 1:1000000))
    
    # Create new emitter list by removing both emitters and adding merged one
    indices_to_keep = [i for i in 1:length(state.emitters) if i != first_idx && i != second_idx]
    new_emitters = [state.emitters[indices_to_keep]; merged_emitter]
    
    # Create temporary state for reallocation
    temp_state = BaGoLState(new_emitters, state.localizations, state.allocations, state.prior, state.log_likelihood)
    
    # Reallocate all localizations optimally to new emitter set
    new_allocations = propose_reallocation(temp_state, rng)
    
    # Create final state with updated likelihood
    new_state = BaGoLState(new_emitters, state.localizations, new_allocations, state.prior, T(0.0))
    new_likelihood = log_likelihood(new_state)
    
    return BaGoLState(new_emitters, state.localizations, new_allocations, state.prior, new_likelihood)
end

function log_acceptance_ratio_split(current::BaGoLState, proposed::BaGoLState)
    @assert length(proposed.emitters) == length(current.emitters) + 1
    
    # Extract priors
    K_prior = current.prior.K_prior
    
    # Prior ratio: P(K+1)/P(K)
    n = length(current.emitters)
    log_prior_ratio = log_prior_K(n + 1, K_prior) - log_prior_K(n, K_prior)
    
    # Likelihood ratio
    log_likelihood_ratio = proposed.log_likelihood - current.log_likelihood
    
    # Proposal ratio: split selects 1/n, merge selects 1/(n+1)  
    # But we also need to account for the reallocation randomness
    # For now, assume the reallocation is deterministic given the emitter positions
    # q(merge)/q(split) = (1/(n+1)) / (1/n) = n/(n+1)
    log_proposal_ratio = log(n) - log(n + 1)
    
    # Note: This is a simplified version. Full detailed balance would require
    # accounting for the Jacobian of the position transformation and 
    # the allocation probability ratios from the refactor-rjmcmc mathematics
    
    return log_prior_ratio + log_likelihood_ratio + log_proposal_ratio
end

function log_acceptance_ratio(::Type{Split}, current::BaGoLState, proposed::BaGoLState)
    log_acceptance_ratio_split(current, proposed)
end

function log_acceptance_ratio(::Type{Merge}, current::BaGoLState, proposed::BaGoLState)
    -log_acceptance_ratio_split(proposed, current)  # Mathematical inverse!
end
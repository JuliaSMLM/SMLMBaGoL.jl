function propose_move(::Type{Birth}, state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    # Extract spatial prior from compound prior
    spatial_prior = isa(state.prior, CompoundPrior) ? state.prior.spatial_prior : 
                   error("Birth move requires CompoundPrior with spatial component")
    
    # Sample new emitter from spatial prior
    new_emitter = sample_emitter_from_prior(state.localizations, E, spatial_prior, rng)
    new_emitters = [state.emitters; new_emitter]
    
    # Create temporary state for reallocation
    temp_state = BaGoLState(new_emitters, state.localizations, state.allocations, state.prior, state.log_likelihood)
    
    # Reallocate all localizations given new emitter set
    new_allocations = propose_reallocation(temp_state, rng)
    
    # Create final state with updated likelihood
    new_state = BaGoLState(new_emitters, state.localizations, new_allocations, state.prior, T(0.0))
    new_likelihood = log_likelihood(new_state)
    
    return BaGoLState(new_emitters, state.localizations, new_allocations, state.prior, new_likelihood)
end

function propose_move(::Type{Death}, state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    length(state.emitters) == 0 && return nothing
    
    idx = rand(rng, 1:length(state.emitters))
    new_emitters = [state.emitters[i] for i in 1:length(state.emitters) if i != idx]
    new_allocations = reallocate_from_removed(state.allocations, idx, rng)
    
    # Create final state with updated likelihood
    new_state = BaGoLState(new_emitters, state.localizations, new_allocations, state.prior, T(0.0))
    new_likelihood = log_likelihood(new_state)
    
    return BaGoLState(new_emitters, state.localizations, new_allocations, state.prior, new_likelihood)
end

function log_acceptance_ratio_birth(current::BaGoLState, proposed::BaGoLState)
    @assert length(proposed.emitters) == length(current.emitters) + 1
    
    # Extract priors
    spatial_prior = current.prior.spatial_prior
    K_prior = current.prior.K_prior
    
    # Prior ratio: P(K+1)/P(K) 
    log_prior_ratio = log_prior_K(length(proposed.emitters), K_prior) - 
                      log_prior_K(length(current.emitters), K_prior)
    
    # Likelihood ratio
    log_likelihood_ratio = proposed.log_likelihood - current.log_likelihood
    
    # Proposal ratio: q(death)/q(birth)
    # q(death) = 1/|K+1| (uniform selection from K+1 emitters)
    # q(birth) = 1/Area (uniform sampling from spatial prior)
    new_emitter = proposed.emitters[end]
    log_q_death = -log(length(proposed.emitters))
    log_q_birth = log_spatial_prior_density(new_emitter, spatial_prior)
    
    return log_prior_ratio + log_likelihood_ratio + log_q_death - log_q_birth
end

function log_acceptance_ratio(::Type{Birth}, current::BaGoLState, proposed::BaGoLState)
    log_acceptance_ratio_birth(current, proposed)
end

function log_acceptance_ratio(::Type{Death}, current::BaGoLState, proposed::BaGoLState)
    -log_acceptance_ratio_birth(proposed, current)  # Mathematical inverse!
end
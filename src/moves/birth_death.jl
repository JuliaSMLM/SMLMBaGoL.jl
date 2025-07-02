"""
Birth and Death moves with integrated position optimization.

These moves add or remove emitters from the model, with automatic position
optimization after the structural change to improve acceptance rates.
"""

function propose_move(::Type{Birth}, state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    # Sample new emitter from spatial prior
    new_emitter = sample_emitter_from_prior(state.localizations, E, state.spatial_prior, rng)
    new_emitters = [state.emitters; new_emitter]
    
    # Create temporary state for reallocation
    temp_state = BaGoLState(new_emitters, state.localizations, state.allocations, 
                           state.spatial_prior, state.count_prior, state.log_likelihood)
    
    # Reallocate all localizations given new emitter set
    new_allocations = propose_reallocation(temp_state, rng)
    
    # Create state after allocation
    allocated_state = BaGoLState(new_emitters, state.localizations, new_allocations, 
                                state.spatial_prior, state.count_prior, T(0.0))
    
    # Now optimize the position of the new emitter using allocated localizations
    new_emitter_idx = length(new_emitters)
    allocated_locs = [state.localizations[i] for i in eachindex(state.localizations) 
                     if new_allocations[i] == new_emitter_idx]
    
    if !isempty(allocated_locs)
        # Calculate optimal position based on allocated localizations
        # This is the posterior mean given the allocated localizations
        x_precision_sum = sum(1 / (loc.σx^2) for loc in allocated_locs)
        y_precision_sum = sum(1 / (loc.σy^2) for loc in allocated_locs)
        
        x_mean = sum(loc.x / (loc.σx^2) for loc in allocated_locs) / x_precision_sum
        y_mean = sum(loc.y / (loc.σy^2) for loc in allocated_locs) / y_precision_sum
        
        # Update emitter position to posterior mean
        optimized_emitter = E(x_mean, y_mean, new_emitter.photons)
        new_emitters[new_emitter_idx] = optimized_emitter
    end
    
    # Create final state with optimized position and updated likelihood
    new_likelihood = log_likelihood(BaGoLState(new_emitters, state.localizations, 
                                              new_allocations, state.spatial_prior, 
                                              state.count_prior, T(0.0)))
    
    return BaGoLState(new_emitters, state.localizations, new_allocations, 
                     state.spatial_prior, state.count_prior, new_likelihood)
end

function propose_move(::Type{Death}, state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    length(state.emitters) == 0 && return nothing
    
    idx = rand(rng, 1:length(state.emitters))
    
    # Before removing, find localizations allocated to this emitter
    allocated_to_removed = [i for i in eachindex(state.allocations) 
                           if state.allocations[i] == idx]
    
    # Remove the emitter
    new_emitters = [state.emitters[i] for i in 1:length(state.emitters) if i != idx]
    new_allocations = reallocate_from_removed(state.allocations, idx, rng)
    
    # If there are remaining emitters and some localizations need reallocation
    if !isempty(new_emitters) && !isempty(allocated_to_removed)
        # Create temporary state
        temp_state = BaGoLState(new_emitters, state.localizations, new_allocations, 
                               state.spatial_prior, state.count_prior, T(0.0))
        
        # Find which emitters received the reallocated localizations
        emitters_to_optimize = unique([new_allocations[i] for i in allocated_to_removed 
                                      if new_allocations[i] > 0])
        
        # Optimize positions of emitters that received new localizations
        for emitter_idx in emitters_to_optimize
            allocated_locs = [state.localizations[i] for i in eachindex(state.localizations) 
                            if new_allocations[i] == emitter_idx]
            
            if !isempty(allocated_locs)
                # Calculate optimal position
                x_precision_sum = sum(1 / (loc.σx^2) for loc in allocated_locs)
                y_precision_sum = sum(1 / (loc.σy^2) for loc in allocated_locs)
                
                x_mean = sum(loc.x / (loc.σx^2) for loc in allocated_locs) / x_precision_sum
                y_mean = sum(loc.y / (loc.σy^2) for loc in allocated_locs) / y_precision_sum
                
                # Update emitter position
                old_emitter = new_emitters[emitter_idx]
                new_emitters[emitter_idx] = E(x_mean, y_mean, old_emitter.photons)
            end
        end
    end
    
    # Create final state with updated likelihood
    new_likelihood = log_likelihood(BaGoLState(new_emitters, state.localizations, 
                                              new_allocations, state.spatial_prior, 
                                              state.count_prior, T(0.0)))
    
    return BaGoLState(new_emitters, state.localizations, new_allocations, 
                     state.spatial_prior, state.count_prior, new_likelihood)
end

# The acceptance ratios account for the position optimization
function log_acceptance_ratio_birth(current::BaGoLState, proposed::BaGoLState)
    @assert length(proposed.emitters) == length(current.emitters) + 1
    
    # Spatial prior ratio for new emitter position
    new_emitter = proposed.emitters[end]
    log_spatial_ratio = log_prior_spatial(new_emitter, current.spatial_prior)
    
    # Likelihood ratio
    log_likelihood_ratio = proposed.log_likelihood - current.log_likelihood
    
    # Proposal ratio (position sampling)
    log_q_death = -log(length(proposed.emitters))  # 1/(K+1)
    log_q_birth = log_spatial_prior_density(new_emitter, current.spatial_prior)
    
    # Note: The Dirichlet-multinomial terms are already incorporated
    # through the Pólya-weighted allocation process
    
    return log_spatial_ratio + log_likelihood_ratio + log_q_death - log_q_birth
end

function log_acceptance_ratio(::Type{Birth}, current::BaGoLState, proposed::BaGoLState)
    log_acceptance_ratio_birth(current, proposed)
end

function log_acceptance_ratio(::Type{Death}, current::BaGoLState, proposed::BaGoLState)
    -log_acceptance_ratio_birth(proposed, current)  # Mathematical inverse!
end
function sample_emitter_from_prior(localizations::Vector{<:AbstractLocalization}, 
                                   EmitterType::Type{<:AbstractEmitter}, 
                                   spatial_prior::UniformSpatialPrior,
                                   rng=Random.GLOBAL_RNG)
    x, y = sample_spatial_prior(spatial_prior, rng)
    # For SMLMData's Emitter2D, we need photons not uncertainties
    # Use a default photon count based on localizations
    photons = isempty(localizations) ? 1000.0 : 
              mean(loc.σx > 0 ? 1000.0 : 1000.0 for loc in localizations)  # Placeholder
    return EmitterType(x, y, photons)
end

function sample_emitter_from_prior(localizations::Vector{<:AbstractLocalization}, 
                                   EmitterType::Type{Emitter2D{T}}, 
                                   spatial_prior::UniformSpatialPrior,
                                   rng=Random.GLOBAL_RNG) where T
    x, y = sample_spatial_prior(spatial_prior, rng)
    # For SMLMData's Emitter2D, use default photon count
    photons = T(1000.0)  # Default photon count
    return Emitter2D{T}(T(x), T(y), photons)
end

# log_spatial_prior_density function is defined in priors.jl

function propose_reallocation(state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    n_emitters = length(state.emitters)
    n_localizations = length(state.localizations)
    
    new_allocations = Vector{Int}(undef, n_localizations)
    
    for i in 1:n_localizations
        if n_emitters == 0
            new_allocations[i] = 0  # No emitters to allocate to
        else
            # Calculate likelihood for each emitter
            log_probs = Vector{T}(undef, n_emitters)
            for j in 1:n_emitters
                log_probs[j] = log_likelihood(state.emitters[j], state.localizations[i])
            end
            
            # Convert to probabilities and sample
            max_log_prob = maximum(log_probs)
            probs = exp.(log_probs .- max_log_prob)
            probs ./= sum(probs)
            
            new_allocations[i] = sample(rng, 1:n_emitters, Weights(probs))
        end
    end
    
    return new_allocations
end

function reallocate_from_removed(allocations::Vector{Int}, removed_idx::Int, rng=Random.GLOBAL_RNG)
    new_allocations = copy(allocations)
    
    # Shift indices down for emitters after the removed one
    for i in eachindex(new_allocations)
        if new_allocations[i] > removed_idx
            new_allocations[i] -= 1
        elseif new_allocations[i] == removed_idx
            # Reallocate localizations that were assigned to removed emitter
            # For now, assign to emitter 1 if available, otherwise 0
            new_allocations[i] = length(unique(allocations)) > 1 ? 1 : 0
        end
    end
    
    return new_allocations
end

"""
    remove_empty_emitters(state::BaGoLState{E,L,T}) where {E,L,T}

Remove emitters that have no allocated localizations.

Returns a new BaGoLState with:
- Empty emitters removed from the emitters vector
- Allocation indices updated to reflect the removal
- All other state components preserved

# Example
```julia
cleaned_state = remove_empty_emitters(state)
```
"""
function remove_empty_emitters(state::BaGoLState{E,L,T}) where {E,L,T}
    # Count allocations for each emitter
    n_emitters = length(state.emitters)
    allocation_counts = zeros(Int, n_emitters)
    
    for alloc in state.allocations
        if 1 ≤ alloc ≤ n_emitters
            allocation_counts[alloc] += 1
        end
    end
    
    # Find non-empty emitters
    non_empty_indices = findall(count -> count > 0, allocation_counts)
    
    # If all emitters have localizations, return state unchanged
    if length(non_empty_indices) == n_emitters
        return state
    end
    
    # Create mapping from old indices to new indices
    old_to_new = Dict{Int,Int}()
    for (new_idx, old_idx) in enumerate(non_empty_indices)
        old_to_new[old_idx] = new_idx
    end
    
    # Filter emitters
    new_emitters = state.emitters[non_empty_indices]
    
    # Update allocations
    new_allocations = similar(state.allocations)
    for (i, old_alloc) in enumerate(state.allocations)
        if haskey(old_to_new, old_alloc)
            new_allocations[i] = old_to_new[old_alloc]
        else
            # This shouldn't happen for valid allocations, but handle it gracefully
            new_allocations[i] = old_alloc > n_emitters ? old_alloc : 0
        end
    end
    
    # Create new state with updated emitters and allocations
    new_state = BaGoLState(
        new_emitters,
        state.localizations,
        new_allocations,
        state.latent_positions,
        state.spatial_prior,
        state.count_prior,
        state.τ²,
        state.log_likelihood  # Will be recalculated if needed
    )
    
    return new_state
end
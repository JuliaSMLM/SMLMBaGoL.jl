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

function log_spatial_prior_density(emitter::AbstractEmitter, spatial_prior::UniformSpatialPrior)
    return log_prior_spatial(emitter, spatial_prior)
end

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
# RJMCMC split and merge emitter moves
using Random
using Distributions
using SMLMData: Emitter2D

"""
Split an emitter into two emitters following refactor-rjmcmc mathematical approach.

The split move takes one emitter and creates two emitters at the same initial position.
This maintains detailed balance with the merge move through proper acceptance ratios.
"""
function split_emitter!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_emitters = length(state)
    n_emitters == 0 && return false, log_prob_current
    
    # Select emitter to split
    split_id = rand(rng, 1:n_emitters)
    original_emitter = state[split_id]
    
    # Store old state for reversion
    old_allocations = copy(allocations)
    
    # Create two new emitters at the same position as the original
    # Following refactor-rjmcmc approach: start with identical positions
    emitter1 = Emitter2D(original_emitter.x, original_emitter.y, original_emitter.photons / 2)
    emitter2 = Emitter2D(original_emitter.x, original_emitter.y, original_emitter.photons / 2)
    
    # Replace the original emitter with two new ones
    state[split_id] = emitter1
    push!(state, emitter2)
    
    # Reallocate observations to account for new emitter
    reallocate_all!(allocations, state, observations, rng)
    
    # Compute new log probability
    log_prob_new = compute_log_posterior(state, allocations, observations, prior)
    
    # RJMCMC acceptance ratio following refactor-rjmcmc mathematical approach
    n_obs = length(observations)
    n_emitters_old = n_emitters
    n_emitters_new = n_emitters + 1
    
    # Prior ratio for number of emitters (using Poisson model)
    λ_mean = prior.α / prior.β
    expected_emitters = n_obs / λ_mean
    prior_ratio_k = logpdf(Poisson(expected_emitters), n_emitters_new) - 
                    logpdf(Poisson(expected_emitters), n_emitters_old)
    
    # Proposal ratio: probability of selecting this split vs reverse merge
    # Split: select 1 out of n_old emitters to split
    # Merge: select 1 out of n_new emitters, then find its nearest neighbor
    proposal_ratio = log(n_emitters_old) - log(n_emitters_new)
    
    # Prior proposal ratio
    prior_proposal_ratio = prior_ratio_k + proposal_ratio
    
    # Likelihood ratio for positions
    likelihood_ratio_position = log_prob_new - log_prob_current
    
    # Likelihood ratio for number of localizations (Dirichlet-multinomial)
    # Compare old state (before split) with new state (after split)
    old_state = copy(state)
    old_state[split_id] = original_emitter
    pop!(old_state)  # Remove the extra emitter we added
    
    likelihood_ratio_number = log_dirichlet_multinomial_pmf(state, allocations, prior) - 
                             log_dirichlet_multinomial_pmf(old_state, old_allocations, prior)
    
    # Combined acceptance ratio
    log_ratio = prior_proposal_ratio + likelihood_ratio_position + likelihood_ratio_number
    
    if log(rand(rng)) < log_ratio
        return true, log_prob_new
    else
        # Revert to original state
        pop!(state)  # Remove the added emitter
        state[split_id] = original_emitter
        copy!(allocations, old_allocations)
        return false, log_prob_current
    end
end

"""
Merge two emitters into one following refactor-rjmcmc mathematical approach.

The merge move selects one emitter, finds its nearest neighbor, and merges them
by averaging their positions and combining their photon counts.
"""
function merge_emitter!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_emitters = length(state)
    n_emitters <= 1 && return false, log_prob_current
    
    # Select first emitter randomly
    id1 = rand(rng, 1:n_emitters)
    emitter1 = state[id1]
    
    # Find nearest neighbor (following refactor-rjmcmc approach)
    min_distance = Inf
    id2 = 0
    
    for i in 1:n_emitters
        if i != id1
            distance = sqrt((emitter1.x - state[i].x)^2 + (emitter1.y - state[i].y)^2)
            if distance < min_distance
                min_distance = distance
                id2 = i
            end
        end
    end
    
    emitter2 = state[id2]
    
    # Store old state for reversion
    old_allocations = copy(allocations)
    old_state = copy(state)
    
    # Create merged emitter by averaging positions and combining photons
    merged_x = (emitter1.x + emitter2.x) / 2
    merged_y = (emitter1.y + emitter2.y) / 2
    merged_photons = emitter1.photons + emitter2.photons
    merged_emitter = Emitter2D(merged_x, merged_y, merged_photons)
    
    # Remove the two emitters and add the merged one
    # Remove higher index first to avoid index shifting issues
    if id1 > id2
        deleteat!(state, id1)
        deleteat!(state, id2)
    else
        deleteat!(state, id2)
        deleteat!(state, id1)
    end
    push!(state, merged_emitter)
    
    # Update allocations: redirect allocations from removed emitters to merged emitter
    merged_id = length(state)  # The merged emitter is now at the end
    for i in eachindex(allocations)
        if allocations[i] == id1 || allocations[i] == id2
            allocations[i] = merged_id
        elseif allocations[i] > max(id1, id2)
            allocations[i] -= 2  # Two emitters were removed
        elseif allocations[i] > min(id1, id2)
            allocations[i] -= 1  # One emitter was removed
        end
    end
    
    # Reallocate observations to optimize assignments
    reallocate_all!(allocations, state, observations, rng)
    
    # Compute new log probability
    log_prob_new = compute_log_posterior(state, allocations, observations, prior)
    
    # RJMCMC acceptance ratio following refactor-rjmcmc mathematical approach
    n_obs = length(observations)
    n_emitters_old = n_emitters
    n_emitters_new = n_emitters - 1
    
    # Prior ratio for number of emitters (using Poisson model)
    λ_mean = prior.α / prior.β
    expected_emitters = n_obs / λ_mean
    prior_ratio_k = logpdf(Poisson(expected_emitters), n_emitters_new) - 
                    logpdf(Poisson(expected_emitters), n_emitters_old)
    
    # Proposal ratio: probability of selecting this merge vs reverse split
    # Merge: select 1 out of n_old emitters, then find nearest neighbor
    # Split: select 1 out of n_new emitters to split
    proposal_ratio = log(n_emitters_new) - log(n_emitters_old)
    
    # Prior proposal ratio
    prior_proposal_ratio = prior_ratio_k + proposal_ratio
    
    # Likelihood ratio for positions
    likelihood_ratio_position = log_prob_new - log_prob_current
    
    # Likelihood ratio for number of localizations (Dirichlet-multinomial)
    likelihood_ratio_number = log_dirichlet_multinomial_pmf(state, allocations, prior) - 
                             log_dirichlet_multinomial_pmf(old_state, old_allocations, prior)
    
    # Combined acceptance ratio
    log_ratio = prior_proposal_ratio + likelihood_ratio_position + likelihood_ratio_number
    
    if log(rand(rng)) < log_ratio
        return true, log_prob_new
    else
        # Revert to original state
        copy!(state, old_state)
        copy!(allocations, old_allocations)
        return false, log_prob_current
    end
end
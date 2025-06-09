# RJMCMC add and remove emitter moves
using Distributions
using Random
using SMLMData: Emitter2D

"""
Add a new emitter with mathematically correct RJMCMC acceptance ratio.
"""
function add_emitter!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_obs = length(observations)
    n_obs == 0 && return false, log_prob_current
    
    # Sample position from prior (mixture of Gaussians at observations)
    i = rand(rng, 1:n_obs)
    obs = observations[i]
    
    # Handle uncertainty if available
    σ_x = hasproperty(obs, :σ_x) ? obs.σ_x : T(0.1)
    σ_y = hasproperty(obs, :σ_y) ? obs.σ_y : T(0.1)
    
    x_new = obs.x + σ_x * randn(rng, T)
    y_new = obs.y + σ_y * randn(rng, T)
    
    # Add new emitter
    new_emitter = Emitter2D(x_new, y_new, obs.photons)
    push!(state, new_emitter)
    
    # Store old allocations for reversion
    old_allocations = copy(allocations)
    
    # Reallocate observations
    reallocate_all!(allocations, state, observations, rng)
    
    # Compute new log probability
    log_prob_new = compute_log_posterior(state, allocations, observations, prior)
    
    # Mathematically correct RJMCMC acceptance ratio using refactor-rjmcmc approach
    n_emitters_old = length(state) - 1
    n_emitters_new = length(state)
    
    # Prior ratio for number of emitters (using convolved prior_k)
    λ_mean = prior.α / prior.β
    expected_emitters = n_obs / λ_mean
    prior_ratio_k = logpdf(Poisson(expected_emitters), n_emitters_new) - 
                    logpdf(Poisson(expected_emitters), n_emitters_old)
    
    # Proposal ratio with proper dimension matching
    prior_proposal_ratio = prior_ratio_k + log(n_emitters_new) - log(n_obs)
    
    # Likelihood ratio for positions
    likelihood_ratio_position = log_prob_new - log_prob_current
    
    # Likelihood ratio for number of localizations (Dirichlet-multinomial)
    likelihood_ratio_number = log_dirichlet_multinomial_pmf(state, allocations, prior) - 
                             log_dirichlet_multinomial_pmf(state[1:end-1], old_allocations, prior)
    
    # Combined acceptance ratio
    log_ratio = prior_proposal_ratio + likelihood_ratio_position + likelihood_ratio_number
    
    if log(rand(rng)) < log_ratio
        return true, log_prob_new
    else
        # Revert
        pop!(state)
        copy!(allocations, old_allocations)
        return false, log_prob_current
    end
end

"""
Remove an emitter with mathematically correct RJMCMC acceptance ratio.
"""
function remove_emitter!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_emitters = length(state)
    n_emitters <= 1 && return false, log_prob_current
    
    # Select emitter to remove
    j = rand(rng, 1:n_emitters)
    
    # Store removed emitter and old allocations
    removed = state[j]
    old_allocations = copy(allocations)
    
    # Remove emitter
    deleteat!(state, j)
    
    # Update allocations
    for i in eachindex(allocations)
        if allocations[i] == j
            allocations[i] = 0  # Unallocated
        elseif allocations[i] > j
            allocations[i] -= 1  # Shift indices
        end
    end
    
    # Reallocate observations
    reallocate_all!(allocations, state, observations, rng)
    
    # Compute new log probability
    log_prob_new = compute_log_posterior(state, allocations, observations, prior)
    
    # Mathematically correct RJMCMC acceptance ratio using refactor-rjmcmc approach
    n_emitters_old = n_emitters
    n_emitters_new = n_emitters - 1
    n_obs = length(observations)
    
    # Create state before removal for comparison
    state_old = copy(state)
    insert!(state_old, j, removed)
    
    # Prior ratio for number of emitters (using convolved prior_k)
    λ_mean = prior.α / prior.β
    expected_emitters = n_obs / λ_mean
    prior_ratio_k = logpdf(Poisson(expected_emitters), n_emitters_new) - 
                    logpdf(Poisson(expected_emitters), n_emitters_old)
    
    # Proposal ratio with proper dimension matching (inverse of birth)
    prior_proposal_ratio = prior_ratio_k + log(n_obs) - log(n_emitters_old)
    
    # Likelihood ratio for positions
    likelihood_ratio_position = log_prob_new - log_prob_current
    
    # Likelihood ratio for number of localizations (Dirichlet-multinomial)
    likelihood_ratio_number = log_dirichlet_multinomial_pmf(state, allocations, prior) - 
                             log_dirichlet_multinomial_pmf(state_old, old_allocations, prior)
    
    # Combined acceptance ratio
    log_ratio = prior_proposal_ratio + likelihood_ratio_position + likelihood_ratio_number
    
    if log(rand(rng)) < log_ratio
        return true, log_prob_new
    else
        # Revert
        insert!(state, j, removed)
        copy!(allocations, old_allocations)
        return false, log_prob_current
    end
end
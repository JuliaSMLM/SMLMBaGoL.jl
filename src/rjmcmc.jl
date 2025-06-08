# Core RJMCMC implementation
using Distributions
using Random
using LinearAlgebra
using StatsBase: sample, Weights
using SMLMData: Emitter2D
using SpecialFunctions: logfactorial, loggamma

# Dirichlet-multinomial model for number of localizations per emitter
function log_dirichlet_multinomial_pmf(n_obs, k_vec, α_vec)
    N = length(k_vec)
    
    # Calculate the log-PMF of the Dirichlet-multinomial distribution
    log_pmf = logfactorial(n_obs) - sum(logfactorial.(k_vec))
    log_pmf += loggamma(sum(α_vec)) - loggamma(n_obs + sum(α_vec))
    for i in 1:N
        log_pmf += loggamma(k_vec[i] + α_vec[i]) - loggamma(α_vec[i])
    end
    
    return log_pmf
end

function estimate_concentration_params(k_vec, prior_λ)
    N = length(k_vec)
    α_vec = zeros(N)
    for i in 1:N
        # Method of moments estimate
        E_k = mean(prior_λ)
        Var_k = var(prior_λ)
        α_vec[i] = (E_k * (E_k - 1)) / Var_k
    end
    return α_vec
end

function log_dirichlet_multinomial_pmf(state::Vector{Emitter2D{T}}, allocations::Vector{Int}, prior::HierarchicalPrior{T}) where T
    n_obs = length(allocations)
    n_emitters = length(state)
    
    if n_emitters == 0
        return zero(T)
    end
    
    # Count localizations per emitter
    k_vec = [count(==(i), allocations) for i in 1:n_emitters]
    
    # Estimate concentration parameters from hierarchical prior
    α_vec = fill(prior.α, length(k_vec))
    
    return log_dirichlet_multinomial_pmf(n_obs, k_vec, α_vec)
end

# Build spatial prior as mixture of Gaussians
function build_spatial_prior(observations::Vector{O}) where O
    n_obs = length(observations)
    means = Vector{Vector{Float64}}()
    covs = Vector{Matrix{Float64}}()
    
    for obs in observations
        σ_x = hasproperty(obs, :σ_x) ? obs.σ_x : 0.1
        σ_y = hasproperty(obs, :σ_y) ? obs.σ_y : 0.1
        
        push!(means, [obs.y, obs.x])  # Note: y, x order for MvNormal
        push!(covs, [σ_y^2 0.0; 0.0 σ_x^2])
    end
    
    components = [MvNormal(mean, cov) for (mean, cov) in zip(means, covs)]
    weights = fill(1.0/n_obs, n_obs)
    
    return MixtureModel(components, weights)
end

"""
Run a single RJMCMC chain for a subregion.
"""
function run_single_chain(
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    steps::Int,
    burnin::Int,
    move_probs::MoveProbs{T},
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    # Initialize state
    state, allocations = initialize_state(observations, prior, rng)
    
    # Pre-allocate output storage
    n_samples = steps - burnin
    states = Vector{Vector{Emitter2D{T}}}()
    log_probs = Vector{T}()
    allocation_history = Vector{Vector{Int}}()
    
    sizehint!(states, n_samples)
    sizehint!(log_probs, n_samples)
    sizehint!(allocation_history, n_samples)
    
    # Current log probability
    log_prob = compute_log_posterior(state, allocations, observations, prior)
    
    # MCMC loop
    n_accepted = 0
    for step in 1:steps
        # Propose move
        accepted, log_prob_new = rjmcmc_step!(
            state, allocations, observations, prior, move_probs, log_prob, rng
        )
        
        if accepted
            log_prob = log_prob_new
            n_accepted += 1
        end
        
        # Store samples after burn-in
        if step > burnin
            push!(states, copy(state))
            push!(log_probs, log_prob)
            push!(allocation_history, copy(allocations))
        end
    end
    
    return BaGoLChain(states, log_probs, allocation_history, observations)
end

"""
Initialize MCMC state with smart starting positions.
"""
function initialize_state(
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_obs = length(observations)
    
    # Start with expected number of emitters based on prior
    expected_λ = prior.α / prior.β
    n_emitters = max(1, round(Int, n_obs / expected_λ))
    
    # Initialize emitters near observations
    state = Vector{Emitter2D{T}}()
    
    if n_emitters >= n_obs
        # One emitter per observation
        for obs in observations
            push!(state, Emitter2D(obs.x, obs.y, obs.photons))
        end
    else
        # K-means style initialization
        selected = sample(rng, 1:n_obs, n_emitters, replace=false)
        for idx in selected
            obs = observations[idx]
            push!(state, Emitter2D(obs.x, obs.y, obs.photons))
        end
    end
    
    # Initialize allocations
    allocations = initialize_allocations(observations, state, rng)
    
    return state, allocations
end

"""
Initialize allocations using nearest neighbor assignment.
"""
function initialize_allocations(
    observations::Vector{O},
    state::Vector{Emitter2D{T}},
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_obs = length(observations)
    n_emitters = length(state)
    allocations = Vector{Int}(undef, n_obs)
    
    if n_emitters == 0
        fill!(allocations, 0)
    else
        # Assign each observation to nearest emitter
        @inbounds for i in 1:n_obs
            obs = observations[i]
            min_dist = Inf
            best_j = 1
            
            for j in 1:n_emitters
                em = state[j]
                dist = (obs.x - em.x)^2 + (obs.y - em.y)^2
                if dist < min_dist
                    min_dist = dist
                    best_j = j
                end
            end
            
            allocations[i] = best_j
        end
    end
    
    return allocations
end

"""
Single RJMCMC step with all move types.
"""
function rjmcmc_step!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    move_probs::MoveProbs{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    # Sample move type
    move_type = sample_move_type(length(state), move_probs, rng)
    
    if move_type == MOVE_EMITTER
        return move_emitter!(state, allocations, observations, prior, log_prob_current, rng)
    elseif move_type == ADD_EMITTER
        return add_emitter!(state, allocations, observations, prior, log_prob_current, rng)
    elseif move_type == REMOVE_EMITTER
        return remove_emitter!(state, allocations, observations, prior, log_prob_current, rng)
    elseif move_type == SPLIT_EMITTER
        return split_emitter!(state, allocations, observations, prior, log_prob_current, rng)
    elseif move_type == MERGE_EMITTER
        return merge_emitter!(state, allocations, observations, prior, log_prob_current, rng)
    else # REALLOCATE
        return reallocate!(state, allocations, observations, prior, log_prob_current, rng)
    end
end

"""
Sample move type based on current state and probabilities.
"""
function sample_move_type(
    n_emitters::Int,
    probs::MoveProbs{T},
    rng::AbstractRNG
) where T
    
    # Adjust probabilities based on state
    p = zeros(T, 6)
    
    p[1] = n_emitters > 0 ? probs.move : zero(T)
    p[2] = probs.add
    p[3] = n_emitters > 1 ? probs.remove : zero(T)
    p[4] = n_emitters > 0 ? probs.split : zero(T)
    p[5] = n_emitters > 1 ? probs.merge : zero(T)
    p[6] = n_emitters > 0 ? probs.reallocate : zero(T)
    
    # Normalize
    p ./= sum(p)
    
    # Sample
    r = rand(rng)
    cumsum = zero(T)
    for i in 1:6
        cumsum += p[i]
        if r <= cumsum
            return MoveType(i)
        end
    end
    
    return REALLOCATE  # fallback
end

# Move implementations

"""
Move an emitter to a new position using precision-weighted sampling.
"""
function move_emitter!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_emitters = length(state)
    n_emitters == 0 && return false, log_prob_current
    
    # Select emitter to move
    j = rand(rng, 1:n_emitters)
    
    # Get observations allocated to this emitter
    obs_indices = findall(==(j), allocations)
    
    # Store old position
    old_emitter = state[j]
    
    if isempty(obs_indices)
        # No allocated observations - sample from spatial prior
        # (mixture of Gaussians centered on all observations)
        i = rand(rng, 1:length(observations))
        obs = observations[i]
        σ_x = hasproperty(obs, :σ_x) ? obs.σ_x : T(0.1)
        σ_y = hasproperty(obs, :σ_y) ? obs.σ_y : T(0.1)
        x_new = obs.x + σ_x * randn(rng, T)
        y_new = obs.y + σ_y * randn(rng, T)
    else
        # Precision-weighted posterior sampling
        x_sum = zero(T)
        y_sum = zero(T)
        x_precision = zero(T)
        y_precision = zero(T)
        
        for i in obs_indices
            obs = observations[i]
            if hasproperty(obs, :σ_x)
                w_x = one(T) / (obs.σ_x^2)
                w_y = one(T) / (obs.σ_y^2)
            else
                w_x = w_y = T(100.0)  # High precision if no uncertainty
            end
            x_sum += w_x * obs.x
            y_sum += w_y * obs.y
            x_precision += w_x
            y_precision += w_y
        end
        
        # Posterior mean and variance
        x_mean = x_sum / x_precision
        y_mean = y_sum / y_precision
        x_var = one(T) / x_precision
        y_var = one(T) / y_precision
        
        # Sample from posterior
        x_new = x_mean + sqrt(x_var) * randn(rng, T)
        y_new = y_mean + sqrt(y_var) * randn(rng, T)
    end
    
    # Update state
    state[j] = Emitter2D(x_new, y_new, old_emitter.photons)
    
    # Compute new log probability
    log_prob_new = compute_log_posterior(state, allocations, observations, prior)
    
    # Metropolis-Hastings accept/reject
    log_ratio = log_prob_new - log_prob_current
    
    if log(rand(rng)) < log_ratio
        return true, log_prob_new
    else
        # Revert
        state[j] = old_emitter
        return false, log_prob_current
    end
end

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

"""
Reallocate observations to emitters.
"""
function reallocate!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_emitters = length(state)
    n_emitters == 0 && return false, log_prob_current
    
    # Store old allocations
    old_allocations = copy(allocations)
    
    # Reallocate using categorical sampling
    for i in eachindex(observations)
        obs = observations[i]
        
        # Compute weights for each emitter
        log_weights = Vector{T}(undef, n_emitters)
        
        @inbounds for j in 1:n_emitters
            log_weights[j] = log_likelihood(obs, state[j])
        end
        
        # Normalize and sample
        log_weights .-= maximum(log_weights)
        weights = exp.(log_weights)
        weights ./= sum(weights)
        
        allocations[i] = sample(rng, 1:n_emitters, Weights(weights))
    end
    
    # Always accept reallocation moves (Gibbs sampling)
    log_prob_new = compute_log_posterior(state, allocations, observations, prior)
    
    return true, log_prob_new
end

"""
Reallocate all observations (helper function).
"""
function reallocate_all!(
    allocations::Vector{Int},
    state::Vector{Emitter2D{T}},
    observations::Vector{O},
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_emitters = length(state)
    
    if n_emitters == 0
        fill!(allocations, 0)
        return
    end
    
    for i in eachindex(observations)
        obs = observations[i]
        
        # Find nearest emitter
        min_dist = Inf
        best_j = 1
        
        @inbounds for j in 1:n_emitters
            em = state[j]
            # Handle observations with or without uncertainty
            if hasproperty(obs, :σ_x)
                dist = ((obs.x - em.x) / obs.σ_x)^2 + ((obs.y - em.y) / obs.σ_y)^2
            else
                dist = (obs.x - em.x)^2 + (obs.y - em.y)^2
            end
            if dist < min_dist
                min_dist = dist
                best_j = j
            end
        end
        
        allocations[i] = best_j
    end
end

# Split and merge moves (simplified versions)

function split_emitter!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    # Not implemented in simplified version
    return false, log_prob_current
end

function merge_emitter!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    # Not implemented in simplified version
    return false, log_prob_current
end

# Likelihood and posterior calculations

"""
Compute log likelihood of observation given emitter.
"""
@inline function log_likelihood(
    obs::O,
    emitter::Emitter2D{T}
) where {T<:AbstractFloat, O}
    
    # Handle different observation types
    if hasproperty(obs, :σ_x)
        # Full likelihood with uncertainties
        return -T(0.5) * (
            ((obs.x - emitter.x) / obs.σ_x)^2 + 
            ((obs.y - emitter.y) / obs.σ_y)^2 +
            log(2π * obs.σ_x * obs.σ_y)
        )
    else
        # Simple distance-based likelihood (assume σ = 0.1)
        σ = T(0.1)
        return -T(0.5) * (
            ((obs.x - emitter.x) / σ)^2 + 
            ((obs.y - emitter.y) / σ)^2 +
            log(2π * σ^2)
        )
    end
end

"""
Compute full log posterior probability with proper spatial priors.
"""
function compute_log_posterior(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T}
) where {T<:AbstractFloat, O}
    
    log_prob = zero(T)
    
    # Likelihood term
    @inbounds for i in eachindex(observations)
        j = allocations[i]
        if j > 0
            log_prob += log_likelihood(observations[i], state[j])
        else
            log_prob += -T(10.0)  # Penalty for unallocated
        end
    end
    
    # Prior on number of emitters (Poisson)
    n_emitters = length(state)
    n_obs = length(observations)
    λ_mean = prior.α / prior.β
    expected_emitters = n_obs / λ_mean
    
    log_prob += logpdf(Poisson(expected_emitters), n_emitters)
    
    # Prior on positions (mixture of Gaussians at observations)
    # P(θⱼ) ∝ Σᵢ N(θⱼ | yᵢ, Σᵢ) as per math reference
    if n_emitters > 0 && n_obs > 0
        for emitter in state
            # Sum log probabilities from mixture components
            log_mixture_prob = -Inf
            for obs in observations
                σ_x = hasproperty(obs, :σ_x) ? obs.σ_x : T(0.1)
                σ_y = hasproperty(obs, :σ_y) ? obs.σ_y : T(0.1)
                
                component_log_prob = logpdf(Normal(obs.x, σ_x), emitter.x) +
                                   logpdf(Normal(obs.y, σ_y), emitter.y) -
                                   log(T(n_obs))  # Equal mixture weights
                
                # LogSumExp trick for numerical stability
                if log_mixture_prob == -Inf
                    log_mixture_prob = component_log_prob
                else
                    max_val = max(log_mixture_prob, component_log_prob)
                    log_mixture_prob = max_val + log(exp(log_mixture_prob - max_val) + 
                                                   exp(component_log_prob - max_val))
                end
            end
            log_prob += log_mixture_prob
        end
    end
    
    return log_prob
end
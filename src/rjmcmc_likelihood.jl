# Likelihood and posterior probability calculations for RJMCMC
using Distributions
using LinearAlgebra
using SpecialFunctions: logfactorial, loggamma

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
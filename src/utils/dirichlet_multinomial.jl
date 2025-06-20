"""
    Dirichlet-Multinomial Distribution Functions
    
    The Dirichlet-multinomial distribution is a compound distribution where:
    - The multinomial probabilities p follow a Dirichlet distribution
    - The counts follow a multinomial distribution given p
    
    This handles overdispersion in allocation counts, allowing for varying
    "attractiveness" of emitters beyond what's captured by spatial likelihood.
"""

"""
    estimate_concentration_param(n_obs::Int, prior_λ)
    
Estimate the concentration parameter α for a single emitter based on the prior distribution
of localizations per emitter and the observed count.
"""
function estimate_concentration_param(n_obs::Int, prior_λ)
    # Method of moments estimation
    # For a Gamma prior λ ~ Gamma(α_λ, β_λ), the expected value is α_λ * β_λ
    # and the variance is α_λ * β_λ²
    
    if isa(prior_λ, GammaPrior)
        # Expected number of localizations per emitter
        E_λ = prior_λ.α * prior_λ.β
        
        # Concentration parameter proportional to expected count
        # Higher α means less overdispersion
        α = E_λ  # Simple choice, could be refined
        
        return α
    else
        # Default concentration parameter
        return 1.0
    end
end

"""
    log_dirichlet_multinomial_pmf(n_total::Int, counts::Vector{Int}, α::Vector{Float64})
    
Calculate the log probability mass function of the Dirichlet-multinomial distribution.

# Arguments
- `n_total`: Total number of observations (sum of counts)
- `counts`: Vector of counts for each category
- `α`: Vector of concentration parameters (same length as counts)
"""
function log_dirichlet_multinomial_pmf(n_total::Int, counts::Vector{Int}, α::Vector{Float64})
    @assert length(counts) == length(α) "counts and α must have same length"
    @assert sum(counts) == n_total "counts must sum to n_total"
    
    K = length(counts)
    α_sum = sum(α)
    
    # Log multinomial coefficient
    log_pmf = logfactorial(n_total)
    for k in counts
        log_pmf -= logfactorial(k)
    end
    
    # Log Dirichlet-multinomial part
    log_pmf += loggamma(α_sum) - loggamma(n_total + α_sum)
    
    for i in 1:K
        log_pmf += loggamma(counts[i] + α[i]) - loggamma(α[i])
    end
    
    return log_pmf
end

"""
    log_dirichlet_multinomial_likelihood(state::BaGoLState, prior_λ)
    
Calculate the log-likelihood of the allocation counts under a Dirichlet-multinomial model.
"""
function log_dirichlet_multinomial_likelihood(state::BaGoLState, prior_λ)
    # Count allocations for each emitter
    K = length(state.emitters)
    counts = zeros(Int, K)
    
    # Count allocated localizations
    n_allocated = 0
    for alloc in state.allocations
        if 1 ≤ alloc ≤ K
            counts[alloc] += 1
            n_allocated += 1
        end
    end
    
    # If no localizations are allocated, return 0 (neutral likelihood)
    if n_allocated == 0
        return 0.0
    end
    
    # Estimate concentration parameters
    α = Float64[]
    for k in 1:K
        push!(α, estimate_concentration_param(counts[k], prior_λ))
    end
    
    # Calculate log probability using only allocated localizations
    return log_dirichlet_multinomial_pmf(n_allocated, counts, α)
end

"""
    log_dirichlet_multinomial_ratio(current::BaGoLState, proposed::BaGoLState, prior_λ)
    
Calculate the log ratio of Dirichlet-multinomial likelihoods for birth/death moves.
"""
function log_dirichlet_multinomial_ratio(current::BaGoLState, proposed::BaGoLState, prior_λ)
    log_proposed = log_dirichlet_multinomial_likelihood(proposed, prior_λ)
    log_current = log_dirichlet_multinomial_likelihood(current, prior_λ)
    return log_proposed - log_current
end
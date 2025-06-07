# Hierarchical Bayesian updates for prior parameters
using Distributions
using Statistics

"""
Update hierarchical prior parameters using posterior samples from all chains.

Following the mathematical formulation:
- α ~ Gamma(a₀, b₀)
- β ~ Gamma(c₀, d₀)
- λᵢⱼ ~ Gamma(α, β) for each emitter

The posterior updates are:
- p(α|Y) ∝ ∏ᵢⱼ λᵢⱼ^(α-1) × α^(a₀-1) × exp(-b₀α)
- p(β|Y) ∝ β^(Nα-1) × exp(-β(Σλᵢⱼ + d₀))
"""
function update_hierarchical_prior(
    chains::Vector{BaGoLChain{T, O}},
    prior::HierarchicalPrior{T}
) where {T<:AbstractFloat, O}
    
    # Collect λ samples (localizations per emitter) from all chains
    λ_samples = collect_lambda_samples(chains)
    
    if isempty(λ_samples)
        # No data to update with
        return prior
    end
    
    # Current values
    α_current = prior.α
    β_current = prior.β
    
    # Posterior parameters for α
    n = length(λ_samples)
    sum_log_λ = sum(log, λ_samples)
    
    # Using method of moments for α (could use MCMC for full Bayesian)
    # E[log(λ)] ≈ ψ(α) - log(β), where ψ is digamma function
    # Var[log(λ)] ≈ ψ'(α), where ψ' is trigamma function
    
    # Simplified update using moment matching
    mean_λ = mean(λ_samples)
    var_λ = var(λ_samples)
    
    # Method of moments estimates
    α_new = mean_λ^2 / var_λ
    β_new = mean_λ / var_λ
    
    # Apply hierarchical shrinkage
    # Weight between prior and data-driven estimates
    w = n / (n + T(10))  # Shrinkage weight (10 is tunable)
    
    α_updated = w * α_new + (1 - w) * α_current
    β_updated = w * β_new + (1 - w) * β_current
    
    # Ensure parameters stay in reasonable range
    α_updated = clamp(α_updated, T(0.1), T(100))
    β_updated = clamp(β_updated, T(0.01), T(100))
    
    return HierarchicalPrior(
        α_updated, β_updated,
        prior.a₀, prior.b₀, prior.c₀, prior.d₀
    )
end

"""
Collect λ samples (number of localizations per emitter) from all chains.
"""
function collect_lambda_samples(chains::Vector{BaGoLChain{T, O}}) where {T, O}
    λ_samples = Vector{T}()
    
    for chain in chains
        n_obs = length(chain.observations)
        
        # For each state in the chain
        for (state, allocations) in zip(chain.states, chain.allocations)
            n_emitters = length(state)
            
            if n_emitters > 0
                # Count localizations per emitter
                counts = zeros(Int, n_emitters)
                
                for j in allocations
                    if j > 0
                        counts[j] += 1
                    end
                end
                
                # Add non-zero counts as λ samples
                for count in counts
                    if count > 0
                        push!(λ_samples, T(count))
                    end
                end
            end
        end
    end
    
    return λ_samples
end

"""
Sample from the posterior distribution of α given observed λ values.
Uses Metropolis-Hastings sampling.
"""
function sample_alpha_posterior(
    λ_samples::Vector{T},
    prior::HierarchicalPrior{T},
    n_samples::Int = 1000;
    rng::AbstractRNG = Random.GLOBAL_RNG
) where T<:AbstractFloat
    
    n = length(λ_samples)
    sum_log_λ = sum(log, λ_samples)
    
    # Initialize at current value
    α = prior.α
    samples = Vector{T}(undef, n_samples)
    
    # Proposal standard deviation
    σ_proposal = T(0.1)
    
    for i in 1:n_samples
        # Propose new α
        α_proposed = α + σ_proposal * randn(rng, T)
        
        if α_proposed > 0
            # Log posterior ratio
            log_ratio = (α_proposed - α) * sum_log_λ
            log_ratio += (prior.a₀ - 1) * (log(α_proposed) - log(α))
            log_ratio -= prior.b₀ * (α_proposed - α)
            log_ratio += n * (lgamma(α) - lgamma(α_proposed))
            
            # Accept/reject
            if log(rand(rng)) < log_ratio
                α = α_proposed
            end
        end
        
        samples[i] = α
    end
    
    return samples
end

"""
Sample from the posterior distribution of β given observed λ values and α.
Uses Gamma conjugacy.
"""
function sample_beta_posterior(
    λ_samples::Vector{T},
    α::T,
    prior::HierarchicalPrior{T};
    rng::AbstractRNG = Random.GLOBAL_RNG
) where T<:AbstractFloat
    
    n = length(λ_samples)
    sum_λ = sum(λ_samples)
    
    # Posterior is Gamma(c₀ + nα, d₀ + Σλ)
    shape = prior.c₀ + n * α
    rate = prior.d₀ + sum_λ
    
    return rand(rng, Gamma(shape, 1/rate))
end

"""
Full hierarchical update using MCMC sampling.
More accurate than method of moments but slower.
"""
function update_hierarchical_prior_mcmc(
    chains::Vector{BaGoLChain{T, O}},
    prior::HierarchicalPrior{T};
    n_samples::Int = 1000,
    rng::AbstractRNG = Random.GLOBAL_RNG
) where {T<:AbstractFloat, O}
    
    # Collect λ samples
    λ_samples = collect_lambda_samples(chains)
    
    if isempty(λ_samples)
        return prior
    end
    
    # Sample from posteriors
    α_samples = sample_alpha_posterior(λ_samples, prior, n_samples; rng)
    
    # For each α sample, sample β
    β_samples = Vector{T}(undef, n_samples)
    for i in 1:n_samples
        β_samples[i] = sample_beta_posterior(λ_samples, α_samples[i], prior; rng)
    end
    
    # Use posterior means as point estimates
    α_updated = mean(α_samples[end÷2:end])  # Use second half (after burn-in)
    β_updated = mean(β_samples[end÷2:end])
    
    return HierarchicalPrior(
        α_updated, β_updated,
        prior.a₀, prior.b₀, prior.c₀, prior.d₀
    )
end

"""
Compute the predictive distribution for number of localizations
given the hierarchical prior.
"""
function predictive_lambda_distribution(prior::HierarchicalPrior{T}) where T
    # The predictive distribution is Negative Binomial
    # when integrating out the Gamma prior
    # NB(r, p) where r = α, p = β/(β+1)
    
    r = prior.α
    p = prior.β / (prior.β + 1)
    
    return NegativeBinomial(r, p)
end

"""
Log probability of observing n localizations given the prior.
"""
function log_prob_n_localizations(
    n::Int,
    prior::HierarchicalPrior{T}
) where T<:AbstractFloat
    
    dist = predictive_lambda_distribution(prior)
    return logpdf(dist, n)
end
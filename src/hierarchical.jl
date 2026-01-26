# Hierarchical Bayesian updates for BaGoL

"""
Estimate α from frame statistics using Fano factor.
For Poisson: Fano = 1. For NegBin: Fano = 1 + μ/α.
"""
function estimate_alpha_from_frames(locs::Vector{<:SMLMData.AbstractEmitter})
    # Count localizations per frame
    frames = [loc.frame for loc in locs]
    if isempty(frames)
        return 2.0  # Default
    end

    frame_min, frame_max = extrema(frames)
    if frame_min == frame_max
        return 2.0  # Single frame, can't estimate
    end

    # Count per frame
    frame_counts = zeros(Int, frame_max - frame_min + 1)
    for f in frames
        frame_counts[f - frame_min + 1] += 1
    end

    # Remove zero frames (may be gaps in data)
    nonzero_counts = filter(x -> x > 0, frame_counts)
    if length(nonzero_counts) < 10
        return 2.0  # Not enough data
    end

    μ = mean(nonzero_counts)
    σ² = var(nonzero_counts)

    # Fano factor
    F = σ² / μ

    if F ≤ 1.2
        # Near Poisson - high α (use μ as rough guide)
        return clamp(μ, 5.0, 50.0)
    else
        # Overdispersed - estimate α from Fano factor
        # F = 1 + μ/α → α = μ/(F-1)
        α_est = μ / (F - 1)
        return clamp(α_est, 0.5, 20.0)
    end
end

"""
Update α (shape parameter) via Metropolis-Hastings step.
Uses log-normal proposal for positive support.
"""
function update_alpha!(chain::RJMCMCChain, locs::Vector{<:SMLMData.AbstractEmitter})
    state = chain.current_state

    # Get current counts
    counts = [length(emitter.allocated) for emitter in state.emitters]
    if isempty(counts)
        return
    end

    α_current = chain.α
    μ = chain.μ

    # Log-normal proposal (multiplicative random walk)
    α_proposed = α_current * exp(randn() * 0.2)

    # Clamp to reasonable range
    if α_proposed < 0.1 || α_proposed > 100.0
        return
    end

    # Log-likelihood ratio for counts
    log_lik_current = sum(log_prior_count(n, μ, α_current) for n in counts)
    log_lik_proposed = sum(log_prior_count(n, μ, α_proposed) for n in counts)

    # Prior on α: Gamma(2, 1) - mode at 1, mean at 2
    log_prior_current = logpdf(Gamma(2.0, 1.0), α_current)
    log_prior_proposed = logpdf(Gamma(2.0, 1.0), α_proposed)

    # Proposal ratio for log-normal (asymmetric proposal)
    log_proposal_ratio = log(α_proposed) - log(α_current)

    log_accept = (log_lik_proposed - log_lik_current) +
                 (log_prior_proposed - log_prior_current) +
                 log_proposal_ratio

    if log(rand()) < log_accept
        chain.α = α_proposed
    end
end

"""
Update μ (mean localizations per emitter) via Gibbs sampling.

Model:
  n_j | μ ~ distribution with E[n_j] = μ
  μ ~ Gamma(a, b)         # Hyperprior on mean

Posterior update (moment-matching approach):
  μ | {n_j} ~ Gamma(a + Σn_j, b + K)

This gives posterior mean ≈ Σn_j / K = sample mean (for weak prior),
which correctly estimates the true mean localizations per emitter.
"""
function update_mu_gibbs!(chain::RJMCMCChain)
    config = chain.config
    state = chain.current_state

    # Hyperprior parameters
    a = config.μ_prior_a
    b = config.μ_prior_b

    # Get counts for each emitter
    counts = [length(emitter.allocated) for emitter in state.emitters]
    k = length(counts)

    if k == 0
        # No emitters - sample from prior
        chain.μ = rand(Gamma(a, 1/b))
        return
    end

    sum_n = sum(counts)

    # Gamma posterior: shape = a + Σn, rate = b + K
    # Posterior mean = (a + Σn) / (b + K) ≈ Σn/K for weak prior
    posterior_shape = a + sum_n
    posterior_rate = b + k

    # Sample new μ (using scale parameterization: Gamma(shape, scale=1/rate))
    # Apply minimum to prevent collapse to single-localization emitters
    chain.μ = max(2.0, rand(Gamma(posterior_shape, 1 / posterior_rate)))
end

# ============================================================================
# Global hierarchical updates for partitioned execution
# ============================================================================

"""
    get_chain_statistics(chain)

Extract sufficient statistics from a chain for global hierarchical updates.
Returns (K, counts) where K = number of emitters, counts = allocations per emitter.
"""
function get_chain_statistics(chain::RJMCMCChain)
    state = chain.current_state
    counts = [length(e.allocated) for e in state.emitters]
    return length(counts), counts
end

"""
    pool_statistics(chains)

Pool sufficient statistics from multiple chains for global updates.
Returns (total_K, all_counts).
"""
function pool_statistics(chains::Vector{RJMCMCChain})
    total_K = 0
    all_counts = Int[]
    for chain in chains
        k, counts = get_chain_statistics(chain)
        total_K += k
        append!(all_counts, counts)
    end
    return total_K, all_counts
end

"""
    update_mu_global!(chains, μ_prior_a, μ_prior_b)

Gibbs update for μ using pooled statistics from all chains.
Updates μ in all chains to the same global value.
"""
function update_mu_global!(chains::Vector{RJMCMCChain}, μ_prior_a::Float64, μ_prior_b::Float64)
    total_K, all_counts = pool_statistics(chains)

    if total_K == 0
        μ_new = rand(Gamma(μ_prior_a, 1/μ_prior_b))
    else
        sum_n = sum(all_counts)
        posterior_shape = μ_prior_a + sum_n
        posterior_rate = μ_prior_b + total_K
        μ_new = max(2.0, rand(Gamma(posterior_shape, 1 / posterior_rate)))
    end

    # Update all chains with global μ
    for chain in chains
        chain.μ = μ_new
    end

    return μ_new
end

"""
    update_alpha_global!(chains)

MH update for α using pooled statistics from all chains.
Updates α in all chains to the same global value.
"""
function update_alpha_global!(chains::Vector{RJMCMCChain})
    if isempty(chains)
        return
    end

    total_K, all_counts = pool_statistics(chains)
    if total_K == 0
        return
    end

    # Use first chain's α as current value (all should be same)
    α_current = chains[1].α
    μ = chains[1].μ

    # Log-normal proposal
    α_proposed = α_current * exp(randn() * 0.2)
    if α_proposed < 0.1 || α_proposed > 100.0
        return
    end

    # Log-likelihood ratio using all counts
    log_lik_current = sum(log_prior_count(n, μ, α_current) for n in all_counts)
    log_lik_proposed = sum(log_prior_count(n, μ, α_proposed) for n in all_counts)

    # Prior on α
    log_prior_current = logpdf(Gamma(2.0, 1.0), α_current)
    log_prior_proposed = logpdf(Gamma(2.0, 1.0), α_proposed)

    # Proposal ratio
    log_proposal_ratio = log(α_proposed) - log(α_current)

    log_accept = (log_lik_proposed - log_lik_current) +
                 (log_prior_proposed - log_prior_current) +
                 log_proposal_ratio

    if log(rand()) < log_accept
        # Update all chains with global α
        for chain in chains
            chain.α = α_proposed
        end
    end
end

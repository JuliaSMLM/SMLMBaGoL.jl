# Hierarchical Bayesian updates for BaGoL
#
# Count model: n_j ~ Gamma(shape, scale=μ/shape)
#   - μ = mean locs per emitter
#   - shape = Gamma shape (1=exponential, higher=peaked)
#
# Both updated via MH with log-normal proposals.

"""
Update μ (mean locs per emitter) via Metropolis-Hastings.
Uses log-normal proposal for positive support.

Uses product of individual count priors: P(n₁,...,n_K | μ, shape) = ∏ᵢ Gamma(nᵢ; shape, μ/shape)
"""
function update_mu!(chain::RJMCMCChain)
    state = chain.current_state
    config = chain.config

    K = length(state.emitters)
    if K == 0
        return
    end

    # Individual counts
    counts = [length(e.allocated) for e in state.emitters]

    μ_current = chain.μ
    shape = chain.shape

    # Log-normal proposal (multiplicative random walk)
    μ_proposed = μ_current * exp(randn() * 0.3)

    # Clamp to reasonable range
    if μ_proposed < 1.0 || μ_proposed > 500.0
        return
    end

    # Log-likelihood ratio using product of individual count priors
    # P(n | μ, shape) = Gamma(n; shape, μ/shape)
    scale_current = μ_current / shape
    scale_proposed = μ_proposed / shape
    dist_current = Gamma(shape, scale_current)
    dist_proposed = Gamma(shape, scale_proposed)

    log_lik_current = sum(logpdf(dist_current, max(n, 0.5)) for n in counts)
    log_lik_proposed = sum(logpdf(dist_proposed, max(n, 0.5)) for n in counts)

    # Prior on μ: Gamma(shape, scale)
    prior_shape = config.μ_prior_shape
    prior_scale = config.μ_prior_scale
    log_prior_current = logpdf(Gamma(prior_shape, prior_scale), μ_current)
    log_prior_proposed = logpdf(Gamma(prior_shape, prior_scale), μ_proposed)

    # Proposal ratio for log-normal (asymmetric)
    log_proposal_ratio = log(μ_proposed) - log(μ_current)

    log_accept = (log_lik_proposed - log_lik_current) +
                 (log_prior_proposed - log_prior_current) +
                 log_proposal_ratio

    if log(rand()) < log_accept
        chain.μ = μ_proposed
    end
end

"""
Update shape parameter via Metropolis-Hastings.
Uses log-normal proposal for positive support.

Uses product of individual count priors: P(n₁,...,n_K | μ, shape) = ∏ᵢ Gamma(nᵢ; shape, μ/shape)
"""
function update_shape!(chain::RJMCMCChain)
    state = chain.current_state
    config = chain.config

    K = length(state.emitters)
    if K == 0
        return
    end

    # Individual counts
    counts = [length(e.allocated) for e in state.emitters]

    shape_current = chain.shape
    μ = chain.μ

    # Log-normal proposal
    shape_proposed = shape_current * exp(randn() * 0.3)

    # Clamp to reasonable range [0.5, 50]
    if shape_proposed < 0.5 || shape_proposed > 50.0
        return
    end

    # Log-likelihood ratio using product of individual count priors
    scale_current = μ / shape_current
    scale_proposed = μ / shape_proposed
    dist_current = Gamma(shape_current, scale_current)
    dist_proposed = Gamma(shape_proposed, scale_proposed)

    log_lik_current = sum(logpdf(dist_current, max(n, 0.5)) for n in counts)
    log_lik_proposed = sum(logpdf(dist_proposed, max(n, 0.5)) for n in counts)

    # Prior on shape: Gamma(shape, scale)
    prior_shape = config.shape_prior_shape
    prior_scale = config.shape_prior_scale
    log_prior_current = logpdf(Gamma(prior_shape, prior_scale), shape_current)
    log_prior_proposed = logpdf(Gamma(prior_shape, prior_scale), shape_proposed)

    # Proposal ratio
    log_proposal_ratio = log(shape_proposed) - log(shape_current)

    log_accept = (log_lik_proposed - log_lik_current) +
                 (log_prior_proposed - log_prior_current) +
                 log_proposal_ratio

    if log(rand()) < log_accept
        chain.shape = shape_proposed
    end
end

"""
Estimate initial shape from count variance.
CV = 1/√shape → shape = 1/CV²
"""
function estimate_initial_shape(counts::Vector{Int})
    if length(counts) < 5
        return 2.0  # Default
    end

    μ = mean(counts)
    σ = std(counts)

    if μ <= 0 || σ <= 0
        return 2.0
    end

    cv = σ / μ
    shape_est = 1.0 / (cv^2)

    return clamp(shape_est, 0.5, 20.0)
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
    update_mu_global!(chains)

MH update for μ using pooled statistics from all chains.
Updates μ in all chains to the same global value.

Uses product of individual count priors across all partitions.
"""
function update_mu_global!(chains::Vector{RJMCMCChain})
    if isempty(chains)
        return
    end

    total_K, all_counts = pool_statistics(chains)
    if total_K == 0
        return
    end

    config = chains[1].config
    μ_current = chains[1].μ
    shape = chains[1].shape

    # Log-normal proposal
    μ_proposed = μ_current * exp(randn() * 0.3)
    if μ_proposed < 1.0 || μ_proposed > 500.0
        return
    end

    # Log-likelihood ratio using product of individual count priors
    scale_current = μ_current / shape
    scale_proposed = μ_proposed / shape
    dist_current = Gamma(shape, scale_current)
    dist_proposed = Gamma(shape, scale_proposed)

    log_lik_current = sum(logpdf(dist_current, max(n, 0.5)) for n in all_counts)
    log_lik_proposed = sum(logpdf(dist_proposed, max(n, 0.5)) for n in all_counts)

    # Prior
    log_prior_current = logpdf(Gamma(config.μ_prior_shape, config.μ_prior_scale), μ_current)
    log_prior_proposed = logpdf(Gamma(config.μ_prior_shape, config.μ_prior_scale), μ_proposed)

    # Proposal ratio
    log_proposal_ratio = log(μ_proposed) - log(μ_current)

    log_accept = (log_lik_proposed - log_lik_current) +
                 (log_prior_proposed - log_prior_current) +
                 log_proposal_ratio

    if log(rand()) < log_accept
        for chain in chains
            chain.μ = μ_proposed
        end
    end
end

"""
    update_shape_global!(chains)

MH update for shape using pooled statistics from all chains.
Updates shape in all chains to the same global value.

Uses product of individual count priors across all partitions.
"""
function update_shape_global!(chains::Vector{RJMCMCChain})
    if isempty(chains)
        return
    end

    total_K, all_counts = pool_statistics(chains)
    if total_K == 0
        return
    end

    config = chains[1].config
    shape_current = chains[1].shape
    μ = chains[1].μ

    # Log-normal proposal
    shape_proposed = shape_current * exp(randn() * 0.3)
    if shape_proposed < 0.5 || shape_proposed > 50.0
        return
    end

    # Log-likelihood ratio using product of individual count priors
    scale_current = μ / shape_current
    scale_proposed = μ / shape_proposed
    dist_current = Gamma(shape_current, scale_current)
    dist_proposed = Gamma(shape_proposed, scale_proposed)

    log_lik_current = sum(logpdf(dist_current, max(n, 0.5)) for n in all_counts)
    log_lik_proposed = sum(logpdf(dist_proposed, max(n, 0.5)) for n in all_counts)

    # Prior
    log_prior_current = logpdf(Gamma(config.shape_prior_shape, config.shape_prior_scale), shape_current)
    log_prior_proposed = logpdf(Gamma(config.shape_prior_shape, config.shape_prior_scale), shape_proposed)

    # Proposal ratio
    log_proposal_ratio = log(shape_proposed) - log(shape_current)

    log_accept = (log_lik_proposed - log_lik_current) +
                 (log_prior_proposed - log_prior_current) +
                 log_proposal_ratio

    if log(rand()) < log_accept
        for chain in chains
            chain.shape = shape_proposed
        end
    end
end

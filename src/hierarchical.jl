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
"""
function update_mu!(chain::RJMCMCChain)
    state = chain.current_state
    config = chain.config

    counts = [length(e.allocated) for e in state.emitters]
    if isempty(counts)
        return
    end

    μ_current = chain.μ
    shape = chain.shape

    # Log-normal proposal (multiplicative random walk)
    μ_proposed = μ_current * exp(randn() * 0.3)

    # Clamp to reasonable range
    if μ_proposed < 1.0 || μ_proposed > 500.0
        return
    end

    # Log-likelihood ratio for counts under Gamma(shape, μ/shape)
    log_lik_current = sum(log_prior_count(n, μ_current, shape) for n in counts)
    log_lik_proposed = sum(log_prior_count(n, μ_proposed, shape) for n in counts)

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
"""
function update_shape!(chain::RJMCMCChain)
    state = chain.current_state
    config = chain.config

    counts = [length(e.allocated) for e in state.emitters]
    if isempty(counts)
        return
    end

    shape_current = chain.shape
    μ = chain.μ

    # Log-normal proposal
    shape_proposed = shape_current * exp(randn() * 0.3)

    # Clamp to reasonable range [0.5, 50]
    if shape_proposed < 0.5 || shape_proposed > 50.0
        return
    end

    # Log-likelihood ratio
    log_lik_current = sum(log_prior_count(n, μ, shape_current) for n in counts)
    log_lik_proposed = sum(log_prior_count(n, μ, shape_proposed) for n in counts)

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

    # Log-likelihood ratio using all counts
    log_lik_current = sum(log_prior_count(n, μ_current, shape) for n in all_counts)
    log_lik_proposed = sum(log_prior_count(n, μ_proposed, shape) for n in all_counts)

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

    # Log-likelihood ratio
    log_lik_current = sum(log_prior_count(n, μ, shape_current) for n in all_counts)
    log_lik_proposed = sum(log_prior_count(n, μ, shape_proposed) for n in all_counts)

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

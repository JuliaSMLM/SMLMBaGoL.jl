# Hierarchical Bayesian updates for BaGoL
#
# Count model: n_j ~ Gamma(shape, scale=μ/shape)
#   - μ = mean locs per emitter
#   - shape = Gamma shape (1=exponential, higher=peaked)
#
# MH updates use counts from recent samples (last chunk) for stable fitting.

"""
Get counts from the last chunk of recorded samples.
Returns empty vector if no samples yet.
"""
function get_recent_counts(chain::RJMCMCChain)
    counts = Int[]
    n_samples = length(chain.samples)
    chunk_size = min(chain.config.hierarchical_interval, n_samples)

    if chunk_size == 0
        # No samples yet (during burn-in), use current state
        for e in chain.current_state.emitters
            push!(counts, length(e.allocated))
        end
    else
        # Use last chunk of samples
        for i in (n_samples - chunk_size + 1):n_samples
            for e in chain.samples[i].emitters
                push!(counts, length(e.allocated))
            end
        end
    end
    return counts
end

"""
Update μ (mean locs per emitter) via Metropolis-Hastings.
Uses log-normal proposal for positive support.

Uses product of individual count priors from recent samples:
P(n₁,...,n_K | μ, shape) = ∏ᵢ Gamma(nᵢ; shape, μ/shape)
"""
function update_mu!(chain::RJMCMCChain)
    config = chain.config

    # Get counts from last chunk of samples
    counts = get_recent_counts(chain)

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

Uses product of individual count priors from recent samples:
P(n₁,...,n_K | μ, shape) = ∏ᵢ Gamma(nᵢ; shape, μ/shape)
"""
function update_shape!(chain::RJMCMCChain)
    config = chain.config

    # Get counts from last chunk of samples
    counts = get_recent_counts(chain)

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
Get pooled counts from recent samples across all chains.
"""
function get_recent_counts_global(chains::Vector{RJMCMCChain})
    all_counts = Int[]
    for chain in chains
        append!(all_counts, get_recent_counts(chain))
    end
    return all_counts
end

"""
    update_mu_global!(chains)

MH update for μ using pooled counts from recent samples across all chains.
Updates μ in all chains to the same global value.

Uses product of individual count priors across all partitions.
"""
function update_mu_global!(chains::Vector{RJMCMCChain})
    if isempty(chains)
        return
    end

    # Get counts from recent samples across all chains
    all_counts = get_recent_counts_global(chains)
    if isempty(all_counts)
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

MH update for shape using pooled counts from recent samples across all chains.
Updates shape in all chains to the same global value.

Uses product of individual count priors across all partitions.
"""
function update_shape_global!(chains::Vector{RJMCMCChain})
    if isempty(chains)
        return
    end

    # Get counts from recent samples across all chains
    all_counts = get_recent_counts_global(chains)
    if isempty(all_counts)
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

# ============================================================================
# Collapsed Gibbs sampler hierarchical updates
# ============================================================================

"""
    _get_collapsed_counts(state::CollapsedState) -> Vector{Int}

Extract counts from active clusters in a collapsed state.
"""
function _get_collapsed_counts(state::CollapsedState)
    counts = Int[]
    for (j, cs) in enumerate(state.clusters)
        state.active[j] || continue
        push!(counts, Int(cs.n))
    end
    return counts
end

"""
    _collapsed_count_loglik(state, dist) -> Float64

Compute sum of logpdf(dist, n) over active cluster counts.
Zero-allocation: iterates directly over clusters without collecting counts.
"""
function _collapsed_count_loglik(state::CollapsedState, dist::Gamma)
    ll = 0.0
    @inbounds for (j, cs) in enumerate(state.clusters)
        state.active[j] || continue
        ll += logpdf(dist, max(Float64(cs.n), 0.5))
    end
    return ll
end

"""
    _update_mu_collapsed(state, μ_current, shape, config) -> Float64

MH update for μ using counts from the current collapsed state.
Returns new μ value. Zero-allocation.
"""
function _update_mu_collapsed(state::CollapsedState, μ_current::Float64,
                               shape::Float64, config::NamedTuple)
    state.n_active == 0 && return μ_current

    μ_proposed = μ_current * exp(randn() * 0.3)
    (μ_proposed < 1.0 || μ_proposed > 500.0) && return μ_current

    dist_current = Gamma(shape, μ_current / shape)
    dist_proposed = Gamma(shape, μ_proposed / shape)

    log_lik_current = _collapsed_count_loglik(state, dist_current)
    log_lik_proposed = _collapsed_count_loglik(state, dist_proposed)

    prior_dist = Gamma(config.μ_prior_shape, config.μ_prior_scale)
    log_prior_current = logpdf(prior_dist, μ_current)
    log_prior_proposed = logpdf(prior_dist, μ_proposed)

    log_proposal_ratio = log(μ_proposed) - log(μ_current)

    log_accept = (log_lik_proposed - log_lik_current) +
                 (log_prior_proposed - log_prior_current) +
                 log_proposal_ratio

    return log(rand()) < log_accept ? μ_proposed : μ_current
end

"""
    _update_shape_collapsed(state, μ, shape_current, config) -> Float64

MH update for shape using counts from the current collapsed state.
Returns new shape value. Zero-allocation.
"""
function _update_shape_collapsed(state::CollapsedState, μ::Float64,
                                  shape_current::Float64, config::NamedTuple)
    state.n_active == 0 && return shape_current

    shape_proposed = shape_current * exp(randn() * 0.3)
    (shape_proposed < 0.5 || shape_proposed > 50.0) && return shape_current

    dist_current = Gamma(shape_current, μ / shape_current)
    dist_proposed = Gamma(shape_proposed, μ / shape_proposed)

    log_lik_current = _collapsed_count_loglik(state, dist_current)
    log_lik_proposed = _collapsed_count_loglik(state, dist_proposed)

    prior_dist = Gamma(config.shape_prior_shape, config.shape_prior_scale)
    log_prior_current = logpdf(prior_dist, shape_current)
    log_prior_proposed = logpdf(prior_dist, shape_proposed)

    log_proposal_ratio = log(shape_proposed) - log(shape_current)

    log_accept = (log_lik_proposed - log_lik_current) +
                 (log_prior_proposed - log_prior_current) +
                 log_proposal_ratio

    return log(rand()) < log_accept ? shape_proposed : shape_current
end

"""
    _update_mu_collapsed_global!(states, μ_current, shape, config) -> Float64

Global MH update for μ using pooled counts across all collapsed partition states.
Zero-allocation: iterates directly over clusters without collecting counts.
"""
function _update_mu_collapsed_global!(states::Vector{CollapsedState},
                                      μ_current::Float64, shape::Float64,
                                      config::NamedTuple)
    # Check any active clusters exist
    total_active = sum(s.n_active for s in states)
    total_active == 0 && return μ_current

    μ_proposed = μ_current * exp(randn() * 0.3)
    (μ_proposed < 1.0 || μ_proposed > 500.0) && return μ_current

    dist_current = Gamma(shape, μ_current / shape)
    dist_proposed = Gamma(shape, μ_proposed / shape)

    log_lik_current = sum(_collapsed_count_loglik(s, dist_current) for s in states)
    log_lik_proposed = sum(_collapsed_count_loglik(s, dist_proposed) for s in states)

    prior_dist = Gamma(config.μ_prior_shape, config.μ_prior_scale)
    log_prior_current = logpdf(prior_dist, μ_current)
    log_prior_proposed = logpdf(prior_dist, μ_proposed)

    log_proposal_ratio = log(μ_proposed) - log(μ_current)

    log_accept = (log_lik_proposed - log_lik_current) +
                 (log_prior_proposed - log_prior_current) +
                 log_proposal_ratio

    return log(rand()) < log_accept ? μ_proposed : μ_current
end

"""
    _update_shape_collapsed_global!(states, μ, shape_current, config) -> Float64

Global MH update for shape using pooled counts across all collapsed partition states.
Zero-allocation: iterates directly over clusters without collecting counts.
"""
function _update_shape_collapsed_global!(states::Vector{CollapsedState},
                                         μ::Float64, shape_current::Float64,
                                         config::NamedTuple)
    total_active = sum(s.n_active for s in states)
    total_active == 0 && return shape_current

    shape_proposed = shape_current * exp(randn() * 0.3)
    (shape_proposed < 0.5 || shape_proposed > 50.0) && return shape_current

    dist_current = Gamma(shape_current, μ / shape_current)
    dist_proposed = Gamma(shape_proposed, μ / shape_proposed)

    log_lik_current = sum(_collapsed_count_loglik(s, dist_current) for s in states)
    log_lik_proposed = sum(_collapsed_count_loglik(s, dist_proposed) for s in states)

    prior_dist = Gamma(config.shape_prior_shape, config.shape_prior_scale)
    log_prior_current = logpdf(prior_dist, shape_current)
    log_prior_proposed = logpdf(prior_dist, shape_proposed)

    log_proposal_ratio = log(shape_proposed) - log(shape_current)

    log_accept = (log_lik_proposed - log_lik_current) +
                 (log_prior_proposed - log_prior_current) +
                 log_proposal_ratio

    return log(rand()) < log_accept ? shape_proposed : shape_current
end


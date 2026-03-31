# Hierarchical Bayesian updates for BaGoL
#
# Count model: n_j ~ Gamma(shape, scale=μ/shape)
#   - μ = mean locs per emitter
#   - shape = Gamma shape (1=exponential, higher=peaked)

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
function _collapsed_count_loglik(state::CollapsedState, dist::UnivariateDistribution)
    ll = 0.0
    @inbounds for (j, cs) in enumerate(state.clusters)
        state.active[j] || continue
        ll += logpdf(dist, Int(cs.n))
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

    p_current = shape / (shape + μ_current)
    p_proposed = shape / (shape + μ_proposed)
    dist_current = NegativeBinomial(shape, p_current)
    dist_proposed = NegativeBinomial(shape, p_proposed)

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

    p_current = shape_current / (shape_current + μ)
    p_proposed = shape_proposed / (shape_proposed + μ)
    dist_current = NegativeBinomial(shape_current, p_current)
    dist_proposed = NegativeBinomial(shape_proposed, p_proposed)

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

# ============================================================================
# Conjugate ρ (emitter density) update
# ============================================================================

"""
    _update_rho_collapsed(state, A, config) -> Float64

Conjugate Gamma update for emitter density ρ given one partition.

Prior: ρ ~ Gamma(a, 1/b)  (shape/rate parameterization)
Likelihood: K | ρ ~ Poisson(ρA)
Posterior: ρ | K, A ~ Gamma(a + K, 1/(b + A))

Returns a sample from the posterior.
"""
function _update_rho_collapsed(state::CollapsedState, A::Float64, config::NamedTuple)
    a = config.ρ_prior_shape
    b = config.ρ_prior_rate
    K = state.n_active
    return rand(Gamma(a + K, 1.0 / (b + A)))
end

"""
    _update_rho_collapsed_global!(states, areas, config) -> Float64

Conjugate Gamma update for ρ pooled across all partitions.

Posterior: ρ | {K_j}, {A_j} ~ Gamma(a + ΣK_j, 1/(b + ΣA_j))
"""
function _update_rho_collapsed_global!(states::AbstractVector{<:CollapsedState},
                                       areas::Vector{Float64},
                                       config::NamedTuple)
    a = config.ρ_prior_shape
    b = config.ρ_prior_rate
    total_K = sum(s.n_active for s in states)
    total_A = sum(areas)
    return rand(Gamma(a + total_K, 1.0 / (b + total_A)))
end

"""
    _update_mu_collapsed_global!(states, μ_current, shape, config) -> Float64

Global MH update for μ using pooled counts across all collapsed partition states.
Zero-allocation: iterates directly over clusters without collecting counts.
"""
function _update_mu_collapsed_global!(states::AbstractVector{<:CollapsedState},
                                      μ_current::Float64, shape::Float64,
                                      config::NamedTuple)
    # Check any active clusters exist
    total_active = sum(s.n_active for s in states)
    total_active == 0 && return μ_current

    μ_proposed = μ_current * exp(randn() * 0.3)
    (μ_proposed < 1.0 || μ_proposed > 500.0) && return μ_current

    p_current = shape / (shape + μ_current)
    p_proposed = shape / (shape + μ_proposed)
    dist_current = NegativeBinomial(shape, p_current)
    dist_proposed = NegativeBinomial(shape, p_proposed)

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
function _update_shape_collapsed_global!(states::AbstractVector{<:CollapsedState},
                                         μ::Float64, shape_current::Float64,
                                         config::NamedTuple)
    total_active = sum(s.n_active for s in states)
    total_active == 0 && return shape_current

    shape_proposed = shape_current * exp(randn() * 0.3)
    (shape_proposed < 0.5 || shape_proposed > 50.0) && return shape_current

    p_current = shape_current / (shape_current + μ)
    p_proposed = shape_proposed / (shape_proposed + μ)
    dist_current = NegativeBinomial(shape_current, p_current)
    dist_proposed = NegativeBinomial(shape_proposed, p_proposed)

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

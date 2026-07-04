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
    _collapsed_count_loglik(state, dist; cluster_mask=nothing) -> Float64

Compute sum of logpdf(dist, n) over active cluster counts.
If `cluster_mask` is provided, skip clusters where `cluster_mask[j] == false`
(used to exclude clusters containing overlap locs).
Zero-allocation: iterates directly over clusters without collecting counts.
"""
function _collapsed_count_loglik(state::CollapsedState, dist::UnivariateDistribution;
                                  cluster_mask::Union{Nothing, BitVector}=nothing)
    ll = 0.0
    @inbounds for (j, cs) in enumerate(state.clusters)
        state.active[j] || continue
        (cluster_mask !== nothing && !cluster_mask[j]) && continue
        ll += logpdf(dist, Int(cs.n))
    end
    return ll
end

"""
    _collapsed_kn(state; cluster_mask=nothing) -> (K, N)

Number of active (unmasked) clusters `K` and their total localization count `N`.
Used by the shape update when the DM concentration is NOT tied to `shape`
(fixed `gamma`, or `:decoupled`/`:categorical` allocation): then `shape` enters
the target only through the total-count model `P(N|K)=NB(N; K·shape, p)`, so the
per-cluster NB product would inject a spurious `P_DM(γ=shape)` factor.
"""
function _collapsed_kn(state::CollapsedState;
                       cluster_mask::Union{Nothing, BitVector}=nothing)
    K = 0; N = 0
    @inbounds for (j, cs) in enumerate(state.clusters)
        state.active[j] || continue
        (cluster_mask !== nothing && !cluster_mask[j]) && continue
        K += 1; N += Int(cs.n)
    end
    return (K, N)
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

    # `shape` enters the DM partition prior only when γ is tied to shape (default
    # :dm with gamma=nothing). Then the per-cluster NB product equals P_count·
    # P_DM(γ=shape) up to a shape-constant, so it is the correct shape likelihood.
    # Otherwise (fixed γ, or :decoupled/:categorical) shape enters ONLY via the
    # total-count model P(N|K)=NB(N; K·shape, p); using the per-cluster product
    # there would double-count a spurious P_DM(γ=shape) factor.
    if config.allocation_model === :dm && config.gamma === nothing
        p_current = shape_current / (shape_current + μ)
        p_proposed = shape_proposed / (shape_proposed + μ)
        log_lik_current = _collapsed_count_loglik(state, NegativeBinomial(shape_current, p_current))
        log_lik_proposed = _collapsed_count_loglik(state, NegativeBinomial(shape_proposed, p_proposed))
    else
        K, Ntot = _collapsed_kn(state)
        log_lik_current = logpdf(NegativeBinomial(K * shape_current, shape_current / (shape_current + μ)), Ntot)
        log_lik_proposed = logpdf(NegativeBinomial(K * shape_proposed, shape_proposed / (shape_proposed + μ)), Ntot)
    end

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
    _update_rho_collapsed_global!(states, areas, config; cluster_masks=nothing) -> Float64

Conjugate Gamma update for ρ pooled across all partitions.
If `cluster_masks` is provided, only counts non-overlap active clusters per partition.
Includes a partition's area only if it contributes ≥1 unmasked active cluster.

Posterior: ρ | {K_j}, {A_j} ~ Gamma(a + ΣK_j, 1/(b + ΣA_j))
"""
function _update_rho_collapsed_global!(states::AbstractVector{<:CollapsedState},
                                       areas::Vector{Float64},
                                       config::NamedTuple;
                                       cluster_masks::Union{Nothing, Vector{BitVector}}=nothing)
    a = config.ρ_prior_shape
    b = config.ρ_prior_rate
    total_K = 0
    total_A = 0.0
    for i in eachindex(states)
        k_i = if cluster_masks === nothing
            states[i].n_active
        else
            count(j -> states[i].active[j] && cluster_masks[i][j],
                  eachindex(states[i].active))
        end
        k_i == 0 && continue  # skip partitions with zero unmasked active clusters
        total_K += k_i
        total_A += areas[i]
    end
    return rand(Gamma(a + total_K, 1.0 / (b + total_A)))
end

"""
    _update_mu_collapsed_global!(states, μ_current, shape, config; cluster_masks=nothing) -> Float64

Global MH update for μ using pooled counts across all collapsed partition states.
If `cluster_masks` is provided (Vector{BitVector}), only clusters where
`cluster_masks[i][j] == true` contribute (excludes clusters with overlap locs).
"""
function _update_mu_collapsed_global!(states::AbstractVector{<:CollapsedState},
                                      μ_current::Float64, shape::Float64,
                                      config::NamedTuple, scale::Float64;
                                      cluster_masks::Union{Nothing, Vector{BitVector}}=nothing,
                                      n_steps::Int=50, adapt::Bool=false)
    # Check any active (unmasked) clusters exist
    total_active = if cluster_masks === nothing
        sum(s.n_active for s in states)
    else
        sum(count(j -> s.active[j] && cluster_masks[i][j], eachindex(s.active))
            for (i, s) in enumerate(states))
    end
    total_active == 0 && return (μ_current, scale)

    cm(i) = cluster_masks !== nothing ? cluster_masks[i] : nothing
    count_loglik(m) = (p = shape / (shape + m);
                       d = NegativeBinomial(shape, p);
                       sum(_collapsed_count_loglik(s, d; cluster_mask=cm(i)) for (i, s) in enumerate(states)))
    prior_dist = Gamma(config.μ_prior_shape, config.μ_prior_scale)

    # N MH steps per sync with the current log-lik CACHED across steps (only the proposal
    # is recomputed). The conditional posterior is tight when there are many clusters, so a
    # large fixed scale rejects almost everything (the old stepwise/stuck behaviour) — hence
    # the adaptive scale below.
    μ = μ_current
    ll_cur = count_loglik(μ); lp_cur = logpdf(prior_dist, μ)
    n_acc = 0
    for _ in 1:n_steps
        μ_prop = μ * exp(randn() * scale)
        (μ_prop < 1.0 || μ_prop > 500.0) && continue
        ll_prop = count_loglik(μ_prop); lp_prop = logpdf(prior_dist, μ_prop)
        if log(rand()) < (ll_prop - ll_cur) + (lp_prop - lp_cur) + (log(μ_prop) - log(μ))
            μ = μ_prop; ll_cur = ll_prop; lp_cur = lp_prop; n_acc += 1
        end
    end
    # Robbins-Monro-style scale adaptation toward ~30% acceptance, BURN-IN ONLY (finite
    # adaptation → post-burn-in chain is fixed-scale MH, so ergodicity is preserved).
    new_scale = adapt ? clamp(scale * exp(0.5 * (n_acc / n_steps - 0.3)), 0.002, 1.0) : scale
    return (μ, new_scale)
end

"""
    _update_shape_collapsed_global!(states, μ, shape_current, config; cluster_masks=nothing) -> Float64

Global MH update for shape using pooled counts across all collapsed partition states.
If `cluster_masks` is provided (Vector{BitVector}), only clusters where
`cluster_masks[i][j] == true` contribute (excludes clusters with overlap locs).
"""
function _update_shape_collapsed_global!(states::AbstractVector{<:CollapsedState},
                                         μ::Float64, shape_current::Float64,
                                         config::NamedTuple, scale::Float64;
                                         cluster_masks::Union{Nothing, Vector{BitVector}}=nothing,
                                         n_steps::Int=50, adapt::Bool=false)
    total_active = if cluster_masks === nothing
        sum(s.n_active for s in states)
    else
        sum(count(j -> s.active[j] && cluster_masks[i][j], eachindex(s.active))
            for (i, s) in enumerate(states))
    end
    total_active == 0 && return (shape_current, scale)

    cm(i) = cluster_masks !== nothing ? cluster_masks[i] : nothing
    # See _update_shape_collapsed: only when γ is tied to shape does the per-cluster
    # NB product give the correct shape likelihood; otherwise shape enters solely
    # through each partition's total-count model NB(N_i; K_i·shape, p).
    _shape_coupled_dm = config.allocation_model === :dm && config.gamma === nothing
    count_loglik(sh) = if _shape_coupled_dm
        p = sh / (sh + μ); d = NegativeBinomial(sh, p)
        sum(_collapsed_count_loglik(s, d; cluster_mask=cm(i)) for (i, s) in enumerate(states))
    else
        p = sh / (sh + μ)
        acc = 0.0
        for (i, s) in enumerate(states)
            Ki, Ni = _collapsed_kn(s; cluster_mask=cm(i))
            Ki == 0 && continue
            acc += logpdf(NegativeBinomial(Ki * sh, p), Ni)
        end
        acc
    end
    prior_dist = Gamma(config.shape_prior_shape, config.shape_prior_scale)

    sh = shape_current
    ll_cur = count_loglik(sh); lp_cur = logpdf(prior_dist, sh)
    n_acc = 0
    for _ in 1:n_steps
        sh_prop = sh * exp(randn() * scale)
        (sh_prop < 0.5 || sh_prop > 50.0) && continue
        ll_prop = count_loglik(sh_prop); lp_prop = logpdf(prior_dist, sh_prop)
        if log(rand()) < (ll_prop - ll_cur) + (lp_prop - lp_cur) + (log(sh_prop) - log(sh))
            sh = sh_prop; ll_cur = ll_prop; lp_cur = lp_prop; n_acc += 1
        end
    end
    new_scale = adapt ? clamp(scale * exp(0.5 * (n_acc / n_steps - 0.3)), 0.002, 1.0) : scale
    return (sh, new_scale)
end

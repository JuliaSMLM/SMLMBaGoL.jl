# Sufficient statistics for collapsed Gibbs sampler
#
# ClusterStats stores the accumulated precision matrix, natural parameters,
# and quadratic form needed to compute marginal likelihood and posterior
# position for a cluster of localizations. All operations are O(1).

"""
    ClusterStats

Immutable sufficient statistics for a cluster of localizations.

Stores the posterior precision matrix (2x2 symmetric), natural parameter vector,
quadratic form, localization count, and sum of log-determinants. These are all
that's needed to compute marginal likelihood, predictive probability, and
posterior position/covariance without storing individual localization data.

All operations (add_loc, remove_loc) are O(1) and exact inverses of each other.
"""
struct ClusterStats
    # Posterior precision matrix (2x2 symmetric): [Λ_xx Λ_xy; Λ_xy Λ_yy]
    Λ_xx::Float64
    Λ_xy::Float64
    Λ_yy::Float64
    # Natural parameter: η = Σ_i Λ_i μ_i
    η_x::Float64
    η_y::Float64
    # Quadratic form: Σ_i μ_i^T Λ_i μ_i
    quad::Float64
    # Number of localizations in this cluster
    n::Int32
    # Sum of log|Σ_i| for normalization
    log_det_sum::Float64
end

"""Empty cluster with no localizations."""
ClusterStats() = ClusterStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Int32(0), 0.0)

"""
    LocPrecision

Precomputed precision contributions for a localization.
Computed once from loc data and cached to avoid repeated field access
on mutable parametric emitter types (which causes dynamic dispatch).

Stores position (x, y) and combined σ for spatial distance checks in block moves.
"""
struct LocPrecision
    λ_xx::Float64
    λ_xy::Float64
    λ_yy::Float64
    η_x::Float64
    η_y::Float64
    quad::Float64
    log_det::Float64
    x::Float64
    y::Float64
    σ::Float64  # sqrt(σ_x² + σ_y²) for distance threshold
end

"""
    _loc_precision(loc) -> (λ_xx, λ_xy, λ_yy, η_x, η_y, quad, log_det)

Extract precision contributions from a single localization.
Returns the precision matrix elements, natural parameters, quadratic form,
and log-determinant of the localization's covariance.
"""
@inline function _loc_precision(loc::SMLMData.AbstractEmitter)
    var_x = loc.σ_x^2
    var_y = loc.σ_y^2
    cov_xy = get_cov_xy(loc)
    det = var_x * var_y - cov_xy^2

    if det <= 0
        # Fallback for degenerate covariance
        det = var_x * var_y
        cov_xy = 0.0
    end

    # Precision matrix: Λ_i = Σ_i^{-1}
    inv_det = 1.0 / det
    λ_xx = var_y * inv_det
    λ_yy = var_x * inv_det
    λ_xy = -cov_xy * inv_det

    # Natural parameter: η_i = Λ_i μ_i
    η_x = λ_xx * loc.x + λ_xy * loc.y
    η_y = λ_xy * loc.x + λ_yy * loc.y

    # Quadratic: μ_i^T Λ_i μ_i = η_i^T μ_i
    q = η_x * loc.x + η_y * loc.y

    # log|Σ_i|
    log_det = log(det)

    return λ_xx, λ_xy, λ_yy, η_x, η_y, q, log_det
end

"""
    add_loc(cs::ClusterStats, loc) -> ClusterStats

Add a localization's precision contribution to the cluster. O(1).
"""
@inline function add_loc(cs::ClusterStats, loc::SMLMData.AbstractEmitter)
    λ_xx, λ_xy, λ_yy, η_x, η_y, q, log_det = _loc_precision(loc)
    ClusterStats(
        cs.Λ_xx + λ_xx,
        cs.Λ_xy + λ_xy,
        cs.Λ_yy + λ_yy,
        cs.η_x + η_x,
        cs.η_y + η_y,
        cs.quad + q,
        cs.n + Int32(1),
        cs.log_det_sum + log_det
    )
end

"""
    remove_loc(cs::ClusterStats, loc) -> ClusterStats

Remove a localization's precision contribution from the cluster. O(1).
Exact inverse of add_loc.
"""
@inline function remove_loc(cs::ClusterStats, loc::SMLMData.AbstractEmitter)
    λ_xx, λ_xy, λ_yy, η_x, η_y, q, log_det = _loc_precision(loc)
    ClusterStats(
        cs.Λ_xx - λ_xx,
        cs.Λ_xy - λ_xy,
        cs.Λ_yy - λ_yy,
        cs.η_x - η_x,
        cs.η_y - η_y,
        cs.quad - q,
        cs.n - Int32(1),
        cs.log_det_sum - log_det
    )
end

"""
    precompute_loc_precisions(locs) -> Vector{LocPrecision}

Precompute precision contributions for all locs. Called once at initialization.
"""
function precompute_loc_precisions(locs::Vector{<:SMLMData.AbstractEmitter})
    precs = Vector{LocPrecision}(undef, length(locs))
    for i in eachindex(locs)
        loc = locs[i]
        λ_xx, λ_xy, λ_yy, η_x, η_y, q, ld = _loc_precision(loc)
        σ = sqrt(loc.σ_x^2 + loc.σ_y^2)
        precs[i] = LocPrecision(λ_xx, λ_xy, λ_yy, η_x, η_y, q, ld,
                                Float64(loc.x), Float64(loc.y), σ)
    end
    return precs
end

"""
    add_loc(cs, lp::LocPrecision) -> ClusterStats

Add precomputed precision contribution. O(1), zero-allocation.
"""
@inline function add_loc(cs::ClusterStats, lp::LocPrecision)
    ClusterStats(
        cs.Λ_xx + lp.λ_xx, cs.Λ_xy + lp.λ_xy, cs.Λ_yy + lp.λ_yy,
        cs.η_x + lp.η_x, cs.η_y + lp.η_y, cs.quad + lp.quad,
        cs.n + Int32(1), cs.log_det_sum + lp.log_det
    )
end

"""
    remove_loc(cs, lp::LocPrecision) -> ClusterStats

Remove precomputed precision contribution. O(1), zero-allocation.
"""
@inline function remove_loc(cs::ClusterStats, lp::LocPrecision)
    ClusterStats(
        cs.Λ_xx - lp.λ_xx, cs.Λ_xy - lp.λ_xy, cs.Λ_yy - lp.λ_yy,
        cs.η_x - lp.η_x, cs.η_y - lp.η_y, cs.quad - lp.quad,
        cs.n - Int32(1), cs.log_det_sum - lp.log_det
    )
end

"""
    log_predictive(cs, lp::LocPrecision, log_area) -> Float64

Predictive using precomputed precision. Zero-allocation.
"""
@inline function log_predictive(cs::ClusterStats, lp::LocPrecision, log_area::Float64)
    if cs.n == 0
        return -log_area
    end
    cs_new = add_loc(cs, lp)
    return log_marginal_likelihood(cs_new, log_area) - log_marginal_likelihood(cs, log_area)
end

"""
    _posterior_precision_inv(cs::ClusterStats) -> (det, S_xx, S_xy, S_yy)

Invert the posterior precision matrix. Returns determinant and covariance elements.
"""
@inline function _posterior_precision_inv(cs::ClusterStats)
    det_Λ = cs.Λ_xx * cs.Λ_yy - cs.Λ_xy^2
    if det_Λ <= 0
        return 0.0, Inf, 0.0, Inf
    end
    inv_det = 1.0 / det_Λ
    S_xx = cs.Λ_yy * inv_det
    S_xy = -cs.Λ_xy * inv_det
    S_yy = cs.Λ_xx * inv_det
    return det_Λ, S_xx, S_xy, S_yy
end

"""
    posterior_mean(cs::ClusterStats) -> (μ_x, μ_y)

Precision-weighted centroid: μ = Λ^{-1} η.
"""
function posterior_mean(cs::ClusterStats)
    det_Λ, S_xx, S_xy, S_yy = _posterior_precision_inv(cs)
    if det_Λ <= 0
        return 0.0, 0.0
    end
    μ_x = S_xx * cs.η_x + S_xy * cs.η_y
    μ_y = S_xy * cs.η_x + S_yy * cs.η_y
    return μ_x, μ_y
end

"""
    posterior_cov(cs::ClusterStats) -> (Σ_xx, Σ_xy, Σ_yy)

Posterior covariance matrix: Σ = Λ^{-1}.
"""
function posterior_cov(cs::ClusterStats)
    det_Λ, S_xx, S_xy, S_yy = _posterior_precision_inv(cs)
    if det_Λ <= 0
        return Inf, 0.0, Inf
    end
    return S_xx, S_xy, S_yy
end

"""
    log_marginal_likelihood(cs::ClusterStats, log_area::Float64) -> Float64

Log marginal likelihood with emitter position integrated out analytically.

log p(data_j | cluster_j) = (1-n_j) log(2π) - ½ log_det_sum
                           - ½(quad - η^T Λ^{-1} η) - ½ log|Λ| - log(A)

This is the key quantity for the collapsed sampler - it replaces the
explicit likelihood + position sampling of the uncollapsed version.
"""
@inline function log_marginal_likelihood(cs::ClusterStats, log_area::Float64)
    n = Int(cs.n)
    if n == 0
        return 0.0
    end

    det_Λ, S_xx, S_xy, S_yy = _posterior_precision_inv(cs)
    if det_Λ <= 0
        return -Inf
    end

    # η^T Λ^{-1} η
    eta_Sinv_eta = S_xx * cs.η_x^2 + 2 * S_xy * cs.η_x * cs.η_y + S_yy * cs.η_y^2

    # log p = (1-n)log(2π) - ½ log_det_sum - ½(quad - η^T Σ η) - ½ log|Λ| - log(A)
    lml = (1 - n) * log(2π) -
          0.5 * cs.log_det_sum -
          0.5 * (cs.quad - eta_Sinv_eta) -
          0.5 * log(det_Λ) -
          log_area

    return lml
end

"""
    log_predictive(cs::ClusterStats, loc, log_area::Float64) -> Float64

Log predictive probability of assigning a new localization to this cluster.

For existing cluster: p(loc | cluster_j) = N(loc; μ_post_j, Σ_loc + Σ_post_j)
For empty cluster (new): p(loc | new) = 1/A

This is the ratio of marginal likelihoods:
  log p = log_marginal(cs + loc) - log_marginal(cs)
which automatically handles the normalization.
"""
@inline function log_predictive(cs::ClusterStats, loc::SMLMData.AbstractEmitter, log_area::Float64)
    if cs.n == 0
        # New cluster: uniform spatial prior
        return -log_area
    end

    # Compute as difference of marginal likelihoods
    cs_new = add_loc(cs, loc)
    return log_marginal_likelihood(cs_new, log_area) - log_marginal_likelihood(cs, log_area)
end

"""
    build_cluster_stats(locs, indices) -> ClusterStats

Build ClusterStats from a set of localization indices. O(n).
"""
function build_cluster_stats(locs::Vector{<:SMLMData.AbstractEmitter}, indices)
    cs = ClusterStats()
    for idx in indices
        cs = add_loc(cs, locs[idx])
    end
    return cs
end

# ============================================================================
# Localization mixture prior functions
#
# Instead of P(θ) = 1/A (uniform), use P(θ) = (1/N) Σⱼ N(θ; dⱼ, Σⱼ).
# This concentrates prior mass near the data, dramatically reducing the
# Occam penalty that penalizes splitting co-located emitters.
#
# The marginal likelihood becomes:
#   ML_k = (1/N) Σⱼ ML_flat(cluster_k ∪ {virtual_loc_j})
# where ML_flat is the marginal likelihood under a flat (improper) prior.
# Each term is a Gaussian integral computable via ClusterStats.
# ============================================================================

"""
    log_ml_flat(cs::ClusterStats) -> Float64

Log marginal likelihood with flat (improper) position prior.
Same as `log_marginal_likelihood` but WITHOUT the `-log(A)` term.

Used as a building block for the localization mixture prior, where
the Gaussian prior components replace the uniform 1/A factor.
"""
@inline function log_ml_flat(cs::ClusterStats)
    n = Int(cs.n)
    if n == 0
        return 0.0
    end

    det_Λ, S_xx, S_xy, S_yy = _posterior_precision_inv(cs)
    if det_Λ <= 0
        return -Inf
    end

    eta_Sinv_eta = S_xx * cs.η_x^2 + 2 * S_xy * cs.η_x * cs.η_y + S_yy * cs.η_y^2

    return (1 - n) * log(2π) -
           0.5 * cs.log_det_sum -
           0.5 * (cs.quad - eta_Sinv_eta) -
           0.5 * log(det_Λ)
end

"""
    log_ml_locmix(cs::ClusterStats, loc_precs::Vector{LocPrecision}) -> Float64

Log marginal likelihood under the localization mixture prior:
  P(θ) = (1/N) Σⱼ N(θ; dⱼ, Σⱼ)

Computed as: -log(N) + logsumexp_j[ log_ml_flat(cs + virtual_j) ]

Each term adds virtual localization j to the cluster and computes
the flat-prior ML. Cost: O(N) per cluster (vs O(1) for uniform prior).
"""
function log_ml_locmix(cs::ClusterStats, loc_precs::Vector{LocPrecision})
    N = length(loc_precs)
    N == 0 && return 0.0

    # Compute log ML_flat for each virtual loc added to the cluster
    max_val = -Inf
    @inbounds for j in 1:N
        cs_aug = add_loc(cs, loc_precs[j])
        val = log_ml_flat(cs_aug)
        if val > max_val
            max_val = val
        end
    end

    if max_val == -Inf
        return -Inf
    end

    # Stable logsumexp
    total = 0.0
    @inbounds for j in 1:N
        cs_aug = add_loc(cs, loc_precs[j])
        total += exp(log_ml_flat(cs_aug) - max_val)
    end

    return -log(N) + max_val + log(total)
end

"""
    log_predictive_locmix(cs, lp::LocPrecision, loc_precs::Vector{LocPrecision}) -> Float64

Predictive probability under localization mixture prior.
Ratio of locmix marginal likelihoods with and without loc.

For empty cluster: returns log[(1/N) Σⱼ N(dᵢ; dⱼ, Σᵢ + Σⱼ)],
which concentrates mass near existing localizations.
"""
@inline function log_predictive_locmix(cs::ClusterStats, lp::LocPrecision,
                                        loc_precs::Vector{LocPrecision})
    cs_new = add_loc(cs, lp)
    return log_ml_locmix(cs_new, loc_precs) - log_ml_locmix(cs, loc_precs)
end

# ============================================================================
# Neighbor-filtered locmix (O(|A|) instead of O(N))
#
# The full locmix sums over all N localizations as virtual anchors, but
# distant anchors contribute exp(-d²/2σ²) ≈ 0 to the logsumexp. By
# precomputing a spatial neighbor list and summing only over nearby anchors,
# we reduce the per-cluster cost from O(N) to O(|A|) ≈ 20-50.
#
# The -log(N) normalization is preserved: we are approximating the same
# probability, just dropping terms that are below floating-point precision.
# ============================================================================

"""
    build_anchor_neighbors(loc_precs; nsigma=10.0) -> Vector{Vector{Int32}}

Build static neighbor lists for locmix anchor filtering.

For each localization i, neighbors[i] contains indices j such that
virtual anchor j could contribute non-negligibly to the locmix sum
of any cluster containing i.

Uses a KDTree with a conservative radius: nsigma × (σ_i + max_σ).
Default nsigma=10 gives truncation error < exp(-50) ≈ 10⁻²² per term.
"""
function build_anchor_neighbors(loc_precs::Vector{LocPrecision};
                                 nsigma::Float64=10.0)
    N = length(loc_precs)
    N == 0 && return Vector{Int32}[]

    # Build KDTree from localization positions
    coords = Matrix{Float64}(undef, 2, N)
    @inbounds for i in 1:N
        coords[1, i] = loc_precs[i].x
        coords[2, i] = loc_precs[i].y
    end
    tree = KDTree(coords)

    max_σ = maximum(lp.σ for lp in loc_precs)

    neighbors = Vector{Vector{Int32}}(undef, N)
    @inbounds for i in 1:N
        # Conservative radius: anchor j matters if ||d_i - d_j|| < nsigma × (σ_i + σ_j)
        # Upper bound using max_σ for σ_j
        radius = nsigma * (loc_precs[i].σ + max_σ)
        idxs = inrange(tree, view(coords, :, i), radius)
        neighbors[i] = Int32.(idxs)
    end

    return neighbors
end

"""
    cluster_anchor_set(assignments, slot, neighbors, N) -> Vector{Int32}

Compute the anchor set A(k) = ∪_{i ∈ C_k} neighbors[i] for cluster k.

For small clusters (typical case), this is O(n × |N(i)|) ≈ O(200).
"""
function cluster_anchor_set(assignments::Vector{Int16}, slot::Int16,
                             neighbors::Vector{Vector{Int32}}, N::Int)
    seen = falses(N)
    anchors = Int32[]
    @inbounds for i in 1:N
        assignments[i] == slot || continue
        for j in neighbors[i]
            if !seen[j]
                seen[j] = true
                push!(anchors, j)
            end
        end
    end
    return anchors
end

"""
    log_ml_locmix_filtered(cs::ClusterStats, loc_precs::Vector{LocPrecision},
                            anchors::AbstractVector{<:Integer}) -> Float64

Neighbor-filtered locmix: same as log_ml_locmix but sums only over anchor indices.

Normalization uses full N (length of loc_precs) to preserve proper probability.
Truncation error is bounded by N_dropped × exp(-Δ_min) where Δ_min is the
minimum dropped quadratic penalty — negligible for nsigma ≥ 8.
"""
function log_ml_locmix_filtered(cs::ClusterStats, loc_precs::Vector{LocPrecision},
                                 anchors::AbstractVector{<:Integer})
    N_total = length(loc_precs)
    N_total == 0 && return 0.0
    n_anchors = length(anchors)
    n_anchors == 0 && return -Inf  # No nearby anchors → negligible probability

    # Pass 1: find max for numerical stability
    max_val = -Inf
    @inbounds for idx in 1:n_anchors
        j = anchors[idx]
        cs_aug = add_loc(cs, loc_precs[j])
        val = log_ml_flat(cs_aug)
        if val > max_val
            max_val = val
        end
    end

    if max_val == -Inf
        return -Inf
    end

    # Pass 2: stable logsumexp over anchors only
    total = 0.0
    @inbounds for idx in 1:n_anchors
        j = anchors[idx]
        cs_aug = add_loc(cs, loc_precs[j])
        total += exp(log_ml_flat(cs_aug) - max_val)
    end

    # Normalization: -log(N_total), not -log(n_anchors)
    return -log(N_total) + max_val + log(total)
end

"""
    log_predictive_locmix_filtered(cs, lp, loc_precs, anchors_without,
                                    anchors_with, cached_lml) -> Float64

Neighbor-filtered predictive with cached "without" term.

- `anchors_without`: anchor set for the cluster without loc i
- `anchors_with`: anchor set for the cluster with loc i (= anchors_without ∪ neighbors[i])
- `cached_lml`: precomputed log_ml_locmix_filtered(cs, loc_precs, anchors_without)
                 Pass nothing to compute fresh.
"""
@inline function log_predictive_locmix_filtered(
    cs::ClusterStats, lp::LocPrecision,
    loc_precs::Vector{LocPrecision},
    anchors_without::AbstractVector{<:Integer},
    anchors_with::AbstractVector{<:Integer},
    cached_lml::Union{Float64, Nothing}=nothing
)
    cs_new = add_loc(cs, lp)
    lml_with = log_ml_locmix_filtered(cs_new, loc_precs, anchors_with)
    lml_without = cached_lml !== nothing ? cached_lml :
                  log_ml_locmix_filtered(cs, loc_precs, anchors_without)
    return lml_with - lml_without
end

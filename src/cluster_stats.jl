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
# Grid-based locmix prior (O(1) per cluster evaluation)
#
# Precompute the localization mixture density P(θ) = (1/N) Σⱼ N(θ; dⱼ, Σⱼ)
# on a spatial grid once before sampling. During MCMC, evaluate the prior at
# a cluster's posterior mean via bilinear interpolation — O(1) per cluster.
#
# This replaces -log(A) in log_marginal_likelihood with log_prior_grid(μ),
# giving the locmix's spatial adaptivity at uniform-prior speed.
# ============================================================================

"""
    LocmixGrid

Precomputed log-density of the localization mixture prior on a 2D grid.

The prior is P(θ) = (1/N) Σⱼ N(θ; dⱼ, Σⱼ), evaluated at each grid point
and stored as log values for direct use in log_marginal_likelihood.

Fields:
- `log_density`: ny × nx matrix of log P(θ) values
- `x0, y0`: lower-left corner of the grid
- `dx, dy`: pixel spacing
- `nx, ny`: grid dimensions
"""
struct LocmixGrid
    log_density::Matrix{Float64}  # ny × nx
    x0::Float64
    y0::Float64
    dx::Float64
    dy::Float64
    nx::Int
    ny::Int
end

"""
    build_locmix_grid(loc_precs; margin_sigma=3.0, pixel_per_sigma=2.0) -> LocmixGrid

Precompute the locmix prior density on a spatial grid.

The grid covers the bounding box of all localizations plus `margin_sigma × max_σ`
margin on each side. Resolution is `max_σ / pixel_per_sigma` — fine enough to
resolve the narrowest Gaussian components.

Cost: O(N × nx × ny), run once per partition at initialization.
"""
function build_locmix_grid(loc_precs::Vector{LocPrecision};
                            margin_sigma::Float64=3.0,
                            pixel_per_sigma::Float64=2.0)
    N = length(loc_precs)
    if N == 0
        return LocmixGrid(fill(-Inf, 1, 1), 0.0, 0.0, 1.0, 1.0, 1, 1)
    end

    # Bounding box with margin
    max_σ = maximum(lp.σ for lp in loc_precs)
    margin = margin_sigma * max_σ
    xmin = minimum(lp.x for lp in loc_precs) - margin
    xmax = maximum(lp.x for lp in loc_precs) + margin
    ymin = minimum(lp.y for lp in loc_precs) - margin
    ymax = maximum(lp.y for lp in loc_precs) + margin

    # Grid resolution: resolve the typical Gaussian width
    dx = max_σ / pixel_per_sigma
    dy = dx
    nx = max(ceil(Int, (xmax - xmin) / dx) + 1, 2)
    ny = max(ceil(Int, (ymax - ymin) / dy) + 1, 2)

    # Evaluate log P(θ) = log( (1/N) Σⱼ N(θ; dⱼ, Σⱼ) ) at each grid point
    # Use logsumexp for numerical stability
    log_density = Matrix{Float64}(undef, ny, nx)

    @inbounds for ix in 1:nx
        gx = xmin + (ix - 1) * dx
        for iy in 1:ny
            gy = ymin + (iy - 1) * dy

            # Compute log of each Gaussian component, then logsumexp
            max_val = -Inf
            for j in 1:N
                lp = loc_precs[j]
                # N(θ; dⱼ, Σⱼ) where Σⱼ is the localization covariance
                # For diagonal cov: log N = -log(2π) - log(σ_x σ_y) - ½((x-μx)²/σx² + (y-μy)²/σy²)
                # We use the full precision from LocPrecision
                ddx = gx - lp.x
                ddy = gy - lp.y
                # Mahalanobis: d^T Λ d = λ_xx dx² + 2 λ_xy dx dy + λ_yy dy²
                maha = lp.λ_xx * ddx^2 + 2 * lp.λ_xy * ddx * ddy + lp.λ_yy * ddy^2
                # log N(θ; dⱼ, Σⱼ) = -log(2π) + ½ log|Λⱼ| - ½ d^T Λⱼ d
                #                   = -log(2π) - ½ log|Σⱼ| - ½ maha
                log_comp = -log(2π) - 0.5 * lp.log_det - 0.5 * maha
                if log_comp > max_val
                    max_val = log_comp
                end
            end

            if max_val == -Inf
                log_density[iy, ix] = -Inf
                continue
            end

            total = 0.0
            for j in 1:N
                lp = loc_precs[j]
                ddx = gx - lp.x
                ddy = gy - lp.y
                maha = lp.λ_xx * ddx^2 + 2 * lp.λ_xy * ddx * ddy + lp.λ_yy * ddy^2
                log_comp = -log(2π) - 0.5 * lp.log_det - 0.5 * maha
                total += exp(log_comp - max_val)
            end

            log_density[iy, ix] = -log(N) + max_val + log(total)
        end
    end

    return LocmixGrid(log_density, xmin, ymin, dx, dy, nx, ny)
end

"""
    log_prior_locmix(grid::LocmixGrid, μ_x::Float64, μ_y::Float64) -> Float64

Evaluate the locmix prior at position (μ_x, μ_y) via bilinear interpolation.
O(1) — just index arithmetic and 4 multiplies.

Returns log P(θ) where P(θ) = (1/N) Σⱼ N(θ; dⱼ, Σⱼ).
"""
@inline function log_prior_locmix(grid::LocmixGrid, μ_x::Float64, μ_y::Float64)
    # Continuous grid coordinates
    fx = (μ_x - grid.x0) / grid.dx + 1.0  # 1-based
    fy = (μ_y - grid.y0) / grid.dy + 1.0

    # Clamp to grid bounds
    fx = clamp(fx, 1.0, Float64(grid.nx))
    fy = clamp(fy, 1.0, Float64(grid.ny))

    # Integer indices and fractional parts
    ix = min(floor(Int, fx), grid.nx - 1)
    iy = min(floor(Int, fy), grid.ny - 1)
    ix = max(ix, 1)
    iy = max(iy, 1)
    sx = fx - ix
    sy = fy - iy

    # Bilinear interpolation in log-space
    @inbounds begin
        v00 = grid.log_density[iy, ix]
        v10 = grid.log_density[iy, ix + 1]
        v01 = grid.log_density[iy + 1, ix]
        v11 = grid.log_density[iy + 1, ix + 1]
    end

    return (1 - sx) * (1 - sy) * v00 +
           sx * (1 - sy) * v10 +
           (1 - sx) * sy * v01 +
           sx * sy * v11
end

"""
    log_marginal_likelihood_locmix(cs::ClusterStats, grid::LocmixGrid) -> Float64

Log marginal likelihood using the grid-based locmix prior.

Same as `log_marginal_likelihood` but replaces `-log(A)` with the
locmix log-prior evaluated at the cluster's posterior mean.

This is the saddle-point approximation to the full locmix integral.
For clusters with tight posteriors (n ≥ 2), the approximation is excellent.
"""
@inline function log_marginal_likelihood_locmix(cs::ClusterStats, grid::LocmixGrid)
    n = Int(cs.n)
    if n == 0
        return 0.0
    end

    det_Λ, S_xx, S_xy, S_yy = _posterior_precision_inv(cs)
    if det_Λ <= 0
        return -Inf
    end

    eta_Sinv_eta = S_xx * cs.η_x^2 + 2 * S_xy * cs.η_x * cs.η_y + S_yy * cs.η_y^2

    # Evaluate locmix prior at posterior mean
    μ_x = S_xx * cs.η_x + S_xy * cs.η_y
    μ_y = S_xy * cs.η_x + S_yy * cs.η_y
    log_prior = log_prior_locmix(grid, μ_x, μ_y)

    # log p = (1-n)log(2π) - ½ log_det_sum - ½(quad - η^T Σ η) - ½ log|Λ| + log_prior
    # Note: +log_prior replaces -log(A). The prior density is P(θ), not 1/A.
    lml = (1 - n) * log(2π) -
          0.5 * cs.log_det_sum -
          0.5 * (cs.quad - eta_Sinv_eta) -
          0.5 * log(det_Λ) +
          log_prior

    return lml
end

"""
    log_predictive_locmix(cs, lp::LocPrecision, grid::LocmixGrid) -> Float64

Predictive probability under the grid-based locmix prior.
O(1) — just two marginal likelihood evaluations with grid lookups.
"""
@inline function log_predictive_locmix(cs::ClusterStats, lp::LocPrecision,
                                        grid::LocmixGrid)
    if cs.n == 0
        # New cluster: evaluate locmix prior at the loc's position
        return log_prior_locmix(grid, lp.x, lp.y)
    end
    cs_new = add_loc(cs, lp)
    return log_marginal_likelihood_locmix(cs_new, grid) -
           log_marginal_likelihood_locmix(cs, grid)
end

# ============================================================================
# Spatial model dispatch (type-based switching between flat and locmix)
# ============================================================================

"""
    FlatSpatial <: AbstractSpatialModel

Flat uniform spatial prior over a region of area exp(log_area).
Each cluster's marginal likelihood includes a -log(A) Occam factor.
"""
struct FlatSpatial <: AbstractSpatialModel
    log_area::Float64
end

"""
    LocmixSpatial <: AbstractSpatialModel

Localization mixture spatial prior via grid-based bilinear interpolation.
Evaluates P(s_j) = (1/N) Σ_i N(s_j; d_i, Σ_i) at the posterior mean.
"""
struct LocmixSpatial <: AbstractSpatialModel
    grid::LocmixGrid
end

"""Cluster marginal likelihood under the spatial model."""
@inline spatial_ml(cs::ClusterStats, sp::FlatSpatial) = log_marginal_likelihood(cs, sp.log_area)
@inline spatial_ml(cs::ClusterStats, sp::LocmixSpatial) = log_marginal_likelihood_locmix(cs, sp.grid)

"""Predictive probability for adding a loc to a cluster."""
@inline spatial_pred(cs::ClusterStats, lp::LocPrecision, sp::FlatSpatial) = log_predictive(cs, lp, sp.log_area)
@inline spatial_pred(cs::ClusterStats, lp::LocPrecision, sp::LocmixSpatial) = log_predictive_locmix(cs, lp, sp.grid)

"""Area of the spatial prior region (for Poisson K prior cancellation)."""
spatial_area(sp::FlatSpatial) = exp(sp.log_area)
spatial_log_area(sp::FlatSpatial) = sp.log_area

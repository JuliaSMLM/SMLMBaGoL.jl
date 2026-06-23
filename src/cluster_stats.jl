# Sufficient statistics for the collapsed Gibbs sampler
#
# ClusterStats stores the accumulated precision matrix, natural parameters,
# and quadratic form needed to compute marginal likelihood and posterior
# position for a cluster of localizations. All operations are O(1).
#
# DIMENSION-PARAMETRIC (feature-model-dispatch branch):
# ClusterStats{D,L} and LocPrecision{D,L} are parameterized over the feature
# dimension D (L = D*D, the StaticArrays length param). D=2 reproduces the
# original 2D-position math exactly. A future 3D position feature is D=3; the
# multi-cue (block-diagonal tuple-of-features) generalization layers on top of
# this single-feature core. See docs/feature-model-architecture.md.

"""
    ClusterStats{D,L}

Immutable sufficient statistics for a cluster of localizations in a
`D`-dimensional Gaussian feature (`L == D*D`).

Stores the posterior precision matrix `Λ` (D×D symmetric), the natural
parameter vector `η = Σ_i Λ_i μ_i`, the quadratic form `Σ_i μ_iᵀ Λ_i μ_i`,
the localization count, and the sum of per-loc `log|Σ_i|`. These are all that
is needed to compute marginal likelihood, predictive probability, and
posterior position/covariance without storing individual localization data.

All operations (`add_loc`, `remove_loc`) are O(1) and exact inverses.
"""
struct ClusterStats{D,L} <: AbstractClusterStats
    Λ::SMatrix{D,D,Float64,L}     # posterior precision (Σ⁻¹), symmetric
    η::SVector{D,Float64}         # natural parameter: Σ_i Λ_i μ_i
    quad::Float64                 # Σ_i μ_iᵀ Λ_i μ_i
    n::Int32                      # number of localizations in this cluster
    log_det_sum::Float64          # Σ_i log|Σ_i| for normalization
end

"""Empty `D`-dimensional cluster with no localizations."""
ClusterStats{D,L}() where {D,L} =
    ClusterStats{D,L}(zero(SMatrix{D,D,Float64,L}), zero(SVector{D,Float64}), 0.0, Int32(0), 0.0)
Base.zero(::Type{ClusterStats{D,L}}) where {D,L} = ClusterStats{D,L}()

# No-arg empty defaults to the 2D position feature. Retained only for the
# 2D-only paths (MAP-N, archive, diagnostics). The sampler engine and chain
# initialization use the dimension-derived `empty_cluster` (defined after the
# LocPrecision struct below), so they run at any feature dimension.
ClusterStats() = ClusterStats{2,4}()

"""Feature dimension for an emitter type (extend for new emitter/feature types)."""
feature_dim(::Type{<:SMLMData.Emitter2DFit}) = 2
feature_dim(::Type{<:SMLMData.Emitter3DFit}) = 3

"""
    LocPrecision{D,L}

Precomputed precision contribution for a single localization.
Computed once from loc data and cached to avoid repeated field access on
mutable parametric emitter types (which causes dynamic dispatch).

Stores the observed position `pos` and combined `σ` for spatial distance
checks and locmix-grid lookups.
"""
struct LocPrecision{D,L} <: AbstractLocPrecision
    Λ::SMatrix{D,D,Float64,L}     # precision Σ⁻¹ of this localization
    η::SVector{D,Float64}         # Λ μ
    quad::Float64                 # μᵀ Λ μ
    log_det::Float64              # log|Σ|
    pos::SVector{D,Float64}       # observed coordinates
    σ::Float64                    # sqrt(Σ tr) proxy for distance thresholds
end

"""
    empty_cluster(container) -> ClusterStats

Dimension-derived empty cluster: the ClusterStats dimension is taken from a
precomputed-contribution vector (`LocPrecision{D,L}`) or a localization vector
(`Emitter2DFit`→2D, `Emitter3DFit`→3D). Used by the engine and chain init so
empties match the data's feature dimension.
"""
@inline empty_cluster(precs::AbstractVector{LocPrecision{D,L}}) where {D,L} = zero(ClusterStats{D,L})
@inline empty_cluster(::AbstractVector{<:SMLMData.Emitter2DFit}) = ClusterStats{2,4}()
@inline empty_cluster(::AbstractVector{<:SMLMData.Emitter3DFit}) = ClusterStats{3,9}()

"""
    _loc_precision(loc::AbstractEmitter) -> LocPrecision{2,4}

Extract the 2D precision contribution from a single localization, via the
`get_sigma`/`get_cov_xy` accessors so any 2D `AbstractEmitter` works (standard
`Emitter2DFit`, GaussMLE `Emitter2DFitSigma`, …). Builds the 2×2 precision
matrix, natural parameter, quadratic form, and log-determinant of the
covariance. `Emitter3DFit` dispatches to its own (more specific) method below.
"""
@inline function _loc_precision(loc::SMLMData.AbstractEmitter)
    s = get_sigma(loc)
    var_x = s[1]^2
    var_y = s[2]^2
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

    Λ = SMatrix{2,2,Float64}(λ_xx, λ_xy, λ_xy, λ_yy)
    pos = SVector{2,Float64}(loc.x, loc.y)
    η = Λ * pos                          # natural parameter Λ μ
    q = dot(η, pos)                      # μᵀ Λ μ
    log_det = log(det)
    σ = sqrt(var_x + var_y)

    return LocPrecision{2,4}(Λ, η, q, log_det, pos, σ)
end

"""
    _loc_precision(loc::Emitter3DFit) -> LocPrecision{3,9}

3D position contribution — full 3×3 covariance from (σ_x,σ_y,σ_z,σ_xy,σ_xz,σ_yz).
"""
@inline function _loc_precision(loc::SMLMData.Emitter3DFit)
    vx = loc.σ_x^2; vy = loc.σ_y^2; vz = loc.σ_z^2
    cxy = loc.σ_xy; cxz = loc.σ_xz; cyz = loc.σ_yz
    Σ = SMatrix{3,3,Float64}(vx, cxy, cxz, cxy, vy, cyz, cxz, cyz, vz)
    dΣ = det(Σ)
    if dΣ <= 0
        # Fallback to diagonal covariance if degenerate
        Σ = SMatrix{3,3,Float64}(vx, 0.0, 0.0, 0.0, vy, 0.0, 0.0, 0.0, vz)
        dΣ = vx * vy * vz
    end
    Λ = inv(Σ)
    pos = SVector{3,Float64}(loc.x, loc.y, loc.z)
    η = Λ * pos
    q = dot(η, pos)
    return LocPrecision{3,9}(Λ, η, q, log(dΣ), pos, sqrt(vx + vy + vz))
end

"""
    add_loc(cs, loc) -> ClusterStats

Add a localization's precision contribution to the cluster. O(1).
"""
@inline add_loc(cs::ClusterStats, loc::SMLMData.AbstractEmitter) = add_loc(cs, _loc_precision(loc))

"""
    remove_loc(cs, loc) -> ClusterStats

Remove a localization's precision contribution from the cluster. O(1).
Exact inverse of `add_loc`.
"""
@inline remove_loc(cs::ClusterStats, loc::SMLMData.AbstractEmitter) = remove_loc(cs, _loc_precision(loc))

"""
    precompute_loc_precisions(locs) -> Vector{LocPrecision}

Precompute precision contributions for all locs. Called once at initialization.
"""
function precompute_loc_precisions(locs::Vector{<:SMLMData.AbstractEmitter})
    precs = [_loc_precision(loc) for loc in locs]
    return precs
end

"""
    add_loc(cs, lp::LocPrecision) -> ClusterStats

Add precomputed precision contribution. O(1), zero-allocation.
"""
@inline function add_loc(cs::ClusterStats{D,L}, lp::LocPrecision{D,L}) where {D,L}
    ClusterStats{D,L}(cs.Λ + lp.Λ, cs.η + lp.η, cs.quad + lp.quad,
                      cs.n + Int32(1), cs.log_det_sum + lp.log_det)
end

"""
    remove_loc(cs, lp::LocPrecision) -> ClusterStats

Remove precomputed precision contribution. O(1), zero-allocation.
"""
@inline function remove_loc(cs::ClusterStats{D,L}, lp::LocPrecision{D,L}) where {D,L}
    ClusterStats{D,L}(cs.Λ - lp.Λ, cs.η - lp.η, cs.quad - lp.quad,
                      cs.n - Int32(1), cs.log_det_sum - lp.log_det)
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
    posterior_mean(cs::ClusterStats{2}) -> (μ_x, μ_y)

Precision-weighted centroid μ = Λ^{-1} η, returned as a 2-tuple. (3D and other
features add their own `posterior_mean` / `feature_summary` methods.)
"""
function posterior_mean(cs::ClusterStats{2})
    det_Λ = det(cs.Λ)
    det_Λ <= 0 && return (0.0, 0.0)
    μ = cs.Λ \ cs.η
    return (μ[1], μ[2])
end

"""3D posterior mean → (x, y, z)."""
function posterior_mean(cs::ClusterStats{3})
    det_Λ = det(cs.Λ)
    det_Λ <= 0 && return (0.0, 0.0, 0.0)
    μ = cs.Λ \ cs.η
    return (μ[1], μ[2], μ[3])
end

"""
    posterior_cov(cs::ClusterStats{2}) -> (Σ_xx, Σ_xy, Σ_yy)

Posterior covariance (2D): Σ = Λ^{-1}, returned as its three unique elements.
"""
function posterior_cov(cs::ClusterStats{2})
    det_Λ = det(cs.Λ)
    if det_Λ <= 0
        return Inf, 0.0, Inf
    end
    Σ = inv(cs.Λ)
    return Σ[1,1], Σ[1,2], Σ[2,2]
end

"""3D posterior covariance → (Σ_xx, Σ_yy, Σ_zz, Σ_xy, Σ_xz, Σ_yz)."""
function posterior_cov(cs::ClusterStats{3})
    det_Λ = det(cs.Λ)
    det_Λ <= 0 && return (Inf, Inf, Inf, 0.0, 0.0, 0.0)
    Σ = inv(cs.Λ)
    return Σ[1,1], Σ[2,2], Σ[3,3], Σ[1,2], Σ[1,3], Σ[2,3]
end

"""
    log_marginal_likelihood(cs::ClusterStats, log_area::Float64) -> Float64

Log marginal likelihood with the emitter position integrated out analytically:

    log p(data_j | cluster_j) = (1-n_j)·(D/2)·log(2π) - ½ log_det_sum
                              - ½(quad - ηᵀ Λ^{-1} η) - ½ log|Λ| - log(A)

This replaces the explicit likelihood + position sampling of the uncollapsed
sampler. (D=2 reduces to the original `(1-n)·log(2π)` form.)
"""
@inline function log_marginal_likelihood(cs::ClusterStats{D}, log_area::Float64) where D
    n = Int(cs.n)
    if n == 0
        return 0.0
    end

    det_Λ = det(cs.Λ)
    if det_Λ <= 0
        return -Inf
    end

    eta_Sinv_eta = dot(cs.η, cs.Λ \ cs.η)   # ηᵀ Λ^{-1} η

    lml = (1 - n) * (D / 2) * log(2π) -
          0.5 * cs.log_det_sum -
          0.5 * (cs.quad - eta_Sinv_eta) -
          0.5 * log(det_Λ) -
          log_area

    return lml
end

"""
    log_predictive(cs::ClusterStats, loc, log_area::Float64) -> Float64

Log predictive probability of assigning a new localization to this cluster.
Ratio of marginal likelihoods with and without the loc; for an empty cluster
returns the uniform `-log(A)`.
"""
@inline function log_predictive(cs::ClusterStats, loc::SMLMData.AbstractEmitter, log_area::Float64)
    if cs.n == 0
        return -log_area
    end
    cs_new = add_loc(cs, loc)
    return log_marginal_likelihood(cs_new, log_area) - log_marginal_likelihood(cs, log_area)
end

"""
    build_cluster_stats(locs, indices) -> ClusterStats

Build ClusterStats from a set of localization indices. O(n).
"""
function build_cluster_stats(locs::Vector{E}, indices) where {E<:SMLMData.AbstractEmitter}
    D = feature_dim(E)
    cs = ClusterStats{D, D*D}()
    for idx in indices
        cs = add_loc(cs, locs[idx])
    end
    return cs
end

# ============================================================================
# Localization mixture prior functions (2D position feature)
#
# Instead of P(θ) = 1/A (uniform), use P(θ) = (1/N) Σⱼ N(θ; dⱼ, Σⱼ).
# This concentrates prior mass near the data, dramatically reducing the Occam
# penalty that penalizes splitting co-located emitters.
# ============================================================================

"""
    log_ml_flat(cs::ClusterStats) -> Float64

Log marginal likelihood with a flat (improper) position prior — same as
`log_marginal_likelihood` but WITHOUT the `-log(A)` term. Building block for
the localization mixture prior.
"""
@inline function log_ml_flat(cs::ClusterStats{D}) where D
    n = Int(cs.n)
    if n == 0
        return 0.0
    end

    det_Λ = det(cs.Λ)
    if det_Λ <= 0
        return -Inf
    end

    eta_Sinv_eta = dot(cs.η, cs.Λ \ cs.η)

    return (1 - n) * (D / 2) * log(2π) -
           0.5 * cs.log_det_sum -
           0.5 * (cs.quad - eta_Sinv_eta) -
           0.5 * log(det_Λ)
end

"""
    log_ml_locmix(cs::ClusterStats, loc_precs) -> Float64

Log marginal likelihood under the localization mixture prior
P(θ) = (1/N) Σⱼ N(θ; dⱼ, Σⱼ), computed as
`-log(N) + logsumexp_j[ log_ml_flat(cs + virtual_j) ]`. O(N) per cluster.
"""
function log_ml_locmix(cs::ClusterStats, loc_precs::Vector{<:LocPrecision})
    N = length(loc_precs)
    N == 0 && return 0.0

    max_val = -Inf
    @inbounds for j in 1:N
        val = log_ml_flat(add_loc(cs, loc_precs[j]))
        if val > max_val
            max_val = val
        end
    end

    if max_val == -Inf
        return -Inf
    end

    total = 0.0
    @inbounds for j in 1:N
        total += exp(log_ml_flat(add_loc(cs, loc_precs[j])) - max_val)
    end

    return -log(N) + max_val + log(total)
end

"""
    log_predictive_locmix(cs, lp::LocPrecision, loc_precs) -> Float64

Predictive probability under the (exact, O(N)) localization mixture prior.
"""
@inline function log_predictive_locmix(cs::ClusterStats, lp::LocPrecision,
                                        loc_precs::Vector{<:LocPrecision})
    cs_new = add_loc(cs, lp)
    return log_ml_locmix(cs_new, loc_precs) - log_ml_locmix(cs, loc_precs)
end

# ============================================================================
# Grid-based locmix prior (O(1) per cluster evaluation) — 2D position feature
# ============================================================================

"""
    LocmixGrid

Precomputed log-density of the 2D localization mixture prior on a grid.
`log_density` is an `ny × nx` matrix of `log P(θ)` values.
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

Precompute the locmix prior density on a spatial grid covering the bounding box
of all localizations plus a margin. Cost O(N × nx × ny), run once per partition.
"""
function build_locmix_grid(loc_precs::Vector{<:LocPrecision};
                            margin_sigma::Float64=3.0,
                            pixel_per_sigma::Float64=2.0)
    N = length(loc_precs)
    if N == 0
        return LocmixGrid(fill(-Inf, 1, 1), 0.0, 0.0, 1.0, 1.0, 1, 1)
    end

    # Bounding box with margin
    max_σ = maximum(lp.σ for lp in loc_precs)
    margin = margin_sigma * max_σ
    xmin = minimum(lp.pos[1] for lp in loc_precs) - margin
    xmax = maximum(lp.pos[1] for lp in loc_precs) + margin
    ymin = minimum(lp.pos[2] for lp in loc_precs) - margin
    ymax = maximum(lp.pos[2] for lp in loc_precs) + margin

    # Grid resolution: resolve the typical Gaussian width
    dx = max_σ / pixel_per_sigma
    dy = dx
    nx = max(ceil(Int, (xmax - xmin) / dx) + 1, 2)
    ny = max(ceil(Int, (ymax - ymin) / dy) + 1, 2)

    log_density = Matrix{Float64}(undef, ny, nx)

    @inbounds for ix in 1:nx
        gx = xmin + (ix - 1) * dx
        for iy in 1:ny
            gy = ymin + (iy - 1) * dy

            # logsumexp of the Gaussian mixture components at (gx, gy)
            max_val = -Inf
            for j in 1:N
                lp = loc_precs[j]
                ddx = gx - lp.pos[1]
                ddy = gy - lp.pos[2]
                # Mahalanobis dᵀ Λ d = λ_xx dx² + 2 λ_xy dx dy + λ_yy dy²
                maha = lp.Λ[1,1] * ddx^2 + 2 * lp.Λ[1,2] * ddx * ddy + lp.Λ[2,2] * ddy^2
                # log N(θ; dⱼ, Σⱼ) = -log(2π) - ½ log|Σⱼ| - ½ maha
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
                ddx = gx - lp.pos[1]
                ddy = gy - lp.pos[2]
                maha = lp.Λ[1,1] * ddx^2 + 2 * lp.Λ[1,2] * ddx * ddy + lp.Λ[2,2] * ddy^2
                log_comp = -log(2π) - 0.5 * lp.log_det - 0.5 * maha
                total += exp(log_comp - max_val)
            end

            log_density[iy, ix] = -log(N) + max_val + log(total)
        end
    end

    return LocmixGrid(log_density, xmin, ymin, dx, dy, nx, ny)
end

"""
    log_prior_locmix(grid::LocmixGrid, μ_x, μ_y) -> Float64

Evaluate the locmix prior at `(μ_x, μ_y)` via bilinear interpolation. O(1).
"""
@inline function log_prior_locmix(grid::LocmixGrid, μ_x::Float64, μ_y::Float64)
    fx = (μ_x - grid.x0) / grid.dx + 1.0  # 1-based continuous grid coords
    fy = (μ_y - grid.y0) / grid.dy + 1.0

    fx = clamp(fx, 1.0, Float64(grid.nx))
    fy = clamp(fy, 1.0, Float64(grid.ny))

    ix = min(floor(Int, fx), grid.nx - 1)
    iy = min(floor(Int, fy), grid.ny - 1)
    ix = max(ix, 1)
    iy = max(iy, 1)
    sx = fx - ix
    sy = fy - iy

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

Log marginal likelihood using the grid-based locmix prior: same as
`log_marginal_likelihood` but replaces `-log(A)` with the locmix log-prior
evaluated at the cluster's posterior mean (saddle-point approximation).
"""
@inline function log_marginal_likelihood_locmix(cs::ClusterStats{D}, grid::LocmixGrid) where D
    n = Int(cs.n)
    if n == 0
        return 0.0
    end

    det_Λ = det(cs.Λ)
    if det_Λ <= 0
        return -Inf
    end

    μ = cs.Λ \ cs.η
    eta_Sinv_eta = dot(cs.η, μ)
    log_prior = log_prior_locmix(grid, μ[1], μ[2])

    lml = (1 - n) * (D / 2) * log(2π) -
          0.5 * cs.log_det_sum -
          0.5 * (cs.quad - eta_Sinv_eta) -
          0.5 * log(det_Λ) +
          log_prior

    return lml
end

"""
    log_predictive_locmix(cs, lp::LocPrecision, grid::LocmixGrid) -> Float64

Predictive probability under the grid-based locmix prior. O(1).
"""
@inline function log_predictive_locmix(cs::ClusterStats, lp::LocPrecision,
                                        grid::LocmixGrid)
    if cs.n == 0
        return log_prior_locmix(grid, lp.pos[1], lp.pos[2])
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

Flat uniform spatial prior over a region of area `exp(log_area)`. Each cluster's
marginal likelihood includes a `-log(A)` Occam factor.
"""
struct FlatSpatial <: AbstractSpatialModel
    log_area::Float64
end

"""
    LocmixSpatial <: AbstractSpatialModel

Localization mixture spatial prior via grid-based bilinear interpolation.
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
spatial_area(sp::LocmixSpatial) = (sp.grid.nx * sp.grid.dx) * (sp.grid.ny * sp.grid.dy)
spatial_log_area(sp::FlatSpatial) = sp.log_area
spatial_log_area(sp::LocmixSpatial) = log(spatial_area(sp))

"""Construct LocmixSpatial from precomputed loc precisions."""
LocmixSpatial(loc_precs::Vector{<:LocPrecision}) = LocmixSpatial(build_locmix_grid(loc_precs))

"""Construct LocmixSpatial from emitters (convenience)."""
function LocmixSpatial(locs::Vector{<:SMLMData.AbstractEmitter})
    LocmixSpatial(precompute_loc_precisions(locs))
end

# motion.jl — per-emitter linear-MOTION model (opt-in, default off)
#
# Each emitter's position is allowed to drift linearly in time:
#     θ_k(t_i) = μ_k + v_k · δ_i ,   δ_i = (frame_i − t0) / span   (≈ [−0.5, 0.5])
# so v_k is the END-TO-END displacement over the acquisition. The latent
# β = [μ; v] ∈ ℝ^{2D} (D = feature dim) is integrated out in closed form per
# cluster (μ flat over area A, v ~ N(0, Σ_v), Λ_v = Σ_v⁻¹). No velocities are
# sampled — the state stays discrete.
#
# MARGINAL (derived + Codex-verified; constants exact) — via the Schur complement
# this is the static marginal plus a D×D velocity correction:
#     log p_motion = log p_static + ½·rᵀS⁻¹r − ½·log|S| + ½·log|Λ_v|
#     S = C + Λ_v − Bmᵀ A⁻¹ Bm ,   r = η_δ − Bmᵀ A⁻¹ η
# Reduces EXACTLY to the static model as σ_v→0 (Λ_v→∞) and at n=1 (→ −log A).
#
# DIMENSION-PARAMETRIC {D,L} (L = D*D), mirroring ClusterStats{D,L}: D=2 for
# Emitter2DFit, D=3 for Emitter3DFit. The math is identical at any D — Bm, C, Λ_v
# are D×D, η_δ is a D-vector, β = [μ; v] is 2D-dimensional. MotionClusterStats
# embeds the static ClusterStats{D,L} (giving A=Λ, η, q, n, log_det_sum and
# `log_marginal_likelihood`) and adds the time-weighted blocks.

# ----------------------------------------------------------------------------
# Per-loc contribution carrying normalized time δ
# ----------------------------------------------------------------------------
"""
    MotionLocPrecision{D,L} <: AbstractLocPrecision

Static `D`-dim precision contribution (`LocPrecision{D,L}`) plus the
localization's normalized time `δ = (frame − t0)/span`.
"""
struct MotionLocPrecision{D,L} <: AbstractLocPrecision
    base::LocPrecision{D,L}
    δ::Float64
end
MotionLocPrecision(base::LocPrecision{D,L}, δ::Real) where {D,L} =
    MotionLocPrecision{D,L}(base, Float64(δ))

# ----------------------------------------------------------------------------
# Motion sufficient statistics
# ----------------------------------------------------------------------------
"""
    MotionClusterStats{D,L} <: AbstractClusterStats

Sufficient statistics for the linear-motion cluster marginal. Embeds the static
`ClusterStats{D,L}` (A=Λ, η, quad=q, n, log_det_sum) and the time-weighted blocks
`Bm = ΣδᵢΛᵢ`, `C = Σδᵢ²Λᵢ`, `η_δ = ΣδᵢΛᵢμᵢ`. All ops O(1).
"""
struct MotionClusterStats{D,L} <: AbstractClusterStats
    base::ClusterStats{D,L}
    Bm::SMatrix{D,D,Float64,L}     # Σ δᵢ Λᵢ
    C::SMatrix{D,D,Float64,L}      # Σ δᵢ² Λᵢ
    η_δ::SVector{D,Float64}        # Σ δᵢ Λᵢ μᵢ
end

MotionClusterStats{D,L}() where {D,L} =
    MotionClusterStats{D,L}(ClusterStats{D,L}(), zero(SMatrix{D,D,Float64,L}),
                            zero(SMatrix{D,D,Float64,L}), zero(SVector{D,Float64}))
Base.zero(::Type{MotionClusterStats{D,L}}) where {D,L} = MotionClusterStats{D,L}()
@inline empty_cluster(::AbstractVector{MotionLocPrecision{D,L}}) where {D,L} = MotionClusterStats{D,L}()

# Forward the static-stat fields (n, Λ, η, quad, log_det_sum) to the embedded base so
# the generic moves — which read cs.n, cs.Λ, … directly — treat a MotionClusterStats
# like a ClusterStats. The motion-only fields (base, Bm, C, η_δ) are read directly.
@inline function Base.getproperty(cs::MotionClusterStats, name::Symbol)
    if name === :base || name === :Bm || name === :C || name === :η_δ
        return getfield(cs, name)
    end
    return getproperty(getfield(cs, :base), name)
end

@inline function add_loc(cs::MotionClusterStats{D,L}, lp::MotionLocPrecision{D,L}) where {D,L}
    δ = lp.δ
    MotionClusterStats{D,L}(add_loc(cs.base, lp.base),
                            cs.Bm + δ * lp.base.Λ,
                            cs.C + (δ * δ) * lp.base.Λ,
                            cs.η_δ + δ * lp.base.η)
end

@inline function remove_loc(cs::MotionClusterStats{D,L}, lp::MotionLocPrecision{D,L}) where {D,L}
    δ = lp.δ
    MotionClusterStats{D,L}(remove_loc(cs.base, lp.base),
                            cs.Bm - δ * lp.base.Λ,
                            cs.C - (δ * δ) * lp.base.Λ,
                            cs.η_δ - δ * lp.base.η)
end

# ----------------------------------------------------------------------------
# Velocity correction (Schur form) — the only term beyond the static marginal
# ----------------------------------------------------------------------------
"""
    velocity_correction(cs::MotionClusterStats{D,L}, Λv) -> Float64

`½·rᵀS⁻¹r − ½·log|S| + ½·log|Λv|`, with `S = C + Λv − Bmᵀ A⁻¹ Bm` and
`r = η_δ − Bmᵀ A⁻¹ η`. The amount the motion model adds to the static cluster
marginal. `Λv` positive-definite guarantees `S` is too (even at n=1 / equal δ).
"""
@inline function velocity_correction(cs::MotionClusterStats{D,L},
                                     Λv::SMatrix{D,D,Float64,L}) where {D,L}
    n = Int(cs.base.n)
    n == 0 && return 0.0
    A = cs.base.Λ
    detA = det(A)
    detA <= 0 && return 0.0        # static marginal already returns -Inf here
    Ainv = inv(A)
    AinvBm = Ainv * cs.Bm
    S = cs.C + Λv - transpose(cs.Bm) * AinvBm
    detS = det(S)
    detS <= 0 && return -Inf
    r = cs.η_δ - transpose(cs.Bm) * (Ainv * cs.base.η)
    rSr = dot(r, S \ r)
    return 0.5 * rSr - 0.5 * log(detS) + 0.5 * log(det(Λv))
end

# ----------------------------------------------------------------------------
# Posterior emitter parameters β̂ = B⁻¹ b  (extraction; not the hot path)
# ----------------------------------------------------------------------------
"""
    motion_posterior(cs::MotionClusterStats{D,L}, Λv) -> (μ̂, v̂, Σμ, Σv)

Posterior position `μ̂` (at the reference time t0), velocity `v̂`, and their D×D
covariances, from `β̂ = B⁻¹ b`. Used to report the moving emitter.
"""
function motion_posterior(cs::MotionClusterStats{D,L},
                          Λv::SMatrix{D,D,Float64,L}) where {D,L}
    A = cs.base.Λ; Bm = cs.Bm
    Ainv = inv(A)
    Sv = cs.C + Λv - transpose(Bm) * (Ainv * Bm)   # Schur complement of A
    v̂ = Sv \ (cs.η_δ - transpose(Bm) * (Ainv * cs.base.η))
    μ̂ = Ainv * (cs.base.η - Bm * v̂)
    Σv = inv(Sv)
    CΛ = cs.C + Λv
    Σμ = inv(A - Bm * (inv(CΛ) * transpose(Bm)))    # top-left block of B⁻¹
    return (μ̂, v̂, Σμ, Σv)
end

# ----------------------------------------------------------------------------
# Spatial model
# ----------------------------------------------------------------------------
"""
    MotionSpatial{D,L} <: AbstractSpatialModel

Linear-motion spatial model: flat μ prior over area `exp(log_area)` plus a
zero-mean Gaussian velocity prior with precision `Λv` (D×D). `t0`/`span` are the
GLOBAL time reference used to normalize each loc's frame to `δ`. (Locmix-motion
deferred.)
"""
struct MotionSpatial{D,L} <: AbstractSpatialModel
    Λv::SMatrix{D,D,Float64,L}
    log_area::Float64
    t0::Float64
    span::Float64
end

"""
    motion_spatial(locs, σv_um, log_area, t0, span) -> MotionSpatial{D,L}

Build an isotropic linear-motion spatial model at the localizations' feature
dimension (`Emitter2DFit`→2D, `Emitter3DFit`→3D, other 2D `AbstractEmitter`→2D).
`σv_um` is the per-axis end-to-end drift SD (μm); `Λv = σv⁻² I`. Radial RMS drift
is `√D · σv_um`.
"""
@inline _check_motion_sigma(σv) =
    (isfinite(σv) && σv > 0) || throw(ArgumentError("motion_sigma must be a finite, positive drift SD in μm; got $σv"))
motion_spatial(::Vector{<:SMLMData.Emitter3DFit}, σv::Real, la::Real, t0::Real, span::Real) =
    (_check_motion_sigma(σv); MotionSpatial{3,9}(SMatrix{3,3,Float64,9}(Float64(σv)^-2 * I), Float64(la), Float64(t0), Float64(span)))
motion_spatial(::Vector{<:SMLMData.AbstractEmitter}, σv::Real, la::Real, t0::Real, span::Real) =
    (_check_motion_sigma(σv); MotionSpatial{2,4}(SMatrix{2,2,Float64,4}(Float64(σv)^-2 * I), Float64(la), Float64(t0), Float64(span)))

# Concrete MotionSpatial{D,L} type for a localization element type — used to give the
# per-partition state vector a concrete element type (mirrors the Flat/Locmix branch).
_motion_sp_type(::Type{<:SMLMData.Emitter3DFit}) = MotionSpatial{3,9}
_motion_sp_type(::Type{<:SMLMData.AbstractEmitter}) = MotionSpatial{2,4}

# Accumulators that need a position without the velocity prior (e.g. PosteriorImage)
# get the time-averaged static-block centroid/cov. The motion μ̂/v̂ (which need Λv) come
# from `motion_posterior` and are surfaced via the diagnostics velocity output.
posterior_mean(cs::MotionClusterStats) = posterior_mean(cs.base)
posterior_cov(cs::MotionClusterStats) = posterior_cov(cs.base)

# Let extract_emitters / make_emitter work on a motion chain's final state (reports the
# static-block centroid; motion μ̂/v̂ come from motion_posterior + the velocity output).
make_emitter(cs::MotionClusterStats, id::Int) = make_emitter(cs.base, id)
output_emitter_type(::Type{<:MotionClusterStats{D,L}}) where {D,L} = output_emitter_type(ClusterStats{D,L})
Base.propertynames(::MotionClusterStats) = (:base, :Bm, :C, :η_δ, :n, :Λ, :η, :quad, :log_det_sum)

@inline function spatial_ml(cs::MotionClusterStats{D,L}, sp::MotionSpatial{D,L}) where {D,L}
    Int(cs.base.n) == 0 && return 0.0
    return log_marginal_likelihood(cs.base, sp.log_area) + velocity_correction(cs, sp.Λv)
end

@inline function spatial_pred(cs::MotionClusterStats{D,L}, lp::MotionLocPrecision{D,L},
                              sp::MotionSpatial{D,L}) where {D,L}
    if cs.base.n == 0
        return -sp.log_area          # single-loc motion marginal = −log A (n=1)
    end
    return spatial_ml(add_loc(cs, lp), sp) - spatial_ml(cs, sp)
end

spatial_area(sp::MotionSpatial) = exp(sp.log_area)
spatial_log_area(sp::MotionSpatial) = sp.log_area

# ----------------------------------------------------------------------------
# Frame-aware precision precompute
# ----------------------------------------------------------------------------
# Default: spatial model carries no time ⇒ static precisions (unchanged path).
@inline precompute_loc_precisions(locs::Vector{<:SMLMData.AbstractEmitter}, ::AbstractSpatialModel) =
    precompute_loc_precisions(locs)

"""
    precompute_loc_precisions(locs, sp::MotionSpatial) -> Vector{MotionLocPrecision}

Build motion precisions, reading each loc's `frame` and normalizing to
`δ = (frame − t0)/span` with the model's GLOBAL time reference.
"""
function precompute_loc_precisions(locs::Vector{<:SMLMData.AbstractEmitter}, sp::MotionSpatial)
    [MotionLocPrecision(_loc_precision(loc), (Float64(loc.frame) - sp.t0) / sp.span) for loc in locs]
end

# ----------------------------------------------------------------------------
# Per-emitter velocity recovery (for the diagnostics velocity output)
# ----------------------------------------------------------------------------
"""
    motion_emitter_params(precs, Λv, dahl_labels) -> Vector{(μ̂, v̂, Σμ)}

Per-emitter motion posterior — position `μ̂` at the reference time, velocity `v̂`, and
position covariance `Σμ` (all μm) — one entry per **sorted unique Dahl label**, in the
same order `estimate_mapn_overlap` / `_emitters_from_assignments` emit their emitters. Each
group's `MotionClusterStats` is rebuilt from the precomputed motion precisions, then
`motion_posterior`. Lets the run_bagol MAP-N path report the model position and carry `v̂`
per final emitter (through the same overlap-filter + boundary-dedup).
"""
function motion_emitter_params(precs::Vector{MotionLocPrecision{D,L}},
                               Λv::SMatrix{D,D,Float64,L},
                               dahl_labels::AbstractVector{<:Integer}) where {D,L}
    labs = sort!(unique(dahl_labels))
    out = Vector{Tuple{SVector{D,Float64}, SVector{D,Float64}, SMatrix{D,D,Float64,L}}}()
    for lab in labs
        cs = MotionClusterStats{D,L}()
        @inbounds for j in eachindex(dahl_labels)
            dahl_labels[j] == lab && (cs = add_loc(cs, precs[j]))
        end
        μ̂, v̂, Σμ, _ = motion_posterior(cs, Λv)
        push!(out, (μ̂, v̂, Σμ))
    end
    return out
end

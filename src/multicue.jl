# Multi-cue (block-diagonal) feature composition.
#
# A MultiClusterStats holds a tuple of INDEPENDENT per-feature Gaussian blocks
# (e.g. position-2D ClusterStats{2,4} + spectral-1D ClusterStats{1,1}). add_loc
# maps over the blocks; the collapsed log-marginal and log-predictive SUM over
# blocks (conditional independence of the features given the shared emitter).
# The blocks reuse the single-feature conjugate core in cluster_stats.jl, so the
# whole sampler engine runs unchanged on the composite. See
# docs/feature-model-architecture.md.

# ---------------------------------------------------------------------------
# Generic 1-D feature contribution (spectral λ, lifetime, any scalar cue)
# ---------------------------------------------------------------------------

"""
    gaussian_contribution(val, σ) -> LocPrecision{1,1}

Precompute a 1-D Gaussian feature contribution from an observed value and its
standard deviation. Used for scalar cues (spectral wavelength, lifetime, ...).
"""
@inline function gaussian_contribution(val::Float64, σ::Float64)
    var = σ^2
    Λ = SMatrix{1,1,Float64}(1.0 / var)
    pos = SVector{1,Float64}(val)
    η = Λ * pos
    q = dot(η, pos)
    return LocPrecision{1,1}(Λ, η, q, log(var), pos, σ)
end

"""1-D posterior mean / variance for a scalar-cue block."""
function posterior_mean1d(cs::ClusterStats{1})
    det(cs.Λ) <= 0 && return 0.0
    return (cs.Λ \ cs.η)[1]
end
function posterior_var1d(cs::ClusterStats{1})
    det(cs.Λ) <= 0 && return Inf
    return inv(cs.Λ)[1, 1]
end

# ---------------------------------------------------------------------------
# Composite types
# ---------------------------------------------------------------------------

"""
    MultiClusterStats{T} <: AbstractClusterStats

Block-diagonal sufficient statistics: a tuple of per-feature `ClusterStats`
blocks plus the shared membership count `n`.
"""
struct MultiClusterStats{T<:Tuple} <: AbstractClusterStats
    blocks::T
    n::Int32
end

"""
    MultiLocPrecision{T} <: AbstractLocPrecision

Per-localization contribution across features: a tuple of per-feature
`LocPrecision` contributions.
"""
struct MultiLocPrecision{T<:Tuple} <: AbstractLocPrecision
    contribs::T
end

"""
    FeatureSet{T} <: AbstractSpatialModel

The per-feature priors (one `AbstractSpatialModel` per block). Occupies the
sampler's `spatial` slot so the engine's `spatial_ml`/`spatial_pred` calls
dispatch to the block-diagonal sums below.
"""
struct FeatureSet{T<:Tuple} <: AbstractSpatialModel
    priors::T
end

# ---------------------------------------------------------------------------
# Seam methods — block-diagonal: map to accumulate, sum to score
# ---------------------------------------------------------------------------

@inline add_loc(mcs::MultiClusterStats, mlp::MultiLocPrecision) =
    MultiClusterStats(map(add_loc, mcs.blocks, mlp.contribs), mcs.n + Int32(1))

@inline remove_loc(mcs::MultiClusterStats, mlp::MultiLocPrecision) =
    MultiClusterStats(map(remove_loc, mcs.blocks, mlp.contribs), mcs.n - Int32(1))

@inline spatial_ml(mcs::MultiClusterStats, fs::FeatureSet) =
    sum(map(spatial_ml, mcs.blocks, fs.priors))

@inline spatial_pred(mcs::MultiClusterStats, mlp::MultiLocPrecision, fs::FeatureSet) =
    sum(map((b, c, p) -> spatial_pred(b, c, p), mcs.blocks, mlp.contribs, fs.priors))

# The emitter-count prior / area come from the first (spatial) feature.
spatial_area(fs::FeatureSet) = spatial_area(fs.priors[1])
spatial_log_area(fs::FeatureSet) = spatial_log_area(fs.priors[1])
_uses_poisson_k_prior(fs::FeatureSet) = _uses_poisson_k_prior(fs.priors[1])

# Dimension/feature-derived empties.
_block_type(::Type{LocPrecision{D,L}}) where {D,L} = ClusterStats{D,L}
Base.zero(::Type{MultiClusterStats{T}}) where {T<:Tuple} =
    MultiClusterStats(map(zero, fieldtypes(T)), Int32(0))
@inline empty_cluster(precs::AbstractVector{MultiLocPrecision{TC}}) where {TC<:Tuple} =
    MultiClusterStats(map(LP -> zero(_block_type(LP)), fieldtypes(TC)), Int32(0))

# ---------------------------------------------------------------------------
# Multi-cue chain
# ---------------------------------------------------------------------------

"""
    build_multicue_precisions(positions, values, σ_values) -> Vector{MultiLocPrecision}

Build per-localization (position-2D + scalar-cue) contributions from a vector of
2D localizations and a parallel scalar cue (`value ± σ`) per localization.
"""
function build_multicue_precisions(positions::Vector{<:SMLMData.Emitter2DFit},
                                    values::AbstractVector{<:Real},
                                    σ_values::AbstractVector{<:Real})
    N = length(positions)
    @assert length(values) == N && length(σ_values) == N "values/σ_values must match positions"
    return [MultiLocPrecision((_loc_precision(positions[i]),
                               gaussian_contribution(Float64(values[i]), Float64(σ_values[i]))))
            for i in 1:N]
end

"""
    initialize_multicue_state(multi_precs, feature_set; am, use_poisson_k_prior) -> CollapsedState

All-in-one initial state for multi-cue grouping (every loc in one cluster).
"""
function initialize_multicue_state(multi_precs::Vector{<:MultiLocPrecision},
                                   feature_set::FeatureSet;
                                   am::AbstractAllocationModel=DMAllocation(),
                                   use_poisson_k_prior::Bool=_uses_poisson_k_prior(feature_set))
    N = length(multi_precs)
    cs = empty_cluster(multi_precs)
    for mlp in multi_precs
        cs = add_loc(cs, mlp)
    end
    assignments = fill(Int16(1), N)
    clusters = [cs]
    active = BitVector([true])
    max_K = max(N, 16)
    return CollapsedState(assignments, clusters, active, 1, feature_set, am,
                          use_poisson_k_prior, multi_precs,
                          collect(1:N), Vector{Int}(undef, max_K),
                          Vector{Float64}(undef, max_K + 1),
                          similar(assignments), similar(clusters), similar(active))
end

"""
    run_multicue_chain(positions, values, σ_values, feature_set; kwargs...) -> CollapsedState

Run the collapsed Gibbs sampler grouping localizations JOINTLY on 2D position +
a scalar cue (e.g. spectral wavelength). Returns the final `CollapsedState`;
read out per-emitter joint estimates with `extract_multicue`.
"""
function run_multicue_chain(positions::Vector{<:SMLMData.Emitter2DFit},
                            values::AbstractVector{<:Real},
                            σ_values::AbstractVector{<:Real},
                            feature_set::FeatureSet;
                            n_iterations::Int=10000, burn_in::Int=2000,
                            μ::Float64=10.0, shape::Float64=2.0, ρ::Float64=2.0,
                            allocation_model::Symbol=:dm, gamma::Union{Nothing,Float64}=nothing,
                            n_restricted_scans::Int=5, n_bd_substeps::Int=5)
    am = allocation_model === :dm ? DMAllocation(gamma) :
         allocation_model === :decoupled ? DecoupledAllocation() : CategoricalAllocation()
    multi_precs = build_multicue_precisions(positions, values, σ_values)
    state = initialize_multicue_state(multi_precs, feature_set; am=am)
    run_collapsed_iterations!(state, positions, n_iterations, μ, shape, ρ,
                              AbstractAccumulator[], burn_in, 0;
                              n_restricted_scans=n_restricted_scans, n_bd_substeps=n_bd_substeps)
    return state
end

"""
    extract_multicue(state) -> Vector{NamedTuple}

Per-emitter joint estimate from a multi-cue `CollapsedState`:
`(x, y, σ_x, σ_y, value, σ_value, n)` — position from block 1, scalar cue from
block 2.
"""
function extract_multicue(state::CollapsedState)
    out = NamedTuple[]
    for (j, mcs) in enumerate(state.clusters)
        state.active[j] || continue
        mcs.n == 0 && continue
        posblock = mcs.blocks[1]
        cueblock = mcs.blocks[2]
        mx, my = posterior_mean(posblock)
        Σxx, Σxy, Σyy = posterior_cov(posblock)
        push!(out, (x = mx, y = my,
                    σ_x = sqrt(max(Σxx, 0.0)), σ_y = sqrt(max(Σyy, 0.0)),
                    value = posterior_mean1d(cueblock),
                    σ_value = sqrt(max(posterior_var1d(cueblock), 0.0)),
                    n = Int(mcs.n)))
    end
    return out
end

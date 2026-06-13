# Core types for BaGoL

# ============================================================================
# Diagnostics (must come before BaGoLResult which references it)
# ============================================================================

"""
Diagnostics from BaGoL analysis for QC and visualization.
"""
struct BaGoLDiagnostics
    n_emitters::Int
    posterior_k::Vector{Int}
    acceptance_rates::Dict{Symbol, Float64}
    final_μ::Float64
    final_shape::Float64
    final_ρ::Float64            # emitter density (emitters per μm²)
    n_partitions::Int  # 1 for non-partitioned runs
    cluster_sizes::Vector{Int}  # active cluster sizes from final state (what Gamma was fit to)
    partition_k::Vector{Int}    # emitters found per partition (from MAP-N extraction)
    partition_ids::Vector{Int}  # partition ID per localization (0 = unassigned)
    posterior_image::Union{Nothing, NamedTuple{(:image, :edges_x, :edges_y, :pixel_size), Tuple{Matrix{Int}, Vector{Float64}, Vector{Float64}, Float64}}}
end

# ============================================================================
# Collapsed Gibbs sampler types
# ============================================================================

"""
    CollapsedState

MCMC state for the collapsed Gibbs sampler. Stores only assignments
(which locs belong to which cluster) — emitter positions are derived
from ClusterStats sufficient statistics.

Cluster slots are pre-allocated and reused via the `active` bitvector.
Workspace buffers (`_perm`, `_active_slots`, `_log_probs`, `_rollback_*`)
are pre-allocated for zero-allocation hot-path operation.
"""
mutable struct CollapsedState{S<:AbstractSpatialModel, A<:AbstractAllocationModel}
    assignments::Vector{Int16}      # assignments[i] = cluster label for loc i
    clusters::Vector{ClusterStats}  # Pre-allocated slots
    active::BitVector               # Which slots are in use
    n_active::Int                   # Number of active clusters
    spatial::S                      # Spatial prior model (FlatSpatial or LocmixSpatial)
    allocation::A                   # Allocation model
    use_poisson_k_prior::Bool       # Whether the target includes Poisson(ρA) K prior + ρ updates

    # Precomputed loc precisions (computed once, locs don't change)
    _loc_precs::Vector{LocPrecision}

    # Workspace buffers (pre-allocated, reused across iterations)
    _perm::Vector{Int}              # Permutation for Gibbs sweep
    _active_slots::Vector{Int}      # Active cluster indices
    _log_probs::Vector{Float64}     # Log probabilities for categorical
    _rollback_assignments::Vector{Int16}  # Rollback buffer for MH moves
    _rollback_clusters::Vector{ClusterStats}
    _rollback_active::BitVector
end

"""
    BaGoLResult

Complete result from collapsed BaGoL analysis.
"""
struct BaGoLResult
    emitters::Vector{<:SMLMData.Emitter2DFit}  # Point estimates
    diagnostics::BaGoLDiagnostics
    accumulators::Dict{Symbol, Any}           # Named accumulator results
    archive_path::Union{Nothing, String}
end

"""
    CollapsedChainResult

Result from a single collapsed chain (one partition).
Lightweight — no sample storage, just final state + accumulator results.
"""
struct CollapsedChainResult
    state::CollapsedState
    μ::Float64
    shape::Float64
    ρ::Float64                 # emitter density (emitters per μm²)
    accumulators::Vector{Any}  # accumulator result objects
    acceptance::Dict{Symbol, Tuple{Int, Int}}  # (accepted, total) per move type
    n_iterations::Int
end

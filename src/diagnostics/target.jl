# Target density abstraction for algorithm-agnostic diagnostic evaluation
#
# Different algorithm variants target different posterior distributions.
# This abstraction lets diagnostic tools (enumeration, detailed balance,
# mixing analysis) work with any target by parameterizing the density.

# ============================================================================
# Abstract interface
# ============================================================================

"""
    AbstractTargetDensity

Base type for target distributions over the partition space.

Subtypes represent different posterior formulations. The diagnostic tools
(brute-force enumeration, detailed balance, mixing analysis) accept any
`AbstractTargetDensity` and evaluate the target density accordingly.

Implement `log_target(td, z, loc_precs, grid, μ, shape)` for new variants.
"""
abstract type AbstractTargetDensity end

# ============================================================================
# Concrete targets
# ============================================================================

"""
    DecoupledTarget <: AbstractTargetDensity

Current `locmix-grid` branch target: count model × collapsed spatial likelihood.

    P(z, K | data) ∝ P_count(N | K, μ, α) × ∏_k ML_locmix_k(z)

No partition prior P_partition(z|K). K is regularized solely by the NegBin
count model. Gibbs sweep uses purely spatial predictive (no CRP weights).
"""
struct DecoupledTarget <: AbstractTargetDensity end

"""
    MFMTarget <: AbstractTargetDensity

Full Mixture of Finite Mixtures target (from `smc-split` branch).

    P(z, K | data) ∝ P_count(N | K) × P_partition(z | K) × ∏_k ML_locmix_k(z)

where P_partition is the Dirichlet-Multinomial partition prior with γ = α (shape).
Miller & Harrison (2018). No double-counting: P_count × P_partition = P(z | K)
from the i.i.d. NegBin blinking model (see dev/math_refs/factorization_check.md).
"""
struct MFMTarget <: AbstractTargetDensity end

"""
    UniformPriorTarget <: AbstractTargetDensity

Uniform spatial prior baseline for comparison.

    P(z, K | data) ∝ P_count(N | K) × ∏_k ML_uniform_k(z)

Uses log_marginal_likelihood with -log(A) instead of locmix grid.
"""
struct UniformPriorTarget <: AbstractTargetDensity
    log_area::Float64
end

# ============================================================================
# Core interface
# ============================================================================

"""
    log_target(td, z, loc_precs, grid, μ, shape) -> Float64

Evaluate the unnormalized log target density for assignment vector `z`.

Arguments:
- `td`: Target density specification
- `z`: Assignment vector (z[i] = cluster label for loc i, 1-based contiguous)
- `loc_precs`: Precomputed LocPrecision for each localization
- `grid`: LocmixGrid (used by DecoupledTarget and MFMTarget)
- `μ`: Mean localizations per emitter
- `shape`: NegBin shape parameter (α)

Returns log π(z | data, μ, α) up to a normalizing constant.
"""
function log_target end

function log_target(::DecoupledTarget, z::AbstractVector{<:Integer},
                    loc_precs::Vector{LocPrecision}, grid::LocmixGrid,
                    μ::Float64, shape::Float64)
    N = length(z)
    K = _count_clusters(z)
    log_count = _log_count_term(K, N, shape, μ)
    log_spatial = _sum_cluster_ml_locmix(z, K, loc_precs, grid)
    return log_count + log_spatial
end

function log_target(::MFMTarget, z::AbstractVector{<:Integer},
                    loc_precs::Vector{LocPrecision}, grid::LocmixGrid,
                    μ::Float64, shape::Float64)
    N = length(z)
    K = _count_clusters(z)
    log_count = _log_count_term(K, N, shape, μ)
    log_partition = _log_mfm_partition(z, K, N, shape)
    log_spatial = _sum_cluster_ml_locmix(z, K, loc_precs, grid)
    return log_count + log_partition + log_spatial
end

function log_target(td::UniformPriorTarget, z::AbstractVector{<:Integer},
                    loc_precs::Vector{LocPrecision}, grid::LocmixGrid,
                    μ::Float64, shape::Float64)
    N = length(z)
    K = _count_clusters(z)
    log_count = _log_count_term(K, N, shape, μ)
    log_spatial = _sum_cluster_ml_uniform(z, K, loc_precs, td.log_area)
    return log_count + log_spatial
end

# ============================================================================
# Convenience wrapper
# ============================================================================

"""
    evaluate_target(td, z, locs; μ, shape) -> Float64

Convenience wrapper that builds LocPrecision and LocmixGrid automatically.
Use for one-off evaluations. For repeated calls with the same locs,
precompute loc_precs and grid and call `log_target` directly.
"""
function evaluate_target(td::AbstractTargetDensity,
                         z::AbstractVector{<:Integer},
                         locs::Vector{<:SMLMData.AbstractEmitter};
                         μ::Float64, shape::Float64)
    loc_precs = precompute_loc_precisions(locs)
    grid = build_locmix_grid(loc_precs)
    return log_target(td, z, loc_precs, grid, μ, shape)
end

# ============================================================================
# Internal helpers
# ============================================================================

"""Count distinct cluster labels in assignment vector."""
function _count_clusters(z::AbstractVector{<:Integer})
    max_k = 0
    @inbounds for i in eachindex(z)
        if z[i] > max_k
            max_k = z[i]
        end
    end
    # Count non-empty clusters (some labels might be unused)
    used = falses(max_k)
    @inbounds for i in eachindex(z)
        used[z[i]] = true
    end
    return count(used)
end

"""
    _log_count_term(K, N, shape, μ) -> Float64

Count-model log-likelihood: P(N | K, μ, α) = NegBin(N; K×α, α/(α+μ)).
"""
function _log_count_term(K::Int, N::Int, shape::Float64, μ::Float64)
    K <= 0 && return -Inf
    N < K && return -Inf
    p = shape / (shape + μ)
    return logpdf(NegativeBinomial(K * shape, p), N)
end

"""
    _build_clusters_from_z(z, loc_precs) -> (clusters, active_labels)

Reconstruct ClusterStats for each cluster from an assignment vector.
Returns a vector of (label, ClusterStats) pairs for non-empty clusters.
"""
function _build_clusters_from_z(z::AbstractVector{<:Integer},
                                loc_precs::Vector{LocPrecision})
    max_k = maximum(z)
    clusters = [ClusterStats() for _ in 1:max_k]
    @inbounds for i in eachindex(z)
        k = Int(z[i])
        clusters[k] = add_loc(clusters[k], loc_precs[i])
    end
    # Collect active clusters
    active = Int[]
    for k in 1:max_k
        if clusters[k].n > 0
            push!(active, k)
        end
    end
    return clusters, active
end

"""
    _sum_cluster_ml_locmix(z, K, loc_precs, grid) -> Float64

Sum of log marginal likelihoods under locmix grid prior across all active clusters.
"""
function _sum_cluster_ml_locmix(z::AbstractVector{<:Integer}, K::Int,
                                 loc_precs::Vector{LocPrecision}, grid::LocmixGrid)
    clusters, active = _build_clusters_from_z(z, loc_precs)
    total = 0.0
    for k in active
        total += log_marginal_likelihood_locmix(clusters[k], grid)
    end
    return total
end

"""
    _sum_cluster_ml_uniform(z, K, loc_precs, log_area) -> Float64

Sum of log marginal likelihoods under uniform spatial prior across all active clusters.
"""
function _sum_cluster_ml_uniform(z::AbstractVector{<:Integer}, K::Int,
                                  loc_precs::Vector{LocPrecision}, log_area::Float64)
    clusters, active = _build_clusters_from_z(z, loc_precs)
    total = 0.0
    for k in active
        total += log_marginal_likelihood(clusters[k], log_area)
    end
    return total
end

# ============================================================================
# MFM partition prior (ported from smc-split branch)
# ============================================================================

"""
    _log_mfm_partition(z, K, N, γ) -> Float64

Log of the Dirichlet-Multinomial partition prior (Miller & Harrison 2018):

    P_partition(z | K) = Γ(Kγ) / [Γ(γ)^K × Γ(N+Kγ)] × ∏_k Γ(n_k + γ)

where γ = shape (NegBin α parameter), NOT a free parameter.
"""
function _log_mfm_partition(z::AbstractVector{<:Integer}, K::Int, N::Int, γ::Float64)
    # Collect cluster sizes
    max_k = maximum(z)
    sizes = zeros(Int, max_k)
    @inbounds for i in eachindex(z)
        sizes[z[i]] += 1
    end

    # Γ(Kγ) / [Γ(γ)^K × Γ(N+Kγ)] × ∏_k Γ(n_k + γ)
    log_p = loggamma(K * γ) - K * loggamma(γ) - loggamma(N + K * γ)
    for k in 1:max_k
        if sizes[k] > 0
            log_p += loggamma(sizes[k] + γ)
        end
    end
    return log_p
end

"""
    log_mfm_partition_ratio(n_a, n_b, n_c, K, N, γ) -> Float64

Log ratio of MFM partition priors for a split C → (A, B) going K → K+1.

    Δ_partition = log[Γ(n_a+γ)Γ(n_b+γ) / (Γ(n_c+γ)Γ(γ))]
                + log[Γ((K+1)γ) / Γ(Kγ)]
                + log[Γ(N+Kγ) / Γ(N+(K+1)γ)]

For merge (K+1 → K): negate the result.
"""
function log_mfm_partition_ratio(n_a::Int, n_b::Int, n_c::Int,
                                  K::Int, N::Int, γ::Float64)
    # Cluster size term
    Δ_sizes = loggamma(n_a + γ) + loggamma(n_b + γ) - loggamma(n_c + γ) - loggamma(γ)
    # Normalization terms
    Kγ = K * γ
    Δ_norm = loggamma((K + 1) * γ) - loggamma(Kγ) +
             loggamma(N + Kγ) - loggamma(N + (K + 1) * γ)
    return Δ_sizes + Δ_norm
end

# Target density for diagnostic evaluation
#
# Evaluates the unnormalized log posterior for assignment vectors
# so that diagnostic tools can compare MCMC output to exact posteriors.

"""
    AbstractTargetDensity

Base type for target distributions over the partition space.
Implement `log_target(td, z, loc_precs, grid, μ, shape)` for new variants.
"""
abstract type AbstractTargetDensity end

"""
    DecoupledTarget <: AbstractTargetDensity

Count model × collapsed spatial likelihood, no partition prior.

    P(z, K | data) ∝ P_count(N | K, μ, α) × ∏_k ML_locmix_k(z)
"""
struct DecoupledTarget <: AbstractTargetDensity end

"""
    log_target(td, z, loc_precs, grid, μ, shape) -> Float64

Evaluate the unnormalized log target density for assignment vector `z`.
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

"""
    evaluate_target(td, z, locs; μ, shape) -> Float64

Convenience wrapper that builds LocPrecision and LocmixGrid automatically.
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

function _count_clusters(z::AbstractVector{<:Integer})
    max_k = 0
    @inbounds for i in eachindex(z)
        if z[i] > max_k
            max_k = z[i]
        end
    end
    used = falses(max_k)
    @inbounds for i in eachindex(z)
        used[z[i]] = true
    end
    return count(used)
end

function _log_count_term(K::Int, N::Int, shape::Float64, μ::Float64)
    K <= 0 && return -Inf
    N < K && return -Inf
    p = shape / (shape + μ)
    return logpdf(NegativeBinomial(K * shape, p), N)
end

function _build_clusters_from_z(z::AbstractVector{<:Integer},
                                loc_precs::Vector{LocPrecision})
    max_k = maximum(z)
    clusters = [ClusterStats() for _ in 1:max_k]
    @inbounds for i in eachindex(z)
        k = Int(z[i])
        clusters[k] = add_loc(clusters[k], loc_precs[i])
    end
    active = Int[]
    for k in 1:max_k
        if clusters[k].n > 0
            push!(active, k)
        end
    end
    return clusters, active
end

function _sum_cluster_ml_locmix(z::AbstractVector{<:Integer}, K::Int,
                                 loc_precs::Vector{LocPrecision}, grid::LocmixGrid)
    clusters, active = _build_clusters_from_z(z, loc_precs)
    total = 0.0
    for k in active
        total += log_marginal_likelihood_locmix(clusters[k], grid)
    end
    return total
end

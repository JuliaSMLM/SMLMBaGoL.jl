# Target density for diagnostic evaluation
#
# Evaluates the unnormalized log posterior for assignment vectors
# so that diagnostic tools can compare MCMC output to exact posteriors.

"""
    AbstractTargetDensity

Base type for target distributions over the partition space.
Implement `log_target(td, z, loc_precs, log_area, μ, shape, ρ)` for new variants.
"""
abstract type AbstractTargetDensity end

"""
    DecoupledTarget <: AbstractTargetDensity

Count model × Poisson K prior × collapsed spatial likelihood.

    P(z, K | data) ∝ P_count(N | K, μ, α) × P_K(K | ρ, A) × ∏_k ML_flat_k(z)

The Poisson(ρA) K prior combined with flat spatial -log(A) per cluster
makes the target area-independent: the A^K from the K prior cancels
the A^{-K} from K clusters' uniform position priors.
"""
struct DecoupledTarget <: AbstractTargetDensity end

"""
    log_target(td, z, loc_precs, log_area, μ, shape, ρ) -> Float64

Evaluate the unnormalized log target density for assignment vector `z`.
"""
function log_target end

function log_target(::DecoupledTarget, z::AbstractVector{<:Integer},
                    loc_precs::Vector{LocPrecision}, log_area::Float64,
                    μ::Float64, shape::Float64, ρ::Float64)
    N = length(z)
    K = _count_clusters(z)
    A = exp(log_area)
    log_count = _log_count_term(K, N, shape, μ)
    log_k_prior = log_prior_k_poisson(K, ρ, A)
    log_spatial = _sum_cluster_ml_flat(z, K, loc_precs, log_area)
    return log_count + log_k_prior + log_spatial
end

"""
    evaluate_target(td, z, locs; μ, shape, ρ) -> Float64

Convenience wrapper that builds LocPrecision automatically.
"""
function evaluate_target(td::AbstractTargetDensity,
                         z::AbstractVector{<:Integer},
                         locs::Vector{<:SMLMData.AbstractEmitter};
                         μ::Float64, shape::Float64, ρ::Float64=2.0)
    loc_precs = precompute_loc_precisions(locs)
    spatial_prior = UniformSpatialPrior(locs)
    log_area = log(area(spatial_prior))
    return log_target(td, z, loc_precs, log_area, μ, shape, ρ)
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

function _sum_cluster_ml_flat(z::AbstractVector{<:Integer}, K::Int,
                               loc_precs::Vector{LocPrecision}, log_area::Float64)
    clusters, active = _build_clusters_from_z(z, loc_precs)
    total = 0.0
    for k in active
        total += log_marginal_likelihood(clusters[k], log_area)
    end
    return total
end

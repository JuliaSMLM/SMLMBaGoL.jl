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
    DMFlatTarget <: AbstractTargetDensity

Collapsed target implied by:
- a Poisson emitter-count prior `K ~ Poisson(ρ A)`,
- iid per-emitter localization counts `n_k ~ NegBin(shape, p)`,
- a flat spatial prior on emitter positions over area `A`,
- and positions integrated out analytically.

For a labeled assignment vector `z` with active cluster sizes `n_1, ..., n_K`,

    P(z, K | data) ∝ P_K(K | ρ, A) × P_count(N | K, μ, shape) ×
                     P_DM(z | N, K, shape) × ∏_k ML_flat_k(z)

This is the collapsed form used to test whether the `DM` partition factor is
the correct consequence of the NegBin generative model.
"""
struct DMFlatTarget <: AbstractTargetDensity end

"""
    DirectNegBinFlatTarget <: AbstractTargetDensity

Direct form of the same flat/NegBin target before rewriting the count model
into `P_count(N | K) × P_DM(z | N, K)`.

For a labeled assignment vector `z` with active cluster sizes `n_1, ..., n_K`,

    P(z, K | data) ∝ P_K(K | ρ, A) × (∏_k NegBin(n_k; shape, p)) ×
                     (∏_k n_k! / N!) × ∏_k ML_flat_k(z)

The factor `∏_k n_k! / N!` is the probability of a specific labeled assignment
vector given the count vector. This target is algebraically identical to
`DMFlatTarget` and is useful as an independent implementation check.
"""
struct DirectNegBinFlatTarget <: AbstractTargetDensity end

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

function log_target(::DMFlatTarget, z::AbstractVector{<:Integer},
                    loc_precs::Vector{LocPrecision}, log_area::Float64,
                    μ::Float64, shape::Float64, ρ::Float64)
    N = length(z)
    K = _count_clusters(z)
    A = exp(log_area)
    counts = _cluster_counts(z)
    log_count = _log_count_term(K, N, shape, μ)
    log_k_prior = log_prior_k_poisson(K, ρ, A)
    log_dm = _log_dm_term(counts, N, shape)
    log_spatial = _sum_cluster_ml_flat(z, K, loc_precs, log_area)
    return log_k_prior + log_count + log_dm + log_spatial
end

function log_target(::DirectNegBinFlatTarget, z::AbstractVector{<:Integer},
                    loc_precs::Vector{LocPrecision}, log_area::Float64,
                    μ::Float64, shape::Float64, ρ::Float64)
    N = length(z)
    K = _count_clusters(z)
    A = exp(log_area)
    counts = _cluster_counts(z)
    log_k_prior = log_prior_k_poisson(K, ρ, A)
    log_counts = _log_product_nb_counts(counts, shape, μ)
    log_assign = _log_assignment_given_counts(counts, N)
    log_spatial = _sum_cluster_ml_flat(z, K, loc_precs, log_area)
    return log_k_prior + log_counts + log_assign + log_spatial
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

function _cluster_counts(z::AbstractVector{<:Integer})
    max_k = maximum(z)
    counts = zeros(Int, max_k)
    @inbounds for i in eachindex(z)
        counts[Int(z[i])] += 1
    end
    return [n for n in counts if n > 0]
end

function _log_dm_term(counts::Vector{Int}, N::Int, shape::Float64)
    K = length(counts)
    lp = loggamma(K * shape) - K * loggamma(shape) - loggamma(N + K * shape)
    @inbounds for n in counts
        lp += loggamma(n + shape)
    end
    return lp
end

function _log_product_nb_counts(counts::Vector{Int}, shape::Float64, μ::Float64)
    p = shape / (shape + μ)
    dist = NegativeBinomial(shape, p)
    lp = 0.0
    @inbounds for n in counts
        lp += logpdf(dist, n)
    end
    return lp
end

function _log_assignment_given_counts(counts::Vector{Int}, N::Int)
    lp = -logfactorial(N)
    @inbounds for n in counts
        lp += logfactorial(n)
    end
    return lp
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

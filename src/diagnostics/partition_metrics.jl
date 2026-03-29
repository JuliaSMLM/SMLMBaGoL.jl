# Partition comparison metrics
#
# Tools for comparing MCMC partition estimates to ground truth and
# assessing posterior quality. Ported from smc-split:examples/viz_metrics.jl.

"""
    variation_of_information(z1, z2) -> (vi_total, overseg, underseg)

Meilă (2007) Variation of Information between two partitions.

    VI(z1, z2) = H(z1 | z2) + H(z2 | z1)

where:
- overseg = H(z1 | z2): z1 splits clusters that z2 keeps together
- underseg = H(z2 | z1): z1 merges clusters that z2 keeps separate

When z1 = estimated and z2 = oracle:
- overseg measures how much the estimate oversegments
- underseg measures how much the estimate undersegments

Properties:
- VI(z, z) = (0, 0, 0)
- VI(z1, z2) = VI(z2, z1) (total is symmetric, but overseg/underseg swap)
- VI is a proper metric on partition space
"""
function variation_of_information(z1::AbstractVector{<:Integer},
                                   z2::AbstractVector{<:Integer})
    N = length(z1)
    @assert length(z2) == N "Partition vectors must have same length"
    N == 0 && return (vi_total=0.0, overseg=0.0, underseg=0.0)

    # Find cluster label ranges
    K1 = maximum(z1)
    K2 = maximum(z2)

    # Build contingency table C[k, l] = |{i : z1[i]=k, z2[i]=l}|
    C = zeros(Int, K1, K2)
    @inbounds for i in 1:N
        C[z1[i], z2[i]] += 1
    end

    # Marginals
    n1 = zeros(Int, K1)  # n1[k] = |cluster k in z1|
    n2 = zeros(Int, K2)  # n2[l] = |cluster l in z2|
    for k in 1:K1
        for l in 1:K2
            n1[k] += C[k, l]
            n2[l] += C[k, l]
        end
    end

    inv_N = 1.0 / N

    # H(z1 | z2) = -Σ_{k,l} (C[k,l]/N) log(C[k,l] / n2[l])
    # = overseg: conditional entropy of z1 given z2
    overseg = 0.0
    @inbounds for l in 1:K2
        n2[l] == 0 && continue
        for k in 1:K1
            c = C[k, l]
            c == 0 && continue
            overseg -= (c * inv_N) * log(c / n2[l])
        end
    end

    # H(z2 | z1) = -Σ_{k,l} (C[k,l]/N) log(C[k,l] / n1[k])
    # = underseg: conditional entropy of z2 given z1
    underseg = 0.0
    @inbounds for k in 1:K1
        n1[k] == 0 && continue
        for l in 1:K2
            c = C[k, l]
            c == 0 && continue
            underseg -= (c * inv_N) * log(c / n1[k])
        end
    end

    return (vi_total=overseg + underseg, overseg=overseg, underseg=underseg)
end

"""
    expected_posterior_loss(candidate, samples; loss=_vi_total_loss) -> Float64

Expected Posterior Loss of a candidate partition under the MCMC posterior.

    EPL(a) = (1/T) Σ_t L(a, Z^(t))

where L is a loss function comparing two partitions (default: VI total).

EPL is valid across different K (unlike unnormalized log posteriors).
Lower EPL = better posterior summary.

Arguments:
- `candidate`: Assignment vector (the partition being evaluated)
- `samples`: Vector of assignment vectors (MCMC samples, post-burn-in)
- `loss`: Loss function (z1, z2) -> Float64 (default: VI total)
"""
function expected_posterior_loss(candidate::AbstractVector{<:Integer},
                                 samples::AbstractVector{<:AbstractVector{<:Integer}};
                                 loss::Function=_vi_total_loss)
    T = length(samples)
    T == 0 && return 0.0
    total = 0.0
    for t in 1:T
        total += loss(candidate, samples[t])
    end
    return total / T
end

"""VI total as a scalar loss function."""
_vi_total_loss(z1, z2) = variation_of_information(z1, z2).vi_total

"""
    partition_diagnostics(dahl_z, oracle_z, samples) -> NamedTuple

Comprehensive partition diagnostic comparing Dahl estimate to oracle.

Returns:
- `vi_total, overseg, underseg`: VI decomposition (Dahl vs Oracle)
- `K_dahl, K_oracle`: Cluster counts
- `epl_dahl`: Expected posterior loss of Dahl partition
- `epl_oracle`: Expected posterior loss of oracle partition
- `epl_best`: Best EPL among stored samples
- `best_sample_idx`: Index of the best sample
- `regret_dahl`: EPL(Dahl) - EPL(best)
- `regret_oracle`: EPL(Oracle) - EPL(best)

Interpretation:
- EPL(Dahl) ≤ EPL(Oracle) → Dahl is genuinely better posterior summary
- EPL(Dahl) > EPL(Oracle) → sampler may be missing modes
- Large regret → estimator is suboptimal vs best stored sample
"""
function partition_diagnostics(dahl_z::AbstractVector{<:Integer},
                                oracle_z::AbstractVector{<:Integer},
                                samples::AbstractVector{<:AbstractVector{<:Integer}})
    # VI decomposition
    vi = variation_of_information(dahl_z, oracle_z)

    # Cluster counts
    K_dahl = length(unique(dahl_z))
    K_oracle = length(unique(oracle_z))

    # EPL computation
    epl_dahl = expected_posterior_loss(dahl_z, samples)
    epl_oracle = expected_posterior_loss(oracle_z, samples)

    # Find best sample
    epl_best = Inf
    best_idx = 0
    for (t, sample) in enumerate(samples)
        epl_t = expected_posterior_loss(sample, samples)
        if epl_t < epl_best
            epl_best = epl_t
            best_idx = t
        end
    end

    return (
        vi_total=vi.vi_total,
        overseg=vi.overseg,
        underseg=vi.underseg,
        K_dahl=K_dahl,
        K_oracle=K_oracle,
        epl_dahl=epl_dahl,
        epl_oracle=epl_oracle,
        epl_best=epl_best,
        best_sample_idx=best_idx,
        regret_dahl=epl_dahl - epl_best,
        regret_oracle=epl_oracle - epl_best,
    )
end

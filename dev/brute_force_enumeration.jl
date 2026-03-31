# Brute-force enumeration of ALL labeled partitions for small N
#
# For N=6 localizations and K_max=4, there are at most 4^6 = 4096 assignment
# vectors. We enumerate every one, compute the exact target density, normalize,
# then run the MCMC sampler and compare empirical visit frequencies to the
# exact posterior.
#
# Usage:
#   julia --project=. dev/brute_force_enumeration.jl

using SMLMBaGoL
using SMLMData
using SpecialFunctions: loggamma
using Distributions
using Random
using Printf

# ============================================================================
# Synthetic data generation
# ============================================================================

"""
Create N localizations from K_true emitters at given positions.
Each emitter gets `n_per` localizations with isotropic uncertainty σ.
"""
function make_locs(emitter_positions::Vector{Tuple{Float64,Float64}},
                   n_per::Int, σ::Float64; seed::Int=42)
    rng = MersenneTwister(seed)
    locs = SMLMData.Emitter2DFit[]
    id = 0
    for (ex, ey) in emitter_positions
        for _ in 1:n_per
            id += 1
            x = ex + σ * randn(rng)
            y = ey + σ * randn(rng)
            push!(locs, SMLMData.Emitter2DFit(
                x, y, 1000.0, 0.0,   # photons, bg
                σ, σ, 0.0,            # σ_x, σ_y, σ_xy
                0.0, 0.0,             # σ_photons, σ_bg
                1, 1, 0, id           # frame, dataset, track_id, id
            ))
        end
    end
    return locs
end

# ============================================================================
# Exact enumeration
# ============================================================================

"""
Enumerate all assignment vectors z ∈ {1,...,K_max}^N that have no empty
clusters (i.e., every label 1..K appears at least once for some K ≤ K_max).

Returns a list of (z, K_effective) where K_effective = number of distinct
labels actually used.

For efficiency, we enumerate ALL K_max^N vectors and keep only those with
contiguous labels starting at 1 and all used. This avoids double-counting
since we canonicalize: labels are in order of first appearance.
"""
function enumerate_canonical_partitions(N::Int, K_max::Int)
    partitions = Vector{Tuple{Vector{Int}, Int}}()
    z = ones(Int, N)

    function _recurse!(pos::Int)
        if pos > N
            # Check: labels must be 1..K contiguous and all used
            K_used = maximum(z)
            if K_used > K_max
                return
            end
            # Canonicalize: first appearance of label k must be before first
            # appearance of label k+1. This ensures each partition is counted
            # exactly once.
            first_appear = fill(N + 1, K_used)
            for i in 1:N
                first_appear[z[i]] = min(first_appear[z[i]], i)
            end
            for k in 1:K_used-1
                if first_appear[k] > first_appear[k+1]
                    return  # Not canonical
                end
            end
            push!(partitions, (copy(z), K_used))
            return
        end
        # The next label can be any existing label or the next new one
        max_so_far = pos > 1 ? maximum(@view z[1:pos-1]) : 0
        for label in 1:min(max_so_far + 1, K_max)
            z[pos] = label
            _recurse!(pos + 1)
        end
    end

    _recurse!(1)
    return partitions
end

"""
Compute the exact (unnormalized) log target density for assignment vector z.

log π(z) = log P_count(N|K) + log P_partition(z|K) + Σ_k log ML_k + log P_K(K|ρ,A)

where:
- P_count uses _log_count_posterior (NegBin)
- P_partition uses the Dirichlet-Multinomial formula (MFM)
- ML_k uses log_marginal_likelihood with log_area (uniform spatial prior)
"""
function log_target_density(z::Vector{Int}, K::Int, N::Int,
                            locs::Vector{<:SMLMData.AbstractEmitter},
                            loc_precs::Vector{SMLMBaGoL.LocPrecision},
                            log_area::Float64,
                            μ::Float64, shape::Float64,
                            ρ::Float64, A::Float64)
    # 1. Count model: P(N|K)
    log_count = SMLMBaGoL._log_count_posterior(K, N, shape, μ)

    # 2. MFM partition prior: P(z|K) under Dirichlet-Multinomial
    # P(z|K) = Γ(Kγ) / [Γ(γ)^K × Γ(N+Kγ)] × ∏_k Γ(n_k + γ)
    γ = shape
    cluster_sizes = zeros(Int, K)
    for i in 1:N
        cluster_sizes[z[i]] += 1
    end

    log_partition = loggamma(K * γ) - K * loggamma(γ) - loggamma(N + K * γ)
    for k in 1:K
        log_partition += loggamma(cluster_sizes[k] + γ)
    end

    # 3. Spatial marginal likelihood: Σ_k log ML_k
    log_spatial = 0.0
    for k in 1:K
        cs = SMLMBaGoL.ClusterStats()
        for i in 1:N
            if z[i] == k
                cs = SMLMBaGoL.add_loc(cs, loc_precs[i])
            end
        end
        log_spatial += SMLMBaGoL.log_marginal_likelihood(cs, log_area)
    end

    # 4. Poisson(ρA) prior on K
    log_k_prior = SMLMBaGoL.log_prior_k_poisson(K, ρ, A)

    return log_count + log_partition + log_spatial + log_k_prior
end

# ============================================================================
# Partition hash for matching MCMC samples to enumerated partitions
# ============================================================================

"""
Convert an assignment vector to a canonical form: relabel so that labels
appear in order of first occurrence (1, 2, 3, ...).
"""
function canonicalize(z::Vector{<:Integer})
    mapping = Dict{Int, Int}()
    next_label = 1
    result = Vector{Int}(undef, length(z))
    for i in eachindex(z)
        label = Int(z[i])
        if !haskey(mapping, label)
            mapping[label] = next_label
            next_label += 1
        end
        result[i] = mapping[label]
    end
    return result
end

"""Hash a canonical assignment vector to a unique integer for Dict lookup."""
function partition_hash(z_canon::Vector{Int}, K_max::Int)
    h = 0
    for i in eachindex(z_canon)
        h = h * (K_max + 1) + z_canon[i]
    end
    return h
end

# ============================================================================
# MCMC sampling with state tracking
# ============================================================================

"""
Run the collapsed Gibbs sampler and record the canonical assignment at each
iteration (post burn-in). Returns a Dict mapping partition_hash -> visit count,
plus a K time series for ESS estimation.
"""
function run_mcmc_with_tracking(locs::Vector{<:SMLMData.AbstractEmitter},
                                 n_iterations::Int, burn_in::Int,
                                 μ::Float64, shape::Float64, ρ::Float64,
                                 K_max::Int;
                                 seed::Int=123)
    Random.seed!(seed)

    spatial_prior = SMLMBaGoL.UniformSpatialPrior(locs)
    state = SMLMBaGoL.initialize_collapsed_state(locs, spatial_prior)

    visit_counts = Dict{Int, Int}()
    k_trace = Int[]  # K time series for ESS estimation
    n_samples = 0

    acceptance = Dict{Symbol, Tuple{Int, Int}}(
        :gibbs_sweep => (0, 0),
        :split => (0, 0),
        :merge => (0, 0),
        :birth => (0, 0),
        :death => (0, 0),
    )

    for iter in 1:n_iterations
        r = rand()
        if r < 0.50
            SMLMBaGoL.gibbs_allocation_sweep!(state, locs, μ, shape)
            prev = acceptance[:gibbs_sweep]
            acceptance[:gibbs_sweep] = (prev[1] + 1, prev[2] + 1)
        elseif r < 0.75
            accepted, move_type = SMLMBaGoL.propose_split_merge!(state, locs, μ, shape, ρ)
            prev = acceptance[move_type]
            acceptance[move_type] = (prev[1] + (accepted ? 1 : 0), prev[2] + 1)
        else
            for _bd in 1:5  # BD burst: 5 substeps per selection
                accepted, move_type = SMLMBaGoL.propose_birth_death!(state, locs, μ, shape, ρ)
                prev = acceptance[move_type]
                acceptance[move_type] = (prev[1] + (accepted ? 1 : 0), prev[2] + 1)
            end
        end

        if iter > burn_in
            z_canon = canonicalize(state.assignments)
            K = state.n_active
            push!(k_trace, K)
            if K <= K_max
                h = partition_hash(z_canon, K_max)
                visit_counts[h] = get(visit_counts, h, 0) + 1
                n_samples += 1
            else
                # Partition has K > K_max, still count it
                h = partition_hash(z_canon, max(K, K_max))
                visit_counts[h] = get(visit_counts, h, 0) + 1
                n_samples += 1
            end
        end
    end

    return visit_counts, n_samples, acceptance, k_trace
end

"""
Estimate effective sample size from a time series using batch means.
"""
function estimate_ess_from_trace(trace::Vector{Int}; n_batch::Int=100)
    n = length(trace)
    n < n_batch && return Float64(n)
    x = Float64.(trace)
    grand_mean = mean(x)
    naive_var = var(x)
    naive_var < 1e-15 && return Float64(n)  # constant chain

    batch_size = n ÷ n_batch
    batch_means = Float64[]
    for b in 1:n_batch
        i0 = (b - 1) * batch_size + 1
        i1 = b * batch_size
        push!(batch_means, mean(@view x[i0:i1]))
    end
    batch_var = var(batch_means)
    batch_var < 1e-15 && return Float64(n)

    # ESS = n × (naive_var / (batch_size × batch_var))
    # The batch_size factor accounts for the variance of means scaling as σ²/m
    ess = n * naive_var / (batch_size * batch_var)
    return clamp(ess, 1.0, Float64(n))
end

# ============================================================================
# Comparison and diagnostics
# ============================================================================

"""
KL divergence D_KL(p || q) where p = exact, q = empirical.
Handles zeros: skip entries where p=0; use Laplace smoothing for q=0.
"""
function kl_divergence(p_exact::Vector{Float64}, p_empirical::Vector{Float64})
    @assert length(p_exact) == length(p_empirical)
    kl = 0.0
    for i in eachindex(p_exact)
        if p_exact[i] > 1e-15
            q = max(p_empirical[i], 1e-15)  # Smoothing to avoid log(0)
            kl += p_exact[i] * log(p_exact[i] / q)
        end
    end
    return kl
end

"""
Total variation distance: TV(p, q) = ½ Σ |p_i - q_i|
"""
function tv_distance(p::Vector{Float64}, q::Vector{Float64})
    return 0.5 * sum(abs.(p .- q))
end


# ============================================================================
# Main test
# ============================================================================

function run_test(;
    emitter_positions::Vector{Tuple{Float64,Float64}},
    n_per::Int,
    σ::Float64,
    μ::Float64,
    shape::Float64,
    K_max::Int,
    n_iterations::Int,
    burn_in::Int,
    test_name::String,
    seed_data::Int=42,
    seed_mcmc::Int=123,
)
    println("\n" * "="^80)
    println("TEST: $test_name")
    println("="^80)

    N = n_per * length(emitter_positions)
    K_true = length(emitter_positions)
    println("  K_true = $K_true, N = $N, σ = $σ, μ = $μ, shape = $shape, K_max = $K_max")

    # Generate data
    locs = make_locs(emitter_positions, n_per, σ; seed=seed_data)
    println("  Locs generated: $(length(locs)) localizations")
    for (i, loc) in enumerate(locs)
        @printf("    loc %d: (%.4f, %.4f)\n", i, loc.x, loc.y)
    end

    # Build spatial prior and compute log_area
    loc_precs = SMLMBaGoL.precompute_loc_precisions(locs)
    spatial_prior = SMLMBaGoL.UniformSpatialPrior(locs)
    log_area = log(SMLMBaGoL.area(spatial_prior))
    ρ = 2.0  # Poisson rate for K prior
    A = exp(log_area)

    # -----------------------------------------------------------------------
    # Step 1: Enumerate all canonical partitions
    # -----------------------------------------------------------------------
    println("\n  Enumerating canonical partitions for N=$N, K_max=$K_max...")
    partitions = enumerate_canonical_partitions(N, K_max)
    println("  Found $(length(partitions)) canonical partitions")

    # Count by K
    k_counts = Dict{Int, Int}()
    for (_, K) in partitions
        k_counts[K] = get(k_counts, K, 0) + 1
    end
    for K in sort(collect(keys(k_counts)))
        println("    K=$K: $(k_counts[K]) partitions")
    end

    # -----------------------------------------------------------------------
    # Step 2: Compute exact log target for each partition
    # -----------------------------------------------------------------------
    println("\n  Computing exact log target densities...")
    log_targets = Float64[]
    hashes = Int[]
    for (z, K) in partitions
        lt = log_target_density(z, K, N, locs, loc_precs, log_area, μ, shape, ρ, A)
        push!(log_targets, lt)
        z_canon = canonicalize(z)
        push!(hashes, partition_hash(z_canon, K_max))
    end

    # Normalize to get exact posterior
    max_lt = maximum(log_targets)
    probs_unnorm = exp.(log_targets .- max_lt)
    Z = sum(probs_unnorm)
    probs_exact = probs_unnorm ./ Z

    # Show top partitions by exact probability
    sorted_idx = sortperm(probs_exact, rev=true)
    println("\n  Top 15 partitions by exact posterior probability:")
    println("  " * "-"^76)
    @printf("  %5s  %6s  %-30s  %12s  %12s\n", "Rank", "K", "z (canonical)", "P(exact)", "log π")
    println("  " * "-"^76)
    for rank in 1:min(15, length(sorted_idx))
        i = sorted_idx[rank]
        z, K = partitions[i]
        @printf("  %5d  %6d  %-30s  %12.6f  %12.2f\n",
                rank, K, string(z), probs_exact[i], log_targets[i])
    end

    # Marginal P(K)
    println("\n  Marginal P(K) from exact enumeration:")
    pk_exact = Dict{Int, Float64}()
    for (i, (_, K)) in enumerate(partitions)
        pk_exact[K] = get(pk_exact, K, 0.0) + probs_exact[i]
    end
    for K in sort(collect(keys(pk_exact)))
        @printf("    P(K=%d) = %.6f\n", K, pk_exact[K])
    end

    # -----------------------------------------------------------------------
    # Step 3: Run MCMC and collect visit frequencies
    # -----------------------------------------------------------------------
    println("\n  Running MCMC: $n_iterations iterations, $burn_in burn-in...")
    visit_counts, n_samples, acceptance, k_trace = run_mcmc_with_tracking(
        locs, n_iterations, burn_in, μ, shape, ρ, K_max; seed=seed_mcmc)

    println("  Collected $n_samples post-burn-in samples")
    ess_k = estimate_ess_from_trace(k_trace)
    @printf("  ESS(K): %.0f  (autocorrelation factor: %.1fx)\n", ess_k, n_samples / ess_k)
    for (move, (acc, tot)) in acceptance
        rate = tot > 0 ? round(100 * acc / tot, digits=1) : 0.0
        println("    $move: $rate% ($acc/$tot)")
    end

    # -----------------------------------------------------------------------
    # Step 4: Build empirical distribution over the same partition space
    # -----------------------------------------------------------------------
    # Build hash -> index mapping
    hash_to_idx = Dict{Int, Int}()
    for (i, h) in enumerate(hashes)
        hash_to_idx[h] = i
    end

    probs_empirical = zeros(Float64, length(partitions))
    unmatched_samples = 0
    for (h, count) in visit_counts
        if haskey(hash_to_idx, h)
            probs_empirical[hash_to_idx[h]] = count / n_samples
        else
            unmatched_samples += count
        end
    end

    if unmatched_samples > 0
        pct = round(100 * unmatched_samples / n_samples, digits=2)
        println("\n  WARNING: $unmatched_samples samples ($pct%) visited partitions with K > $K_max")
    end

    # Renormalize empirical to only the enumerated space (for fair comparison)
    matched_mass = sum(probs_empirical)
    if matched_mass > 0
        probs_empirical ./= matched_mass
    end

    # -----------------------------------------------------------------------
    # Step 5: Compare exact vs empirical
    # -----------------------------------------------------------------------
    println("\n  Comparison: top 20 partitions")
    println("  " * "-"^86)
    @printf("  %5s  %6s  %-25s  %12s  %12s  %12s\n",
            "Rank", "K", "z", "P(exact)", "P(MCMC)", "Ratio")
    println("  " * "-"^86)
    for rank in 1:min(20, length(sorted_idx))
        i = sorted_idx[rank]
        z, K = partitions[i]
        pe = probs_exact[i]
        pm = probs_empirical[i]
        ratio = pm > 0 ? pe / pm : Inf
        @printf("  %5d  %6d  %-25s  %12.6f  %12.6f  %12.4f\n",
                rank, K, string(z), pe, pm, ratio)
    end

    # Marginal P(K) comparison
    println("\n  Marginal P(K) comparison:")
    pk_mcmc = Dict{Int, Float64}()
    for (i, (_, K)) in enumerate(partitions)
        pk_mcmc[K] = get(pk_mcmc, K, 0.0) + probs_empirical[i]
    end
    @printf("  %6s  %12s  %12s  %12s\n", "K", "P_exact", "P_MCMC", "Ratio")
    for K in sort(collect(keys(pk_exact)))
        pe = pk_exact[K]
        pm = get(pk_mcmc, K, 0.0)
        ratio = pm > 0 ? pe / pm : Inf
        @printf("  %6d  %12.6f  %12.6f  %12.4f\n", K, pe, pm, ratio)
    end

    # -----------------------------------------------------------------------
    # Step 5: Metrics
    # -----------------------------------------------------------------------

    # Per-partition metrics
    kl = kl_divergence(probs_exact, probs_empirical)
    tv = tv_distance(probs_exact, probs_empirical)

    # Marginal P(K) metrics (more robust — aggregates over partition permutations)
    K_vals = sort(collect(keys(pk_exact)))
    pk_exact_vec = [pk_exact[K] for K in K_vals]
    pk_mcmc_vec = [get(pk_mcmc, K, 0.0) for K in K_vals]
    # Renormalize marginal MCMC
    pk_mcmc_sum = sum(pk_mcmc_vec)
    if pk_mcmc_sum > 0
        pk_mcmc_vec ./= pk_mcmc_sum
    end
    kl_marginal = kl_divergence(pk_exact_vec, pk_mcmc_vec)
    tv_marginal = tv_distance(pk_exact_vec, pk_mcmc_vec)

    # Chi-squared on marginal P(K) with ESS correction for autocorrelation
    n_eff = ess_k
    chi2_marginal = 0.0
    n_k_bins = 0
    for i in eachindex(pk_exact_vec)
        expected = pk_exact_vec[i] * n_eff
        if expected >= 3  # relaxed threshold for small K_max
            observed = pk_mcmc_vec[i] * n_eff
            chi2_marginal += (observed - expected)^2 / expected
            n_k_bins += 1
        end
    end
    dof_k = max(n_k_bins - 1, 1)
    chi2_k_pval = n_k_bins > 1 ? ccdf(Chisq(dof_k), chi2_marginal) : NaN

    # Maximum ratio deviation on marginal P(K) for bins with P > 0.01
    max_ratio_dev = 0.0
    for i in eachindex(pk_exact_vec)
        if pk_exact_vec[i] > 0.01 && pk_mcmc_vec[i] > 1e-10
            ratio = pk_exact_vec[i] / pk_mcmc_vec[i]
            max_ratio_dev = max(max_ratio_dev, abs(log(ratio)))
        end
    end

    println("\n  " * "="^60)
    println("  METRICS")
    println("  " * "="^60)
    println("  Per-partition:")
    @printf("    KL divergence (exact || empirical):  %.6f\n", kl)
    @printf("    Total variation distance:            %.6f\n", tv)
    println("  Marginal P(K):")
    @printf("    KL divergence:                       %.6f\n", kl_marginal)
    @printf("    Total variation distance:            %.6f\n", tv_marginal)
    @printf("    Chi-squared (ESS-corrected):         %.2f  (dof=%d, ESS=%.0f, p=%.4f)\n",
            chi2_marginal, dof_k, n_eff, chi2_k_pval)
    @printf("    Max log-ratio deviation:             %.4f  (= %.2fx)\n",
            max_ratio_dev, exp(max_ratio_dev))
    @printf("  Matched sample mass:                   %.4f\n", matched_mass)

    # Verdict based on marginal P(K) — the quantity that actually matters for BaGoL
    #
    # Thresholds:
    # - KL(marginal) < 0.05: the K distribution is close
    # - TV(marginal) < 0.10: absolute deviation bounded
    # - max_ratio_dev < 0.5 (i.e., ratio within [0.6, 1.6]): no bin off by > 60%
    #
    # The per-partition chi-squared will always fail for correlated MCMC,
    # so we report it but don't use it for the verdict.
    pass_kl = kl_marginal < 0.05
    pass_tv = tv_marginal < 0.10
    pass_ratio = max_ratio_dev < 0.50
    pass = pass_kl && pass_tv && pass_ratio

    verdict = pass ? "PASS" : "FAIL"
    color_start = pass ? "\033[32m" : "\033[31m"
    color_end = "\033[0m"

    println()
    println("  $(color_start)>>> VERDICT: $verdict <<<$(color_end)")
    if !pass
        println("  Failing criteria:")
        if !pass_kl
            @printf("    Marginal KL %.4f > 0.05 threshold\n", kl_marginal)
        end
        if !pass_tv
            @printf("    Marginal TV %.4f > 0.10 threshold\n", tv_marginal)
        end
        if !pass_ratio
            @printf("    Max log-ratio %.4f > 0.50 (ratio = %.2fx)\n",
                    max_ratio_dev, exp(max_ratio_dev))
        end
    end

    return pass, kl_marginal, tv_marginal, max_ratio_dev
end

# ============================================================================
# Run tests
# ============================================================================

function main()
    println("Brute-Force Enumeration Test for Collapsed Gibbs Sampler")
    println("Comparing exact posterior P(z|data) to MCMC visit frequencies")

    all_pass = true

    # Test 1: Well-separated dimer (easy case — should be K=2 dominated)
    pass1, _, _, _ = run_test(
        emitter_positions = [(0.0, 0.0), (0.05, 0.0)],  # 50nm apart
        n_per = 3,
        σ = 0.005,  # 5nm uncertainty
        μ = 3.0,
        shape = 2.0,
        K_max = 4,
        n_iterations = 200_000,
        burn_in = 10_000,
        test_name = "Well-separated dimer (50nm apart, σ=5nm, d/σ=10)",
        seed_data = 42,
        seed_mcmc = 123,
    )
    all_pass &= pass1

    # Test 2: Close dimer (hard case — K=1 and K=2 both have significant mass)
    pass2, _, _, _ = run_test(
        emitter_positions = [(0.0, 0.0), (0.015, 0.0)],  # 15nm apart
        n_per = 3,
        σ = 0.005,  # 5nm uncertainty => d/σ = 3
        μ = 3.0,
        shape = 2.0,
        K_max = 4,
        n_iterations = 500_000,
        burn_in = 20_000,
        test_name = "Close dimer (15nm apart, σ=5nm, d/σ=3)",
        seed_data = 42,
        seed_mcmc = 456,
    )
    all_pass &= pass2

    # Test 3: Single emitter (should be K=1 dominated)
    pass3, _, _, _ = run_test(
        emitter_positions = [(0.0, 0.0)],
        n_per = 6,
        σ = 0.005,
        μ = 6.0,
        shape = 2.0,
        K_max = 4,
        n_iterations = 200_000,
        burn_in = 10_000,
        test_name = "Single emitter (N=6, μ=6)",
        seed_data = 42,
        seed_mcmc = 789,
    )
    all_pass &= pass3

    # Test 4: Well-separated dimer with N=8 (larger state space)
    pass4, _, _, _ = run_test(
        emitter_positions = [(0.0, 0.0), (0.04, 0.0)],  # 40nm apart
        n_per = 4,
        σ = 0.005,  # d/σ = 8
        μ = 4.0,
        shape = 2.0,
        K_max = 4,
        n_iterations = 500_000,
        burn_in = 20_000,
        test_name = "Well-separated dimer N=8 (40nm apart, σ=5nm, d/σ=8)",
        seed_data = 42,
        seed_mcmc = 321,
    )
    all_pass &= pass4

    println("\n" * "="^80)
    if all_pass
        println("\033[32mALL TESTS PASSED\033[0m")
    else
        println("\033[31mSOME TESTS FAILED\033[0m")
    end
    println("="^80)
end

main()

# ============================================================================
# SPLIT/MERGE ONLY mode — isolate whether bias comes from Gibbs or split/merge
# ============================================================================

"""
Run ONLY split/merge moves (no Gibbs allocation sweeps) and record the
canonical assignment at each iteration (post burn-in).

This isolates the split/merge kernel: if the marginal P(K) bias persists
here, the split/merge kernel is the culprit. If it disappears, the Gibbs
sweep is at fault.
"""
function run_splitmerge_only_tracking(locs::Vector{<:SMLMData.AbstractEmitter},
                                      n_iterations::Int, burn_in::Int,
                                      μ::Float64, shape::Float64, ρ::Float64,
                                      K_max::Int;
                                      seed::Int=123)
    Random.seed!(seed)

    spatial_prior = SMLMBaGoL.UniformSpatialPrior(locs)
    state = SMLMBaGoL.initialize_collapsed_state(locs, spatial_prior)

    visit_counts = Dict{Int, Int}()
    k_trace = Int[]
    n_samples = 0

    acceptance = Dict{Symbol, Tuple{Int, Int}}(
        :split => (0, 0),
        :merge => (0, 0),
    )

    for iter in 1:n_iterations
        # ONLY split/merge — no Gibbs sweep
        accepted, move_type = SMLMBaGoL.propose_split_merge!(state, locs, μ, shape, ρ)
        prev = acceptance[move_type]
        acceptance[move_type] = (prev[1] + (accepted ? 1 : 0), prev[2] + 1)

        if iter > burn_in
            z_canon = canonicalize(state.assignments)
            K = state.n_active
            push!(k_trace, K)
            if K <= K_max
                h = partition_hash(z_canon, K_max)
                visit_counts[h] = get(visit_counts, h, 0) + 1
                n_samples += 1
            else
                h = partition_hash(z_canon, max(K, K_max))
                visit_counts[h] = get(visit_counts, h, 0) + 1
                n_samples += 1
            end
        end
    end

    return visit_counts, n_samples, acceptance, k_trace
end

# ============================================================================
# Split/merge-only test
# ============================================================================

function run_splitmerge_only_test(;
    emitter_positions::Vector{Tuple{Float64,Float64}},
    n_per::Int,
    σ::Float64,
    μ::Float64,
    shape::Float64,
    K_max::Int,
    n_iterations::Int,
    burn_in::Int,
    test_name::String,
    seed_data::Int=42,
    seed_mcmc::Int=123,
)
    println("\n" * "="^80)
    println("TEST: $test_name")
    println("="^80)

    N = n_per * length(emitter_positions)
    K_true = length(emitter_positions)
    println("  K_true = $K_true, N = $N, σ = $σ, μ = $μ, shape = $shape, K_max = $K_max")

    # Generate data
    locs = make_locs(emitter_positions, n_per, σ; seed=seed_data)
    println("  Locs generated: $(length(locs)) localizations")
    for (i, loc) in enumerate(locs)
        @printf("    loc %d: (%.4f, %.4f)\n", i, loc.x, loc.y)
    end

    # Build spatial prior and compute log_area
    loc_precs = SMLMBaGoL.precompute_loc_precisions(locs)
    spatial_prior = SMLMBaGoL.UniformSpatialPrior(locs)
    log_area = log(SMLMBaGoL.area(spatial_prior))
    ρ = 2.0
    A = exp(log_area)

    # -----------------------------------------------------------------------
    # Step 1: Enumerate all canonical partitions
    # -----------------------------------------------------------------------
    println("\n  Enumerating canonical partitions for N=$N, K_max=$K_max...")
    partitions = enumerate_canonical_partitions(N, K_max)
    println("  Found $(length(partitions)) canonical partitions")

    k_counts = Dict{Int, Int}()
    for (_, K) in partitions
        k_counts[K] = get(k_counts, K, 0) + 1
    end
    for K in sort(collect(keys(k_counts)))
        println("    K=$K: $(k_counts[K]) partitions")
    end

    # -----------------------------------------------------------------------
    # Step 2: Compute exact log target for each partition
    # -----------------------------------------------------------------------
    println("\n  Computing exact log target densities...")
    log_targets = Float64[]
    hashes = Int[]
    for (z, K) in partitions
        lt = log_target_density(z, K, N, locs, loc_precs, log_area, μ, shape, ρ, A)
        push!(log_targets, lt)
        z_canon = canonicalize(z)
        push!(hashes, partition_hash(z_canon, K_max))
    end

    max_lt = maximum(log_targets)
    probs_unnorm = exp.(log_targets .- max_lt)
    Z = sum(probs_unnorm)
    probs_exact = probs_unnorm ./ Z

    # Marginal P(K) from exact enumeration
    println("\n  Marginal P(K) from exact enumeration:")
    pk_exact = Dict{Int, Float64}()
    for (i, (_, K)) in enumerate(partitions)
        pk_exact[K] = get(pk_exact, K, 0.0) + probs_exact[i]
    end
    for K in sort(collect(keys(pk_exact)))
        @printf("    P(K=%d) = %.6f\n", K, pk_exact[K])
    end

    # -----------------------------------------------------------------------
    # Step 3: Run split/merge-only MCMC
    # -----------------------------------------------------------------------
    println("\n  Running SPLIT/MERGE ONLY MCMC: $n_iterations iterations, $burn_in burn-in...")
    visit_counts, n_samples, acceptance, k_trace = run_splitmerge_only_tracking(
        locs, n_iterations, burn_in, μ, shape, ρ, K_max; seed=seed_mcmc)

    println("  Collected $n_samples post-burn-in samples")
    ess_k = estimate_ess_from_trace(k_trace)
    @printf("  ESS(K): %.0f  (autocorrelation factor: %.1fx)\n", ess_k, n_samples / ess_k)
    for (move, (acc, tot)) in acceptance
        rate = tot > 0 ? round(100 * acc / tot, digits=1) : 0.0
        println("    $move: $rate% ($acc/$tot)")
    end

    # -----------------------------------------------------------------------
    # Step 4: Build empirical distribution
    # -----------------------------------------------------------------------
    hash_to_idx = Dict{Int, Int}()
    for (i, h) in enumerate(hashes)
        hash_to_idx[h] = i
    end

    probs_empirical = zeros(Float64, length(partitions))
    unmatched_samples = 0
    for (h, count) in visit_counts
        if haskey(hash_to_idx, h)
            probs_empirical[hash_to_idx[h]] = count / n_samples
        else
            unmatched_samples += count
        end
    end

    if unmatched_samples > 0
        pct = round(100 * unmatched_samples / n_samples, digits=2)
        println("\n  WARNING: $unmatched_samples samples ($pct%) visited partitions with K > $K_max")
    end

    matched_mass = sum(probs_empirical)
    if matched_mass > 0
        probs_empirical ./= matched_mass
    end

    # -----------------------------------------------------------------------
    # Step 5: Compare exact vs empirical
    # -----------------------------------------------------------------------
    sorted_idx = sortperm(probs_exact, rev=true)

    println("\n  Comparison: top 20 partitions")
    println("  " * "-"^86)
    @printf("  %5s  %6s  %-25s  %12s  %12s  %12s\n",
            "Rank", "K", "z", "P(exact)", "P(SM-only)", "Ratio")
    println("  " * "-"^86)
    for rank in 1:min(20, length(sorted_idx))
        i = sorted_idx[rank]
        z, K = partitions[i]
        pe = probs_exact[i]
        pm = probs_empirical[i]
        ratio = pm > 0 ? pe / pm : Inf
        @printf("  %5d  %6d  %-25s  %12.6f  %12.6f  %12.4f\n",
                rank, K, string(z), pe, pm, ratio)
    end

    # Marginal P(K) comparison
    println("\n  Marginal P(K) comparison:")
    pk_mcmc = Dict{Int, Float64}()
    for (i, (_, K)) in enumerate(partitions)
        pk_mcmc[K] = get(pk_mcmc, K, 0.0) + probs_empirical[i]
    end
    @printf("  %6s  %12s  %12s  %12s\n", "K", "P_exact", "P_SM-only", "Ratio")
    for K in sort(collect(keys(pk_exact)))
        pe = pk_exact[K]
        pm = get(pk_mcmc, K, 0.0)
        ratio = pm > 0 ? pe / pm : Inf
        @printf("  %6d  %12.6f  %12.6f  %12.4f\n", K, pe, pm, ratio)
    end

    # -----------------------------------------------------------------------
    # Step 6: Metrics
    # -----------------------------------------------------------------------
    kl = kl_divergence(probs_exact, probs_empirical)
    tv = tv_distance(probs_exact, probs_empirical)

    K_vals = sort(collect(keys(pk_exact)))
    pk_exact_vec = [pk_exact[K] for K in K_vals]
    pk_mcmc_vec = [get(pk_mcmc, K, 0.0) for K in K_vals]
    pk_mcmc_sum = sum(pk_mcmc_vec)
    if pk_mcmc_sum > 0
        pk_mcmc_vec ./= pk_mcmc_sum
    end
    kl_marginal = kl_divergence(pk_exact_vec, pk_mcmc_vec)
    tv_marginal = tv_distance(pk_exact_vec, pk_mcmc_vec)

    n_eff = ess_k
    chi2_marginal = 0.0
    n_k_bins = 0
    for i in eachindex(pk_exact_vec)
        expected = pk_exact_vec[i] * n_eff
        if expected >= 3
            observed = pk_mcmc_vec[i] * n_eff
            chi2_marginal += (observed - expected)^2 / expected
            n_k_bins += 1
        end
    end
    dof_k = max(n_k_bins - 1, 1)
    chi2_k_pval = n_k_bins > 1 ? ccdf(Chisq(dof_k), chi2_marginal) : NaN

    max_ratio_dev = 0.0
    for i in eachindex(pk_exact_vec)
        if pk_exact_vec[i] > 0.01 && pk_mcmc_vec[i] > 1e-10
            ratio = pk_exact_vec[i] / pk_mcmc_vec[i]
            max_ratio_dev = max(max_ratio_dev, abs(log(ratio)))
        end
    end

    println("\n  " * "="^60)
    println("  METRICS (SPLIT/MERGE ONLY)")
    println("  " * "="^60)
    println("  Per-partition:")
    @printf("    KL divergence (exact || empirical):  %.6f\n", kl)
    @printf("    Total variation distance:            %.6f\n", tv)
    println("  Marginal P(K):")
    @printf("    KL divergence:                       %.6f\n", kl_marginal)
    @printf("    Total variation distance:            %.6f\n", tv_marginal)
    @printf("    Chi-squared (ESS-corrected):         %.2f  (dof=%d, ESS=%.0f, p=%.4f)\n",
            chi2_marginal, dof_k, n_eff, chi2_k_pval)
    @printf("    Max log-ratio deviation:             %.4f  (= %.2fx)\n",
            max_ratio_dev, exp(max_ratio_dev))
    @printf("  Matched sample mass:                   %.4f\n", matched_mass)

    pass_kl = kl_marginal < 0.05
    pass_tv = tv_marginal < 0.10
    pass_ratio = max_ratio_dev < 0.50
    pass = pass_kl && pass_tv && pass_ratio

    verdict = pass ? "PASS" : "FAIL"
    color_start = pass ? "\033[32m" : "\033[31m"
    color_end = "\033[0m"

    println()
    println("  $(color_start)>>> VERDICT (SPLIT/MERGE ONLY): $verdict <<<$(color_end)")
    if !pass
        println("  Failing criteria:")
        if !pass_kl
            @printf("    Marginal KL %.4f > 0.05 threshold\n", kl_marginal)
        end
        if !pass_tv
            @printf("    Marginal TV %.4f > 0.10 threshold\n", tv_marginal)
        end
        if !pass_ratio
            @printf("    Max log-ratio %.4f > 0.50 (ratio = %.2fx)\n",
                    max_ratio_dev, exp(max_ratio_dev))
        end
    end

    println("\n  INTERPRETATION:")
    println("    If FAIL: split/merge kernel has a bias (wrong acceptance ratio or proposal)")
    println("    If PASS: Gibbs sweep is the culprit for any bias seen in full MCMC")

    return pass, kl_marginal, tv_marginal, max_ratio_dev
end

# ============================================================================
# Run split/merge-only diagnostic
# ============================================================================

println("\n\n" * "#"^80)
println("# SPLIT/MERGE ONLY DIAGNOSTIC")
println("# Isolating whether bias comes from Gibbs sweep or split/merge kernel")
println("#"^80)

pass_sm, _, _, _ = run_splitmerge_only_test(
    emitter_positions = [(0.0, 0.0), (0.05, 0.0)],  # 50nm apart (same as Test 1)
    n_per = 3,
    σ = 0.005,  # 5nm uncertainty
    μ = 3.0,
    shape = 2.0,
    K_max = 4,
    n_iterations = 500_000,
    burn_in = 50_000,
    test_name = "SPLIT/MERGE ONLY (no Gibbs) — Well-separated dimer (50nm, σ=5nm, d/σ=10)",
    seed_data = 42,
    seed_mcmc = 123,
)

println("\n" * "="^80)
if pass_sm
    println("\033[32mSPLIT/MERGE ONLY: PASS — bias likely comes from Gibbs sweep\033[0m")
else
    println("\033[31mSPLIT/MERGE ONLY: FAIL — split/merge kernel has a bias\033[0m")
end
println("="^80)

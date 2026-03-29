# Brute-force partition enumeration
#
# For small systems (N ≤ 8, K_max ≤ 4), enumerate all canonical partitions,
# compute the exact posterior under any AbstractTargetDensity, then compare
# to MCMC empirical frequencies. This is the gold-standard correctness test:
# if MCMC matches exact enumeration, the sampler is correct.

"""
    enumerate_canonical_partitions(N, K_max) -> Vector{Vector{Int}}

Enumerate all canonical partitions of N items into at most K_max groups.

A canonical partition has labels assigned in order of first appearance:
item 1 is always in group 1, item 2 is in group 1 or 2, etc.
The number of canonical partitions equals the sum of Stirling numbers
of the second kind: Σ_{k=1}^{K_max} S(N, k).

Examples:
- enumerate_canonical_partitions(3, 3) → 5 partitions (Bell number B₃)
- enumerate_canonical_partitions(4, 2) → 8 partitions (S(4,1) + S(4,2))

Warning: grows rapidly! N=8, K_max=4 → 32,768 partitions.
N=10 is practical upper limit.
"""
function enumerate_canonical_partitions(N::Int, K_max::Int)
    partitions = Vector{Vector{Int}}()
    N == 0 && return partitions
    z = ones(Int, N)
    if N == 1
        push!(partitions, copy(z))
        return partitions
    end
    # Item 1 is always label 1 (canonical). Recurse from item 2.
    _enumerate_recursive!(partitions, z, 2, 1, N, K_max)
    return partitions
end

function _enumerate_recursive!(partitions::Vector{Vector{Int}},
                                z::Vector{Int}, pos::Int, max_label::Int,
                                N::Int, K_max::Int)
    if pos > N
        push!(partitions, copy(z))
        return
    end

    # Item at position pos can go into any existing group (1..max_label)
    # or start a new group (max_label + 1) if K_max allows
    for k in 1:max_label
        z[pos] = k
        _enumerate_recursive!(partitions, z, pos + 1, max_label, N, K_max)
    end

    if max_label < K_max
        z[pos] = max_label + 1
        _enumerate_recursive!(partitions, z, pos + 1, max_label + 1, N, K_max)
    end
end

"""
    canonicalize(z) -> Vector{Int}

Relabel assignment vector so labels appear in order of first occurrence.

    canonicalize([3, 3, 1, 1]) → [1, 1, 2, 2]
    canonicalize([2, 1, 2, 1]) → [1, 2, 1, 2]
"""
function canonicalize(z::AbstractVector{<:Integer})
    result = similar(z, Int)
    label_map = Dict{eltype(z), Int}()
    next_label = 1
    for i in eachindex(z)
        if !haskey(label_map, z[i])
            label_map[z[i]] = next_label
            next_label += 1
        end
        result[i] = label_map[z[i]]
    end
    return result
end

"""
    partition_hash(z, K_max) -> Int

Hash a canonical partition vector to a unique integer index.
Used for fast lookup in exact posterior comparison.
"""
function partition_hash(z::AbstractVector{<:Integer}, K_max::Int)
    h = 0
    for i in eachindex(z)
        h = h * K_max + (z[i] - 1)
    end
    return h
end

"""
    exact_posterior(partitions, locs, td; μ, shape) -> (probs, log_targets)

Compute the normalized exact posterior for all enumerated partitions
under target density `td`.

Returns:
- `probs`: Normalized posterior probabilities (sum to 1)
- `log_targets`: Unnormalized log target densities
"""
function exact_posterior(partitions::Vector{Vector{Int}},
                          locs::Vector{<:SMLMData.AbstractEmitter},
                          td::AbstractTargetDensity;
                          μ::Float64, shape::Float64)
    loc_precs = precompute_loc_precisions(locs)
    grid = build_locmix_grid(loc_precs)

    n_parts = length(partitions)
    log_targets = Vector{Float64}(undef, n_parts)

    for i in 1:n_parts
        log_targets[i] = log_target(td, partitions[i], loc_precs, grid, μ, shape)
    end

    # Normalize via logsumexp
    max_lt = maximum(log_targets)
    probs = exp.(log_targets .- max_lt)
    total = sum(probs)
    probs ./= total

    return probs, log_targets
end

"""
    compare_mcmc_to_exact(visit_counts, n_samples, exact_probs, partitions, K_max)
        -> NamedTuple

Compare MCMC empirical frequencies to exact posterior.

Arguments:
- `visit_counts`: Dict mapping partition hash → visit count
- `n_samples`: Total number of post-burn-in samples
- `exact_probs`: Normalized exact probabilities (from `exact_posterior`)
- `partitions`: List of canonical partitions
- `K_max`: Maximum K for hashing

Returns:
- `kl_div`: KL(exact || empirical) — measures information loss
- `tv_dist`: Total variation distance — max probability of distinguishing
- `chi_sq`: Chi-squared statistic (with continuity correction)
- `per_K_exact`: Marginal P(K) from exact posterior
- `per_K_empirical`: Marginal P(K) from MCMC
- `max_ratio_deviation`: Max |log(exact/empirical)| over K marginals
- `verdict`: true if marginal KL < 0.05 and TV < 0.10
"""
function compare_mcmc_to_exact(visit_counts::Dict{Int, Int},
                                n_samples::Int,
                                exact_probs::Vector{Float64},
                                partitions::Vector{Vector{Int}},
                                K_max::Int)
    n_parts = length(partitions)

    # Per-partition comparison
    kl = 0.0
    tv = 0.0
    chi_sq = 0.0

    for i in 1:n_parts
        p_exact = exact_probs[i]
        h = partition_hash(partitions[i], K_max)
        count = get(visit_counts, h, 0)
        p_emp = count / n_samples

        # KL(exact || empirical)
        if p_exact > 0
            if p_emp > 0
                kl += p_exact * log(p_exact / p_emp)
            else
                kl += p_exact * log(p_exact / (1.0 / n_samples))  # smoothed
            end
        end

        # Total variation
        tv += abs(p_exact - p_emp)

        # Chi-squared
        expected = p_exact * n_samples
        if expected > 0
            chi_sq += (count - expected)^2 / expected
        end
    end
    tv *= 0.5

    # Marginal P(K)
    per_K_exact = zeros(K_max)
    per_K_empirical = zeros(K_max)

    for i in 1:n_parts
        k = _count_clusters(partitions[i])
        if k <= K_max
            per_K_exact[k] += exact_probs[i]

            h = partition_hash(partitions[i], K_max)
            count = get(visit_counts, h, 0)
            per_K_empirical[k] += count / n_samples
        end
    end

    # Max log-ratio deviation on K marginals
    max_ratio = 0.0
    kl_marginal = 0.0
    for k in 1:K_max
        pe = per_K_exact[k]
        pm = per_K_empirical[k]
        if pe > 1e-6
            if pm > 0
                ratio = abs(log(pe / pm))
                max_ratio = max(max_ratio, ratio)
                kl_marginal += pe * log(pe / pm)
            else
                max_ratio = Inf
                kl_marginal += pe * log(pe / (1.0 / n_samples))
            end
        end
    end

    verdict = kl_marginal < 0.05 && tv < 0.10

    return (
        kl_div=kl,
        kl_marginal=kl_marginal,
        tv_dist=tv,
        chi_sq=chi_sq,
        per_K_exact=per_K_exact,
        per_K_empirical=per_K_empirical,
        max_ratio_deviation=max_ratio,
        verdict=verdict,
    )
end

"""
    run_enumeration_test(locs, td; μ, shape, K_max=3,
                         n_iterations=100000, burn_in=10000,
                         seed=nothing, verbose=false) -> NamedTuple

Full brute-force enumeration test pipeline.

1. Enumerate all canonical partitions for N locs with K_max clusters
2. Compute exact posterior under target density `td`
3. Run MCMC chain and tally partition visit frequencies
4. Compare empirical to exact

Returns everything from `compare_mcmc_to_exact` plus:
- `n_partitions`: Number of canonical partitions enumerated
- `n_samples`: Number of post-burn-in MCMC samples
- `exact_probs`: Exact posterior probabilities
- `partitions`: The enumerated partitions
"""
function run_enumeration_test(locs::Vector{<:SMLMData.AbstractEmitter},
                               td::AbstractTargetDensity;
                               μ::Float64,
                               shape::Float64,
                               K_max::Int=3,
                               n_iterations::Int=100_000,
                               burn_in::Int=10_000,
                               seed::Union{Int, Nothing}=nothing,
                               verbose::Bool=false)
    N = length(locs)

    if seed !== nothing
        Random.seed!(seed)
    end

    # Step 1: Enumerate
    verbose && println("Enumerating canonical partitions (N=$N, K_max=$K_max)...")
    partitions = enumerate_canonical_partitions(N, K_max)
    verbose && println("  Found $(length(partitions)) partitions")

    # Step 2: Exact posterior
    verbose && println("Computing exact posterior...")
    exact_probs, log_targets = exact_posterior(partitions, locs, td; μ=μ, shape=shape)

    # Step 3: Run MCMC
    verbose && println("Running MCMC ($n_iterations iterations, $burn_in burn-in)...")
    ps_acc = PartitionSamples(; thin=1)
    run_collapsed_chain(locs;
        n_iterations=n_iterations,
        burn_in=burn_in,
        shape=shape,
        learn_distribution=false,
        accumulators=AbstractAccumulator[ps_acc],
        verbose=false)
    samples = accumulator_result(ps_acc)

    # Step 4: Tally visits
    n_samples = length(samples)
    visit_counts = Dict{Int, Int}()
    for sample in samples
        z_canon = canonicalize(sample)
        h = partition_hash(z_canon, K_max)
        visit_counts[h] = get(visit_counts, h, 0) + 1
    end

    # Step 5: Compare
    verbose && println("Comparing MCMC to exact posterior...")
    result = compare_mcmc_to_exact(visit_counts, n_samples, exact_probs, partitions, K_max)

    verbose && begin
        println("Results:")
        println("  KL(exact||empirical) = $(round(result.kl_div; digits=4))")
        println("  KL(marginal K)       = $(round(result.kl_marginal; digits=4))")
        println("  TV distance          = $(round(result.tv_dist; digits=4))")
        println("  Max ratio deviation  = $(round(result.max_ratio_deviation; digits=4))")
        println("  Verdict              = $(result.verdict ? "PASS" : "FAIL")")
        println("  P(K) exact:    $(round.(result.per_K_exact; digits=3))")
        println("  P(K) empirical:$(round.(result.per_K_empirical; digits=3))")
    end

    return (;
        result...,
        n_partitions=length(partitions),
        n_samples=n_samples,
        exact_probs=exact_probs,
        partitions=partitions,
        log_targets=log_targets,
    )
end

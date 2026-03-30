# Finite-state sampler validation
#
# For small systems (N ≤ 8): enumerate all canonical partitions, compute
# exact posterior, build empirical transition matrix, and check:
# (a) does the chain sample from π?
# (b) does the kernel preserve π?  (π^T P = π^T)
# (c) is the support graph connected? (empirical irreducibility)

# ============================================================================
# Canonical partition enumeration
# ============================================================================

"""
    enumerate_canonical_partitions(N, K_max) -> Vector{Vector{Int}}

Enumerate all canonical partitions of N items into at most K_max groups.
Labels assigned in order of first appearance. Count = Σ S(N,k) for k=1..K_max.
"""
function enumerate_canonical_partitions(N::Int, K_max::Int)
    partitions = Vector{Vector{Int}}()
    N == 0 && return partitions
    z = ones(Int, N)
    if N == 1
        push!(partitions, copy(z))
        return partitions
    end
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

Relabel so labels appear in order of first occurrence.
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

# ============================================================================
# Exact posterior (with K! multiplicity)
# ============================================================================

"""
    exact_posterior(partitions, locs, td; μ, shape) -> (probs, log_targets)

Normalized exact posterior over canonical partitions. Includes K!
label-multiplicity correction for comparison to canonicalized MCMC samples.
"""
function exact_posterior(partitions::Vector{Vector{Int}},
                          locs::Vector{<:SMLMData.AbstractEmitter},
                          td::AbstractTargetDensity;
                          μ::Float64, shape::Float64)
    isempty(partitions) && return (Float64[], Float64[])

    loc_precs = precompute_loc_precisions(locs)
    grid = build_locmix_grid(loc_precs)

    n_parts = length(partitions)
    log_targets = Vector{Float64}(undef, n_parts)

    for i in 1:n_parts
        K = _count_clusters(partitions[i])
        log_targets[i] = log_target(td, partitions[i], loc_precs, grid, μ, shape) +
                          logfactorial(K)
    end

    max_lt = maximum(log_targets)
    probs = exp.(log_targets .- max_lt)
    probs ./= sum(probs)
    return probs, log_targets
end

# ============================================================================
# Kernel invariance test (the primary correctness diagnostic)
# ============================================================================

"""
    run_kernel_invariance_test(locs, td; μ, shape, K_max=3,
                                n_steps_per_state=1500,
                                seed=nothing, verbose=false) -> NamedTuple

Build empirical one-step transition matrix P and verify:
1. Invariance: π^T P ≈ π^T (Wald test, not arbitrary threshold)
2. Irreducibility: support graph is strongly connected (empirical evidence)
3. MCMC vs exact: K-marginal z-tests with ESS correction

Also reports overflow fraction and within-K conditional TV.

Uses `initial_assignments` + `learn_distribution=false` to hold (μ, shape)
fixed. The transition matrix is for the z-marginal kernel only.
"""
function run_kernel_invariance_test(locs::Vector{<:SMLMData.AbstractEmitter},
                                     td::AbstractTargetDensity;
                                     μ::Float64, shape::Float64,
                                     K_max::Int=3,
                                     n_steps_per_state::Int=1500,
                                     n_mcmc_iterations::Int=100_000,
                                     mcmc_burn_in::Int=10_000,
                                     seed::Union{Int, Nothing}=nothing,
                                     verbose::Bool=false)
    if seed !== nothing
        Random.seed!(seed)
    end

    N = length(locs)
    partitions = enumerate_canonical_partitions(N, K_max)
    n_parts = length(partitions)
    exact_probs, log_targets = exact_posterior(partitions, locs, td; μ=μ, shape=shape)

    # K-marginals from exact posterior
    per_K_exact = zeros(K_max)
    for (i, z) in enumerate(partitions)
        k = _count_clusters(z)
        k <= K_max && (per_K_exact[k] += exact_probs[i])
    end

    # --- Part 1: Transition matrix ---
    part_to_idx = Dict{Tuple, Int}()
    for (i, p) in enumerate(partitions)
        part_to_idx[Tuple(p)] = i
    end

    verbose && println("Building transition matrix: $n_parts states × $n_steps_per_state steps each")

    P = zeros(n_parts, n_parts)
    n_overflow_tm = zeros(Int, n_parts)

    for (i, z_start) in enumerate(partitions)
        for _ in 1:n_steps_per_state
            ps_acc = PartitionSamples(; thin=1)
            run_collapsed_chain(locs;
                n_iterations=1, burn_in=0,
                initial_assignments=z_start,
                shape=shape, learn_distribution=false,
                accumulators=AbstractAccumulator[ps_acc],
                verbose=false)
            samples = accumulator_result(ps_acc)
            if !isempty(samples)
                z_end = Tuple(canonicalize(samples[end]))
                j = get(part_to_idx, z_end, 0)
                if j > 0
                    P[i, j] += 1.0
                else
                    n_overflow_tm[i] += 1
                end
            end
        end
        row_sum = sum(P[i, :])
        row_sum > 0 && (P[i, :] ./= row_sum)
        verbose && (i % 20 == 0) && println("  State $i / $n_parts done")
    end

    total_tm = n_parts * n_steps_per_state
    overflow_tm = sum(n_overflow_tm)
    overflow_fraction = total_tm > 0 ? overflow_tm / total_tm : 0.0
    overflow_ok = overflow_fraction < 0.05

    # Wald test for π^T P = π^T
    wald = invariance_wald_test(P, exact_probs, n_steps_per_state)

    # Support graph connectivity
    support_graph = P .> 0
    is_irreducible = _is_strongly_connected(support_graph)

    # --- Part 2: Long-run MCMC vs exact ---
    verbose && println("Running MCMC ($n_mcmc_iterations iterations)...")
    ps_acc = PartitionSamples(; thin=1)
    diag_acc = ChainDiagnosticAccumulator(; thin=1)
    run_collapsed_chain(locs;
        n_iterations=n_mcmc_iterations, burn_in=mcmc_burn_in,
        shape=shape, learn_distribution=false,
        accumulators=AbstractAccumulator[ps_acc, diag_acc],
        verbose=false)
    mcmc_samples = accumulator_result(ps_acc)
    k_trace = accumulator_result(diag_acc).k_trace

    # Tally visits
    visit_counts = Dict{Tuple, Int}()
    n_valid = 0
    n_overflow_mc = 0
    for sample in mcmc_samples
        z_canon = canonicalize(sample)
        K_s = length(z_canon) > 0 ? maximum(z_canon) : 0
        if K_s > K_max
            n_overflow_mc += 1
            continue
        end
        key = Tuple(z_canon)
        visit_counts[key] = get(visit_counts, key, 0) + 1
        n_valid += 1
    end

    # Calibrated K-marginal test
    k_test = calibrated_k_test(k_trace, per_K_exact)

    # Within-K conditional TV
    within_k = within_k_conditional_tv(visit_counts, n_valid, exact_probs,
                                        partitions, K_max)

    # --- Verdict ---
    verdict = wald.verdict == :pass && is_irreducible && overflow_ok &&
              k_test.verdict == :pass

    verbose && begin
        println("\nResults:")
        println("  Kernel invariance (Wald): $(wald.verdict)  TV=$(round(wald.tv_hat; digits=4))  p=$(round(wald.wald_pvalue; sigdigits=3))")
        println("  Irreducible (empirical): $is_irreducible")
        println("  Overflow (transition matrix): $(round(100*overflow_fraction; digits=1))%")
        println("  K-marginal test: $(k_test.verdict)  ($(k_test.n_tested) tested, $(k_test.n_rejected) rejected, $(k_test.n_inconclusive) inconclusive)")
        for (k, r) in sort(collect(k_test.per_k))
            println("    K=$k: p̂=$(round(r.p_hat; digits=3)) π=$(round(r.p_exact; digits=3)) ESS=$(round(r.ess; digits=0)) $(r.status)")
        end
        !isempty(within_k) && println("  Within-K TV: $(Dict(k => round(v.tv; digits=3) for (k,v) in within_k))")
        println("  Overall: $(verdict ? "PASS" : "FAIL")")
    end

    return (
        # Kernel invariance
        transition_matrix=P,
        wald=wald,
        is_irreducible=is_irreducible,
        overflow_fraction=overflow_fraction,
        support_graph=support_graph,
        # MCMC vs exact
        k_test=k_test,
        within_k_tv=within_k,
        # Shared
        pi_exact=exact_probs,
        per_K_exact=per_K_exact,
        partitions=partitions,
        verdict=verdict,
    )
end

# ============================================================================
# Calibrated K-marginal z-tests
# ============================================================================

"""
    calibrated_k_test(k_trace, per_K_exact; alpha=0.05) -> NamedTuple

Per-K indicator z-tests with ESS-corrected standard errors and Bonferroni
correction. Categories with ESS < 30 marked `:inconclusive`.
"""
function calibrated_k_test(k_trace::AbstractVector{<:Integer},
                            per_K_exact::Vector{Float64};
                            alpha::Float64=0.05,
                            min_ess::Float64=30.0,
                            min_expected::Float64=5.0)
    K_max = length(per_K_exact)
    n = length(k_trace)

    testable = Int[]
    results = Dict{Int, NamedTuple}()

    for k in 1:K_max
        pe = per_K_exact[k]
        pe < 1e-6 && continue

        indicator = Float64[t == k ? 1.0 : 0.0 for t in k_trace]
        p_hat = mean(indicator)
        ess = var(indicator) > 0 ?
            effective_sample_size(indicator; method=:initial_sequence) : Float64(n)

        if ess < min_ess || ess * pe < min_expected || ess * (1 - pe) < min_expected
            results[k] = (p_hat=p_hat, p_exact=pe, ess=ess, z=NaN,
                          pvalue=NaN, adjusted_pvalue=NaN, status=:inconclusive)
        else
            se = sqrt(pe * (1 - pe) / ess)
            z = (p_hat - pe) / se
            pval = 2.0 * (1.0 - cdf(Normal(), abs(z)))
            results[k] = (p_hat=p_hat, p_exact=pe, ess=ess, z=z,
                          pvalue=pval, adjusted_pvalue=NaN, status=:pending)
            push!(testable, k)
        end
    end

    m = length(testable)
    n_rejected = 0
    for k in testable
        r = results[k]
        adj_p = min(1.0, m * r.pvalue)
        status = adj_p < alpha ? :fail : :pass
        status == :fail && (n_rejected += 1)
        results[k] = (p_hat=r.p_hat, p_exact=r.p_exact, ess=r.ess, z=r.z,
                       pvalue=r.pvalue, adjusted_pvalue=adj_p, status=status)
    end

    verdict = m == 0 ? :inconclusive : n_rejected > 0 ? :fail : :pass

    return (per_k=results, n_tested=m,
            n_inconclusive=count(k -> results[k].status == :inconclusive, keys(results)),
            n_rejected=n_rejected, verdict=verdict)
end

# ============================================================================
# Within-K conditional TV
# ============================================================================

"""
    within_k_conditional_tv(visit_counts, n_samples, exact_probs,
                             partitions, K_max; min_mass=0.05) -> Dict

Conditional TV between exact and empirical within each K stratum.
Detects within-K bias that K-marginal tests miss.
"""
function within_k_conditional_tv(visit_counts::Dict{Tuple, Int},
                                  n_samples::Int,
                                  exact_probs::Vector{Float64},
                                  partitions::Vector{Vector{Int}},
                                  K_max::Int;
                                  min_mass::Float64=0.05)
    result = Dict{Int, NamedTuple}()
    for k in 1:K_max
        idx_k = [i for i in eachindex(partitions) if _count_clusters(partitions[i]) == k]
        isempty(idx_k) && continue
        mass_k = sum(exact_probs[i] for i in idx_k)
        mass_k < min_mass && continue
        cond_exact = [exact_probs[i] / mass_k for i in idx_k]
        emp_counts = [get(visit_counts, Tuple(partitions[i]), 0) for i in idx_k]
        total_emp = sum(emp_counts)
        total_emp == 0 && (result[k] = (tv=1.0, n_partitions=length(idx_k)); continue)
        cond_emp = emp_counts ./ total_emp
        tv = 0.5 * sum(abs.(cond_exact .- cond_emp))
        result[k] = (tv=tv, n_partitions=length(idx_k))
    end
    return result
end

# ============================================================================
# Invariance Wald test
# ============================================================================

"""
    invariance_wald_test(P_hat, pi_exact, n_per_state; alpha=0.05) -> NamedTuple

Wald test for H0: π^T P = π^T. Uses multinomial plug-in covariance.
"""
function invariance_wald_test(P_hat::Matrix{Float64},
                               pi_exact::Vector{Float64},
                               n_per_state::Int;
                               alpha::Float64=0.05)
    s = size(P_hat, 1)
    delta = vec(pi_exact' * P_hat) .- pi_exact
    tv_hat = 0.5 * sum(abs.(delta))

    # Covariance under multinomial model
    Sigma = zeros(s, s)
    for i in 1:s
        pi_i = pi_exact[i]
        row = @view P_hat[i, :]
        for j in 1:s, k in 1:s
            cov_jk = (j == k ? row[j] : 0.0) - row[j] * row[k]
            Sigma[j, k] += (pi_i^2 / n_per_state) * cov_jk
        end
    end

    # TV upper bound
    se_tv = 0.5 * sum(sqrt(max(Sigma[j, j], 0.0)) for j in 1:s)
    tv_upper = tv_hat + quantile(Normal(), 1.0 - alpha / 2) * se_tv

    # Wald statistic (drop last component for non-singularity)
    if s > 1
        delta_r = delta[1:end-1]
        Sigma_r = Sigma[1:end-1, 1:end-1]
        min_diag = maximum(abs.(Sigma_r)) * 1e-10
        for j in axes(Sigma_r, 1)
            Sigma_r[j, j] = max(Sigma_r[j, j], min_diag)
        end
        wald_stat = try dot(delta_r, inv(Sigma_r) * delta_r) catch; NaN end
        wald_df = s - 1
    else
        wald_stat = Sigma[1, 1] > 0 ? delta[1]^2 / Sigma[1, 1] : 0.0
        wald_df = 1
    end

    wald_pvalue = isnan(wald_stat) ? NaN : 1.0 - cdf(Chisq(wald_df), wald_stat)
    verdict = isnan(wald_pvalue) ? :inconclusive :
              wald_pvalue < alpha ? :fail : :pass

    return (delta=delta, tv_hat=tv_hat, tv_upper_95=tv_upper,
            wald_stat=wald_stat, wald_df=wald_df, wald_pvalue=wald_pvalue,
            verdict=verdict)
end

# ============================================================================
# Strong connectivity (BFS)
# ============================================================================

function _is_strongly_connected(adj::AbstractMatrix{Bool})
    n = size(adj, 1)
    n <= 1 && return true
    visited = falses(n)
    queue = Int[1]
    visited[1] = true
    while !isempty(queue)
        u = popfirst!(queue)
        for v in 1:n
            adj[u, v] && !visited[v] && (visited[v] = true; push!(queue, v))
        end
    end
    all(visited) || return false
    visited .= false
    queue = Int[1]
    visited[1] = true
    while !isempty(queue)
        u = popfirst!(queue)
        for v in 1:n
            adj[v, u] && !visited[v] && (visited[v] = true; push!(queue, v))
        end
    end
    return all(visited)
end

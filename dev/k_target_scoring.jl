# Target-vs-Kernel Diagnostic for the Octamer Problem
#
# Scores representative allocation states at K=4..8 under the full target.
# Compares uniform spatial (log_area) vs exact locmix (Gaussian integral).
#
# If the target itself prefers K<8, it's a model problem.
# If it prefers K=8 but the chain drifts down, it's a kernel/mixing problem.
#
# Usage: julia --project=dev dev/k_target_scoring.jl

using SMLMBaGoL
using SMLMData
using Random
using Printf
using SpecialFunctions: loggamma
using Statistics

# ============================================================================
# Setup
# ============================================================================

function make_octamer_locs(; NN_over_sigma=1.9, σ=0.007, n_per=5, seed=42)
    NN = NN_over_sigma * σ
    R = NN / (2 * sin(π / 8))
    positions = [(R * cos(2π * k / 8), R * sin(2π * k / 8)) for k in 0:7]

    rng = MersenneTwister(seed)
    locs = SMLMData.Emitter2DFit[]
    true_z = Int[]
    for (eid, (ex, ey)) in enumerate(positions)
        for _ in 1:n_per
            x = ex + σ * randn(rng)
            y = ey + σ * randn(rng)
            push!(locs, SMLMData.Emitter2DFit(
                x, y, 1000.0, 0.0,
                σ, σ, 0.0, 0.0, 0.0,
                1, 1, eid, length(locs) + 1))
            push!(true_z, eid)
        end
    end
    return locs, true_z, positions
end

# ============================================================================
# Flat ML (no prior term — just the Gaussian integral over emitter position)
# ============================================================================

function log_ml_flat(cs::SMLMBaGoL.ClusterStats)
    n = Int(cs.n)
    n == 0 && return 0.0
    det_Λ = cs.Λ_xx * cs.Λ_yy - cs.Λ_xy^2
    det_Λ <= 0 && return -Inf
    inv_det = 1.0 / det_Λ
    S_xx = cs.Λ_yy * inv_det
    S_xy = -cs.Λ_xy * inv_det
    S_yy = cs.Λ_xx * inv_det
    eta_Sinv_eta = S_xx * cs.η_x^2 + 2 * S_xy * cs.η_x * cs.η_y + S_yy * cs.η_y^2
    return (1 - n) * log(2π) - 0.5 * cs.log_det_sum -
           0.5 * (cs.quad - eta_Sinv_eta) - 0.5 * log(det_Λ)
end

# ============================================================================
# Exact locmix ML (integrate P_locmix over posterior analytically)
#
# ML_exact(j) = (1/N) Σ_ℓ ∫ N(s; d_ℓ, Σ_ℓ) × L(data_j | s) ds
#             = (1/N) Σ_ℓ ML_flat(j ∪ {virtual_ℓ})
#
# Each term adds loc ℓ as a "virtual prior component" to the cluster.
# ============================================================================

function log_ml_exact_locmix(cs::SMLMBaGoL.ClusterStats,
                             all_loc_precs::Vector{SMLMBaGoL.LocPrecision},
                             N_total::Int)
    log_terms = Vector{Float64}(undef, N_total)
    for ℓ in 1:N_total
        cs_ext = SMLMBaGoL.add_loc(cs, all_loc_precs[ℓ])
        log_terms[ℓ] = log_ml_flat(cs_ext)
    end
    # logsumexp - log(N)
    max_lt = maximum(log_terms)
    lse = max_lt + log(sum(exp.(log_terms .- max_lt)))
    return lse - log(N_total)
end

# ============================================================================
# Score an allocation under various targets
# ============================================================================

function score_allocation(z::Vector{Int}, K::Int, N::Int,
                          loc_precs::Vector{SMLMBaGoL.LocPrecision},
                          log_area::Float64,
                          μ::Float64, shape::Float64, ρ::Float64)
    γ = shape

    A = exp(log_area)

    # 1. Count model: P(N|K)
    log_count = SMLMBaGoL._log_count_posterior(K, N, shape, μ)

    # 1b. Poisson(ρA) K prior
    log_k_prior = SMLMBaGoL.log_prior_k_poisson(K, ρ, A)

    # 2. DM partition prior: P(z|K)
    sizes = zeros(Int, K)
    for i in 1:N
        sizes[z[i]] += 1
    end
    log_dm = loggamma(K * γ) - K * loggamma(γ) - loggamma(N + K * γ)
    for k in 1:K
        log_dm += loggamma(sizes[k] + γ)
    end

    # 3. Spatial ML: grid locmix (saddle-point) and exact locmix
    log_spatial_grid = 0.0
    log_spatial_exact = 0.0
    log_spatial_flat = 0.0
    per_cluster = Vector{NamedTuple{(:n, :grid, :exact, :flat), NTuple{4, Float64}}}()

    for k in 1:K
        cs = SMLMBaGoL.ClusterStats()
        for i in 1:N
            if z[i] == k
                cs = SMLMBaGoL.add_loc(cs, loc_precs[i])
            end
        end
        ml_grid = SMLMBaGoL.log_marginal_likelihood(cs, log_area)
        ml_exact = log_ml_exact_locmix(cs, loc_precs, N)
        ml_flat = log_ml_flat(cs)
        log_spatial_grid += ml_grid
        log_spatial_exact += ml_exact
        log_spatial_flat += ml_flat
        push!(per_cluster, (n=Float64(cs.n), grid=ml_grid, exact=ml_exact, flat=ml_flat))
    end

    return (K=K,
            sizes=sort(sizes, rev=true),
            count=log_count,
            k_prior=log_k_prior,
            dm=log_dm,
            spatial_grid=log_spatial_grid,
            spatial_exact=log_spatial_exact,
            spatial_flat=log_spatial_flat,
            total_grid=log_count + log_k_prior + log_dm + log_spatial_grid,
            total_exact=log_count + log_k_prior + log_dm + log_spatial_exact,
            count_only=log_count,
            per_cluster=per_cluster)
end

# ============================================================================
# Construct allocations at each K by merging adjacent emitters
# ============================================================================

function make_merged_allocation(true_z::Vector{Int}, K_target::Int)
    # Start from oracle (K=8), merge adjacent pairs using ORIGINAL emitter IDs
    z = copy(true_z)
    K = maximum(z)
    # Merge pairs: (1,2), (3,4), (5,6), (7,8), then (1,3), (5,7), etc.
    merge_sequence = [(1, 2), (3, 4), (5, 6), (7, 8), (1, 3), (5, 7), (1, 5)]
    idx = 0
    while K > K_target && idx < length(merge_sequence)
        idx += 1
        a, b = merge_sequence[idx]
        for i in eachindex(z)
            if z[i] == b
                z[i] = a
            end
        end
        K -= 1
    end
    # Relabel to 1:K at the end only
    labels = sort(unique(z))
    mapping = Dict(l => i for (i, l) in enumerate(labels))
    z = [mapping[zi] for zi in z]
    K = length(labels)
    return z, K
end

# ============================================================================
# Gibbs optimization at fixed K (find best allocation at that K)
# ============================================================================

function gibbs_optimize!(state::SMLMBaGoL.CollapsedState,
                         locs::Vector{<:SMLMData.AbstractEmitter},
                         μ::Float64, shape::Float64, n_sweeps::Int)
    for _ in 1:n_sweeps
        SMLMBaGoL.gibbs_allocation_sweep!(state, locs, μ, shape)
    end
end

function make_state_from_allocation(z::Vector{Int}, K::Int,
                                    locs::Vector{<:SMLMData.AbstractEmitter})
    sp = SMLMBaGoL.UniformSpatialPrior(locs)
    state = SMLMBaGoL.initialize_collapsed_state(locs, sp)

    # Rebuild state with given allocation
    N = length(locs)
    loc_precs = state._loc_precs

    # Clear all clusters
    for j in eachindex(state.active)
        if state.active[j]
            state.clusters[j] = SMLMBaGoL.ClusterStats()
            state.active[j] = false
        end
    end
    state.n_active = 0

    # Ensure enough cluster slots
    while length(state.clusters) < K
        push!(state.clusters, SMLMBaGoL.ClusterStats())
        push!(state.active, false)
    end
    if length(state._rollback_clusters) < K
        resize!(state._rollback_clusters, K)
        resize!(state._rollback_active, K)
    end

    # Activate K clusters
    for k in 1:K
        state.active[k] = true
        state.n_active += 1
    end

    # Assign locs
    for i in 1:N
        k = z[i]
        state.assignments[i] = Int16(k)
        state.clusters[k] = SMLMBaGoL.add_loc(state.clusters[k], loc_precs[i])
    end

    return state
end

function get_allocation(state::SMLMBaGoL.CollapsedState, N::Int)
    z = [Int(state.assignments[i]) for i in 1:N]
    # Relabel to 1:K
    labels = sort(unique(z))
    mapping = Dict(l => i for (i, l) in enumerate(labels))
    return [mapping[zi] for zi in z]
end

# ============================================================================
# Main
# ============================================================================

function main()
    println("="^80)
    println("K-TARGET SCORING DIAGNOSTIC")
    println("="^80)

    # Parameters
    σ = 0.007
    μ = 5.0
    shape = 2.0
    ρ = 2.0
    n_per = 5
    NN_over_sigma = 1.9

    println("\nSetup: 8 emitters on regular octagon, NN/σ = $NN_over_sigma")
    println("  σ = $(σ*1000) nm, μ = $μ, shape = $shape, n_per = $n_per")
    println("  N = $(8 * n_per), NN = $(round(NN_over_sigma * σ * 1000, digits=1)) nm")

    # Generate data
    locs, true_z, positions = make_octamer_locs(;
        NN_over_sigma=NN_over_sigma, σ=σ, n_per=n_per, seed=42)
    N = length(locs)

    # Build infrastructure
    loc_precs = SMLMBaGoL.precompute_loc_precisions(locs)
    spatial_prior = SMLMBaGoL.UniformSpatialPrior(locs)
    log_area = log(SMLMBaGoL.area(spatial_prior))

    # Q-PAINT reference
    println("\n--- Q-PAINT (count-only) ---")
    for K in 1:12
        lp = SMLMBaGoL._log_count_posterior(K, N, shape, μ)
        @printf("  P(N=%d|K=%d) = %.4f (logpdf = %.2f)\n", N, K, exp(lp), lp)
    end
    qpaint_K = argmax([SMLMBaGoL._log_count_posterior(K, N, shape, μ) for K in 1:12])
    println("  Q-PAINT MAP K = $qpaint_K")

    # Score allocations at K=4..8
    println("\n" * "="^80)
    println("ALLOCATION SCORING")
    println("="^80)

    results = []

    for K_target in [8, 7, 6, 5, 4, 3, 2, 1]
        println("\n--- K=$K_target ---")

        if K_target == 8
            z = copy(true_z)
            label = "oracle"
        else
            z, _ = make_merged_allocation(true_z, K_target)
            label = "merged"
        end

        # Score the merged/oracle allocation
        s = score_allocation(z, K_target, N, loc_precs, log_area, μ, shape, ρ)
        @printf("  %s: sizes=%s\n", label, string(s.sizes))
        @printf("    count=%+9.2f  dm=%+9.2f  spatial_grid=%+9.2f  TOTAL_grid=%+9.2f\n",
                s.count, s.dm, s.spatial_grid, s.total_grid)
        @printf("    spatial_exact=%+9.2f  TOTAL_exact=%+9.2f\n",
                s.spatial_exact, s.total_exact)
        @printf("    saddle-point error (grid-exact) = %+.4f per cluster avg\n",
                (s.spatial_grid - s.spatial_exact) / K_target)

        push!(results, (K=K_target, label=label, scores=s))

        # Gibbs-optimized allocation (run 500 sweeps at fixed K)
        if K_target > 1
            state = make_state_from_allocation(z, K_target, locs)
            gibbs_optimize!(state, locs, μ, shape, 500)
            z_opt = get_allocation(state, N)
            s_opt = score_allocation(z_opt, K_target, N, loc_precs, log_area, μ, shape, ρ)
            @printf("  gibbs-opt: sizes=%s\n", string(s_opt.sizes))
            @printf("    count=%+9.2f  dm=%+9.2f  spatial_grid=%+9.2f  TOTAL_grid=%+9.2f\n",
                    s_opt.count, s_opt.dm, s_opt.spatial_grid, s_opt.total_grid)
            @printf("    spatial_exact=%+9.2f  TOTAL_exact=%+9.2f\n",
                    s_opt.spatial_exact, s_opt.total_exact)
            push!(results, (K=K_target, label="gibbs-opt", scores=s_opt))
        end
    end

    # Summary table
    println("\n" * "="^80)
    println("SUMMARY TABLE")
    println("="^80)
    @printf("  %3s  %10s  %9s  %9s  %9s  %9s  %9s  %9s\n",
            "K", "label", "count", "dm", "spat_grid", "spat_exact", "tot_grid", "tot_exact")
    println("  " * "-"^76)

    ref_grid = results[1].scores.total_grid
    ref_exact = results[1].scores.total_exact
    for r in results
        s = r.scores
        @printf("  %3d  %10s  %+9.2f  %+9.2f  %+9.2f  %+9.2f  %+9.2f  %+9.2f\n",
                r.K, r.label, s.count, s.dm, s.spatial_grid, s.spatial_exact,
                s.total_grid, s.total_exact)
    end

    # Relative to K=8 oracle
    println("\n  Relative to K=8 oracle (positive = better than K=8):")
    @printf("  %3s  %10s  %9s  %9s  %9s  %9s  %9s  %9s\n",
            "K", "label", "Δcount", "Δdm", "Δspat_g", "Δspat_e", "Δtot_g", "Δtot_e")
    println("  " * "-"^76)
    for r in results
        s = r.scores
        @printf("  %3d  %10s  %+9.2f  %+9.2f  %+9.2f  %+9.2f  %+9.2f  %+9.2f\n",
                r.K, r.label,
                s.count - results[1].scores.count,
                s.dm - results[1].scores.dm,
                s.spatial_grid - results[1].scores.spatial_grid,
                s.spatial_exact - results[1].scores.spatial_exact,
                s.total_grid - ref_grid,
                s.total_exact - ref_exact)
    end

    # Per-cluster breakdown for K=8 oracle
    println("\n  Per-cluster locmix comparison (K=8 oracle):")
    @printf("  %4s  %3s  %10s  %10s  %10s\n", "k", "n", "grid", "exact", "error")
    for (k, pc) in enumerate(results[1].scores.per_cluster)
        @printf("  %4d  %3.0f  %10.3f  %10.3f  %+10.4f\n",
                k, pc.n, pc.grid, pc.exact, pc.grid - pc.exact)
    end

    # Co-located test
    println("\n" * "="^80)
    println("CO-LOCATED TEST (d=0)")
    println("="^80)
    println("Same N=$N locs but all emitters at (0,0)")

    locs_coloc, _, _ = make_octamer_locs(; NN_over_sigma=0.0, σ=σ, n_per=n_per, seed=42)
    lp_coloc = SMLMBaGoL.precompute_loc_precisions(locs_coloc)
    spatial_prior_coloc = SMLMBaGoL.UniformSpatialPrior(locs_coloc)
    log_area_coloc = log(SMLMBaGoL.area(spatial_prior_coloc))

    # Score co-located at K=1..10
    println("\n  Co-located allocations (balanced, no Gibbs opt):")
    @printf("  %3s  %9s  %9s  %9s  %9s  %9s\n",
            "K", "count", "dm", "spat_grid", "tot_grid", "tot_exact")
    println("  " * "-"^56)
    coloc_z_base = repeat(1:n_per*8, inner=1)  # placeholder

    for K in [1, 2, 4, 6, 8, 10]
        K > N && continue
        # Create balanced allocation: each loc i → cluster ((i-1) % K) + 1
        z_coloc = [(mod(i - 1, K) + 1) for i in 1:N]
        s = score_allocation(z_coloc, K, N, lp_coloc, log_area_coloc, μ, shape, ρ)
        @printf("  %3d  %+9.2f  %+9.2f  %+9.2f  %+9.2f  %+9.2f\n",
                K, s.count, s.dm, s.spatial_grid, s.total_grid, s.total_exact)
    end

    # Run sampler on co-located data from both K=1 and K=8
    for (init_label, init_K) in [("K=1 start", 1), ("oracle K=8", 8)]
        println("\n  Co-located sampler ($init_label, 200K iters)...")
        Random.seed!(789)
        if init_K == 1
            z_init = ones(Int, N)
        else
            z_init = [(mod(i - 1, 8) + 1) for i in 1:N]
        end
        state_coloc = make_state_from_allocation(z_init, init_K, locs_coloc)
        k_trace = Int[]
        for iter in 1:200_000
            r = rand()
            if r < 0.50
                SMLMBaGoL.gibbs_allocation_sweep!(state_coloc, locs_coloc, μ, shape)
            elseif r < 0.75
                SMLMBaGoL.propose_split_merge!(state_coloc, locs_coloc, μ, shape, ρ)
            else
                for _ in 1:5
                    SMLMBaGoL.propose_birth_death!(state_coloc, locs_coloc, μ, shape, ρ)
                end
            end
            if iter > 20_000
                push!(k_trace, state_coloc.n_active)
            end
        end
        km = round(mean(k_trace), digits=2)
        kmed = median(k_trace)
        kmode = sort(collect(countmap(k_trace)), by=x -> -x[2])[1][1]
        println("  $init_label: mean K=$km, median K=$kmed, mode K=$kmode (Q-PAINT=$qpaint_K)")
    end

    # Octamer sampler test (from K=1 and from oracle K=8)
    println("\n" * "="^80)
    println("OCTAMER SAMPLER K-TRACE")
    println("="^80)

    for (init_label, init_z) in [("K=1", ones(Int, N)), ("oracle K=8", copy(true_z))]
        K_init = maximum(init_z)
        state = make_state_from_allocation(init_z, K_init, locs)
        k_trace_oct = Int[]
        Random.seed!(456)
        for iter in 1:100_000
            r = rand()
            if r < 0.50
                SMLMBaGoL.gibbs_allocation_sweep!(state, locs, μ, shape)
            elseif r < 0.75
                SMLMBaGoL.propose_split_merge!(state, locs, μ, shape, ρ)
            else
                for _ in 1:5
                    SMLMBaGoL.propose_birth_death!(state, locs, μ, shape, ρ)
                end
            end
            if iter > 10_000
                push!(k_trace_oct, state.n_active)
            end
        end
        km = round(mean(k_trace_oct), digits=2)
        kmed = median(k_trace_oct)
        println("  Init $init_label: mean K=$km, median K=$kmed (Q-PAINT=$qpaint_K, true=8)")
    end

    println("\nDone.")
end

# Helper
function countmap(v)
    d = Dict{eltype(v), Int}()
    for x in v
        d[x] = get(d, x, 0) + 1
    end
    return d
end

main()

# Mixing proof: show the chain is stuck at a lower-probability state
#
# Strategy:
# 1. Run the sampler to equilibrium (K≈3 at d=0, K≈6 at octamer)
# 2. Score the chain's final state under the exact target
# 3. Score the oracle K=8 state under the same target
# 4. If π(oracle) > π(chain), the chain is visiting a less-probable state → mixing failure
#
# Also: construct a specific K=7→K=8 birth move and show its MH ratio > 0
# (the move WOULD be accepted if proposed correctly)
#
# Usage: julia --project=dev dev/mixing_proof.jl

using SMLMBaGoL
using SMLMData
using Random
using Printf
using SpecialFunctions: loggamma
using Statistics

# ============================================================================
# Setup (same as k_target_scoring.jl)
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
                x, y, 1000.0, 0.0, σ, σ, 0.0, 0.0, 0.0,
                1, 1, eid, length(locs) + 1))
            push!(true_z, eid)
        end
    end
    return locs, true_z, positions
end

function log_target(z::Vector{Int}, K::Int, N::Int,
                    loc_precs::Vector{SMLMBaGoL.LocPrecision},
                    grid::SMLMBaGoL.LocmixGrid,
                    μ::Float64, shape::Float64)
    γ = shape
    log_count = SMLMBaGoL._log_count_posterior(K, N, shape, μ)
    sizes = zeros(Int, K)
    for i in 1:N; sizes[z[i]] += 1; end
    log_dm = loggamma(K * γ) - K * loggamma(γ) - loggamma(N + K * γ)
    for k in 1:K; log_dm += loggamma(sizes[k] + γ); end
    log_spatial = 0.0
    for k in 1:K
        cs = SMLMBaGoL.ClusterStats()
        for i in 1:N
            if z[i] == k
                cs = SMLMBaGoL.add_loc(cs, loc_precs[i])
            end
        end
        log_spatial += SMLMBaGoL.log_marginal_likelihood_locmix(cs, grid)
    end
    return log_count + log_dm + log_spatial
end

function make_state_from_z(z::Vector{Int}, K::Int, locs)
    sp = SMLMBaGoL.UniformSpatialPrior(locs)
    state = SMLMBaGoL.initialize_collapsed_state(locs, sp)
    N = length(locs)
    loc_precs = state._loc_precs
    for j in eachindex(state.active)
        if state.active[j]
            state.clusters[j] = SMLMBaGoL.ClusterStats()
            state.active[j] = false
        end
    end
    state.n_active = 0
    while length(state.clusters) < K
        push!(state.clusters, SMLMBaGoL.ClusterStats())
        push!(state.active, false)
    end
    if length(state._rollback_clusters) < K
        resize!(state._rollback_clusters, K)
        resize!(state._rollback_active, K)
    end
    for k in 1:K
        state.active[k] = true
        state.n_active += 1
    end
    for i in 1:N
        k = z[i]
        state.assignments[i] = Int16(k)
        state.clusters[k] = SMLMBaGoL.add_loc(state.clusters[k], loc_precs[i])
    end
    return state
end

function get_z(state, N)
    z = [Int(state.assignments[i]) for i in 1:N]
    labels = sort(unique(z))
    mapping = Dict(l => i for (i, l) in enumerate(labels))
    return [mapping[zi] for zi in z]
end

function countmap(v)
    d = Dict{eltype(v), Int}()
    for x in v; d[x] = get(d, x, 0) + 1; end
    return d
end

# ============================================================================
# Test 1: Co-located — chain state vs oracle
# ============================================================================

function test_colocated()
    println("="^70)
    println("TEST 1: CO-LOCATED (d=0) — Chain state vs Oracle")
    println("="^70)

    σ = 0.007; μ = 5.0; shape = 2.0; n_per = 5
    locs, true_z, _ = make_octamer_locs(; NN_over_sigma=0.0, σ=σ, n_per=n_per, seed=42)
    N = length(locs)
    loc_precs = SMLMBaGoL.precompute_loc_precisions(locs)
    grid = SMLMBaGoL.build_locmix_grid(loc_precs)

    # Run sampler to "equilibrium"
    println("\n  Running sampler from K=1 (200K iters)...")
    Random.seed!(999)
    state = make_state_from_z(ones(Int, N), 1, locs)
    for iter in 1:200_000
        r = rand()
        if r < 0.50
            SMLMBaGoL.gibbs_allocation_sweep!(state, locs, μ, shape)
        elseif r < 0.75
            SMLMBaGoL.propose_split_merge!(state, locs, μ, shape)
        else
            for _ in 1:5
                SMLMBaGoL.propose_birth_death!(state, locs, μ, shape)
            end
        end
    end
    z_chain = get_z(state, N)
    K_chain = maximum(z_chain)

    # Score chain state
    π_chain = log_target(z_chain, K_chain, N, loc_precs, grid, μ, shape)

    # Score oracle K=8 balanced allocation
    z_oracle = [(mod(i - 1, 8) + 1) for i in 1:N]
    π_oracle = log_target(z_oracle, 8, N, loc_precs, grid, μ, shape)

    # Score oracle after Gibbs optimization at K=8
    state_oracle = make_state_from_z(z_oracle, 8, locs)
    for _ in 1:1000
        SMLMBaGoL.gibbs_allocation_sweep!(state_oracle, locs, μ, shape)
    end
    z_oracle_opt = get_z(state_oracle, N)
    K_oracle_opt = maximum(z_oracle_opt)
    π_oracle_opt = log_target(z_oracle_opt, K_oracle_opt, N, loc_precs, grid, μ, shape)

    sizes_chain = sort([count(==(k), z_chain) for k in 1:K_chain], rev=true)
    sizes_oracle_opt = sort([count(==(k), z_oracle_opt) for k in 1:K_oracle_opt], rev=true)

    println("\n  Chain final state: K=$K_chain, sizes=$sizes_chain")
    @printf("    log π(chain) = %.4f\n", π_chain)
    println("  Oracle K=8 balanced: sizes=[5,5,5,5,5,5,5,5]")
    @printf("    log π(oracle) = %.4f\n", π_oracle)
    println("  Oracle K=$K_oracle_opt Gibbs-opt: sizes=$sizes_oracle_opt")
    @printf("    log π(oracle_opt) = %.4f\n", π_oracle_opt)

    Δ = π_oracle_opt - π_chain
    println("\n  *** log π(oracle_opt) - log π(chain) = $(@sprintf("%.4f", Δ)) ***")
    if Δ > 0
        println("  *** Oracle K=$K_oracle_opt state has HIGHER probability than chain K=$K_chain ***")
        println("  *** Probability ratio: exp($(@sprintf("%.1f", Δ))) = $(@sprintf("%.1e", exp(Δ))) ***")
        println("  *** This PROVES the chain is NOT at the target mode — MIXING FAILURE ***")
    else
        println("  Chain state has higher probability — target may genuinely prefer K=$K_chain")
    end
end

# ============================================================================
# Test 2: Octamer — chain state vs oracle
# ============================================================================

function test_octamer()
    println("\n" * "="^70)
    println("TEST 2: OCTAMER (NN=1.9σ) — Chain state vs Oracle")
    println("="^70)

    σ = 0.007; μ = 5.0; shape = 2.0; n_per = 5
    locs, true_z, _ = make_octamer_locs(; NN_over_sigma=1.9, σ=σ, n_per=n_per, seed=42)
    N = length(locs)
    loc_precs = SMLMBaGoL.precompute_loc_precisions(locs)
    grid = SMLMBaGoL.build_locmix_grid(loc_precs)

    # Run sampler from K=1
    println("\n  Running sampler from K=1 (200K iters)...")
    Random.seed!(999)
    state = make_state_from_z(ones(Int, N), 1, locs)
    for iter in 1:200_000
        r = rand()
        if r < 0.50
            SMLMBaGoL.gibbs_allocation_sweep!(state, locs, μ, shape)
        elseif r < 0.75
            SMLMBaGoL.propose_split_merge!(state, locs, μ, shape)
        else
            for _ in 1:5
                SMLMBaGoL.propose_birth_death!(state, locs, μ, shape)
            end
        end
    end
    z_chain = get_z(state, N)
    K_chain = maximum(z_chain)
    π_chain = log_target(z_chain, K_chain, N, loc_precs, grid, μ, shape)

    # Score oracle
    π_oracle = log_target(true_z, 8, N, loc_precs, grid, μ, shape)

    # Gibbs-optimize oracle at K=8
    state_oracle = make_state_from_z(copy(true_z), 8, locs)
    for _ in 1:1000
        SMLMBaGoL.gibbs_allocation_sweep!(state_oracle, locs, μ, shape)
    end
    z_oracle_opt = get_z(state_oracle, N)
    K_oracle_opt = maximum(z_oracle_opt)
    π_oracle_opt = log_target(z_oracle_opt, K_oracle_opt, N, loc_precs, grid, μ, shape)

    sizes_chain = sort([count(==(k), z_chain) for k in 1:K_chain], rev=true)
    sizes_oracle_opt = sort([count(==(k), z_oracle_opt) for k in 1:K_oracle_opt], rev=true)

    println("\n  Chain final state: K=$K_chain, sizes=$sizes_chain")
    @printf("    log π(chain) = %.4f\n", π_chain)
    println("  Oracle K=8: sizes=[5,5,5,5,5,5,5,5]")
    @printf("    log π(oracle) = %.4f\n", π_oracle)
    println("  Oracle K=$K_oracle_opt Gibbs-opt: sizes=$sizes_oracle_opt")
    @printf("    log π(oracle_opt) = %.4f\n", π_oracle_opt)

    Δ = π_oracle_opt - π_chain
    println("\n  *** log π(oracle_opt) - log π(chain) = $(@sprintf("%.4f", Δ)) ***")
    if Δ > 0
        println("  *** Oracle state has HIGHER probability — MIXING FAILURE ***")
        @printf("  *** Probability ratio: exp(%.1f) = %.1e ***\n", Δ, exp(Δ))
    else
        println("  *** Chain state has higher probability ***")
        println("  *** This does NOT prove mixing failure — target may prefer K=$K_chain ***")
        println("  *** (But recall: this compares specific allocations, not marginals) ***")
    end
end

# ============================================================================
# Test 3: Designed birth move — would it be accepted?
# ============================================================================

function test_designed_move()
    println("\n" * "="^70)
    println("TEST 3: DESIGNED BIRTH MOVE — Would it be accepted?")
    println("="^70)

    σ = 0.007; μ = 5.0; shape = 2.0; n_per = 5
    locs, true_z, positions = make_octamer_locs(; NN_over_sigma=0.0, σ=σ, n_per=n_per, seed=42)
    N = length(locs)
    loc_precs = SMLMBaGoL.precompute_loc_precisions(locs)
    grid = SMLMBaGoL.build_locmix_grid(loc_precs)
    γ = shape

    println("\n  Co-located case: construct K=7 → K=8 birth")
    println("  Start: K=7 balanced-ish allocation")
    println("  End: K=8 by detaching one loc from largest cluster")

    # Create K=7 allocation: merge emitters 1,2 into one cluster
    z7 = copy(true_z)
    for i in eachindex(z7)
        if z7[i] == 2; z7[i] = 1; end
        if z7[i] > 2; z7[i] -= 1; end
    end
    K7 = 7

    # Score K=7
    π_7 = log_target(z7, K7, N, loc_precs, grid, μ, shape)

    # Birth: detach loc 1 (from cluster 1, which has 10 locs) → singleton
    z8 = copy(z7)
    z8[1] = 8  # new singleton cluster
    K8 = 8

    π_8 = log_target(z8, K8, N, loc_precs, grid, μ, shape)

    Δ_target = π_8 - π_7

    # Compute birth MH proposal ratio
    # Forward (birth): p_birth × (1/N_eligible)
    n_singletons_7 = 0  # at K=7, no singletons in balanced allocation
    n_eligible_7 = N - n_singletons_7
    p_birth_7 = n_singletons_7 == 0 ? 1.0 : 0.5
    log_q_fwd = log(p_birth_7) - log(Float64(n_eligible_7))

    # Reverse (death from K=8): p_death × (1/n_singletons) × w(dest)/Σw
    n_singletons_8 = 1  # just created one
    n_eligible_8 = N - n_singletons_8
    p_death_8 = 0.5  # both birth and death feasible

    # For the destination weight: need w(k) = (n_k + γ) × predictive
    # The singleton loc should go back to cluster 1 (its original)
    # Compute all weights
    state8 = make_state_from_z(z8, K8, locs)
    lp_birth = state8._loc_precs[1]
    singleton_slot = 8

    log_weights = Float64[]
    dest_names = Int[]
    for k in 1:K8
        k == singleton_slot && continue
        cs = SMLMBaGoL.ClusterStats()
        for i in 1:N
            if z8[i] == k
                cs = SMLMBaGoL.add_loc(cs, loc_precs[i])
            end
        end
        lw = log(Float64(cs.n) + γ) + SMLMBaGoL.log_predictive_locmix(cs, lp_birth, grid)
        push!(log_weights, lw)
        push!(dest_names, k)
    end
    max_lw = maximum(log_weights)
    log_sum_w = max_lw + log(sum(exp.(log_weights .- max_lw)))
    # Weight for dest=1 (original cluster)
    log_w_dest = log_weights[1]  # cluster 1 is first

    log_q_rev = log(p_death_8) - log(Float64(n_singletons_8)) + log_w_dest - log_sum_w

    Δ_proposal = log_q_rev - log_q_fwd
    log_α = Δ_target + Δ_proposal

    @printf("\n  K=7 state: log π = %.4f\n", π_7)
    @printf("  K=8 state: log π = %.4f\n", π_8)
    @printf("  Δ_target (K=8 - K=7) = %+.4f\n", Δ_target)
    @printf("  log q_fwd (birth) = %.4f\n", log_q_fwd)
    @printf("  log q_rev (death) = %.4f\n", log_q_rev)
    @printf("  Δ_proposal = %+.4f\n", Δ_proposal)
    @printf("  log α = Δ_target + Δ_proposal = %+.4f\n", log_α)
    @printf("  Acceptance probability = %.4f\n", min(1.0, exp(log_α)))

    println("\n  Now run 1000 actual birth proposals from K=7 state and measure acceptance:")
    n_proposed = 0
    n_accepted = 0
    for trial in 1:1000
        Random.seed!(trial)
        state7 = make_state_from_z(copy(z7), K7, locs)
        accepted, move_type = SMLMBaGoL.propose_birth_death!(state7, locs, μ, shape)
        if move_type == :birth
            n_proposed += 1
            if accepted
                n_accepted += 1
            end
        end
    end
    @printf("  Birth proposals: %d/%d, accepted: %d (%.1f%%)\n",
            n_proposed, 1000, n_accepted,
            n_proposed > 0 ? 100.0 * n_accepted / n_proposed : 0.0)

    # Now the key test: run many BD steps from the Gibbs-optimized K=7 state
    # and see if ANY birth reaches K=8
    println("\n  Run 10000 BD substeps from K=7. How many times does K reach 8?")
    k_reached_8 = 0
    for trial in 1:10
        Random.seed!(trial + 100)
        state7 = make_state_from_z(copy(z7), K7, locs)
        # First Gibbs-optimize
        for _ in 1:100
            SMLMBaGoL.gibbs_allocation_sweep!(state7, locs, μ, shape)
        end
        for step in 1:1000
            SMLMBaGoL.propose_birth_death!(state7, locs, μ, shape)
            if state7.n_active >= 8
                k_reached_8 += 1
                break
            end
        end
    end
    println("  Reached K≥8: $k_reached_8 / 10 trials (1000 BD steps each)")
end

# ============================================================================
# Test 4: Direct comparison — oracle vs chain at SAME K
# ============================================================================

function test_same_K_comparison()
    println("\n" * "="^70)
    println("TEST 4: SAME-K COMPARISON — Oracle K=8 vs Chain's K=8 allocation")
    println("="^70)

    σ = 0.007; μ = 5.0; shape = 2.0; n_per = 5

    for (label, nn_sigma) in [("co-located (d=0)", 0.0), ("octamer (NN=1.9σ)", 1.9)]
        println("\n  --- $label ---")
        locs, true_z, _ = make_octamer_locs(; NN_over_sigma=nn_sigma, σ=σ, n_per=n_per, seed=42)
        N = length(locs)
        loc_precs = SMLMBaGoL.precompute_loc_precisions(locs)
        grid = SMLMBaGoL.build_locmix_grid(loc_precs)

        # Oracle K=8
        z_oracle = nn_sigma > 0 ? copy(true_z) : [(mod(i-1, 8)+1) for i in 1:N]
        π_oracle = log_target(z_oracle, 8, N, loc_precs, grid, μ, shape)

        # Gibbs-optimize at K=8 (find the BEST K=8 allocation)
        state8 = make_state_from_z(copy(z_oracle), 8, locs)
        best_π = π_oracle
        best_z = copy(z_oracle)
        for sweep in 1:2000
            SMLMBaGoL.gibbs_allocation_sweep!(state8, locs, μ, shape)
            z_curr = get_z(state8, N)
            π_curr = log_target(z_curr, 8, N, loc_precs, grid, μ, shape)
            if π_curr > best_π
                best_π = π_curr
                best_z = copy(z_curr)
            end
        end
        sizes_best = sort([count(==(k), best_z) for k in 1:8], rev=true)

        # Run full sampler from K=1
        Random.seed!(777)
        state_chain = make_state_from_z(ones(Int, N), 1, locs)
        best_chain_π = -Inf
        best_chain_z = ones(Int, N)
        best_chain_K = 1
        for iter in 1:200_000
            r = rand()
            if r < 0.50
                SMLMBaGoL.gibbs_allocation_sweep!(state_chain, locs, μ, shape)
            elseif r < 0.75
                SMLMBaGoL.propose_split_merge!(state_chain, locs, μ, shape)
            else
                for _ in 1:5
                    SMLMBaGoL.propose_birth_death!(state_chain, locs, μ, shape)
                end
            end
            if iter > 50_000 && iter % 100 == 0
                z_c = get_z(state_chain, N)
                K_c = maximum(z_c)
                π_c = log_target(z_c, K_c, N, loc_precs, grid, μ, shape)
                if π_c > best_chain_π
                    best_chain_π = π_c
                    best_chain_z = copy(z_c)
                    best_chain_K = K_c
                end
            end
        end
        sizes_chain = sort([count(==(k), best_chain_z) for k in 1:best_chain_K], rev=true)

        @printf("  Best K=8 (Gibbs-opt): sizes=%s, log π = %.4f\n", string(sizes_best), best_π)
        @printf("  Best from chain:      K=%d, sizes=%s, log π = %.4f\n",
                best_chain_K, string(sizes_chain), best_chain_π)

        Δ = best_π - best_chain_π
        if Δ > 0
            @printf("  *** K=8 Gibbs-opt has HIGHER probability by %.2f ***\n", Δ)
            println("  *** Chain never found this state — MIXING PROOF ***")
        else
            @printf("  Chain found better state (Δ = %.2f) — no mixing proof at this level\n", Δ)
        end
    end
end

# ============================================================================
# Main
# ============================================================================

test_colocated()
test_octamer()
test_designed_move()
test_same_K_comparison()

println("\nDone.")

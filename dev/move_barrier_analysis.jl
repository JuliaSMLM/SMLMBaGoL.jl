#!/usr/bin/env julia
#
# Move Barrier Analysis
#
# Decomposes K→K+1 MH acceptance into individual Δ components:
#   Δ_spatial, Δ_partition, Δ_count, Δ_K_prior, Δ_proposal
#
# Diagnoses whether K-mixing failure is target-side (Δ_partition + Δ_spatial
# dominate) or transport-side (Δ_proposal dominates).
#
# Run: julia --project=dev dev/move_barrier_analysis.jl
# Time: ~5 seconds

using SMLMBaGoL, SMLMData, Random, Statistics, SpecialFunctions

Random.seed!(42)

# ============================================================================
# Configuration
# ============================================================================

N = 40           # total localizations
μ = 10.0         # mean locs per emitter
shape = 2.0      # NegBin shape
γ = shape        # DM concentration
ρ = 2.0          # Poisson K prior rate density
N_EMPIRICAL = 1000  # number of random birth attempts

# ============================================================================
# Setup
# ============================================================================

locs = [SMLMData.Emitter2DFit(
    0.5 + randn() * 0.010, 0.5 + randn() * 0.010,
    1000.0, 10.0, 0.010, 0.010, 0.0, 100.0, 5.0,
    i, 1, 1, i
) for i in 1:N]

prior = SMLMBaGoL.UniformSpatialPrior(locs)
A = SMLMBaGoL.area(prior)
log_area = log(A)

function target_components(state, N, γ, μ, shape, ρ, A)
    K = state.n_active
    lml = SMLMBaGoL._total_spatial_lml(state)
    dm = SMLMBaGoL._log_dm_partition(state, N, γ)
    count = SMLMBaGoL._log_count_posterior(K, N, shape, μ)
    kprior = SMLMBaGoL.log_prior_k_poisson(K, ρ, A)
    return (spatial=lml, partition=dm, count=count, k_prior=kprior, K=K)
end

# Simulate a birth: detach one loc as singleton from cluster `from_cluster`
function simulate_birth(state, loc_idx)
    st = deepcopy(state)
    lp = st._loc_precs[loc_idx]
    old_cluster = Int(st.assignments[loc_idx])
    st.clusters[old_cluster] = SMLMBaGoL.remove_loc(st.clusters[old_cluster], lp)
    new_slot = length(st.clusters) + 1
    push!(st.clusters, SMLMBaGoL.add_loc(SMLMBaGoL.ClusterStats(), lp))
    push!(st.active, true)
    st.n_active += 1
    st.assignments[loc_idx] = Int16(new_slot)
    return st
end

# ============================================================================
# Analysis
# ============================================================================

println("=" ^ 70)
println("Move Barrier Analysis: N=$N co-located")
println("=" ^ 70)
println("  μ=$μ, shape=$shape, γ=$γ, ρ=$ρ")
println("  Area A = $(round(A, digits=6)), log(A) = $(round(log_area, digits=2))")
qpaint_k = argmax([SMLMBaGoL.log_prior_total_count(N, k, μ, shape) for k in 1:20])
println("  Q-PAINT MAP K = $qpaint_k")
println()

# --- Section 1: Count + K_prior landscape ---
println("=" ^ 70)
println("Section 1: TARGET LANDSCAPE (count + K_prior across K)")
println("=" ^ 70)
println()
println("  K   log P(N|K)   log P(K|ρ,A)   sum          Δ_from_K-1")
global landscape_peak = 1
global landscape_peak_val = -Inf
for k in 1:12
    lc = SMLMBaGoL.log_prior_total_count(N, k, μ, shape)
    lk = SMLMBaGoL.log_prior_k_poisson(k, ρ, A)
    total = lc + lk
    if total > landscape_peak_val
        global landscape_peak = k
        global landscape_peak_val = total
    end
    if k > 1
        lc_prev = SMLMBaGoL.log_prior_total_count(N, k-1, μ, shape)
        lk_prev = SMLMBaGoL.log_prior_k_poisson(k-1, ρ, A)
        delta = total - (lc_prev + lk_prev)
        println("  $(lpad(k,2))   $(lpad(round(lc, digits=2), 8))   $(lpad(round(lk, digits=2), 12))   $(lpad(round(total, digits=2), 8))      $(round(delta, digits=3))")
    else
        println("  $(lpad(k,2))   $(lpad(round(lc, digits=2), 8))   $(lpad(round(lk, digits=2), 12))   $(lpad(round(total, digits=2), 8))      —")
    end
end
println()
println("  Count+K_prior peak: K=$landscape_peak")
println()

# --- Section 2: DM partition barrier at each K ---
println("=" ^ 70)
println("Section 2: DM PARTITION BARRIER (balanced K → K+1 singleton birth)")
println("=" ^ 70)
println()
println("  K→K+1    n_per   Δ_partition")
for k in 1:8
    n_per = N ÷ k
    dm_before = loggamma(k * γ) - k * loggamma(γ) - loggamma(N + k * γ) + k * loggamma(n_per + γ)
    dm_after = loggamma((k+1) * γ) - (k+1) * loggamma(γ) - loggamma(N + (k+1) * γ) +
               (k-1) * loggamma(n_per + γ) + loggamma(n_per - 1 + γ) + loggamma(1 + γ)
    Δ = dm_after - dm_before
    println("  K=$(lpad(k,2))→$(lpad(k+1,2))  n_per=$(lpad(n_per,3))  Δ_partition = $(round(Δ, digits=3))")
end
println()

# --- Section 3: Full MH decomposition for birth K=1→K=2 ---
println("=" ^ 70)
println("Section 3: BIRTH K=1→K=2 (full MH decomposition)")
println("=" ^ 70)
println()

state_k1 = SMLMBaGoL.initialize_collapsed_state(locs, prior)
before = target_components(state_k1, N, γ, μ, shape, ρ, A)
state_k2 = simulate_birth(state_k1, 1)
after = target_components(state_k2, N, γ, μ, shape, ρ, A)

Δ_spatial = after.spatial - before.spatial
Δ_partition = after.partition - before.partition
Δ_count = after.count - before.count
Δ_K_prior = after.k_prior - before.k_prior

# Birth proposal: p_birth=1 at K=1, uniform loc
log_q_fwd = log(1.0) - log(Float64(N))
# Reverse death: p_death=0.5, 1 singleton, 1 destination
log_q_rev = log(0.5) - log(1.0)
Δ_proposal = log_q_rev - log_q_fwd

log_α = Δ_spatial + Δ_partition + Δ_count + Δ_K_prior + Δ_proposal
accept_prob = min(1.0, exp(log_α))

println("  Δ_spatial:    $(lpad(round(Δ_spatial, digits=3), 8))")
println("  Δ_partition:  $(lpad(round(Δ_partition, digits=3), 8))")
println("  Δ_count:      $(lpad(round(Δ_count, digits=3), 8))")
println("  Δ_K_prior:    $(lpad(round(Δ_K_prior, digits=3), 8))")
println("  Δ_proposal:   $(lpad(round(Δ_proposal, digits=3), 8))")
println("  ──────────────────────")
println("  log α:        $(lpad(round(log_α, digits=3), 8))  → accept $(round(100*accept_prob, digits=3))%")
println()

# Classify the barrier
target_penalty = Δ_spatial + Δ_partition + Δ_K_prior  # target-side terms (negative = barrier)
count_help = Δ_count                                    # count model (positive = help)
proposal_term = Δ_proposal                              # proposal term
total_barrier = abs(min(0.0, log_α))

println("  TARGET penalty  (spatial + partition + K_prior): $(round(target_penalty, digits=3))")
println("  COUNT help      (Δ_count):                      $(round(count_help, digits=3))")
println("  PROPOSAL term   (Δ_proposal):                   $(round(proposal_term, digits=3))")
println()

# --- Section 4: Empirical birth Δ distribution ---
println("=" ^ 70)
println("Section 4: EMPIRICAL ($N_EMPIRICAL random births from K=1)")
println("=" ^ 70)
println()

Δs_hist = Float64[]
Δp_hist = Float64[]
Δc_hist = Float64[]
Δk_hist = Float64[]
Δprop_hist = Float64[]
α_hist = Float64[]

for trial in 1:N_EMPIRICAL
    st = SMLMBaGoL.initialize_collapsed_state(locs, prior)
    lml_bef = SMLMBaGoL._total_spatial_lml(st)
    dm_bef = SMLMBaGoL._log_dm_partition(st, N, γ)
    K_bef = st.n_active

    chosen = rand(1:N)
    st_after = simulate_birth(st, chosen)

    lml_aft = SMLMBaGoL._total_spatial_lml(st_after)
    dm_aft = SMLMBaGoL._log_dm_partition(st_after, N, γ)

    ds = lml_aft - lml_bef
    dp = dm_aft - dm_bef
    dc = SMLMBaGoL._log_count_posterior(K_bef+1, N, shape, μ) -
         SMLMBaGoL._log_count_posterior(K_bef, N, shape, μ)
    dk = SMLMBaGoL.log_prior_k_poisson(K_bef+1, ρ, A) -
         SMLMBaGoL.log_prior_k_poisson(K_bef, ρ, A)
    dprop = log(0.5) - log(1.0) - (log(1.0) - log(Float64(N)))

    push!(Δs_hist, ds)
    push!(Δp_hist, dp)
    push!(Δc_hist, dc)
    push!(Δk_hist, dk)
    push!(Δprop_hist, dprop)
    push!(α_hist, ds + dp + dc + dk + dprop)
end

println("  Component statistics (mean ± std):")
println("    Δ_spatial:    $(round(mean(Δs_hist), digits=3)) ± $(round(std(Δs_hist), digits=3))")
println("    Δ_partition:  $(round(mean(Δp_hist), digits=3)) ± $(round(std(Δp_hist), digits=3))")
println("    Δ_count:      $(round(mean(Δc_hist), digits=3)) ± $(round(std(Δc_hist), digits=3))")
println("    Δ_K_prior:    $(round(mean(Δk_hist), digits=3)) ± $(round(std(Δk_hist), digits=3))")
println("    Δ_proposal:   $(round(mean(Δprop_hist), digits=3)) ± $(round(std(Δprop_hist), digits=3))")
println("    ──────────────────────────────────────")
println("    log α:        $(round(mean(α_hist), digits=3)) ± $(round(std(α_hist), digits=3))")
mean_accept = mean(min.(1.0, exp.(α_hist)))
println("    accept prob:  $(round(mean_accept, digits=6))")
println()

# Ranked by mean
means = [("spatial", mean(Δs_hist)), ("partition", mean(Δp_hist)),
         ("count", mean(Δc_hist)), ("K_prior", mean(Δk_hist)), ("proposal", mean(Δprop_hist))]
sort!(means, by=x->x[2])
total_abs = sum(abs(v) for (_, v) in means)
println("  Ranked by mean (most negative = biggest barrier):")
for (name, val) in means
    pct = round(100 * abs(val) / total_abs, digits=1)
    marker = val < -2.0 ? " <<<" : ""
    println("    $(rpad(name, 12)) $(lpad(round(val, digits=3), 8))  ($(lpad(pct, 5))% of |total|)$marker")
end

# ============================================================================
# Verdict
# ============================================================================
println()
println("=" ^ 70)

# Determine if target or transport dominates
mean_target = mean(Δs_hist) + mean(Δp_hist) + mean(Δk_hist)
mean_transport = mean(Δprop_hist)
mean_count = mean(Δc_hist)

if mean_target < -3.0 && abs(mean_target) > 2 * abs(mean_transport)
    verdict = "TARGET-DOMINATED"
    explanation = "Δ_partition + Δ_spatial = $(round(mean_target, digits=1)) overwhelms count help = $(round(mean_count, digits=1)). The collapsed target density penalizes K-increasing moves at this N."
elseif mean_transport < -3.0 && abs(mean_transport) > 2 * abs(mean_target)
    verdict = "TRANSPORT-DOMINATED"
    explanation = "Δ_proposal = $(round(mean_transport, digits=1)) overwhelms target ratio. Local K±1 proposals cannot find good transitions even though the target may favor higher K."
else
    verdict = "MIXED"
    explanation = "Both target ($(round(mean_target, digits=1))) and transport ($(round(mean_transport, digits=1))) contribute comparably to the barrier."
end

println("VERDICT: $verdict")
println()
println("  $explanation")
println()
println("  Count+K_prior landscape peak: K=$landscape_peak (Q-PAINT MAP K=$qpaint_k)")
if landscape_peak <= 2
    println("  WARNING: target landscape itself peaks at K≤2. Higher K may not be favored.")
end
println("  Mean birth acceptance: $(round(100*mean_accept, digits=4))%")
println("=" ^ 70)

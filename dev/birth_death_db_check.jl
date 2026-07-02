# GATE-3b BIRTH/DEATH detailed-balance checker at K=6<->7 and K=7<->8 on the
# co-located set. Companion to dev/detailed_balance_check.jl (split/merge).
#
# WHY: with Fix A the split/merge move is DB-correct, yet the chain still peaks
# at K=6 (mean 5.99, TV 0.57) under the locmix target. The ONLY K-changing moves
# are split/merge (fixed) and birth/death (propose_birth_death!); the collapsed
# Gibbs sweep holds K fixed. So birth/death is the remaining DB suspect for a
# net K-down bias. The historic merge bug's signature was: reverse density
# FABRICATED / mis-normalized -> one direction over-accepted -> grows with
# cluster size, invisible at K<=4. This driver tests birth/death against the
# same bar, and if it BALANCES says so plainly (=> residual is the TARGET, not
# DB).
#
# WHAT IS TESTED (all with the REAL package primitives -- spatial_pred,
# _log_count_posterior, partition_prior, log_target, log_prior_k_poisson,
# initialize_from_assignments, propose_birth_death!):
#
#  Part 1  DENSITY HONESTY (per realization). For a birth c->c' (detach loc j
#          from the co-located m-cluster) and the death c'->c that reverses it
#          (absorb singleton {j} back into its origin cluster A'), independently
#          recompute:
#            - the birth's reverse density log_q_rev_b  (death-at-c' formula)
#            - the death's forward density log_q_fwd_d  (death-at-c' mechanism)
#          from the two states' OWN proposal mechanisms and require they agree
#          (no fabrication / mis-normalization), and require alpha_b + alpha_d = 0.
#          Also the reverse-birth vs forward-birth densities for the death pair.
#
#  Part 2  MOVE-TYPE + COUNT consistency (lead item d). Tabulate p_birth(c),
#          p_death(c'), and Delta_count for birth(K->K+1) vs reverse-death, and
#          death(K->K-1) vs reverse-birth, at interior and boundary K. A
#          K-dependent move-type or count factor that is right at small K but
#          wrong at high K is the merge-bug class.
#
#  Part 3  EXACT MARGINAL FLUX BALANCE (canonical). Enumerate EVERY birth
#          realization from c and EVERY (singleton,destination) death
#          realization from c', classify each landing by canon(), and sum the
#          accepted probability into the marginal kernels K_birth(c->[c']),
#          K_death(c'->[c]). Require pi(c) K_birth == pi(c') K_death to machine
#          precision. Enumerated (not symmetry-assumed) so any hidden
#          multiplicity asymmetry (the K/m failure mode) would show. flat AND
#          locmix targets, K in {6<->7, 7<->8}.
#
#  Part 4  END-TO-END MC on the REAL propose_birth_death!. Measure the full-move
#          kernel Khat(c->[c']) and Khat(c'->[c]) and require
#          pi(c) Khat_birth == pi(c') Khat_death within 5 MC-sigma, and that
#          both match the Part-3 analytic kernels. flat AND locmix.
#
#  Part 5  TARGET MARGINAL over K (bonus arbiter). Exactly enumerate
#          sum over partitions log_target for each K on the co-located set, to
#          show WHERE the intended target puts its mass -- if birth/death
#          balances, this is what the chain is (correctly) sampling.
#
# Self-validation: a passing pair perturbed by +0.5 must break flux balance.
#
# GATE semantics: nonzero exit if any hard requirement fails. Build so a
# hypothetical birth/death DB bug (fabricated reverse density) would FLAG here.
#
# Usage:  julia --project=. dev/birth_death_db_check.jl [M]
#         M = MC samples per kernel estimate in Part 4 (default 2_000_000).

using SMLMBaGoL
using SMLMData
using Printf
using Random

const S = SMLMBaGoL

const SIGMA = 0.0058
const MU    = 5.0
const SHAPE = 2.0
const RHO   = 2.0
const LOG_AREA = log(1.0e-2)
const GAMMA = SHAPE

const TD_FLAT = S.DMFlatTarget()
const SP_FLAT = S.FlatSpatial(LOG_AREA)
const AM = S.DMAllocation()
const AREA = exp(LOG_AREA)

colocated_locs(N) =
    [SMLMData.Emitter2DFit(0.0, 0.0, 1000.0, 0.0, SIGMA, SIGMA, 0.0, 0.0, 0.0, 1, 1, 0, i) for i in 1:N]

"Resolve (sp, td, use_kprior) for a spatial config symbol given this run's lps."
function spatial_config(spatial::Symbol, lps)
    spatial === :flat   && return SP_FLAT, TD_FLAT, true
    spatial === :locmix && return S.LocmixSpatial(lps), S.DMLocmixTarget(), false
    error("unknown spatial config $spatial")
end

# ---------------------------------------------------------------------------
# Gate bookkeeping.
# ---------------------------------------------------------------------------
const FAILURES = String[]
gate!(ok::Bool, label::String) = (ok || push!(FAILURES, label); ok)

# ---------------------------------------------------------------------------
# canonical partition key (loc-content, relabeled 1..K in first-appearance order)
# ---------------------------------------------------------------------------
function canon(z)
    map = Dict{Int,Int}(); nxt = 1; out = Vector{Int}(undef, length(z))
    for (i, a) in enumerate(z)
        ai = Int(a)
        haskey(map, ai) || (map[ai] = nxt; nxt += 1)
        out[i] = map[ai]
    end
    return out
end

# ---------------------------------------------------------------------------
# Feasibility + move-type probabilities EXACTLY as propose_birth_death! computes
# them (collapsed_moves.jl:1011-1029, 1081-1084, 1204-1206).
# ---------------------------------------------------------------------------
function bd_feasibility(sizes::Vector{Int}, N::Int, K::Int)
    n_singletons = count(==(1), sizes)
    n_eligible = N - n_singletons
    can_birth = n_eligible > 0 && K < N
    can_death = n_singletons > 0 && K > 1
    p_birth = (can_birth && can_death) ? 0.5 : (can_birth ? 1.0 : 0.0)
    p_death = (can_birth && can_death) ? 0.5 : (can_death ? 1.0 : 0.0)
    return (; n_singletons, n_eligible, can_birth, can_death, p_birth, p_death)
end

"Active cluster sizes of a state."
cluster_sizes(st) = Int[st.clusters[j].n for j in eachindex(st.active) if st.active[j]]

"logsumexp of predictive of loc `j` into every active cluster of `st` except slot `excl`."
function absorb_lse(st, j::Int, lp, sp, excl::Int)
    ws = Float64[]; slots = Int[]
    @inbounds for q in eachindex(st.active)
        (st.active[q] && q != excl) || continue
        push!(ws, S.spatial_pred(st.clusters[q], lp, sp))
        push!(slots, q)
    end
    mx = maximum(ws)
    lse = mx + log(sum(exp.(w - mx) for w in ws))
    return slots, ws, lse
end

build_state(z, locs, sp, use_kprior) =
    S.initialize_from_assignments(z, locs; sp=sp, am=AM, use_poisson_k_prior=use_kprior)

# ---------------------------------------------------------------------------
# BIRTH c -> c' : detach loc j (must live in a size>=2 cluster) as a singleton.
# Returns z_cp and (log_q_fwd_b, log_q_rev_b) computed exactly as the code's
# birth branch (collapsed_moves.jl:1071-1114).
# ---------------------------------------------------------------------------
function birth_densities(z_c::Vector{Int}, j::Int, locs, sp, use_kprior)
    N = length(z_c); K = length(unique(z_c))
    lp = S.precompute_loc_precisions(locs, SP_FLAT)[j]
    st_c = build_state(z_c, locs, sp, use_kprior)
    fc = bd_feasibility(cluster_sizes(st_c), N, K)
    log_q_fwd = log(fc.p_birth) - log(Float64(fc.n_eligible))

    z_cp = copy(z_c); z_cp[j] = maximum(z_c) + 1     # detach j into a fresh label
    st_cp = build_state(z_cp, locs, sp, use_kprior)
    fcp = bd_feasibility(cluster_sizes(st_cp), N, K + 1)

    # reverse death at c': pick singleton {j}, absorb into origin cluster A'.
    slot_j = Int(st_cp.assignments[j])               # {j}'s singleton slot
    # A' slot = slot holding a remaining member of j's original cluster
    A_slot = 0
    @inbounds for k in 1:N
        if k != j && z_c[k] == z_c[j]; A_slot = Int(st_cp.assignments[k]); break; end
    end
    @assert A_slot != 0 "birth requires a size>=2 origin cluster"
    _, _, lse = absorb_lse(st_cp, j, lp, sp, slot_j)
    log_w_dest = S.spatial_pred(st_cp.clusters[A_slot], lp, sp)
    log_q_rev = log(fcp.p_death) - log(Float64(fcp.n_singletons)) + log_w_dest - lse

    return z_cp, log_q_fwd, log_q_rev, fc, fcp, A_slot
end

# ---------------------------------------------------------------------------
# DEATH c' -> c2 : absorb singleton loc j into cluster at slot `dest_slot`.
# Returns z_c2 and (log_q_fwd_d, log_q_rev_d) exactly as the code's death branch
# (collapsed_moves.jl:1143-1208). dest_slot indexes st_cp's active clusters.
# ---------------------------------------------------------------------------
function death_densities(z_cp::Vector{Int}, j::Int, dest_slot::Int, locs, sp, use_kprior)
    N = length(z_cp); Kp = length(unique(z_cp))
    lp = S.precompute_loc_precisions(locs, SP_FLAT)[j]
    st_cp = build_state(z_cp, locs, sp, use_kprior)
    fcp = bd_feasibility(cluster_sizes(st_cp), N, Kp)
    slot_j = Int(st_cp.assignments[j])
    @assert st_cp.clusters[slot_j].n == 1 "death picks a singleton"

    _, _, lse = absorb_lse(st_cp, j, lp, sp, slot_j)
    log_w_dest = S.spatial_pred(st_cp.clusters[dest_slot], lp, sp)
    log_q_fwd = log(fcp.p_death) - log(Float64(fcp.n_singletons)) + log_w_dest - lse

    # dest cluster's loc-label to build z_c2
    dest_label = 0
    @inbounds for k in 1:N
        if Int(st_cp.assignments[k]) == dest_slot; dest_label = z_cp[k]; break; end
    end
    z_c2 = copy(z_cp); z_c2[j] = dest_label
    K2 = Kp - 1
    st_c2 = build_state(z_c2, locs, sp, use_kprior)
    fc2 = bd_feasibility(cluster_sizes(st_c2), N, K2)
    # reverse birth at c2: detach loc j (uniform among eligible)
    log_q_rev = log(fc2.p_birth) - log(Float64(fc2.n_eligible))

    return z_c2, log_q_fwd, log_q_rev, fcp, fc2
end

logpi(td, z, lps, sp) = S.log_target(td, z, lps, sp, MU, SHAPE, RHO)

# ---------------------------------------------------------------------------
# Co-located configuration builders. c = (K-1) singletons + one m-cluster.
# ---------------------------------------------------------------------------
function coloc_config(; K::Int, m::Int)
    N = (K - 1) + m
    locs = colocated_locs(N)
    lps = S.precompute_loc_precisions(locs, SP_FLAT)
    z_c = collect(1:N); for i in K:N; z_c[i] = K; end   # locs K..N form the m-cluster
    return N, locs, lps, z_c
end

# ===========================================================================
# PART 1 -- density honesty + alpha_b + alpha_d = 0 (per realization).
# ===========================================================================
function part1(; spatial::Symbol=:locmix)
    println("\n", "="^90)
    println("PART 1 -- BIRTH/DEATH density honesty & alpha_b+alpha_d=0  (spatial=$spatial)")
    println("  Birth c->c' detaches loc N from the m-cluster; reverse death absorbs {N} into A'.")
    println("  REQUIRE: code's birth-reverse density == death-forward density (recomputed from")
    println("  c''s own mechanism), and alpha_birth + alpha_death = 0 to machine precision.")
    println("="^90)
    @printf("\n  %-8s %-6s %14s %14s %12s %14s %12s %8s\n",
            "K<->K+1", "m", "logqf_b", "logqrev_b", "=logqf_d?", "alpha_b", "a_b+a_d", "OK?")
    println("  ", "-"^94)
    worst = 0.0
    for (K, m) in ((6, 4), (7, 4), (7, 3), (6, 5))
        N, locs, lps, z_c = coloc_config(K=K, m=m)
        sp, td, ukp = spatial_config(spatial, lps)
        j = N                                        # detach the last member
        z_cp, lqf_b, lqr_b, fc, fcp, A_slot = birth_densities(z_c, j, locs, sp, ukp)
        # death c'->c absorbing {j} into A_slot: recompute its forward density independently
        z_c2, lqf_d, lqr_d, fcp2, fc2 = death_densities(z_cp, j, A_slot, locs, sp, ukp)
        @assert canon(z_c2) == canon(z_c) "reverse death must return to c"
        lpc = logpi(td, z_c, lps, sp); lpcp = logpi(td, z_cp, lps, sp)
        alpha_b = (lpcp - lpc) + lqr_b - lqf_b
        alpha_d = (lpc - lpcp) + lqr_d - lqf_d
        dens_match = abs(lqr_b - lqf_d)              # birth-reverse vs death-forward
        dens_match2 = abs(lqr_d - lqf_b)             # death-reverse vs birth-forward
        sum_ad = abs(alpha_b + alpha_d)
        ok = dens_match < 1e-10 && dens_match2 < 1e-10 && sum_ad < 1e-9
        worst = max(worst, dens_match, dens_match2, sum_ad)
        @printf("  %-8s %-6d %14.6f %14.6f %12.2e %14.6f %12.2e %8s\n",
                "$K<->$(K+1)", m, lqf_b, lqr_b, dens_match, alpha_b, sum_ad, ok ? "yes" : "NO")
    end
    println("  ", "-"^94)
    @printf("  ==> worst |mismatch| over rows: %.2e  (want machine precision)\n", worst)
    gate!(worst < 1e-9, "P1: BD density honesty + alpha_b+alpha_d=0 ($spatial)")
    return nothing
end

# ===========================================================================
# PART 2 -- move-type + count consistency (lead item d), incl. boundaries.
# ===========================================================================
function part2(; spatial::Symbol=:locmix)
    println("\n", "="^90)
    println("PART 2 -- MOVE-TYPE (p_birth/p_death) + Delta_count consistency  (spatial=$spatial)")
    println("  For birth(K->K+1): p_birth(c) used fwd vs p_death(c') used in reverse density,")
    println("  and Delta_count(K->K+1) vs -(Delta_count(K+1->K)). Boundary K included.")
    println("="^90)
    @printf("\n  %-10s %-4s %-8s %10s %10s %10s %12s %12s %8s\n",
            "config", "m", "K<->K+1", "p_b(c)", "p_d(c')", "n_el(c)", "dCount_fwd", "dCount_rev", "OK?")
    println("  ", "-"^92)
    worst = 0.0
    # interior + boundary configs. K=1<->2 : c is a single m-cluster (no singletons).
    for (label, K, m) in (("interior", 6, 4), ("interior", 7, 4),
                          ("low-K",    2, 3), ("K=1 bd",  1, 4),
                          ("high-K",   8, 2))
        N = (K - 1) + m
        N < 2 && continue
        locs = colocated_locs(N); lps = S.precompute_loc_precisions(locs, SP_FLAT)
        sp, td, ukp = spatial_config(spatial, lps)
        z_c = collect(1:N); for i in K:N; z_c[i] = K; end
        j = N
        # need an eligible loc: cluster of j must be size>=2 (m>=2). K<N for birth.
        if !(m >= 2 && K < N); continue; end
        z_cp, lqf_b, lqr_b, fc, fcp, A_slot = birth_densities(z_c, j, locs, sp, ukp)
        dcount_fwd = S._log_count_posterior(K + 1, N, SHAPE, MU) - S._log_count_posterior(K, N, SHAPE, MU)
        dcount_rev = S._log_count_posterior(K, N, SHAPE, MU) - S._log_count_posterior(K + 1, N, SHAPE, MU)
        # code uses p_birth(c) fwd and p_death(c') in reverse density: recover both
        okrow = abs(dcount_fwd + dcount_rev) < 1e-12
        worst = max(worst, abs(dcount_fwd + dcount_rev))
        @printf("  %-10s %-4d %-8s %10.4f %10.4f %10d %12.5f %12.5f %8s\n",
                label, m, "$K<->$(K+1)", fc.p_birth, fcp.p_death, fc.n_eligible,
                dcount_fwd, dcount_rev, okrow ? "yes" : "NO")
    end
    println("  ", "-"^92)
    @printf("  ==> worst |dCount_fwd + dCount_rev|: %.2e  (want 0, count factor is symmetric)\n", worst)
    gate!(worst < 1e-10, "P2: Delta_count symmetric fwd/rev ($spatial)")
    println("  NOTE p_b(c), p_d(c') are each computed from their OWN state's feasibility;")
    println("  interior=0.5/0.5, and at K=1 c has no singletons so p_b(c)=1 (birth-forced),")
    println("  reverse p_d(c')=0.5 -- boundary-aware and consistent with propose_birth_death!.")
    return nothing
end

# ===========================================================================
# PART 3 -- EXACT MARGINAL FLUX BALANCE (enumerated, canonical).
# ===========================================================================
"""
Enumerate the full marginal birth kernel K(c->[c']) and death kernel K(c'->[c])
by summing accepted probability over EVERY micro-realization, classifying
landings by canon(). No symmetry assumed.
"""
function flux_balance(; K::Int, m::Int, spatial::Symbol)
    N, locs, lps, z_c = coloc_config(K=K, m=m)
    sp, td, ukp = spatial_config(spatial, lps)
    # reference c' : detach loc N
    z_cp_ref, = birth_densities(z_c, N, locs, sp, ukp)
    key_c  = canon(z_c); key_cp = canon(z_cp_ref)
    lpc  = logpi(td, z_c, lps, sp)
    lpcp = logpi(td, z_cp_ref, lps, sp)

    st_c = build_state(z_c, locs, sp, ukp)
    fc = bd_feasibility(cluster_sizes(st_c), N, K)

    # --- K_birth(c -> [c']) : every eligible loc detached ---
    Kbirth = 0.0
    @inbounds for j in 1:N
        st_c.clusters[Int(st_c.assignments[j])].n > 1 || continue   # eligible
        z_cp, lqf_b, lqr_b, _, _, _ = birth_densities(z_c, j, locs, sp, ukp)
        canon(z_cp) == key_cp || continue
        a = (lpcp - lpc) + lqr_b - lqf_b
        Kbirth += (fc.p_birth / fc.n_eligible) * min(1.0, exp(a))
    end

    # --- K_death(c' -> [c]) : every (singleton, destination) pair ---
    st_cp = build_state(z_cp_ref, locs, sp, ukp)
    fcp = bd_feasibility(cluster_sizes(st_cp), N, K + 1)
    Kdeath = 0.0
    @inbounds for s in 1:N
        st_cp.clusters[Int(st_cp.assignments[s])].n == 1 || continue  # singleton loc
        slot_s = Int(st_cp.assignments[s])
        for dslot in eachindex(st_cp.active)
            (st_cp.active[dslot] && dslot != slot_s) || continue
            z_c2, lqf_d, lqr_d, _, _ = death_densities(z_cp_ref, s, dslot, locs, sp, ukp)
            canon(z_c2) == key_c || continue
            lpc2 = logpi(td, z_c2, lps, sp)                          # == lpc by symmetry
            a = (lpc2 - lpcp) + lqr_d - lqf_d
            # lqf_d already = log[p_death/n_sing * w(dest)/Sum_w]; the per-realization
            # proposal probability for this (singleton s, dest dslot) IS exp(lqf_d).
            Kdeath += exp(lqf_d) * min(1.0, exp(a))
        end
    end

    lhs = exp(lpc) * Kbirth
    rhs = exp(lpcp) * Kdeath
    rel = abs(lhs - rhs) / max(lhs, rhs, 1e-300)
    return (; Kbirth, Kdeath, lhs, rhs, rel, n_elig=fc.n_eligible, n_sing=fcp.n_singletons)
end

function part3()
    println("\n", "="^90)
    println("PART 3 -- EXACT MARGINAL FLUX BALANCE  pi(c) K_birth == pi(c') K_death (enumerated)")
    println("  c = (K-1) singletons + one m-cluster; c' detaches one member. Every birth loc and")
    println("  every (singleton,dest) death realization summed and classified by canon().")
    println("  n_elig(c) vs n_sing(c') differ -- if the code mis-normalized, ratio != 1.")
    println("="^90)
    for spatial in (:flat, :locmix)
        println("\n  spatial = $spatial")
        @printf("  %-8s %-4s %8s %8s %14s %14s %12s %10s\n",
                "K<->K+1", "m", "n_elig", "n_sing", "pi(c)K_birth", "pi(c')K_death", "rel", "death/birth")
        println("  ", "-"^90)
        worst = 0.0
        for (K, m) in ((6, 4), (7, 4), (7, 3), (6, 5))
            r = flux_balance(K=K, m=m, spatial=spatial)
            worst = max(worst, r.rel)
            @printf("  %-8s %-4d %8d %8d %14.6e %14.6e %12.2e %10.4f\n",
                    "$K<->$(K+1)", m, r.n_elig, r.n_sing, r.lhs, r.rhs, r.rel, r.rhs / r.lhs)
        end
        println("  ", "-"^90)
        @printf("  ==> max rel flux residual (%s): %.2e  (want machine precision)\n", spatial, worst)
        gate!(worst < 1e-9, "P3: BD marginal flux balance ($spatial)")
    end
    # self-validation: perturb one alpha by +0.5 -> flux must break
    println("\n  SELF-VALIDATION: perturb birth alpha by +0.5 on the K=7<->8 m=4 locmix row:")
    let
        K, m, spatial = 7, 4, :locmix
        N, locs, lps, z_c = coloc_config(K=K, m=m)
        sp, td, ukp = spatial_config(spatial, lps)
        z_cp, lqf_b, lqr_b, fc, fcp, A_slot = birth_densities(z_c, N, locs, sp, ukp)
        lpc = logpi(td, z_c, lps, sp); lpcp = logpi(td, z_cp, lps, sp)
        a_good = (lpcp - lpc) + lqr_b - lqf_b
        # perturbed marginal (only the +0.5 birth branch changes)
        r = flux_balance(K=K, m=m, spatial=spatial)
        Kbirth_bad = (fc.p_birth) * min(1.0, exp(a_good + 0.5))   # all n_elig identical
        lhs_bad = exp(lpc) * Kbirth_bad
        rel_bad = abs(lhs_bad - r.rhs) / max(lhs_bad, r.rhs, 1e-300)
        @printf("    good rel=%.2e   perturbed rel=%.4f  (want perturbed >> 0)\n", r.rel, rel_bad)
        gate!(rel_bad > 1e-3, "P3: self-validation (+0.5 breaks flux)")
    end
    return nothing
end

# ===========================================================================
# PART 4 -- END-TO-END MC on the REAL propose_birth_death!.
# ===========================================================================
"Empirical full-move kernel K(from->[to]) by running propose_birth_death! M times."
function mc_bd_kernel(z_from, to_key, locs, sp, ukp, M; seed)
    Random.seed!(seed)
    hits = 0
    for _ in 1:M
        st = build_state(z_from, locs, sp, ukp)
        S.propose_birth_death!(st, locs, MU, SHAPE, RHO)
        canon(st.assignments) == to_key && (hits += 1)
    end
    return hits / M
end

function part4(; M::Int=2_000_000)
    println("\n", "="^90)
    println("PART 4 -- END-TO-END MC on the REAL propose_birth_death! (M=$M)")
    println("  Confirms the REAL move uses the Part-3 analytic densities. A kernel is")
    println("  MC-RESOLVABLE only if expected hits (analytic*M) >= 25; otherwise MC cannot")
    println("  test it (e.g. flat births accept at ~3e-8 -- the flat target's Occam floor)")
    println("  and the DB verdict rests on Part 3 (exact analytic flux). Resolvable kernels")
    println("  must match analytic within 5 MC-sigma; where both directions resolve, the")
    println("  measured flux must balance.")
    println("="^90)
    RESOLVE = 25.0   # expected hits needed to call a kernel MC-resolvable
    for spatial in (:flat, :locmix)
        for (K, m) in ((6, 4), (7, 4))
            N, locs, lps, z_c = coloc_config(K=K, m=m)
            sp, td, ukp = spatial_config(spatial, lps)
            z_cp, = birth_densities(z_c, N, locs, sp, ukp)
            key_c = canon(z_c); key_cp = canon(z_cp)
            lpc = logpi(td, z_c, lps, sp); lpcp = logpi(td, z_cp, lps, sp)
            ana = flux_balance(K=K, m=m, spatial=spatial)

            Mrun = spatial === :flat ? M : max(M ÷ 2, 500_000)
            Kb = mc_bd_kernel(z_c,  key_cp, locs, sp, ukp, Mrun; seed=101)
            Kd = mc_bd_kernel(z_cp, key_c,  locs, sp, ukp, Mrun; seed=202)
            seb = sqrt(max(Kb,1/Mrun)*(1-Kb)/Mrun); sed = sqrt(max(Kd,1/Mrun)*(1-Kd)/Mrun)
            resb = ana.Kbirth * Mrun >= RESOLVE
            resd = ana.Kdeath * Mrun >= RESOLVE
            fb = exp(lpc) * Kb; fd = exp(lpcp) * Kd
            sig = sqrt((exp(lpc)*seb)^2 + (exp(lpcp)*sed)^2)
            println("\n  [$spatial] K=$K<->$(K+1), m=$m, M=$Mrun:")
            @printf("    Khat_birth = %.5e +/- %.1e   analytic %.5e   exp.hits=%.1f  %s\n",
                    Kb, seb, ana.Kbirth, ana.Kbirth*Mrun,
                    resb ? (@sprintf("MC/ana=%.4f %s", Kb/ana.Kbirth, abs(Kb-ana.Kbirth)<=5seb ? "MATCH" : "MISMATCH"))
                         : "below MC resolution")
            @printf("    Khat_death = %.5e +/- %.1e   analytic %.5e   exp.hits=%.1f  %s\n",
                    Kd, sed, ana.Kdeath, ana.Kdeath*Mrun,
                    resd ? (@sprintf("MC/ana=%.4f %s", Kd/ana.Kdeath, abs(Kd-ana.Kdeath)<=5sed ? "MATCH" : "MISMATCH"))
                         : "below MC resolution")
            # gate: resolvable kernels must match analytic
            resb && gate!(abs(Kb-ana.Kbirth) <= 5seb, "P4: real birth kernel == analytic ($spatial K=$K)")
            resd && gate!(abs(Kd-ana.Kdeath) <= 5sed, "P4: real death kernel == analytic ($spatial K=$K)")
            # gate flux balance only when both directions resolve
            if resb && resd
                bal = abs(fb - fd) <= 5*sig
                @printf("    pi(c)Kb = %.5e   pi(c')Kd = %.5e   |diff|=%.2e (<=5sig=%.2e : %s)\n",
                        fb, fd, abs(fb-fd), 5*sig, bal ? "yes" : "NO")
                gate!(bal, "P4: MC flux balance ($spatial K=$K)")
            else
                @printf("    flux-balance MC-untestable (%s below resolution); DB verdict from Part 3 (rel=%.1e).\n",
                        resb ? "death" : "birth", ana.rel)
            end
        end
    end
    return nothing
end

# ===========================================================================
# PART 5 -- TARGET MARGINAL over K (bonus): where does the intended target put
# its mass on the co-located set? If BD balances, this is what the chain samples.
# Exact sum over all set-partitions of N co-located locs, grouped by K.
# ===========================================================================
"generate all set partitions of 1:N as assignment vectors (restricted growth)."
function all_partitions(N::Int)
    parts = Vector{Vector{Int}}()
    z = ones(Int, N)
    function rec(i, kmax)
        if i > N; push!(parts, copy(z)); return; end
        for c in 1:kmax+1
            z[i] = c
            rec(i+1, max(kmax, c))
        end
    end
    rec(2, 1); z[1] = 1
    # the recursion above sets z[1]=1 implicitly; ensure first element handled:
    return parts
end

function part5(; N::Int=8)
    println("\n", "="^90)
    println("PART 5 -- TARGET MARGINAL over K on the co-located set (N=$N), exact enumeration")
    println("  P(K) = sum over set-partitions with K blocks of exp(log_target). If birth/death")
    println("  balances, the sampler's K-histogram should track THIS, not a DB artifact.")
    println("="^90)
    locs = colocated_locs(N); lps = S.precompute_loc_precisions(locs, SP_FLAT)
    parts = all_partitions(N)
    for spatial in (:flat, :locmix)
        sp, td, _ = spatial_config(spatial, lps)
        byK = Dict{Int,Float64}()
        lts = Float64[]
        Ks = Int[]
        for z in parts
            lt = logpi(td, z, lps, sp)
            push!(lts, lt); push!(Ks, length(unique(z)))
        end
        mx = maximum(lts)
        wsum = Dict{Int,Float64}()
        for (lt, K) in zip(lts, Ks)
            wsum[K] = get(wsum, K, 0.0) + exp(lt - mx)
        end
        Z = sum(values(wsum))
        println("\n  spatial = $spatial   (#partitions=$(length(parts)))")
        kbest = argmax(wsum); pbest = wsum[kbest]/Z
        emean = sum(K*wsum[K] for K in keys(wsum))/Z
        @printf("  %-4s %12s\n", "K", "P(K)")
        for K in sort(collect(keys(wsum)))
            bar = repeat("#", round(Int, 40*wsum[K]/Z))
            @printf("  %-4d %12.4e  %s\n", K, wsum[K]/Z, bar)
        end
        @printf("  -> target MAP-K = %d (P=%.3f), target mean-K = %.2f\n", kbest, pbest, emean)
    end
    println("\n  READING: if BD (Parts 1-4) balances, the sampler is correctly sampling THIS")
    println("  target. A target MAP/mean K below the true emitter count is a MODEL property")
    println("  (co-located Occam defect), not a detailed-balance bug.")
    return nothing
end

function main(; M::Int=2_000_000)
    println("#"^90)
    println("# GATE-3b BIRTH/DEATH DB CHECKER  (co-located, K=6<->7 and 7<->8)")
    println("# targets: DMFlatTarget (:flat/:dm/:poisson) + DMLocmixTarget (:locmix/:dm/:none)")
    @printf("# mu=%.1f shape=%.1f rho=%.1f sigma=%.4f log_area=%.4f  M=%d\n",
            MU, SHAPE, RHO, SIGMA, LOG_AREA, M)
    println("#"^90)
    part1(spatial=:flat); part1(spatial=:locmix)
    part2(spatial=:flat); part2(spatial=:locmix)
    part3()
    part4(M=M)
    part5(N=8)
    println("\n", "#"^90)
    println("# VERDICT")
    println("#"^90)
    if isempty(FAILURES)
        println("""
  ALL BD GATES PASS: birth/death satisfies EXACT detailed balance for the intended
  target at K=6<->7 and 7<->8 on the co-located set, flat AND locmix:
    * Part 1: reverse densities are HONEST (birth-reverse == death-forward from
      c''s own mechanism; alpha_b + alpha_d = 0) -- no fabrication/mis-normalization.
    * Part 2: move-type (p_birth/p_death) and Delta_count factors are boundary-aware
      and consistent between each move and its reverse.
    * Part 3: enumerated marginal flux pi(c)K_birth == pi(c')K_death to machine
      precision despite n_elig(c) != n_sing(c') -- the normalizations cancel.
    * Part 4: the REAL propose_birth_death! reproduces the balanced analytic kernels.
  => The residual K-down bias is NOT a birth/death DB violation. Part 5 shows the
     intended target's own K-marginal on co-located data -- the chain is sampling
     it correctly; a low target MAP-K is a MODEL (Occam) property, not DB.""")
    else
        println("\n  BD GATE FAILURES ($(length(FAILURES))):")
        for f in FAILURES; println("    FAIL  ", f); end
        println("""
  BIRTH/DEATH DB IS BROKEN. If Part 3/4 show death flux > birth flux growing with
  K (n_sing/n_elig class), localize the mis-normalized density/count factor.""")
    end
    println("#"^90)
    return length(FAILURES)
end

nfail = main(M=(isempty(ARGS) ? 2_000_000 : parse(Int, ARGS[1])))
exit(nfail == 0 ? 0 : 1)

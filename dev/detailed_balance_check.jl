# GATE-3 detailed-balance checker for the K-changing moves (split/merge,
# birth/death) at K=7 <-> K=8, targeted at the K-marginal deficit regime.
#
# WHY: brute_force_enumeration.jl only validates DB up to K_max=4. The historic
# merge-move DB violation only fires for merged clusters with m>2 members, so it
# was invisible there. This driver localizes DB at K=7<->8 per move and per
# proposal sub-factor, and gives an overall programmatic PASS/FAIL verdict.
#
# THE (FIXED) BUG = merge reverse-split SEED COMPATIBILITY. The merge draws BOTH
# reverse-split seeds uniformly from all m union members with NO one-per-cluster
# constraint. A split pins seed1->A, seed2->B (collapsed_moves.jl
# _do_sequential_split!), so a reverse split can recreate the pre-merge A|B
# partition only if the two seeds lie in DIFFERENT pre-merge clusters. When both
# seeds fall in the SAME cluster (is_in_b[1]==true) the true reverse-split
# density is 0, but the low-level density helpers only score members 3:m and
# would fabricate a finite q_rev -> merge over-accepted by ~m(m-1)/(2 nA nB)
# (2x for a (3,1) merge) -> downward K pressure growing with cluster size.
#
# THE FIX (src/collapsed_moves.jl, merge branch of propose_split_merge!):
#   if is_in_b[1]; log_q_alloc_rev = -Inf; elseif n_restricted_scans>0; ...; end
# i.e. the compatibility indicator is enforced in the ACCEPTANCE PATH
# (propose_split_merge!), not in _restricted_gibbs_transition_density, which is
# unchanged and still fabricates in isolation.
#
# WHAT THIS CHECKER TESTS (fixed-path-aware):
#   Parts A/B/B2 : split(7->8) forward + per-compatible-seed marginal DB
#                  (final-sweep-only q_fwd = canonical Jain-Neal; unaffected).
#   Part C       : birth/death exact DB (unaffected).
#   Part E       : per-ordered-seed-pair table on the FIXED acceptance branch
#                  (guard mirrored line-for-line): incompatible draws must give
#                  q_rev = -Inf (auto-reject), compatible draws unchanged.
#   Part F       : seed-MARGINALIZED flux balance pi(c) K_split == pi(c') K_merge
#                  with the FIXED code kernel, for (nA,nB) in
#                  {(3,1),(2,2),(4,1),(5,1)}, flat AND locmix targets. The
#                  PRE-FIX fabricated kernel is reported alongside as reference.
#   Part D       : END-TO-END MC on the REAL propose_split_merge! -- the only
#                  part that exercises the shipped code path, hence the part
#                  that FLAGS unfixed code. Measured merge kernel must match the
#                  DB-TRUE (indicator-enforced) prediction, not the fabricated
#                  one. Run for flat AND locmix targets. Death kernel likewise.
#   Self-check   : a passing log_ratio_code perturbed by +0.5 must FAIL.
#
# GATE SEMANTICS: run against UNFIXED code, Part D fails (measured merge rate
# ~2x DB-true) -> nonzero exit. Run against FIXED code, all parts pass -> exit 0.
# NOTE Parts E/F mirror the FIXED source, so on unfixed code they stay green by
# construction; the code-path sensitivity lives in Part D.
#
# All density calculations reuse the actual package functions
# (_log_sequential_allocation, _restricted_gibbs_transition_density,
# _sample_sequential_launch, _restricted_gibbs_sweep!, partition_prior,
# _log_count_posterior, log_prior_k_poisson, spatial_ml, spatial_pred,
# check_detailed_balance, log_target) -- no re-derivation of the density math.
#
# Usage:  julia --project=. dev/detailed_balance_check.jl [M]
#         M = MC samples per kernel estimate in Part D (default 2_000_000).

using SMLMBaGoL
using SMLMData
using Printf
using Random
using LinearAlgebra: I

const S = SMLMBaGoL

# ---------------------------------------------------------------------------
# Fixed model parameters (values are irrelevant to whether DB holds; DB must
# hold for ANY target). Chosen in a plausible co-located regime.
# ---------------------------------------------------------------------------
const SIGMA = 0.0058          # isotropic localization SE (um)
const MU    = 5.0             # count-model prior mean
const SHAPE = 2.0             # count-model shape (= DM gamma)
const RHO   = 2.0             # Poisson(rho*A) K-prior rate
const LOG_AREA = log(1.0e-2)  # area 0.01 um^2 (finite; co-located pts otherwise degenerate)
const GAMMA = SHAPE           # dm_gamma(DMAllocation(nothing), shape) = shape

# Targets:
#   :flat   -> DMFlatTarget   = flat spatial x DM partition x NegBin count x Poisson-K
#              (sampler config spatial=:flat,   allocation=:dm, k_prior=:poisson)
#   :locmix -> DMLocmixTarget = locmix spatial x DM partition x NegBin count, NO K prior
#              (sampler config spatial=:locmix, allocation=:dm, k_prior=:none)
# Both are exact acceptance targets of propose_split_merge!/propose_birth_death!
# under the corresponding state configuration.
const TD = S.DMFlatTarget()
const SP = S.FlatSpatial(LOG_AREA)
const AM = S.DMAllocation()
const AREA = exp(LOG_AREA)

colocated_locs(N) =
    [SMLMData.Emitter2DFit(0.0, 0.0, 1000.0, 0.0, SIGMA, SIGMA, 0.0, 0.0, 0.0, 1, 1, 0, i) for i in 1:N]

"Resolve (sp, td, use_kprior) for a spatial config symbol given this run's lps."
function spatial_config(spatial::Symbol, lps)
    spatial === :flat   && return SP, TD, true
    spatial === :locmix && return S.LocmixSpatial(lps), S.DMLocmixTarget(), false
    error("unknown spatial config $spatial")
end

# ---------------------------------------------------------------------------
# Gate bookkeeping: every hard requirement registers through gate!.
# ---------------------------------------------------------------------------
const FAILURES = String[]
function gate!(ok::Bool, label::String)
    ok || push!(FAILURES, label)
    return ok
end

# ---------------------------------------------------------------------------
# Target log-density, two independent ways (consistency cross-check).
#   log_target(DMFlatTarget)  vs  the code's own component functions
#   (spatial_ml sum + partition_prior + _log_count_posterior + log_prior_k_poisson)
# ---------------------------------------------------------------------------
"log pi via the diagnostic target density (flat config)."
function logpi_target(z, lps)
    return S.log_target(TD, z, lps, LOG_AREA, MU, SHAPE, RHO)
end

"log pi via the SAMPLER's own component functions (state-based)."
function logpi_code_components(z, locs)
    st = S.initialize_from_assignments(z, locs; sp=SP, am=AM, use_poisson_k_prior=true)
    K = st.n_active
    N = length(z)
    lml   = S._total_spatial_lml(st)
    dm    = S.partition_prior(AM, st, N, SHAPE)
    count = S._log_count_posterior(K, N, SHAPE, MU)
    kpri  = S.log_prior_k_poisson(K, RHO, AREA)
    return lml + dm + count + kpri, (lml=lml, dm=dm, count=count, kpri=kpri, K=K)
end

# ---------------------------------------------------------------------------
# Restricted-Gibbs allocation state machinery for a co-located m-member cluster
# with FIXED seeds at member positions 1,2 (seed1 -> sub-cluster A,
# seed2 -> sub-cluster B). Non-seed members are positions 3..m.
# State s in 0..2^(m-2)-1 : bit (j-3) set  <=>  member j in sub-cluster B.
# ---------------------------------------------------------------------------
full_isb(s::Int, m::Int) = Bool[j == 1 ? false : j == 2 ? true : (((s >> (j - 3)) & 1) == 1) for j in 1:m]

"Launch distribution v[s] = P(sequential predictive launch produces allocation s)."
function launch_dist(m, mi, lps, sp=SP)
    ns = 2^(m - 2)
    v = Vector{Float64}(undef, ns)
    for s in 0:ns-1
        v[s+1] = exp(S._log_sequential_allocation(mi, full_isb(s, m), lps, sp, GAMMA))
    end
    return v
end

"Full sub-cluster stats (cs_a, cs_b) for allocation state s (all members allocated)."
function full_cs(s, m, mi, lps)
    fs = full_isb(s, m)
    csa = S.empty_cluster(lps); csb = S.empty_cluster(lps)
    for j in 1:m
        lp = lps[mi[j]]
        if fs[j]; csb = S.add_loc(csb, lp); else; csa = S.add_loc(csa, lp); end
    end
    return csa, csb
end

"""
One-restricted-Gibbs-sweep transition matrix P[s,t] (rows sum to 1).
`_restricted_gibbs_transition_density` removes each member from its START
position, so it must be given the FULL start-allocation cluster stats for s.
"""
function sweep_matrix(m, mi, lps, sp=SP)
    ns = 2^(m - 2)
    P = Matrix{Float64}(undef, ns, ns)
    for s in 0:ns-1
        fs = full_isb(s, m)
        csa, csb = full_cs(s, m, mi, lps)
        for t in 0:ns-1
            ft = full_isb(t, m)
            P[s+1, t+1] = exp(S._restricted_gibbs_transition_density(fs, ft, csa, csb, mi, lps, sp, GAMMA, AM))
        end
    end
    return P
end

"""
Validate P against the ACTUAL sampler: run the real `_restricted_gibbs_sweep!`
n_trials times from start state s0 and compare empirical landing frequencies to
P[s0, :]. Confirms the transition-density matrix equals the sampler's kernel.
"""
function validate_sweep_matrix(m, mi, lps; s0::Int=0, n_trials::Int=400_000, seed::Int=7)
    ns = 2^(m - 2)
    P = sweep_matrix(m, mi, lps)
    counts = zeros(Int, ns)
    Random.seed!(seed)
    for _ in 1:n_trials
        isb = full_isb(s0, m)
        csa, csb = full_cs(s0, m, mi, lps)
        csa, csb, _ = S._restricted_gibbs_sweep!(isb, csa, csb, mi, lps, SP, GAMMA, true, AM)
        # decode landed state from isb (positions 3..m)
        st = 0
        for j in 3:m
            if isb[j]; st |= (1 << (j - 3)); end
        end
        counts[st+1] += 1
    end
    emp = counts ./ n_trials
    maxdev = maximum(abs.(emp .- P[s0+1, :]))
    return (P_row=P[s0+1, :], emp=emp, maxdev=maxdev)
end

"""
Post-intermediate launch distribution seen by the FINAL scan:
  n_scans == 0 : no launch/scan machinery -> the sequential allocation IS the
                 proposal (return v_launch; caller treats it as the final density).
  n_scans >= 1 : launch, then (n_scans-1) intermediate sweeps -> v * P^(n_scans-1).
"""
function post_intermediate_dist(v, P, n_scans)
    n_scans == 0 && return v                 # handled specially by callers
    n_scans == 1 && return v
    return vec(transpose(v) * P^(n_scans - 1))
end

# ---------------------------------------------------------------------------
# EXACT marginal split/merge detailed-balance test at K <-> K+1, fixed seeds.
#
# Uses the code's per-realization proposal densities:
#   forward split : log_q_fwd(L) = -log(K)   + log P[L, s*]   (final-sweep only)
#   reverse merge : log_q_rev    = -log(#pairs at c')          (merge deterministic)
# and the reverse direction (merge move) reconstruction density
#   log_q_rev_merge(L') = -log(K_new) + log P[L', s*]          (final-sweep only)
#
# Then forms the EXACT marginal kernel entries by summing per-launch acceptance:
#   K_split(c->c') = b_K (1/K)   sum_L  piL(L) P[L,s*] min(1, exp(logR_split(L)))
#   K_merge(c'->c) = d_{K+1} (1/#pairs) sum_L' piL(L') min(1, exp(logR_merge(L')))
# and checks pi(c) K_split == pi(c') K_merge.
# NOTE: fixed COMPATIBLE seeds -> the seed-compatibility guard never fires here;
# this part isolates the final-sweep-only q_fwd question (unchanged by the fix).
# ---------------------------------------------------------------------------
function split_merge_marginal_db(; m::Int, n_scans::Int, s_star::Int,
                                   K::Int, verbose::Bool=false)
    # base state: (K-1) singletons + one co-located cluster of m members
    N = (K - 1) + m
    locs = colocated_locs(N)
    lps  = S.precompute_loc_precisions(locs, SP)

    # assignment vectors
    # clusters 1..K-1 are singletons (locs 1..K-1); cluster K holds locs K..N
    z_c = collect(1:N); for i in K:N; z_c[i] = K; end
    fs = full_isb(s_star, m)                    # allocation of the m members
    z_cp = copy(z_c)
    for (pos, loc) in enumerate(K:N)            # member pos -> loc index
        z_cp[loc] = fs[pos] ? (K + 1) : K       # B-side -> new cluster K+1
    end

    # sanity: c' must have K+1 clusters (both sub-clusters non-empty)
    Kp = maximum(z_cp)
    @assert Kp == K + 1 "s_star=$s_star does not yield a genuine K+1 split (got K'=$Kp)"

    logpi_c  = logpi_target(z_c, lps)
    logpi_cp = logpi_target(z_cp, lps)
    Tt = logpi_cp - logpi_c                     # target log-ratio

    mi = collect(K:N)                           # member loc indices (positions 1..m)
    v = launch_dist(m, mi, lps)
    P = sweep_matrix(m, mi, lps)

    # move-type selection probabilities (boundary-aware)
    b_K   = (K <= 1) ? 1.0 : (K >= N ? 0.0 : 0.5)
    d_Kp1 = (K + 1 >= N) ? 1.0 : 0.5
    npairs = (K + 1) * K / 2

    if n_scans == 0
        # deterministic proposal density = sequential allocation
        q_alloc = v[s_star + 1]
        logqf = -log(K) + log(q_alloc)
        logqr = -log(npairs)
        logR_split = Tt + (logqr - logqf) + (log(d_Kp1) - log(b_K))
        Ksplit = b_K * (1 / K) * q_alloc * min(1.0, exp(logR_split))
        logR_merge = -Tt + ((-log(K) + log(q_alloc)) - (-log(npairs))) + (log(b_K) - log(d_Kp1))
        Kmerge = d_Kp1 * (1 / npairs) * min(1.0, exp(logR_merge))
    else
        piL = post_intermediate_dist(v, P, n_scans)
        Ksplit = 0.0
        Kmerge = 0.0
        for L in 0:length(piL)-1
            pL = piL[L+1]
            pLs = P[L+1, s_star+1]              # final-scan density landing on s*
            # forward split, realized launch L, code's final-sweep-only q_fwd
            logqf = -log(K) + log(pLs)
            logqr = -log(npairs)
            logR_split = Tt + (logqr - logqf) + (log(d_Kp1) - log(b_K))
            Ksplit += pL * pLs * min(1.0, exp(logR_split))
            # reverse: merge move, fresh launch L', reconstruction density P[L',s*]
            logqr_m = -log(K) + log(pLs)        # reverse-split density (-log K_new + finalscan)
            logqf_m = -log(npairs)
            logR_merge = -Tt + (logqr_m - logqf_m) + (log(b_K) - log(d_Kp1))
            Kmerge += pL * min(1.0, exp(logR_merge))
        end
        Ksplit *= b_K * (1 / K)
        Kmerge *= d_Kp1 * (1 / npairs)
    end

    lhs = exp(logpi_c)  * Ksplit
    rhs = exp(logpi_cp) * Kmerge
    resid = abs(lhs - rhs)
    rel = resid / max(abs(lhs), abs(rhs), 1e-300)

    if verbose
        @printf("      logpi_c=%.6f  logpi_c'=%.6f  T=%.6f\n", logpi_c, logpi_cp, Tt)
        @printf("      K_split=%.6e  K_merge=%.6e\n", Ksplit, Kmerge)
        @printf("      pi(c)K_split=%.6e  pi(c')K_merge=%.6e\n", lhs, rhs)
    end
    return (resid=resid, rel=rel, lhs=lhs, rhs=rhs, Ksplit=Ksplit, Kmerge=Kmerge)
end

# ---------------------------------------------------------------------------
# FULL-PATH-PRODUCT variant (the codex-proposed "correct" q_fwd): q_fwd uses the
# density of the WHOLE realized launch+scan path, not just the final sweep.
# We enumerate all launch+intermediate+final paths landing on s* and form the
# marginal kernel using per-path acceptance with the full-path density.
# The reverse merge is treated symmetrically (full reconstruction-path density).
# ---------------------------------------------------------------------------
function split_merge_marginal_db_fullpath(; m::Int, n_scans::Int, s_star::Int, K::Int)
    @assert n_scans >= 1
    N = (K - 1) + m
    locs = colocated_locs(N)
    lps  = S.precompute_loc_precisions(locs, SP)
    z_c = collect(1:N); for i in K:N; z_c[i] = K; end
    fs = full_isb(s_star, m); z_cp = copy(z_c)
    for (pos, loc) in enumerate(K:N); z_cp[loc] = fs[pos] ? (K + 1) : K; end
    logpi_c = logpi_target(z_c, lps); logpi_cp = logpi_target(z_cp, lps); Tt = logpi_cp - logpi_c
    mi = collect(K:N); v = launch_dist(m, mi, lps); P = sweep_matrix(m, mi, lps)
    b_K = (K <= 1) ? 1.0 : (K >= N ? 0.0 : 0.5)
    d_Kp1 = (K + 1 >= N) ? 1.0 : 0.5
    npairs = (K + 1) * K / 2
    ns = length(v)

    # enumerate all paths L0 -> L1 -> ... -> L_{n_scans-1} -> s*
    # path prob = v[L0] * prod P[L_{i-1},L_i] ; last hop is the final scan to s*
    # forward full-path density (allocation) = path prob (launch+all sweeps)
    Ksplit = 0.0; Kmerge = 0.0
    # states visited: n_scans hops total (launch is L0; then n_scans sweeps land on s*)
    # sequence of intermediate states has length n_scans (L1..L_{n_scans}=s*)
    function recurse(prevstate, hop, pathprob)
        if hop == n_scans
            # prevstate must equal s* (final landing)
            if prevstate == s_star
                # forward: full path density
                logqf = -log(K) + log(pathprob)
                logqr = -log(npairs)
                logR_split = Tt + (logqr - logqf) + (log(d_Kp1) - log(b_K))
                Ksplit += b_K * (1 / K) * pathprob * min(1.0, exp(logR_split))
                # reverse merge: symmetric full reconstruction-path density
                logqr_m = -log(K) + log(pathprob)
                logqf_m = -log(npairs)
                logR_merge = -Tt + (logqr_m - logqf_m) + (log(b_K) - log(d_Kp1))
                Kmerge += d_Kp1 * (1 / npairs) * pathprob * min(1.0, exp(logR_merge))
            end
            return
        end
        for nxt in 0:ns-1
            recurse(nxt, hop + 1, pathprob * P[prevstate+1, nxt+1])
        end
    end
    for L0 in 0:ns-1
        recurse(L0, 1, v[L0+1])
    end
    lhs = exp(logpi_c) * Ksplit; rhs = exp(logpi_cp) * Kmerge
    return (resid=abs(lhs - rhs), rel=abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300),
            lhs=lhs, rhs=rhs)
end

# ---------------------------------------------------------------------------
# Birth/death exact detailed balance at K <-> K+1.
# Densities are deterministic given the transition (no launch aux), so we
# replicate the code's forward/reverse density formulas faithfully with the
# real spatial_pred, then check pi(c) q_birth alpha == pi(c') q_death alpha'.
# ---------------------------------------------------------------------------
function birth_death_db(; K::Int, m::Int, verbose::Bool=false)
    # c: (K-1) singletons + one co-located m-cluster (locs K..N).  Birth detaches
    # loc N from cluster K -> singleton (cluster K+1).
    N = (K - 1) + m
    locs = colocated_locs(N)
    lps  = S.precompute_loc_precisions(locs, SP)
    z_c = collect(1:N); for i in K:N; z_c[i] = K; end
    z_cp = copy(z_c); z_cp[N] = K + 1          # detach loc N

    logpi_c  = logpi_target(z_c, lps)
    logpi_cp = logpi_target(z_cp, lps)
    Tt = logpi_cp - logpi_c

    # ----- BIRTH move at c (K -> K+1) : detach loc N -----
    n_singletons_c = K - 1                     # locs 1..K-1
    n_eligible_c   = N - n_singletons_c        # locs in n>=2 clusters = m
    can_birth_c = n_eligible_c > 0 && K < N
    can_death_c = n_singletons_c > 0 && K > 1
    p_birth_c = (can_birth_c && can_death_c) ? 0.5 : (can_birth_c ? 1.0 : 0.0)
    logqf_birth = log(p_birth_c) - log(n_eligible_c)

    # reverse death (from c'): absorb the new singleton (loc N) back into cluster K.
    # cluster K at c' has m-1 members. destination weights = predictive over all
    # active clusters except the new singleton.
    n_sing_cp = n_singletons_c + 1             # + new singleton (loc N)
    # (cluster K still has m-1 >= 1 members; only becomes singleton if m-1==1)
    if m - 1 == 1; n_sing_cp += 1; end
    can_d_cp = n_sing_cp > 0 && (K + 1) > 1
    can_b_cp = (N - n_sing_cp) > 0 && (K + 1) < N
    p_death_cp = (can_d_cp && can_b_cp) ? 0.5 : (can_d_cp ? 1.0 : 0.0)

    st_cp = S.initialize_from_assignments(z_cp, locs; sp=SP, am=AM, use_poisson_k_prior=true)
    lpN = lps[N]
    # predictive of absorbing loc N into each active cluster except its own singleton
    logw_dest = -Inf; lse_terms = Float64[]
    for j in eachindex(st_cp.active)
        st_cp.active[j] || continue
        j == z_cp[N] && continue               # skip the new singleton itself
        w = S.spatial_pred(st_cp.clusters[j], lpN, SP)
        push!(lse_terms, w)
        if j == K                              # loc N's original cluster (the dest)
            logw_dest = w
        end
    end
    mx = maximum(lse_terms); logsum = mx + log(sum(exp.(lse_terms .- mx)))
    logqr_birth = log(p_death_cp) - log(n_sing_cp) + logw_dest - logsum

    logR_birth = Tt + (logqr_birth - logqf_birth)   # birth/death fold move-type into q

    # ----- DEATH move at c' (K+1 -> K) : absorb loc N into cluster K -----
    logqf_death = log(p_death_cp) - log(n_sing_cp) + logw_dest - logsum   # == logqr_birth
    # reverse birth (from c): detach loc N again.
    can_b_c2 = n_eligible_c > 0 && K < N
    can_d_c2 = n_singletons_c > 0 && K > 1
    p_birth_c2 = (can_b_c2 && can_d_c2) ? 0.5 : (can_b_c2 ? 1.0 : 0.0)
    logqr_death = log(p_birth_c2) - log(n_eligible_c)                     # == logqf_birth
    logR_death = -Tt + (logqr_death - logqf_death)

    # DB via the marginal kernel entries (deterministic proposals)
    Kbirth = exp(logqf_birth) * min(1.0, exp(logR_birth))
    Kdeath = exp(logqf_death) * min(1.0, exp(logR_death))
    lhs = exp(logpi_c) * Kbirth
    rhs = exp(logpi_cp) * Kdeath
    resid = abs(lhs - rhs)

    if verbose
        @printf("      logqf_birth=%.6f logqr_birth=%.6f logR_birth=%.6f\n", logqf_birth, logqr_birth, logR_birth)
        @printf("      logqf_death=%.6f logqr_death=%.6f logR_death=%.6f\n", logqf_death, logqr_death, logR_death)
        @printf("      logR_birth + logR_death = %.3e (want 0)\n", logR_birth + logR_death)
    end
    return (resid=resid, rel=resid / max(abs(lhs), abs(rhs), 1e-300),
            logR_birth=logR_birth, logR_death=logR_death,
            logqf_birth=logqf_birth, logqr_birth=logqr_birth,
            logqf_death=logqf_death, logqr_death=logqr_death,
            Kbirth=Kbirth, Kdeath=Kdeath,
            z_c=z_c, z_cp=z_cp, locs=locs, lps=lps, Tt=Tt)
end

# ---------------------------------------------------------------------------
# Part A: formal check_detailed_balance table (per move, per direction) with
# target-vs-proposal decomposition and self-validation.
# Uses n_scans=0 for split/merge so proposal densities are deterministic.
# ---------------------------------------------------------------------------
function part_A()
    println("\n", "="^88)
    println("PART A -- check_detailed_balance table (per move / per direction), K=7<->8")
    println("  Split/merge use n_scans=0 (deterministic proposal densities).")
    println("="^88)

    K = 7; m = 4; s_star = 1               # member3->B (partition sizes 1..)
    # --- build the split transition c(K=7) -> c'(K=8) ---
    N = (K - 1) + m; locs = colocated_locs(N); lps = S.precompute_loc_precisions(locs, SP)
    z_c = collect(1:N); for i in K:N; z_c[i] = K; end
    fs = full_isb(s_star, m); z_cp = copy(z_c)
    for (pos, loc) in enumerate(K:N); z_cp[loc] = fs[pos] ? (K+1) : K; end

    # target consistency: log_target vs code components
    lt_c  = logpi_target(z_c, lps);  cc_c,  parts_c  = logpi_code_components(z_c, locs)
    lt_cp = logpi_target(z_cp, lps); cc_cp, parts_cp = logpi_code_components(z_cp, locs)
    @printf("\n  Target consistency (log_target vs sampler component sum):\n")
    @printf("    z_c  : log_target=%.9f  code_sum=%.9f  |diff|=%.2e\n", lt_c, cc_c, abs(lt_c-cc_c))
    @printf("    z_c' : log_target=%.9f  code_sum=%.9f  |diff|=%.2e\n", lt_cp, cc_cp, abs(lt_cp-cc_cp))
    gate!(abs(lt_c - cc_c) < 1e-9 && abs(lt_cp - cc_cp) < 1e-9,
          "A: log_target vs sampler component sum")

    mi = collect(K:N); v = launch_dist(m, mi, lps)
    q_alloc = v[s_star+1]
    b_K = 0.5; d_Kp1 = 0.5; npairs = (K+1)*K/2

    # proposal densities INCLUDING move-type selection (so log_ratio_direct == code log_alpha)
    # forward split total: b_K * (1/K) * q_alloc ; reverse (merge) total: d_{K+1} * (1/#pairs)
    logqf_split = log(b_K) - log(K) + log(q_alloc)
    logqr_split = log(d_Kp1) - log(npairs)
    # code's log_alpha for split:
    Δtarget = lt_cp - lt_c
    logα_split = Δtarget + (logqr_split - logqf_split)

    loc_precs = lps
    r = S.check_detailed_balance(z_c, z_cp, loc_precs, LOG_AREA, TD;
            μ=MU, shape=SHAPE, ρ=RHO,
            log_q_fwd=logqf_split, log_q_rev=logqr_split,
            log_ratio_code=logα_split, tol=1e-8)

    println("\n  ", "-"^84)
    @printf("  %-14s %-9s %14s %14s %14s %10s %6s\n",
            "move", "dir", "Δ_target", "Δ_proposal", "|Δd-Δcode|", "Δ_direct", "PASS")
    println("  ", "-"^84)
    Δprop_split = logqr_split - logqf_split
    @printf("  %-14s %-9s %14.6f %14.6f %14.2e %10.4f %6s\n",
            "split", "7->8", Δtarget, Δprop_split, r.difference, r.log_ratio_direct, r.pass ? "yes" : "NO")
    gate!(r.pass, "A: split 7->8 check_detailed_balance")

    # merge move 8->7 (reverse): forward = merge, reverse = split reconstruction
    logqf_merge = log(d_Kp1) - log(npairs)
    logqr_merge = log(b_K) - log(K) + log(q_alloc)
    logα_merge = (lt_c - lt_cp) + (logqr_merge - logqf_merge)
    rm = S.check_detailed_balance(z_cp, z_c, loc_precs, LOG_AREA, TD;
            μ=MU, shape=SHAPE, ρ=RHO,
            log_q_fwd=logqf_merge, log_q_rev=logqr_merge,
            log_ratio_code=logα_merge, tol=1e-8)
    @printf("  %-14s %-9s %14.6f %14.6f %14.2e %10.4f %6s\n",
            "merge", "8->7", lt_c - lt_cp, logqr_merge - logqf_merge, rm.difference, rm.log_ratio_direct, rm.pass ? "yes" : "NO")
    gate!(rm.pass, "A: merge 8->7 check_detailed_balance (compatible seeds)")

    # --- birth 7->8 and death 8->7 ---
    bd = birth_death_db(K=7, m=4)
    rb = S.check_detailed_balance(bd.z_c, bd.z_cp, bd.lps, LOG_AREA, TD;
            μ=MU, shape=SHAPE, ρ=RHO,
            log_q_fwd=bd.logqf_birth, log_q_rev=bd.logqr_birth,
            log_ratio_code=bd.logR_birth, tol=1e-8)
    @printf("  %-14s %-9s %14.6f %14.6f %14.2e %10.4f %6s\n",
            "birth", "7->8", bd.Tt, bd.logqr_birth - bd.logqf_birth, rb.difference, rb.log_ratio_direct, rb.pass ? "yes" : "NO")
    gate!(rb.pass, "A: birth 7->8 check_detailed_balance")
    rd = S.check_detailed_balance(bd.z_cp, bd.z_c, bd.lps, LOG_AREA, TD;
            μ=MU, shape=SHAPE, ρ=RHO,
            log_q_fwd=bd.logqf_death, log_q_rev=bd.logqr_death,
            log_ratio_code=bd.logR_death, tol=1e-8)
    @printf("  %-14s %-9s %14.6f %14.6f %14.2e %10.4f %6s\n",
            "death", "8->7", -bd.Tt, bd.logqr_death - bd.logqf_death, rd.difference, rd.log_ratio_direct, rd.pass ? "yes" : "NO")
    gate!(rd.pass, "A: death 8->7 check_detailed_balance")
    println("  ", "-"^84)

    # --- SELF-VALIDATION: perturb a passing case's log_ratio_code by +0.5 -> must FAIL ---
    rbad = S.check_detailed_balance(z_c, z_cp, loc_precs, LOG_AREA, TD;
            μ=MU, shape=SHAPE, ρ=RHO,
            log_q_fwd=logqf_split, log_q_rev=logqr_split,
            log_ratio_code=logα_split + 0.5, tol=1e-8)
    @printf("\n  SELF-VALIDATION: split with log_ratio_code += 0.5 -> |diff|=%.4f  pass=%s (want NO)\n",
            rbad.difference, rbad.pass ? "yes(BAD)" : "NO(good)")
    gate!(!rbad.pass && abs(rbad.difference - 0.5) < 1e-6,
          "A: self-validation (+0.5 perturbation must FAIL)")
    return nothing
end

# ---------------------------------------------------------------------------
# Part B: exact marginal split/merge DB across n_scans and cluster size m.
# ---------------------------------------------------------------------------
function part_B()
    println("\n", "="^88)
    println("PART B -- marginal split<->merge DB at K=7<->8, CONDITIONED on a fixed")
    println("  COMPATIBLE seed pair (launch/intermediate-scan randomness marginalized).")
    println("  This isolates the final-sweep-only q_fwd question (unchanged by the fix;")
    println("  the seed-compatibility guard never fires on compatible seeds -- see E/F).")
    println("  Residual = | pi(c) K_split(c->c') - pi(c') K_merge(c'->c) |. DB holds iff ~0.")
    println("="^88)

    # validate the transition matrix P against the REAL _restricted_gibbs_sweep! sampler
    println("\n  Validating transition matrix P vs real _restricted_gibbs_sweep! (m=4, start=0):")
    let m = 4, K = 7
        N = (K - 1) + m; locs = colocated_locs(N); lps = S.precompute_loc_precisions(locs, SP)
        mi = collect(K:N)
        vv = validate_sweep_matrix(m, mi, lps; s0=0, n_trials=400_000)
        @printf("    P[0,:]      = %s\n", string(round.(vv.P_row, digits=5)))
        @printf("    empirical   = %s\n", string(round.(vv.emp, digits=5)))
        @printf("    max |dev|   = %.4e  (MC noise ~ 1/sqrt(4e5) ~ 1.6e-3)\n", vv.maxdev)
        gate!(vv.maxdev < 0.01, "B: sweep matrix vs real _restricted_gibbs_sweep!")
    end
    println()
    @printf("  %-4s %-9s %-7s %14s %14s %12s %10s\n",
            "m", "sizes", "n_scans", "pi(c)K_split", "pi(c')K_merge", "|resid|", "rel")
    println("  ", "-"^80)
    sizes_label(m, s) = begin
        fs = full_isb(s, m); nb = count(fs); "$(m-nb)+$(nb)"
    end
    maxresid = 0.0
    for m in (3, 4, 5, 6)
        s_star = 0                              # B = {seed2} : an (m-1, 1) split
        for n_scans in (0, 1, 2, 5)
            r = split_merge_marginal_db(m=m, n_scans=n_scans, s_star=s_star, K=7)
            maxresid = max(maxresid, r.rel)
            @printf("  %-4d %-9s %-7d %14.6e %14.6e %12.3e %10.2e\n",
                    m, sizes_label(m, s_star), n_scans, r.lhs, r.rhs, r.resid, r.rel)
        end
    end
    println("  ", "-"^80)
    for n_scans in (0, 1, 2, 5)                  # a balanced (2,2) split for m=4
        r = split_merge_marginal_db(m=4, n_scans=n_scans, s_star=1, K=7)
        maxresid = max(maxresid, r.rel)
        @printf("  %-4d %-9s %-7d %14.6e %14.6e %12.3e %10.2e\n",
                4, sizes_label(4, 1), n_scans, r.lhs, r.rhs, r.resid, r.rel)
    end
    @printf("\n  ==> max relative DB residual over all rows: %.2e  (want machine precision)\n", maxresid)
    gate!(maxresid < 1e-10, "B: per-compatible-seed marginal split/merge DB")
    println("  ==> GIVEN a compatible seed pair, final-sweep-only q_fwd is exactly correct")
    println("      (canonical Jain-Neal). Launch recipe is IDENTICAL on both sides:")
    println("      split launch = sequential predictive in _do_sequential_split! (seeds pinned")
    println("      1->A, 2->B), merge reverse launch = _sample_sequential_launch (same recipe).")
    return nothing
end

# ---------------------------------------------------------------------------
# Part B2: final-sweep-only vs full-path-product for q_fwd (informational).
# ---------------------------------------------------------------------------
function part_B2()
    println("\n", "="^88)
    println("PART B2 -- q_fwd: FINAL-SWEEP-ONLY (code) vs FULL-PATH-PRODUCT, m=4, K=7<->8")
    println("="^88)
    println()
    @printf("  %-7s %18s %18s\n", "n_scans", "final-sweep |resid|", "full-path |resid|")
    println("  ", "-"^48)
    for n_scans in (1, 2, 3, 5)
        rf = split_merge_marginal_db(m=4, n_scans=n_scans, s_star=1, K=7)
        rp = split_merge_marginal_db_fullpath(m=4, n_scans=n_scans, s_star=1, K=7)
        @printf("  %-7d %18.3e %18.3e\n", n_scans, rf.resid, rp.resid)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Part C: birth/death exact DB.
# ---------------------------------------------------------------------------
function part_C()
    println("\n", "="^88)
    println("PART C -- EXACT birth<->death detailed balance at K=7<->8")
    println("="^88)
    println()
    @printf("  %-4s %14s %14s %14s %12s\n", "m", "logR_birth", "logR_death", "sum(want 0)", "|resid|")
    println("  ", "-"^64)
    worst = 0.0
    for m in (2, 3, 4, 5)
        bd = birth_death_db(K=7, m=m)
        worst = max(worst, abs(bd.logR_birth + bd.logR_death), bd.rel)
        @printf("  %-4d %14.6f %14.6f %14.3e %12.3e\n",
                m, bd.logR_birth, bd.logR_death, bd.logR_birth + bd.logR_death, bd.resid)
    end
    gate!(worst < 1e-9, "C: birth/death exact DB")
    return nothing
end

# ---------------------------------------------------------------------------
# Seed-pair machinery (Parts E/F).
#
# Configuration: c = (K-1) singletons + one co-located m-cluster (locs K..N);
# c' = same but the last nB members split off into cluster K+1 (an (m-nB, nB)
# split). The merge move at c' selects the pair and must compute q_rev = the
# density of the reverse split reconstructing c'. The code draws the two
# reverse-split seeds uniformly from ALL m union members and builds is_in_b by
# seed2's cluster; the FIXED acceptance path then auto-rejects when
# is_in_b[1]==true (both seeds in the same pre-merge cluster).
# ---------------------------------------------------------------------------
function seed_setup(; K::Int, m::Int, nB::Int)
    N = (K - 1) + m
    locs = colocated_locs(N)
    lps  = S.precompute_loc_precisions(locs, SP)
    z_c = collect(1:N); for i in K:N; z_c[i] = K; end
    z_cp = copy(z_c)
    for i in (N - nB + 1):N; z_cp[i] = K + 1; end
    return N, locs, lps, z_c, z_cp
end

"""
Per-seed-pair data, mirroring the merge code exactly:
  member_indices = [s1, s2, sorted rest]
  is_in_b[i] = (cluster of mi[i] == cluster of s2)
  compatible  <=>  s1 and s2 in DIFFERENT pre-merge clusters (is_in_b[1]==false)
  tstate = the is_in_b bits at positions 3..m as a state index.
"""
function pair_data(s1::Int, s2::Int, union_::Vector{Int}, z_cp, m::Int)
    rest = sort([u for u in union_ if u != s1 && u != s2])
    mi = vcat([s1, s2], rest)
    tfull = Bool[z_cp[i] == z_cp[s2] for i in mi]
    compat = !tfull[1]
    tstate = 0
    for j in 3:m
        if tfull[j]; tstate |= 1 << (j - 3); end
    end
    return mi, tfull, compat, tstate
end

incompat_frac(m, nA, nB) = (nA * (nA - 1) + nB * (nB - 1)) / (m * (m - 1))

"""
Seed-MARGINALIZED kernels at K<->K+1 for the (m-nB, nB) split, in CANONICAL
state space (both seed orderings of a compatible pair reconstruct c').

  K_split       : true split kernel c->c' (only compatible seed draws can
                  produce c'; the split side has no defect).
  K_merge_fixed : the FIXED code's merge kernel c'->c -- mirrors the guard
                  `if is_in_b[1]; log_q_alloc_rev = -Inf` in
                  propose_split_merge!: incompatible seed draws auto-reject
                  (zero flux); compatible draws use the real density path.
                  This is identical to the indicator-enforced (DB-true) kernel.
  K_merge_fab   : the PRE-FIX kernel -- incompatible seed draws use the
                  FABRICATED finite reconstruction density (positions 1-2
                  ignored), exactly as the unguarded density helpers compute
                  it. Kept as the reference the unfixed code reproduces.

DB requires pi(c) K_split == pi(c') K_merge. Which of fixed/fab satisfies it
is the arbiter; Part D checks which one the REAL move reproduces.
"""
function seed_marginal_kernels(; K::Int, m::Int, nB::Int, n_scans::Int,
                                 spatial::Symbol=:flat)
    N, locs, lps, z_c, z_cp = seed_setup(K=K, m=m, nB=nB)
    sp, td, _ = spatial_config(spatial, lps)
    union_ = collect(K:N)
    logpi_c = S.log_target(td, z_c, lps, sp, MU, SHAPE, RHO)
    logpi_cp = S.log_target(td, z_cp, lps, sp, MU, SHAPE, RHO)
    Δt = logpi_cp - logpi_c
    b_K = 0.5; d_Kp1 = 0.5                     # interior K
    npairs = (K + 1) * K / 2
    w = 1.0 / (m * (m - 1))                    # ordered seed-pair density

    tot_split = 0.0; tot_mfab = 0.0; tot_mfixed = 0.0
    n_pairs = 0; n_incompat = 0
    for s1 in union_, s2 in union_
        s1 == s2 && continue
        n_pairs += 1
        mi, tfull, compat, tstate = pair_data(s1, s2, union_, z_cp, m)
        compat || (n_incompat += 1)
        v = launch_dist(m, mi, lps, sp)
        if n_scans == 0
            q = v[tstate + 1]
            logα_m = -Δt + ((-log(K) + log(q)) - (-log(npairs))) + (log(b_K) - log(d_Kp1))
            mc = min(1.0, exp(logα_m))
            tot_mfab += mc
            if compat                          # FIXED path: guard passes -> density path
                logα_s = Δt + ((-log(npairs)) - (-log(K) + log(q))) + (log(d_Kp1) - log(b_K))
                tot_split += q * min(1.0, exp(logα_s))
                tot_mfixed += mc
            end                                # else guard fires: q_rev=-Inf, zero flux
        else
            P = sweep_matrix(m, mi, lps, sp)
            piL = post_intermediate_dist(v, P, n_scans)
            sc = 0.0; mc = 0.0
            for L in 0:length(piL)-1
                pL = piL[L+1]; pLt = P[L+1, tstate+1]
                logα_m = -Δt + ((-log(K) + log(pLt)) - (-log(npairs))) + (log(b_K) - log(d_Kp1))
                mc += pL * min(1.0, exp(logα_m))
                if compat
                    logα_s = Δt + ((-log(npairs)) - (-log(K) + log(pLt))) + (log(d_Kp1) - log(b_K))
                    sc += pL * pLt * min(1.0, exp(logα_s))
                end
            end
            tot_mfab += mc
            if compat
                tot_split += sc
                tot_mfixed += mc
            end
        end
    end

    Ksplit = b_K * (1 / K) * w * tot_split
    Kmfab   = d_Kp1 * (1 / npairs) * w * tot_mfab
    Kmfixed = d_Kp1 * (1 / npairs) * w * tot_mfixed
    lhs = exp(logpi_c) * Ksplit
    rhs_fab   = exp(logpi_cp) * Kmfab
    rhs_fixed = exp(logpi_cp) * Kmfixed
    return (lhs=lhs, rhs_fab=rhs_fab, rhs_fixed=rhs_fixed,
            Ksplit=Ksplit, Kmfab=Kmfab, Kmfixed=Kmfixed,
            frac_incompat=n_incompat / n_pairs,
            rel_fixed=abs(lhs - rhs_fixed) / max(lhs, rhs_fixed, 1e-300),
            infl_fab=rhs_fab / lhs)
end

# ---------------------------------------------------------------------------
# Part E: per-seed-pair test on the FIXED merge(8->7) acceptance path.
# For every ordered reverse-seed pair, mirror the FIXED merge branch of
# propose_split_merge! line-for-line (guard FIRST, then launch + n_scans-1
# intermediate sweeps + final transition density, all with the REAL package
# functions) to get log q_rev_code, and compare Δ_code (the merge
# log-acceptance the fixed code computes) against Δ_direct (with the TRUE
# q_rev, which is -Inf when the seeds are incompatible).
# NOTE: this mirrors the FIXED source; that the mirror equals the shipped
# kernel is established end-to-end by Part D on the real propose_split_merge!.
# ---------------------------------------------------------------------------
function part_E(; n_scans::Int=5)
    println("\n", "="^88)
    println("PART E -- merge(8->7) reverse-split SEED COMPATIBILITY on the FIXED path")
    println("  c' = (K=8): 6 singletons + A={locs 7,8,9} + B={loc 10};  merge pair (A,B), m=4.")
    println("  For each ordered reverse-seed draw (s1,s2): the FIXED code's q_rev (guard")
    println("  `if is_in_b[1] -> -Inf` mirrored, else real density path, one stochastic")
    println("  realization) vs TRUE q_rev (compatibility indicator). Δ = merge log-acceptance.")
    println("  REQUIRE: incompatible -> Δ_code = -Inf (auto-reject); compatible -> match, tol 1e-8.")
    println("="^88)

    K = 7; m = 4; nB = 1
    N, locs, lps, z_c, z_cp = seed_setup(K=K, m=m, nB=nB)
    union_ = collect(K:N)
    logpi_c = logpi_target(z_c, lps); logpi_cp = logpi_target(z_cp, lps)
    Δt_rev = logpi_c - logpi_cp
    b_K = 0.5; d_Kp1 = 0.5; npairs = (K + 1) * K / 2

    println()
    @printf("  %-10s %-7s %14s %14s %12s %12s %10s %6s\n",
            "(s1,s2)", "compat", "logqrev_code", "logqrev_true", "Δ_code", "Δ_direct", "|diff|", "OK?")
    println("  ", "-"^92)
    Random.seed!(4242)
    n_bad = 0; n_pairs = 0; n_incompat = 0; n_inf_on_incompat = 0
    for s1 in union_, s2 in union_
        s1 == s2 && continue
        n_pairs += 1
        mi, tfull, compat, _ = pair_data(s1, s2, union_, z_cp, m)
        compat || (n_incompat += 1)
        # mirror the FIXED merge branch of propose_split_merge! exactly:
        # guard first, then launch/scans/transition density only if compatible
        lq_code = if tfull[1]                  # is_in_b[1] -> guard fires
            -Inf
        else
            launch_isb, cs_a, cs_b = S._sample_sequential_launch(mi, lps, SP, GAMMA)
            for _ in 1:(n_scans - 1)
                cs_a, cs_b, _ = S._restricted_gibbs_sweep!(launch_isb, cs_a, cs_b,
                    mi, lps, SP, GAMMA, false, AM)
            end
            S._restricted_gibbs_transition_density(launch_isb, tfull,
                cs_a, cs_b, mi, lps, SP, GAMMA, AM)
        end
        lq_true = compat ? lq_code : -Inf
        Δ_code   = Δt_rev + ((-log(K) + lq_code) - (-log(npairs))) + (log(b_K) - log(d_Kp1))
        Δ_direct = Δt_rev + ((-log(K) + lq_true) - (-log(npairs))) + (log(b_K) - log(d_Kp1))
        ok = (Δ_code == Δ_direct == -Inf) || abs(Δ_code - Δ_direct) < 1e-8
        ok || (n_bad += 1)
        if !compat && lq_code == -Inf; n_inf_on_incompat += 1; end
        diffstr = (Δ_code == Δ_direct == -Inf) ? "0 (both)" :
                  isfinite(Δ_code - Δ_direct) ? @sprintf("%.2e", abs(Δ_code - Δ_direct)) : "Inf"
        @printf("  %-10s %-7s %14s %14s %12s %12s %10s %6s\n",
                "($s1,$s2)", compat ? "yes" : "NO",
                isfinite(lq_code) ? @sprintf("%.6f", lq_code) : "-Inf",
                compat ? @sprintf("%.6f", lq_true) : "-Inf",
                isfinite(Δ_code) ? @sprintf("%.4f", Δ_code) : "-Inf",
                isfinite(Δ_direct) ? @sprintf("%.4f", Δ_direct) : "-Inf",
                diffstr, ok ? "yes" : "FAIL")
    end
    println("  ", "-"^92)
    @printf("\n  %d / %d ordered seed draws are INCOMPATIBLE; the FIXED path returned q_rev=-Inf\n",
            n_incompat, n_pairs)
    @printf("  (auto-reject) on %d / %d of them. Mismatches vs TRUE q_rev: %d (want 0).\n",
            n_inf_on_incompat, n_incompat, n_bad)
    gate!(n_bad == 0, "E: fixed path == true q_rev on all seed draws")
    gate!(n_inf_on_incompat == n_incompat && n_incompat == 6,
          "E: guard fires (q_rev=-Inf) on exactly the 6 incompatible draws")

    println("\n  Incompatible-draw fraction (nA(nA-1)+nB(nB-1))/(m(m-1)) for representative merges:")
    for (mm, nb) in ((2, 1), (4, 1), (4, 2), (5, 1), (6, 1), (8, 1))
        na = mm - nb
        @printf("    (nA,nB)=(%d,%d)  m=%-2d :  %.3f\n", na, nb, mm, incompat_frac(mm, na, nb))
    end
    println("  -> zero at m=2 (all pairs compatible): why small-K brute-force tests passed,")
    println("     and why the guard changes nothing for pair merges.")
    return nothing
end

# ---------------------------------------------------------------------------
# Part F: seed-MARGINALIZED kernel flux balance against the FIXED code kernel.
# ---------------------------------------------------------------------------
function part_F()
    println("\n", "="^88)
    println("PART F -- seed-MARGINALIZED flux balance at K=7<->8 (canonical states)")
    println("  DB requires pi(c) K_split == pi(c') K_merge. K_merge_FIXED mirrors the fixed")
    println("  acceptance path (guard -> zero flux on incompatible draws); K_merge_FAB is the")
    println("  pre-fix fabricated-q_rev kernel, reported as reference. REQUIRE rel_FIXED ~ 0.")
    println("="^88)
    for spatial in (:flat, :locmix)
        println()
        println("  spatial = $(spatial)  " *
                (spatial === :flat ? "(DMFlatTarget: :flat/:dm/:poisson)" :
                                     "(DMLocmixTarget: :locmix/:dm/:none)"))
        @printf("  %-9s %-7s %13s %13s %10s %13s %10s %9s\n",
                "(nA,nB)", "n_scans", "pi(c)Ksplit", "pi(c')Km_FIX", "rel_FIX",
                "pi(c')Km_FAB", "FAB/lhs", "1/compat")
        println("  ", "-"^92)
        worst = 0.0
        for (m, nB) in ((4, 1), (4, 2), (5, 1), (6, 1))
            na = m - nB
            pred = 1.0 / (1.0 - incompat_frac(m, na, nB))
            for n_scans in (0, 5)
                r = seed_marginal_kernels(K=7, m=m, nB=nB, n_scans=n_scans, spatial=spatial)
                worst = max(worst, r.rel_fixed)
                @printf("  %-9s %-7d %13.4e %13.4e %10.2e %13.4e %10.4f %9.4f\n",
                        "($na,$nB)", n_scans, r.lhs, r.rhs_fixed, r.rel_fixed,
                        r.rhs_fab, r.infl_fab, pred)
            end
        end
        println("  ", "-"^92)
        @printf("  ==> max rel_FIXED (%s): %.2e  (want machine precision)\n", spatial, worst)
        gate!(worst < 1e-8, "F: flux balance with FIXED kernel ($spatial)")
    end
    println("""
  READING: 'rel_FIX' ~ machine precision => the guard-enforced merge kernel balances
  the split flux EXACTLY (DB restored). 'FAB/lhs' > 1 => the PRE-FIX kernel produced
  MORE c'->c (merge) flux than DB allows, by the predicted factor 1/compat_frac when
  acceptance saturates (excess merges = downward K pressure, growing with m).""")
    return nothing
end

# ---------------------------------------------------------------------------
# Part D: Monte-Carlo cross-check that runs the ACTUAL moves (no replication).
# Estimates the marginal kernel entries K(c'->c) directly by calling
# propose_split_merge! / propose_birth_death! from fresh copies of the state
# and counting landings. THE code-path-sensitive gate: unfixed code reproduces
# the FABRICATED kernel (ratio ~2x DB-true), fixed code the DB-TRUE kernel.
# ---------------------------------------------------------------------------
canon(z) = begin
    map = Dict{Int,Int}(); nxt = 1; out = similar(collect(z))
    for (i, a) in enumerate(z)
        ai = Int(a)
        if !haskey(map, ai); map[ai] = nxt; nxt += 1; end
        out[i] = map[ai]
    end
    out
end

"Empirical kernel entry K(from->to) by running `move!` from a fresh `from` state M times."
function mc_kernel(z_from, z_to_canon, locs, move!, M; seed=11, sp=SP, use_kprior=true)
    Random.seed!(seed)
    hits = 0
    for _ in 1:M
        st = S.initialize_from_assignments(z_from, locs; sp=sp, am=AM, use_poisson_k_prior=use_kprior)
        move!(st)
        if canon(st.assignments) == z_to_canon; hits += 1; end
    end
    return hits / M
end

function part_D(; M::Int=2_000_000)
    println("\n", "="^88)
    println("PART D -- end-to-end MC on the REAL propose_split_merge! (M=$M): does the")
    println("  shipped merge kernel match the DB-TRUE prediction (guard/indicator enforced,")
    println("  = fixed code) or the FABRICATED-q_rev prediction (= unfixed code)?")
    println("  GATE: |MC - TRUE| <= 5 MC-sigma  AND  MC separated from FAB by > 5 sigma.")
    println("="^88)

    K = 7; m = 4; N = (K-1)+m
    locs = colocated_locs(N)
    z_c = collect(1:N); for i in K:N; z_c[i] = K; end
    z_cp = copy(z_c); z_cp[N] = K+1                      # c' = (3,1): loc N detached
    cc = canon(z_c)

    for spatial in (:flat, :locmix)
        lps = S.precompute_loc_precisions(locs, SP)
        sp, td, use_kprior = spatial_config(spatial, lps)
        Mrun = spatial === :flat ? M : max(M ÷ 2, 500_000)
        r = seed_marginal_kernels(K=7, m=4, nB=1, n_scans=5, spatial=spatial)
        sm5!(st) = (S.propose_split_merge!(st, locs, MU, SHAPE, RHO; n_restricted_scans=5); nothing)
        KrMC = mc_kernel(z_cp, cc, locs, sm5!, Mrun; seed=202, sp=sp, use_kprior=use_kprior)
        serM = sqrt(KrMC*(1-KrMC)/Mrun)
        println("\n  [$(spatial)] MERGE kernel  Khat(c'->c)  [c'=(K=8, 3+1) -> c=(K=7, 4-cluster)], M=$Mrun:")
        @printf("    MC measured (real move)         = %.5e  +/- %.1e\n", KrMC, serM)
        @printf("    predicted DB-TRUE kernel (fix)  = %.5e   ratio MC/TRUE = %.4f  (want 1)\n",
                r.Kmfixed, KrMC / r.Kmfixed)
        @printf("    predicted FABRICATED kernel     = %.5e   ratio MC/FAB  = %.4f\n",
                r.Kmfab, KrMC / r.Kmfab)
        ok_true = abs(KrMC - r.Kmfixed) <= 5 * serM
        ok_sep  = abs(KrMC - r.Kmfab)  >  5 * serM
        @printf("    |MC-TRUE|=%.2e (<= 5 sigma=%.2e : %s)   |MC-FAB|=%.2e (> 5 sigma : %s)\n",
                abs(KrMC - r.Kmfixed), 5serM, ok_true ? "yes" : "NO",
                abs(KrMC - r.Kmfab), ok_sep ? "yes" : "NO")
        gate!(ok_true, "D: real merge kernel == DB-TRUE prediction ($spatial)")
        gate!(ok_sep,  "D: real merge kernel != FABRICATED prediction ($spatial)")
    end

    # death direction (no seeds -> should match its exact prediction; flat target)
    bdx = birth_death_db(K=7, m=4)
    bd!(st) = (S.propose_birth_death!(st, locs, MU, SHAPE, RHO); nothing)
    KrbMC = mc_kernel(z_cp, cc, locs, bd!, M; seed=404)
    serb = sqrt(KrbMC*(1-KrbMC)/M)
    println("\n  [flat] DEATH kernel  Khat(c'->c)  [absorb singleton back; no seed draw involved]:")
    @printf("    MC measured = %.5e  +/- %.1e\n", KrbMC, serb)
    @printf("    exact (C)   = %.5e   ratio = %.4f  (want 1)\n", bdx.Kdeath, KrbMC/bdx.Kdeath)
    gate!(abs(KrbMC - bdx.Kdeath) <= 5 * serb, "D: real death kernel == exact prediction")
    return nothing
end

function main(; M::Int=2_000_000)
    println("#"^88)
    println("# GATE-3 DETAILED-BALANCE CHECKER for K-changing moves at K=7<->8")
    println("# targets: DMFlatTarget (:flat/:dm/:poisson) + DMLocmixTarget (:locmix/:dm/:none)")
    @printf("# mu=%.1f shape=%.1f rho=%.1f  sigma=%.4f um  log_area=%.4f  M=%d\n",
            MU, SHAPE, RHO, SIGMA, LOG_AREA, M)
    println("#"^88)
    part_A()
    part_B()
    part_B2()
    part_C()
    part_E()
    part_F()
    part_D(M=M)
    println("\n", "#"^88)
    println("# VERDICT")
    println("#"^88)
    if isempty(FAILURES)
        println("""
  ALL GATES PASS: the merge move's seed-compatibility guard (log_q_alloc_rev =
  -Inf when is_in_b[1], in propose_split_merge!) restores EXACT detailed balance
  on the fixed code path:
    * Part E: guard fires (q_rev=-Inf, auto-reject) on exactly the incompatible
      reverse-seed draws; compatible draws unchanged at machine precision.
    * Part F: guard-enforced merge kernel balances the split flux exactly
      (flat AND locmix targets, all (nA,nB), n_scans in {0,5}).
    * Part D: the REAL propose_split_merge! merge kernel now matches the
      DB-TRUE prediction (ratio ~1), NOT the pre-fix fabricated kernel (~2x).
    * Split(7->8), birth, death unchanged and exact (Parts A/B/C).
    * Self-validation intact (+0.5 perturbation FAILS).""")
    else
        println("\n  DB GATE FAILURES ($(length(FAILURES))):")
        for f in FAILURES
            println("    FAIL  ", f)
        end
        println("""
  DB IS BROKEN on the tested code path. If Part D shows ratio MC/TRUE ~ 2 with
  MC matching the FABRICATED kernel, the seed-compatibility guard is missing
  from the merge branch of propose_split_merge! (unfixed code).""")
    end
    println("#"^88)
    return length(FAILURES)
end

nfail = main(M=(isempty(ARGS) ? 2_000_000 : parse(Int, ARGS[1])))
exit(nfail == 0 ? 0 : 1)

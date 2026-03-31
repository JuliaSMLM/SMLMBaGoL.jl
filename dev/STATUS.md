# Sampler Research Status

## Current State (2026-03-31, post-Round 12)

Round 12 tested predictive-only proposals (remove DM weighting from all proposal kernels, MH-correct). **Conclusion: DM proposal dynamics are NOT the large-N bottleneck.** The N=40 co-located K=1 trap exists identically in both DM-weighted and predictive-only proposals (birth acceptance 0.12% in both cases). The bottleneck is the target landscape under Poisson K prior + flat spatial.

**Brute-force: 4/4 PASS.** Small-N sampler correct with predictive-only proposals. Close dimer improved 1.30x→1.27x.
**Practical: UNCHANGED.** smlmsim recall 0.478 (baseline 0.475). Poisson K prior itself causes worse recall than locmix (0.592).

### Root Cause: Per-Allocation Energy Barriers (Round 11, refined)

The spatial ML has a per-cluster Occam factor `-log(n_k)` (partition-dependent part, after cancelling terms that depend only on N). For any SPECIFIC allocation at K=8 vs K=4, the DM relief (+30) overwhelms the count model (+2) and spatial penalty (+2), making K=4 score higher.

**But:** the MARGINAL P(K|data) sums over all allocations at each K. The entropy of allocations at higher K compensates. Exact computation of the co-located marginal (composition enumeration for K=1..6) shows P(K) monotonically increasing through K=6:

**Thermodynamic integration (TI)** computes the exact marginal P(K|data) = P(N|K) × Z(K) where Z(K) = Σ_z P_DM(z|K) × exp(ML(z)), estimated by tempering the spatial likelihood β∈[0,1] and integrating E_β[ML]:

| K | Co-located P/P(max) | Octamer P/P(max) |
|---|--------------------:|------------------:|
| 1 | 0.001 | 0.000 |
| 4 | 0.352 | 0.000 |
| 6 | 0.916 | 0.023 |
| **7** | **1.000** | 0.106 |
| 8 | 0.896 | 0.298 |
| 9 | 0.565 | 0.632 |
| **10** | 0.459 | **1.000** |

- Co-located: TI MAP K=**7** (Q-PAINT=8), sampler gives K≈**3**
- Octamer: TI MAP K=**10** (Q-PAINT=8, spatial helps!), sampler gives K≈**6**

**This definitively proves MIXING FAILURE.** The marginal peaks at K=7-10, but the sampler is trapped at K=3-6. The target is correct — the sampler cannot reach it.

**Why specific-allocation scoring was misleading:** K=4 gibbs-opt scores +26 above K=8 oracle for a SPECIFIC allocation. But P(K) requires summing over ALL allocations. The number of good allocations at K=8 vastly exceeds K=4, and the DM-weighted entropy compensates for the per-allocation penalty.

### Key Evidence (Round 11 diagnostics, `dev/k_target_scoring.jl`)

**Per-allocation scoring (informative but NOT the marginal):**

| K | Label | Count | DM | Spatial | Total | Δ vs K=8 oracle |
|---|-------|-------|-----|---------|-------|-----------------|
| 8 | oracle | -3.40 | -87.79 | +257.08 | +165.89 | 0.00 |
| 8 | gibbs-opt | -3.40 | -81.47 | +256.83 | +171.97 | +6.07 |
| 7 | gibbs-opt | -3.54 | -78.92 | +264.46 | +181.99 | +16.10 |
| 4 | gibbs-opt | -5.52 | -57.17 | +254.76 | +192.07 | +26.18 |

Each K-change in the allocation faces a large energy barrier (DM penalty ≈ -6 to -30 per step). Even though the MARGINAL favors high K, the chain's K-transition moves see the per-allocation landscape, which has valleys at K≈3-6. The chain gets trapped.

**Saddle-point is fine:** Grid vs exact locmix differ by <0.02 per cluster. Not a factor.

**Co-located test (d=0):** Both K=1 and K=8 starts converge to K≈3 (200K iters). This is NOT because the target prefers K=3 (the marginal increases through K=6+). It is because the chain mixes too slowly to reach equilibrium at N=40.

**Octamer (NN=1.9σ):** Both starts converge to K≈6. Same mixing explanation — chain is trapped below the target's preferred K.

### Why Mixing Fails at Large N

At N=6 (brute-force), the DM energy barriers are ≈2-5 per K step, and birth/death with BD burst overcome them. At N=40:
- Each K step faces ≈6-30 in DM penalty for a specific allocation
- Birth acceptance ≈5-20% (creates singletons → vulnerable to immediate death)
- Split acceptance ≈3-7% (DM penalty compounds with allocation cost)
- The chain oscillates locally but cannot make sustained net progress toward high K

The mixing time scales exponentially with N/K — the energy barrier per K-step grows with cluster sizes, making higher K increasingly hard to reach.

### Brute-Force Results (Round 12, Poisson K prior + predictive-only proposals)

| Test | R12 KL | R12 Max | Verdict | Baseline KL | Baseline Max |
|------|--------|---------|---------|-------------|--------------|
| Well-separated (d/σ=10) | 0.000 | **1.08x** | **PASS** | 0.000 | 1.05x |
| Close dimer (d/σ=3) | **0.009** | **1.27x** | **PASS** | 0.010 | 1.30x |
| Single emitter | 0.002 | **1.48x** | **PASS** | 0.001 | 1.37x |
| Large dimer (d/σ=8) | 0.000 | **1.08x** | **PASS** | 0.000 | 1.00x |
| SM-only (well-sep) | 0.002 | **1.47x** | **PASS** | 0.002 | 1.45x |

**ESS improvement (Round 9 vs Round 8):**

| Test | R9 ESS | R8 ESS | Improvement |
|------|--------|--------|-------------|
| Well-sep | 25062 | 21599 | +16% |
| Close dimer | 84997 | 43803 | +94% |
| Single emitter | 23002 | 17490 | +32% |
| Large dimer | 65639 | 23528 | +179% |

**Single emitter P(K) detail (N=6, μ=6) — exact vs MCMC:**

| K | P_exact | P_MCMC | Ratio | R8 Ratio | Direction |
|---|---------|--------|-------|----------|-----------|
| 1 | 0.791 | 0.841 | 0.94 | 0.92 | Over-visited (improved) |
| 2 | 0.194 | 0.148 | 1.31 | 1.46 | Under-visited (improved) |
| 3 | 0.014 | 0.010 | 1.40 | 1.97 | Under-visited (improved) |
| 4 | 0.000 | 0.000 | — | — | Negligible mass |

### Practical Benchmarks

**smlmsim_highdensity** (331 hexamers, 25nm diameter, d/σ≈3.4):

| Metric | R12 (pred-only) | Poisson baseline | R9 (locmix) |
|--------|-----------------|------------------|-------------|
| True emitters | 1986 | 1986 | 1986 |
| Precision | 1.000 | 1.000 | 0.999 |
| Recall | 0.478 | 0.475 | 0.592 |
| RMSE | 4.5 nm | 4.4 nm | 4.8 nm |
| Learned μ | 19.93 | 19.93 | 17.79 |
| Learned shape | 24.31 | 20.97 | 21.92 |

Note: Poisson K prior branch has worse recall than locmix (0.475 vs 0.592).

BD burst doesn't help practical recall because smlmsim uses large partitions (median K=4, up to K=13) where the K-mixing bottleneck is different — dominated by split/merge dynamics, not birth/death acceptance.

**genmab** (GenMAb HexaBody, ROI ~2×2 μm): 19501 locs → 2104 emitters, μ=8.44, shape=2.74 (R8).

### Architecture Summary

```
Move mix: Gibbs allocation (50%) + Split/Merge (25%) + Birth/Death (25% × 5 substeps)
Gibbs:    q(z_i = k) ∝ predictive  [pred-only proposal, MH-correct with (n+γ) ratio]
Split/Merge:
  K proposal: Random |ΔK|=1 (coin flip split/merge, boundary-aware)
  Split:    Random seeds + sequential launch + restricted Gibbs scans (Jain-Neal)
            Final scan density enters MH ratio; intermediate scans free.
  Merge:    Uniform pair + random seeds + reverse density via restricted Gibbs
            Launch → intermediate scans → transition density to current allocation
  MH:       log α = Δ_spatial + Δ_partition + Δ_proposal + Δ_count + Δ_move_type
Birth/Death:
  Birth:    Pick random non-sole-occupant loc, detach as singleton (K+1)
  Death:    Pick random singleton, absorb via DM-weighted predictive (K-1)
  MH:       log α = Δ_spatial + Δ_partition + Δ_proposal + Δ_count
  q_birth = p_birth × 1/N_eligible
  q_death = p_death × 1/n_singletons × w(dest)/Σw  [w = (n_k+γ) × pred]
Spatial prior: Flat uniform + Poisson(ρA) K prior (area cancels)
Count model: NegBin(N; K*shape, p)
DM prior: γ = shape (currently 2.0 default)
Proposals: All predictive-only (no DM in proposals), MH-corrected
Hierarchical: Global MH updates for μ, shape across partitions
n_restricted_scans: 5 (default) — 4 intermediate + 1 final sweep
```

### What Works

- Gibbs allocation sweep with DM weighting: targets correct conditional
- RJMCMC split/merge: all terms computable, proper MH ratio
- **Jain-Neal restricted Gibbs:** intermediate sweeps improve split quality, final sweep density is proposal. Reverse density for merges uses matching framework.
- Random seed selection: seed density cancels via RJMCMC bijection, no coverage gaps
- **Birth/death moves (Round 8) + BD burst (Round 9):** cheap K±1 transitions with ~-1.2 DM penalty (vs -2.5 to -5 for split). Birth acceptance 5-20%, death 100%. BD burst (n_bd_substeps=5) amplifies K-throughput. **4/4 brute-force PASS.**
- Locmix prior: area-invariant, O(1) via grid
- Hierarchical learning: μ and shape converge to reasonable values
- MAP-N estimation: Dahl+overlap, functional
- Partitioned execution: correct boundary dedup, synchronized globals
- All 170 tests passing

### Residual Under-Visiting K > K_mode (Brute-Force: RESOLVED)

All 4 brute-force tests now PASS. The residual under-visiting (1.2-1.4× for worst-case bins) is within the PASS threshold (max 1.65×). BD burst provides sufficient K-throughput to overcome the DM energy barrier.

**Practical impact:** ~41% recall loss on dense 6-mers (d/σ≈3.4), unchanged from R8. Large partitions are dominated by split/merge dynamics where BD burst has limited impact — the bottleneck is split/merge acceptance, not birth/death frequency.

**Acceptable limitation:** Under-splitting of truly co-located emitters (d/σ ≈ 0) — identifiability limit, not a sampler deficiency.

### Validation Infrastructure

| Script | What it tests | Status |
|--------|--------------|--------|
| `brute_force_enumeration.jl` | Exact posterior comparison | WORKING, **4/4 PASS** |
| `prior_sensitivity.jl` | Exact P(K) under different priors | Working (Round 6) |
| `mh_component_analysis.jl` | MH ratio component distributions | Working (Round 5) |
| `move_barrier_analysis.jl` | Per-move Δ decomposition + best-split search at N=40 | WORKING, TARGET-DOMINATED (best K=2: Δ=-7.6) |
| `detailed_balance_check.jl` | DB for specific state pairs | STALE |
| `smlmsim_highdensity.jl` | Synthetic with ground truth | Re-run Round 9: Recall=0.592, RMSE=4.8nm |
| `genmab_bagol.jl` | Real antibody data | Re-run Round 8: 2104 emitters, μ=8.44 |
| `test/runtests.jl` | Unit + integration tests | ALL PASSING (173/173) |

## Active Research Threads

### Thread 1: K-Mixing at Large N — OPEN (Critical)

**Status:** Round 12 proved that DM proposal dynamics are NOT the bottleneck. Predictive-only proposals (remove all DM from proposals, MH-correct) give identical mixing at N=40: birth acceptance 0.12% in both cases, K stuck at 1.

**Key insight (Round 12):** DM-weighted vs predictive-only proposals give identical N=40 behavior. Best-split search over ALL two-way splits of N=40 shows the **best possible** K=2 state has Δ_target = -7.6 (Δ_partition=-11.3 dominates, even when Δ_spatial is positive). **The collapsed target itself prefers K=1 at N=40 co-located.** No proposal of any flavor can fix this — it's the target.

The count+K_prior landscape peaks at K=3, but the DM partition penalty for splitting any cluster grows faster than the count model's help. This is a fundamental tension between the DM prior (which penalizes unbalanced allocations at high K) and the count model (which wants K ≈ N/μ).

**Evidence (Round 12):**
- Brute-force 4/4 PASS (both DM-weighted and predictive-only)
- N=40 co-located: K=1 stuck, birth 0.12% — identical baseline vs predictive-only
- smlmsim: recall 0.478 (baseline) vs 0.475 (pred-only) — within noise
- Close dimer improved slightly (1.30x → 1.27x) — small-N benefit only
- SM-only PASS: 1.45x (baseline) vs 1.47x (pred-only) — equivalent

**Next directions:**
1. **RESOLVED: Target is wrong.** Best-split search shows ALL K=2 states have Δ_target < -7.6. The collapsed target under Poisson+flat prefers K=1 at N=40 co-located. DM partition penalty dominates.
2. **TI on Poisson+flat target** at N=40: confirm marginal P(K) peaks near K=1 (expected given best-split result). Compare to locmix TI which peaked at K=7-10.
3. **CRITICAL: Restore locmix or fix the target.** Poisson+flat gives wrong target at large N. Either (a) go back to locmix (recall 0.592) or (b) find a target where the marginal favors K≈N/μ.
4. The DM prior γ=shape is mathematically correct but creates O(N) partition barriers that the count model cannot overcome. May need to revisit the DM concentration or the count model factorization.

### Thread 2: Hierarchical Learner Feedback

**Status:** smlmsim_highdensity still shows μ drifts from 8.7 → 17.79 and shape from 1.5 → 21.92 when the sampler under-splits. Unchanged from R8. BD burst doesn't help large partitions.

### Thread 3: Practical Validation

**Status:** Re-run in Round 9. smlmsim: Recall=0.592, Precision=0.999, RMSE=4.8nm (~same as R8). genmab: pending re-run.

## Round History

| Round | Date | Focus | Key Finding |
|-------|------|-------|-------------|
| 0 | 2026-03-29 | Framework setup | Established STATUS.md + KNOWLEDGE_BASE.md |
| 1 | 2026-03-29 | Baseline diagnostics | Systematic over-splitting bias confirmed. Root cause attributed to missing Δ_proposal in MH. Brute-force: 4/4 FAIL. |
| 2 | 2026-03-29 | Fix root cause | TRUE root cause: missing DM partition prior + intractable proposal density. Fixed with DM-weighted Gibbs + proper RJMCMC sequential allocation. 7-10x improvement in marginal KL. |
| 3 | 2026-03-29 | Random seeds | Fixed seed coverage gap (35x under-visiting → 1.4x). SM-only per-partition KL improved 14x. Full MCMC: 3/4 tests improved, close dimer KL halved. K=2 test accuracy 53.5% → 80%. |
| 4 | 2026-03-29 | Restricted Gibbs | Jain-Neal restricted Gibbs scans (n=5). **First brute-force PASS** (well-separated). K=1 accuracy converged to theory (92→80%=79.5%). K=4: 76→84%. 3/4 MCMC tests improved. |
| 5 | 2026-03-30 | MH component analysis | **Root cause of under-splitting identified:** DM partition prior penalty (-2.5 to -5) exceeds proposal density compensation (+0.2 to +2.7). Informed seed selection tried and failed (KB #14). Diagnostic script `mh_component_analysis.jl` created. |
| 6 | 2026-03-30 | Model vs mixing diagnosis | **DM prior is correct (not the problem).** Multi-prior enumeration: DM γ=2 gives best exact P(K_true), uniform over-splits. Problem is slow K-mixing (1.5-2.5× under-visit per K step). Practical: 40% recall loss on dense 6-mers. |
| 7 | 2026-03-30 | |ΔK|=1 proposals | Replaced count-model K sampling with random ±1 split/merge. Split acceptance doubled (2.7%→6.7%), close dimer improved (2.18x→2.01x). Fundamental DM barrier persists. |
| 8 | 2026-03-30 | Birth/death moves | Added B/D as third move type (50/25/25 mix). **3/4 brute-force PASS** (was 1/4). Close dimer FAIL→PASS, KL improved 2-15× across all tests. Birth acc. 5-20%, death 100%. Practical benchmarks unchanged (~59% recall on dense 6-mers). |
| 9 | 2026-03-30 | BD burst | BD burst (n_bd_substeps=5): 5 sequential BD per selection. **4/4 brute-force PASS** (was 3/4). Single emitter FAIL→PASS (1.97×→1.40×). ESS +16-179%. Practical recall unchanged (59%). |
| 9+ | 2026-03-30 | DM/Polya analysis | Optimality sweep + oracle init revealed DM creates systematic ↓K pressure. BaGoL worse than Q-PAINT for octamers at NN≈2σ. Root cause: Gibbs destabilizes balanced partitions, compounds with death/merge bias. Fix: time-scale separation. |
| 10 | 2026-03-30 | MH-Gibbs diagnostic | Tested MH-corrected pred-only Gibbs. Both Gibbs variants give K≈4.5 from oracle K=8. Initially concluded "target is wrong" — **Codex corrected: diagnostic only changed within-K sweep, not K-changing moves.** K-kernel bias (SM-only FAIL at 1.76x) is the real suspect. γ=shape stays fixed. |
| 11 | 2026-03-30 | Target scoring + TI marginal | **TI proves mixing failure.** Per-allocation K=4 MAP > K=8 oracle (misleading). TI marginal: co-located peak K=7, octamer peak K=10. Sampler: K≈3 and K≈6 respectively. Joint mode ≠ marginal mode — sampler tracks joint, can't reach marginal peak. Saddle-point fine (<0.02/cluster). |
| 12 | 2026-03-31 | Predictive-only proposals | Removed DM `(n+γ)` from all proposal kernels (Gibbs, split, restricted Gibbs, death). MH-corrected against unchanged target. **4/4 brute-force PASS** (close dimer improved 1.30x→1.27x). Practical recall unchanged (0.478 vs 0.475 baseline). N=40 co-located stuck at K=1 — same as baseline (Poisson K prior issue). Conclusion: DM proposal dynamics are not the bottleneck at practical N. |

## Future Priorities

1. **CRITICAL:** Poisson K prior + flat spatial creates deep K=1 well at large N. Birth acceptance <0.2% at N=40 co-located, identical before and after predictive-only changes. Need to address the target/prior landscape, not just proposal dynamics.
2. **HIGH:** Design K-proposals that see the marginal landscape (not per-allocation). Possible: count-model K proposal + full reallocation via SMC or annealing.
3. **HIGH:** Parallel tempering / annealed importance sampling for the K dimension.
4. **MEDIUM:** Block birth (approach #2 from Round 12 analysis) — create nontrivial new clusters via micro-split instead of singleton detach.
5. **MEDIUM:** Validate octamer marginal P(K) (composition enumeration too expensive at K=8 — use SMC or thermodynamic integration).
6. **COMPLETED (Round 12):** Predictive-only proposals everywhere. Correct but neutral — DM proposal dynamics are not the large-N bottleneck. Close dimer improved slightly.
7. **COMPLETED (Round 11):** Target is correct at d=0 (marginal increases through K=6+). Saddle-point fine. Per-allocation scoring is misleading — must use marginal.
8. **COMPLETED (Round 10):** MH-corrected Gibbs doesn't help (same K-changing kernel).
9. **COMPLETED (Round 9):** Brute-force 4/4 PASS — small-N sampler correct.
10. **DOCUMENTED:** DM/Polya derivation — see docs/dm-polya-proof.md. γ=shape is correct.

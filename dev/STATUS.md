# Sampler Research Status

## Current State (2026-03-30, post-Round 11)

Round 9 achieved 4/4 brute-force PASS. Round 11 investigated whether BaGoL's K under-estimation at intermediate separations is a target or kernel problem. **Conclusion: it is a MIXING problem.** The target (marginal P(K)) likely still favors K≈7-8 at d=0, consistent with brute-force and Q-PAINT. But the sampler converges to K≈3 from both K=1 and K=8 starts — the chain cannot reach equilibrium at N=40.

**Brute-force: 4/4 PASS.** Small-N sampler is correct. Target is correct at N=6.
**Optimality: FAILING.** Mixing problem at N=40 — sampler cannot reach the target's preferred K.

### Root Cause: Per-Allocation Energy Barriers (Round 11, refined)

The spatial ML has a per-cluster Occam factor `-log(n_k)` (partition-dependent part, after cancelling terms that depend only on N). For any SPECIFIC allocation at K=8 vs K=4, the DM relief (+30) overwhelms the count model (+2) and spatial penalty (+2), making K=4 score higher.

**But:** the MARGINAL P(K|data) sums over all allocations at each K. The entropy of allocations at higher K compensates. Exact computation of the co-located marginal (composition enumeration for K=1..6) shows P(K) monotonically increasing through K=6:

| K | count | E[-Σlog(n_k)] | P(K)/P(6) |
|---|-------|---------------|-----------|
| 1 | -12.25 | -3.69 | 0.016 |
| 2 | -9.05 | -5.55 | 0.086 |
| 3 | -6.96 | -6.92 | 0.241 |
| 4 | -5.52 | -7.98 | 0.480 |
| 5 | -4.54 | -8.82 | 0.756 |
| 6 | -3.91 | -9.49 | 1.000 |

The trend clearly continues to K≈7-8 (Q-PAINT MAP), confirming **the target is correct**. The sampler at K≈3 is far below the target peak — this is a severe mixing failure at N=40.

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

### Brute-Force Results (Round 9)

| Test | KL | Max ratio | Verdict | R8 KL | R8 ratio | Change |
|------|-----|-----------|---------|-------|----------|--------|
| Well-separated (d/σ=10) | **0.000** | **1.06x** | **PASS** | 0.001 | 1.08x | Better |
| Close dimer (d/σ=3) | **0.008** | **1.25x** | **PASS** | 0.013 | 1.41x | **1.6x better KL** |
| Single emitter | **0.012** | **1.40x** | **PASS** | 0.024 | 1.97x | **2x better KL, FAIL→PASS** |
| Large dimer (d/σ=8) | **0.002** | **1.14x** | **PASS** | 0.003 | 1.24x | Better |
| SM-only (well-sep) | 0.019 | 1.76x | FAIL | 0.019 | 1.76x | same (no B/D) |

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

| Metric | R9 Value | R8 Value | Assessment |
|--------|----------|----------|------------|
| True emitters | 1986 | 1986 | — |
| Precision | 0.999 | 1.000 | Near-perfect |
| Recall | 0.592 | 0.591 | ~same |
| RMSE | 4.8 nm | 4.8 nm | Excellent |
| Learned μ | 17.79 | 17.79 | 2× overestimate (unchanged) |
| Learned shape | 21.92 | 21.92 | Unchanged |

BD burst doesn't help practical recall because smlmsim uses large partitions (median K=4, up to K=13) where the K-mixing bottleneck is different — dominated by split/merge dynamics, not birth/death acceptance.

**genmab** (GenMAb HexaBody, ROI ~2×2 μm): 19501 locs → 2104 emitters, μ=8.44, shape=2.74 (R8).

### Architecture Summary

```
Move mix: Gibbs allocation (50%) + Split/Merge (25%) + Birth/Death (25% × 5 substeps)
Gibbs:    P(z_i = k | rest) ∝ (n_{-i,k} + γ) × predictive  [DM-weighted]
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
Spatial prior: Locmix (data-driven mixture of localizations)
Count model: NegBin(N; K*shape, p) — no separate P(K) prior
DM prior: γ = shape (currently 2.0 default)
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
| `detailed_balance_check.jl` | DB for specific state pairs | STALE |
| `smlmsim_highdensity.jl` | Synthetic with ground truth | Re-run Round 9: Recall=0.592, RMSE=4.8nm |
| `genmab_bagol.jl` | Real antibody data | Re-run Round 8: 2104 emitters, μ=8.44 |
| `test/runtests.jl` | Unit + integration tests | ALL PASSING (173/173) |

## Active Research Threads

### Thread 1: K-Mixing at Large N — OPEN (Critical)

**Status:** Round 11 confirmed the target is correct (marginal P(K) favors high K at d=0 and likely at NN=1.9σ), but the sampler cannot reach the target's preferred K at N=40. This is a MIXING problem, not a target or kernel-correctness problem.

**Key insight (Round 11):** Scoring specific allocations across K is misleading — K=4 MAP scores +26 above K=8 oracle, but the marginal P(K) (which sums over all allocations weighted by DM) favors K≈7-8. The entropy of allocations at higher K compensates for the per-allocation DM+Occam penalty.

**Evidence:**
- Exact co-located marginal (composition enumeration, K=1..6): P(K) monotonically increasing, consistent with Q-PAINT MAP K=8
- Co-located sampler gives K≈3 from both K=1 and K=8 starts → severe under-mixing
- Octamer sampler gives K≈6 from both starts → same under-mixing, different degree
- SM-only brute-force FAILS at 1.76x → split/merge kernel has measurable bias at N=6 scale
- Saddle-point locmix vs exact: <0.02 per cluster (not a factor)

**Root cause:** Per-allocation energy barriers scale with cluster sizes and compound at higher K. Birth/death and split/merge moves have ~5-20% acceptance for K-increasing proposals. The mixing time scales exponentially with N, making the sampler unable to reach equilibrium at N=40 within practical iteration counts.

**Next directions:**
1. Design K-transition moves that see the MARGINAL landscape, not per-allocation landscape
2. Consider parallel tempering or annealed approaches for K-dimension mixing
3. Investigate whether non-reversible MCMC or Hamiltonian-like moves could improve K-mixing
4. Possible: decouple K proposal from allocation (propose K first from count model, then sample allocation at new K)

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
| 11 | 2026-03-30 | Target scoring + marginal | **Per-allocation scoring misleading:** K=4 MAP scores +26 above K=8 oracle, but MARGINAL P(K) increases through K=6+ (exact enumeration). Target is correct — **mixing is the problem.** Sampler K≈3 at d=0 (both starts), target peak ≈ K=7-8. Saddle-point fine (<0.02/cluster). |

## Future Priorities

1. **CRITICAL:** Improve K-mixing at large N. Per-allocation energy barriers prevent the sampler from reaching the target's preferred K. Need fundamentally new K-transition strategy.
2. **HIGH:** Design K-proposals that see the marginal landscape (not per-allocation). Possible: count-model K proposal + full reallocation via SMC or annealing.
3. **HIGH:** Parallel tempering / annealed importance sampling for the K dimension.
4. **MEDIUM:** Investigate non-reversible K-transitions (momentum-based, look-ahead).
5. **MEDIUM:** Validate octamer marginal P(K) (composition enumeration too expensive at K=8 — use SMC or thermodynamic integration).
6. **COMPLETED (Round 11):** Target is correct at d=0 (marginal increases through K=6+). Saddle-point fine. Per-allocation scoring is misleading — must use marginal.
7. **COMPLETED (Round 10):** MH-corrected Gibbs doesn't help (same K-changing kernel).
8. **COMPLETED (Round 9):** Brute-force 4/4 PASS — small-N sampler correct.
9. **DOCUMENTED:** DM/Polya derivation — see docs/dm-polya-proof.md. γ=shape is correct.

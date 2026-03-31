# Sampler Research Status

## Current State (2026-03-30, post-Round 10)

Round 10 tested MH-corrected predictive-only Gibbs as a diagnostic for the DM/Polya under-splitting problem. **Result: the target distribution itself is wrong, not the mixing dynamics.** Both standard DM Gibbs and MH-corrected Gibbs converge from oracle K=8 to K≈4.5 for octamers at NN=1.9σ. Changing the Gibbs kernel doesn't help because both target the same DM posterior, which genuinely under-estimates K.

**Brute-force: 4/4 PASS.** Small-N sampler is correct (both Gibbs variants).
**Optimality: FAILING.** Octamers at d/σ=2 — Q-PAINT MAP K=8, BaGoL K≈4.5 (from oracle K=8, both Gibbs variants). **Confirmed: target problem, not mixing problem.**

### Root Cause: DM Posterior Under-Weights Large K

The DM partition prior (γ=shape) is mathematically derived from the NegBin count model. But the resulting posterior genuinely under-estimates K for many-emitter configurations at intermediate separations:
- For N=40, K_true=8, NN=1.9σ: DM posterior peaks at K≈4-5
- Q-PAINT (count-only) correctly gives K=8
- The DM penalty compounds: going K=1→8 accumulates ~7×(-2.5) ≈ -17.5 in log-space
- Spatial evidence at NN≈2σ adds only +1 to +2 per split — overwhelmed by DM penalty
- **Round 10 confirmed:** replacing Gibbs sampling strategy (MH vs exact) doesn't change the posterior mode — both converge to K≈4.5

The problem is NOT the Gibbs "rich get richer" dynamics (which was the Round 9 hypothesis). The Gibbs correctly samples the DM conditional. The DM conditional itself places too much mass on unbalanced, low-K partitions.

See `docs/dm-polya-proof.md` for derivation and evidence.

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

### Thread 1: DM Target Distribution — OPEN (Critical)

**Status:** Round 10 confirmed that the DM posterior is the problem, not the mixing dynamics. Both standard DM Gibbs and MH-corrected predictive-only Gibbs converge to K≈4.5 from oracle K=8 (octamers at NN=1.9σ). Q-PAINT correctly gives K=8.

**Round 9 hypothesis (DISPROVED):** Time-scale mismatch — Gibbs destabilizes balanced partitions before birth/split can act. This was wrong: replacing the Gibbs with a slower MH variant doesn't change the posterior mode.

**Corrected diagnosis:** The DM posterior P(z|data) ∝ P(data|z) × P_DM(z|K) × P(N|K) genuinely peaks at K≈4-5 for this configuration. The DM penalty compounds across K transitions, overwhelming weak spatial evidence at NN≈2σ.

**Next: Round 11 — Change the Target Distribution**

The fix is to decouple the partition prior concentration from the NegBin shape. Options:
1. **Separate γ_alloc parameter:** Keep DM in Gibbs but with γ_alloc < shape (weaker concentration). Must re-validate brute-force with modified target.
2. **Predictive-only Gibbs + count model in MH only:** Remove DM from Gibbs conditional (uniform partition at fixed K), keep NegBin count model only in K-changing MH moves. Risk: over-splitting (KB #15 showed uniform catastrophically over-splits for N=6).
3. **Scaled γ:** Use γ = shape/K or γ = shape/√K to reduce compounding at large K.

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
| 10 | 2026-03-30 | MH-Gibbs diagnostic | MH-corrected predictive-only Gibbs (propose ∝ pred, accept with DM ratio). **Diagnostic result: target is wrong, not mixing.** Both Gibbs variants converge from oracle K=8 to K≈4.5 for octamers at NN=1.9σ. Brute-force 4/4 PASS. Code reverted (no benefit). |

## Future Priorities

1. **CRITICAL — Round 11:** Change the target distribution — decouple γ_alloc from NegBin shape to reduce compounding K penalty at large K
2. **HIGH:** Re-validate brute-force after target change (new exact posterior)
3. **HIGH:** Re-run optimality sweep to measure octamer improvement
4. **COMPLETED:** MH-corrected Gibbs — targets same distribution, doesn't help (Round 10)
5. **COMPLETED:** Time-scale separation hypothesis — disproved (Round 10, not a mixing problem)
6. **COMPLETED:** Brute-force 4/4 PASS — small-N sampler is correct
7. **DOCUMENTED:** DM/Polya root cause — see docs/dm-polya-proof.md

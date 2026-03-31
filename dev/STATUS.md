# Sampler Research Status

## Current State (2026-03-30, post-Round 10)

Round 9 achieved 4/4 brute-force PASS via BD burst (n_bd_substeps=5). However, subsequent optimality sweep and oracle-init testing revealed a fundamental issue: **the DM/Polya partition prior creates systematic downward K pressure** that makes BaGoL worse than Q-PAINT for octamers at intermediate separations (NN≈2σ).

**Brute-force: 4/4 PASS.** Small-N sampler is correct.
**Optimality: FAILING.** Octamers at d/σ=5 — Q-PAINT 100%, BaGoL 0% (from oracle K=8, chain drops to K≈6).

### Root Cause: DM/Polya Dynamics

The DM partition prior (γ=shape) is mathematically correct (proven derivation from NegBin count model). But it creates compounding K-reducing forces:
- Gibbs: (n_k+γ) rich-get-richer destabilizes balanced partitions at NN≈2σ
- Every birth: Δ_DM ≈ -4 (opposed). Every death: Δ_DM ≈ +4 (favored)
- Every split: Δ_DM ≈ -6 (opposed). Every merge: Δ_DM ≈ +6 (favored)
- log Γ convexity means Polya density favors UNBALANCED sizes

At K=8 with NN=1.9σ: Gibbs creates sizes [9,7,7,6,4,3,2,2] from balanced [5,5,5,5,5,5,5,5]. Small clusters become death targets. Chain drops to K≈6. This is NOT mixing failure — chain converges to K≈6 from both K=1 and oracle K=8 in 500K iterations.

See `docs/dm-polya-proof.md` for full derivation and evidence.

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

### Thread 1: K-Changing Kernel Bias — OPEN (Critical)

**Status:** Round 10 tested MH-corrected predictive-only Gibbs (propose ∝ pred, accept with DM ratio). Both standard and MH Gibbs converge from oracle K=8 to K≈4.5 for octamers at NN=1.9σ. Initially this was misinterpreted as "the target is wrong." Codex review corrected this: **the diagnostic only changed the within-K allocation sweep, not the K-changing moves.** Both variants share identical split/merge and birth/death kernels, which are the actual bottleneck.

**Key insight (Codex Round 10 review):** P(N|K) and the DM prior are NOT two forces fighting — they are a factorization of the same NegBin count model. After summing over allocations at fixed K, the DM collapses back into the count model. If Q-PAINT (count-only) correctly favors K=8, the full posterior with correct spatial likelihood should also favor K≥8. The problem must be in the K-changing kernel or the spatial likelihood approximation.

**Evidence of K-kernel bias:** SM-only diagnostic FAILS at 1.76x max ratio for well-separated dimers. This directly shows the split/merge kernel has bias that persists regardless of which Gibbs variant is used.

**Next: Round 11 — Audit K-Changing Moves**

Priority investigation:
1. Measure K=8↔7↔6↔5 transition flux and acceptance rates individually (not just end-state K)
2. Check locmix saddle-point approximation vs exact integration on octamer states
3. Validate K-moves on small octamer-like toy (e.g., N=8, K=4, d/σ=2 brute-force)
4. Verify co-located limit: full sampler must match Q-PAINT exactly at d=0

γ=shape is mathematically correct and stays fixed. The target distribution is NOT the problem.

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

## Future Priorities

1. **CRITICAL — Round 11:** Audit K-changing kernel — measure transition flux K=8↔7↔...↔4, identify asymmetry source
2. **HIGH:** Validate K-moves on small octamer-like brute-force (e.g., N=8, K=4, d/σ=2)
3. **HIGH:** Check locmix saddle-point approximation vs exact integration on octamer states
4. **HIGH:** Verify co-located limit: full sampler must match Q-PAINT at d=0
5. **COMPLETED (Round 10):** MH-corrected Gibbs — doesn't isolate the problem (same K-changing kernel)
6. **COMPLETED:** Brute-force 4/4 PASS — small-N sampler correct
7. **DOCUMENTED:** DM/Polya derivation — see docs/dm-polya-proof.md. γ=shape is correct.

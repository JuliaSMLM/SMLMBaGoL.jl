# Sampler Research Status

## Current State (2026-03-30, Round 8)

Round 8 added birth/death moves as a third move type supplementing Gibbs allocation and split/merge. Birth detaches a random non-sole-occupant loc as a singleton (K+1); death absorbs a random singleton via DM-weighted predictive (K-1). The DM penalty per birth is ~-1.2 (vs -2.5 to -5 for split), providing much cheaper K-mobility. Subsequent Gibbs sweeps grow newborn singletons into proper clusters.

**Result: 3/4 brute-force PASS (was 1/4).** Close dimer flipped from FAIL to PASS. KL improved 2-15x across all tests. Birth acceptance 5-20%, death acceptance 100%.

### Brute-Force Results (Round 8)

| Test | KL | Max ratio | Verdict | R7 KL | R7 ratio | Change |
|------|-----|-----------|---------|-------|----------|--------|
| Well-separated (d/σ=10) | **0.001** | **1.08x** | **PASS** | 0.015 | 1.55x | **15x better KL** |
| Close dimer (d/σ=3) | **0.013** | **1.41x** | **PASS** | 0.036 | 2.01x | **2.8x better, FAIL→PASS** |
| Single emitter | 0.024 | 1.97x | FAIL | 0.054 | 2.81x | **2.3x better KL** |
| Large dimer (d/σ=8) | **0.003** | **1.24x** | **PASS** | 0.038 | 2.56x | **15x better, FAIL→PASS** |
| SM-only (well-sep) | 0.019 | 1.76x | FAIL | 0.019 | 1.76x | same (no B/D) |

**Acceptance rates (Round 8):**

| Test | Birth | Death | Split | Merge |
|------|-------|-------|-------|-------|
| Well-sep (full MCMC) | 16.1% | 100% | 6.4% | 8.6% |
| Close dimer | 20.4% | 100% | 34.5% | 64.9% |
| Single emitter | 5.3% | 100% | 5.5% | 100% |
| Large dimer (d/σ=8) | 9.9% | 100% | 4.8% | 7.4% |
| SM-only (well-sep) | — | — | 6.7% | 6.7% |

**Close dimer P(K) detail (d/σ=3) — exact vs MCMC:**

| K | P_exact | P_MCMC | Ratio | R7 Ratio | Direction |
|---|---------|--------|-------|----------|-----------|
| 1 | 0.186 | 0.241 | 0.77 | 0.68 | Over-visited (1.29×, was 1.46×) |
| 2 | 0.603 | 0.590 | 1.02 | 1.02 | **Correct** |
| 3 | 0.191 | 0.155 | 1.23 | 1.54 | Under-visited (improved) |
| 4 | 0.020 | 0.014 | 1.41 | 2.01 | Under-visited (improved) |

### Practical Benchmarks

**smlmsim_highdensity** (331 hexamers, 25nm diameter, d/σ≈3.4):

| Metric | R8 Value | R7 Value | Assessment |
|--------|----------|----------|------------|
| True emitters | 1986 | 1986 | — |
| Precision | 1.000 | 0.999 | Near-perfect |
| Recall | 0.591 | 0.594 | ~same |
| RMSE | 4.8 nm | 4.9 nm | Excellent |
| Learned μ | 17.79 | 17.79 | 2× overestimate (unchanged) |
| Learned shape | 21.92 | 21.92 | Unchanged |

**genmab** (GenMAb HexaBody, ROI ~2×2 μm): 19501 locs → 2104 emitters, μ=8.44, shape=2.74 (R7: 2034 emitters, μ=9.43, shape=3.02).

### Architecture Summary

```
Move mix: Gibbs allocation (50%) + Split/Merge (25%) + Birth/Death (25%)
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
- **Birth/death moves (Round 8):** cheap K±1 transitions with ~-1.2 DM penalty (vs -2.5 to -5 for split). Birth acceptance 5-20%, death 100%. 3/4 brute-force PASS.
- Locmix prior: area-invariant, O(1) via grid
- Hierarchical learning: μ and shape converge to reasonable values
- MAP-N estimation: Dahl+overlap, functional
- Partitioned execution: correct boundary dedup, synchronized globals
- All 170 tests passing

### Known Issue: Residual Under-Visiting K > K_mode

The sampler still slightly under-visits K > K_mode (1.2-1.4× for close dimers, up to 2× for single emitter). Dramatically improved from Round 7 (was 1.5-2.8×). Birth/death reduced the DM energy barrier but didn't eliminate it entirely.

**Practical impact:** ~41% recall loss on dense 6-mers (d/σ≈3.4), essentially unchanged from R7. Large partitions are dominated by split/merge dynamics where birth/death has less impact. The hierarchical learner still adapts μ/shape upward.

**Remaining bottleneck:** Single emitter test (KL=0.024, max ratio 1.97×). When the true K=1, births are rarely accepted (5.3%) since removing a loc from the sole cluster creates a poor singleton. This is the hardest case for birth/death.

**Acceptable limitation:** Under-splitting of truly co-located emitters (d/σ ≈ 0) — identifiability limit, not a sampler deficiency.

### Validation Infrastructure

| Script | What it tests | Status |
|--------|--------------|--------|
| `brute_force_enumeration.jl` | Exact posterior comparison | WORKING, **3/4 PASS** |
| `prior_sensitivity.jl` | Exact P(K) under different priors | Working (Round 6) |
| `mh_component_analysis.jl` | MH ratio component distributions | Working (Round 5) |
| `detailed_balance_check.jl` | DB for specific state pairs | STALE |
| `smlmsim_highdensity.jl` | Synthetic with ground truth | Re-run Round 8: Recall=0.59, RMSE=4.8nm |
| `genmab_bagol.jl` | Real antibody data | Re-run Round 8: 2104 emitters, μ=8.44 |
| `test/runtests.jl` | Unit + integration tests | ALL PASSING (170/170) |

## Active Research Threads

### Thread 1: Improve K-Mixing — OPEN

**Status:** Round 8 added birth/death moves. Brute-force: 3/4 PASS (was 1/4). Close dimer KL improved from 0.036 to 0.013. Birth acceptance 5-20%, death 100%. The remaining FAIL is single emitter (KL=0.024, max ratio 1.97×) where births from a single large cluster are rarely accepted.

**Next steps:**
1. **Targeted birth** — bias birth toward locs with lowest within-cluster fit, rather than uniform selection. Should improve birth acceptance for single-emitter and large-cluster cases.
2. **Multiple-try MH** for split/merge — propose M=5-10 independent split allocations, select best.
3. **Count-informed ±1 proposal** for split/merge direction.

### Thread 2: Hierarchical Learner Feedback

**Status:** smlmsim_highdensity still shows μ drifts from 8.7 → 17.79 and shape from 1.5 → 21.92 when the sampler under-splits. Unchanged from R7/R6. Birth/death improved brute-force (small N) but large partitions in smlmsim still dominated by split/merge.

### Thread 3: Practical Validation

**Status:** Re-run in Round 8. smlmsim: Recall=0.591, Precision=1.000, RMSE=4.8nm (~same as R7). genmab: pending re-run.

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

## Future Priorities

1. **HIGH:** Targeted birth (bias toward locs with low within-cluster fit) — should help single-emitter case (only remaining FAIL)
2. **MEDIUM:** Multiple-try MH for split/merge
3. **MEDIUM:** Count-informed ±1 proposal for split/merge
4. **LOW:** Tune move mix (40/30/30) by ESS/sec for K
5. **LOW:** Parallel tempering or non-reversible lifting

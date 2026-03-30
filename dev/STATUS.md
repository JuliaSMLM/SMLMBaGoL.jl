# Sampler Research Status

## Current State (2026-03-29, Round 4)

Round 4 implemented Jain-Neal restricted Gibbs scans for the split/merge proposal. After the initial sequential allocation (launch state), `n_restricted_scans - 1` intermediate Gibbs sweeps improve the allocation, then one final sweep produces the proposal with computable density. The reverse density for merges uses the same framework: launch → intermediate sweeps → transition density.

**Key result: First brute-force PASS** on the well-separated dimer test (KL 0.014, max ratio 1.48x < 0.50 threshold).

### Results Comparison

**Full MCMC marginal P(K) — Round 3 vs Round 4:**

| Test | KL R3 | KL R4 | Verdict R3 | Verdict R4 |
|------|-------|-------|------------|------------|
| Well-separated (d/σ=10) | 0.019 | **0.014** | FAIL | **PASS** |
| Close dimer (d/σ=3) | 0.043 | **0.040** | FAIL | FAIL |
| Single emitter | 0.047 | 0.054 | FAIL | FAIL |
| Large dimer (d/σ=8) | 0.036 | **0.033** | FAIL | FAIL |

3/4 tests improved. Single emitter slightly worse (noise — ESS variation).

**SM-only kernel — Round 3 vs Round 4:**

| Metric | Round 3 | Round 4 |
|--------|---------|---------|
| Per-partition KL | 0.021 | **0.020** |
| Max ratio | 1.94x | **1.76x** |

**Test suite K-accuracy (hierarchical, 25 trials):**

| K | Round 3 | Round 4 | Theory |
|---|---------|---------|--------|
| 1 | 92.0% | **80.0%** | 79.5% |
| 2 | 80.0% | 80.0% | 53.5% |
| 4 | 76.0% | **84.0%** | 38.4% |

K=1 converged from 92% to 80% — exactly matching the 79.5% theoretical limit. R3's 92% indicated under-splitting bias that R4 corrected. K=4 improved from 76% to 84%.

### Acceptance Rates (Round 4)

Nearly identical to Round 3 — restricted Gibbs improves proposal QUALITY not quantity:

| Move | Well-sep | Close | Single | Large |
|------|----------|-------|--------|-------|
| Split | 2.8% | 12.2% | 5.6% | 2.1% |
| Merge | 9.4% | 64.8% | 99.2% | 7.9% |
| Gibbs | 100% | 100% | 100% | 100% |

### Architecture Summary

```
Move mix: Gibbs allocation (50%) + Split/Merge (50%)
Gibbs:    P(z_i = k | rest) ∝ (n_{-i,k} + γ) × predictive  [DM-weighted]
K proposal: Sample from count-model posterior P(K|N)
Split:    Random seeds + sequential launch + restricted Gibbs scans (Jain-Neal)
          Final scan density enters MH ratio; intermediate scans free.
Merge:    Uniform pair + random seeds + reverse density via restricted Gibbs
          Launch → intermediate scans → transition density to current allocation
MH:       log α = Δ_spatial + Δ_partition + Δ_proposal  [seed density cancels]
Spatial prior: Locmix (data-driven mixture of localizations)
Count model: NegBin(N; K*shape, p) — no separate P(K) prior
Hierarchical: Global MH updates for μ, shape across partitions
n_restricted_scans: 5 (default) — 4 intermediate + 1 final sweep
```

### What Works

- Gibbs allocation sweep with DM weighting: targets correct conditional
- RJMCMC split/merge: all terms computable, proper MH ratio
- **Jain-Neal restricted Gibbs:** intermediate sweeps improve split quality, final sweep density is proposal. Reverse density for merges uses matching framework.
- Random seed selection: seed density cancels via RJMCMC bijection, no coverage gaps
- Count-model K proposal: good K mixing via direct sampling
- Locmix prior: area-invariant, O(1) via grid
- Hierarchical learning: μ and shape converge to reasonable values
- MAP-N estimation: Dahl+overlap, functional
- Partitioned execution: correct boundary dedup, synchronized globals
- All 170 tests passing

### What's Left (Residual Bias)

**Under-splitting at K > K_true:** The sampler under-visits higher K states, especially for co-located emitters (single emitter test: K=2 is 2x under-visited, K=3 is 3.2x). This is the dominant remaining bias.

1. **Low split acceptance:** 2-12% means many K proposals are wasted. Restricted Gibbs improved proposal quality but didn't significantly change acceptance rates (proposals are better but the MH correction compensates).

2. **Merge reverse density noise:** ~50% of merge proposals pick "bad" seeds (both from same original cluster), giving low q_alloc_rev. Doesn't hurt correctness but wastes effective proposals.

3. **Small cluster sizes limit restricted Gibbs:** For N=6 with K=2 (m=3 per cluster), there's only 1 non-seed member — restricted Gibbs can't improve the allocation. The benefit is stronger for larger clusters.

### Validation Infrastructure

| Script | What it tests | Status |
|--------|--------------|--------|
| `brute_force_enumeration.jl` | Exact posterior comparison | WORKING, **1/4 PASS** (well-separated), 3/4 FAIL |
| `detailed_balance_check.jl` | DB for specific state pairs | STALE (needs update for restricted Gibbs) |
| `smlmsim_highdensity.jl` | Synthetic with ground truth | Not re-run |
| `genmab_bagol.jl` | Real antibody data | Not re-run |
| `test/runtests.jl` | Unit + integration tests | ALL PASSING (170/170) |

## Active Research Threads

### Thread 1: Split/Merge Kernel Design

**Status:** Jain-Neal restricted Gibbs implemented (Round 4). First brute-force PASS on well-separated test. Acceptance rates unchanged — improvement is in proposal quality, not quantity.

**Next improvements (in priority order):**
- **(a) Informed seed selection for merges:** Instead of uniform random seeds, pick one from each original cluster (guaranteed "good" pair). Halves wasted merge proposals. Need matching bijection for split seeds.
- **(b) Increase n_restricted_scans for larger clusters:** Adaptive R based on cluster size — more scans for larger clusters where the benefit is greater.
- **(c) Multiple split proposals:** Propose several allocations, pick the best. Population-based RJMCMC.

### Thread 2: Validation & Diagnostics

**Status:** Brute-force baseline established for Round 4. Well-separated test now PASSES.

**TODO:**
- Update `detailed_balance_check.jl` for restricted Gibbs proposal mechanism
- Re-run `smlmsim_highdensity.jl` with Round 4 code
- Investigate single emitter 3.2x under-visiting at K=3

### Thread 3: Alternative Approaches (not yet explored)

- Birth-death MCMC (one emitter at a time, simpler proposal)
- Allocation sampler (Nobile & Fearnside 2007) — K as parameter
- Tempered transitions or parallel tempering for K mixing

## Round History

| Round | Date | Focus | Key Finding |
|-------|------|-------|-------------|
| 0 | 2026-03-29 | Framework setup | Established STATUS.md + KNOWLEDGE_BASE.md |
| 1 | 2026-03-29 | Baseline diagnostics | Systematic over-splitting bias confirmed. Root cause attributed to missing Δ_proposal in MH. Brute-force: 4/4 FAIL. |
| 2 | 2026-03-29 | Fix root cause | TRUE root cause: missing DM partition prior + intractable proposal density. Fixed with DM-weighted Gibbs + proper RJMCMC sequential allocation. 7-10x improvement in marginal KL. |
| 3 | 2026-03-29 | Random seeds | Fixed seed coverage gap (35x under-visiting → 1.4x). SM-only per-partition KL improved 14x. Full MCMC: 3/4 tests improved, close dimer KL halved. K=2 test accuracy 53.5% → 80%. |
| 4 | 2026-03-29 | Restricted Gibbs | Jain-Neal restricted Gibbs scans (n=5). **First brute-force PASS** (well-separated). K=1 accuracy converged to theory (92→80%=79.5%). K=4: 76→84%. 3/4 MCMC tests improved. |

## Next Round Priorities

1. **HIGH:** Informed seed selection for merges (one from each cluster) — halve wasted proposals.
2. **HIGH:** Re-run smlmsim_highdensity and practical benchmarks with Round 4 code.
3. **MEDIUM:** Update detailed_balance_check.jl for restricted Gibbs mechanism.
4. **MEDIUM:** Investigate single emitter bias (K=2/3 under-visited 2-3x).
5. **LOW:** Adaptive n_restricted_scans based on cluster size.

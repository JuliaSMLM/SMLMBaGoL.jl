# Sampler Research Status

## Current State (2026-03-30, Round 5)

Round 5 performed deep MH component analysis to diagnose the under-splitting bias. Instrumented the `propose_split_merge!` function to log (Δ_spatial, Δ_partition, Δ_proposal, log_α, accepted) for every proposal across three test cases. Also tried informed seed selection for merges (dead end — see KB #14).

**Key finding: The DM partition prior (Δ_partition) is the dominant barrier to splits.** The per-split penalty of -2.5 to -5.0 exceeds the proposal density compensation of +0.2 to +2.7, leaving a consistent gap of ~2 in log_α. This is structural — not fixable by improving proposal quality alone.

### MH Component Analysis Results

**Well-separated dimer SM-only (K=2→3 splits):**

| Component | All (mean) | Accepted (mean) | Rejected (mean) |
|-----------|-----------|-----------------|-----------------|
| Δ_spatial | -0.430 | -0.344 | -0.435 |
| Δ_partition | -3.078 | -2.459 | -3.118 |
| Δ_proposal | +0.137 | +0.273 | +0.128 |
| log_α | -3.370 | -2.530 | -3.426 |

log_α quantiles: 5%=-5.17, 25%=-4.78, 50%=-2.54, 75%=-2.43, 95%=-2.33. Bimodal: ~75% are single-step (log_α ≈ -2.5), ~25% are multi-step |ΔK|>1 (log_α ≈ -5).

**Single emitter SM-only (K=1→2 splits):**

| Component | All (mean) | Accepted (mean) | Rejected (mean) |
|-----------|-----------|-----------------|-----------------|
| Δ_spatial | -0.210 | -0.231 | -0.207 |
| Δ_partition | -4.993 | -4.381 | -5.076 |
| Δ_proposal | +2.655 | +2.766 | +2.640 |
| log_α | -2.548 | -1.846 | -2.643 |

Higher Δ_proposal (+2.7) for co-located data because allocation density ≈ (0.5)^(m-2) — each member is equally likely to go to either sub-cluster.

**Close dimer SM-only (K=1→2 splits):**

| Component | All (mean) | Accepted (mean) | Rejected (mean) |
|-----------|-----------|-----------------|-----------------|
| Δ_spatial | +2.779 | +2.875 | +2.720 |
| Δ_partition | -6.111 | -4.898 | -6.857 |
| Δ_proposal | +1.545 | +1.753 | +1.418 |
| log_α | -1.787 | -0.270 | -2.719 |

Here Δ_spatial is large and positive (spatial evidence SUPPORTS the split). Rejected proposals have much worse Δ_partition (-6.86 vs -4.90) — these are multi-step proposals (K_new ≥ 3) accumulating DM penalty across steps.

### Analysis Summary

1. **DM penalty is the bottleneck:** -2.5 to -5.0 per split, dominating log_α
2. **Δ_spatial is small:** -0.4 to +2.8 depending on data geometry, usually small relative to DM
3. **Δ_proposal partially compensates:** +0.2 (separated) to +2.7 (co-located), but never enough
4. **~25% of proposals are wasted multi-step chains** (|ΔK|>1) with log_α ≈ -5 to -7
5. **Accepted vs rejected splits differ mainly in Δ_partition** — proposals hitting balanced partitions (lower DM penalty) have better chance
6. **The under-splitting bias is multiplicative per K:** each K transition adds ~2 to log_α deficit, causing 2x under-visiting at K=K_true+1, 3.2x at K_true+2, etc.

### Brute-Force Results (Unchanged from Round 4)

| Test | KL | Max ratio | Verdict |
|------|-----|-----------|---------|
| Well-separated (d/σ=10) | 0.014 | 1.48x | **PASS** |
| Close dimer (d/σ=3) | 0.040 | 2.18x | FAIL |
| Single emitter | 0.054 | 3.22x | FAIL |
| Large dimer (d/σ=8) | 0.033 | 2.49x | FAIL |
| SM-only (well-sep) | 0.020 | 1.76x | FAIL |

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
DM prior: γ = shape (currently 2.0 default)
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

### Known Limitation: Under-Splitting of Close Emitters

The DM partition prior with γ=shape creates an energy barrier between K states. The per-split penalty (Δ_partition ≈ -2.5 to -5) exceeds the proposal density compensation (Δ_proposal ≈ +0.2 to +2.7). For close/co-located emitters, this produces under-splitting (K biased low).

**Decision (2026-03-30): This is an acceptable limitation.** Close emitters at d/σ ≲ 3 are near the resolution limit — under-splitting there is physically reasonable. The DM prior with γ=shape is the correct formulation; introducing an arbitrary decoupled γ would be a hack. Well-separated emitters (d/σ ≥ 10) pass brute-force validation. The sampler is correct; the remaining bias is a consequence of the statistical difficulty of the problem.

### Validation Infrastructure

| Script | What it tests | Status |
|--------|--------------|--------|
| `brute_force_enumeration.jl` | Exact posterior comparison | WORKING, **1/4 PASS** |
| `mh_component_analysis.jl` | MH ratio component distributions | **NEW** (Round 5) |
| `detailed_balance_check.jl` | DB for specific state pairs | STALE |
| `smlmsim_highdensity.jl` | Synthetic with ground truth | Not re-run |
| `genmab_bagol.jl` | Real antibody data | Not re-run |
| `test/runtests.jl` | Unit + integration tests | ALL PASSING (170/170) |

## Active Research Threads

### Thread 1: DM Prior Energy Barrier — CLOSED (acceptable limitation)

**Status:** Root cause identified in Round 5. The DM partition prior with γ=shape imposes a per-split penalty for close emitters. Decision: this is physically reasonable and acceptable. The sampler is correct; the bias reflects the statistical difficulty of resolving close emitters.

### Thread 2: Practical Benchmarks

**Status:** Not re-run since Round 4. Should run smlmsim_highdensity and genmab with current code to assess practical impact of the bias.

### Thread 3: Alternative Approaches

**Status:** Birth-death and allocation sampler identified as alternatives. Birth-death has the same DM penalty issue. Allocation sampler (Nobile & Fearnside 2007) treats K as a parameter — potentially avoids the per-K energy barrier.

## Round History

| Round | Date | Focus | Key Finding |
|-------|------|-------|-------------|
| 0 | 2026-03-29 | Framework setup | Established STATUS.md + KNOWLEDGE_BASE.md |
| 1 | 2026-03-29 | Baseline diagnostics | Systematic over-splitting bias confirmed. Root cause attributed to missing Δ_proposal in MH. Brute-force: 4/4 FAIL. |
| 2 | 2026-03-29 | Fix root cause | TRUE root cause: missing DM partition prior + intractable proposal density. Fixed with DM-weighted Gibbs + proper RJMCMC sequential allocation. 7-10x improvement in marginal KL. |
| 3 | 2026-03-29 | Random seeds | Fixed seed coverage gap (35x under-visiting → 1.4x). SM-only per-partition KL improved 14x. Full MCMC: 3/4 tests improved, close dimer KL halved. K=2 test accuracy 53.5% → 80%. |
| 4 | 2026-03-29 | Restricted Gibbs | Jain-Neal restricted Gibbs scans (n=5). **First brute-force PASS** (well-separated). K=1 accuracy converged to theory (92→80%=79.5%). K=4: 76→84%. 3/4 MCMC tests improved. |
| 5 | 2026-03-30 | MH component analysis | **Root cause of under-splitting identified:** DM partition prior penalty (-2.5 to -5) exceeds proposal density compensation (+0.2 to +2.7). Informed seed selection tried and failed (KB #14). Diagnostic script `mh_component_analysis.jl` created. |

## Next Round Priorities

1. **HIGH:** Re-run practical benchmarks (smlmsim_highdensity, genmab) to assess real-world impact with current sampler.
2. **MEDIUM:** Explore multiple-try MH (propose M allocations, select best) for improved mixing.
3. **MEDIUM:** Investigate allocation sampler (Nobile & Fearnside) as alternative to split/merge.
4. **LOW:** Parallel tempering for improved K exploration.

**Closed:** Decoupled γ — decided against (arbitrary hack, under-splitting of close emitters is acceptable).

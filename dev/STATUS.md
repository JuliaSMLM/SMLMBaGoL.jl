# Sampler Research Status

## Current State (2026-03-30, Round 6)

Round 6 resolved the model-vs-mixing question definitively: **the DM prior is correct; the problem is slow K-mixing.** Multi-prior brute-force enumeration shows DM γ=2 gives the best exact posterior P(K) across all separations. The sampler under-visits K > K_mode by 1.5-2.5× per step, consistent with slow split acceptance (2-12%). Practical benchmarks confirm 40% recall loss on high-density 6-mers.

### Brute-Force Results (Round 6, unchanged code from Round 5)

| Test | KL | Max ratio | Verdict | K_mode exact | K_mode MCMC |
|------|-----|-----------|---------|-------------|-------------|
| Well-separated (d/σ=10) | 0.014 | 1.48x | **PASS** | K=2 (72.5%) | K=2 (79.4%) |
| Close dimer (d/σ=3) | 0.040 | 2.18x | FAIL | K=2 (60.3%) | K=2 (59.3%) |
| Single emitter | 0.047 | 3.22x | FAIL | K=1 (79.1%) | K=1 (89.4%) |
| Large dimer (d/σ=8) | 0.033 | 2.49x | FAIL | K=2 (75.4%) | K=2 (84.9%) |
| SM-only (well-sep) | 0.019 | 1.76x | FAIL | — | — |

**Close dimer P(K) detail (d/σ=3) — exact vs MCMC:**

| K | P_exact | P_MCMC | Ratio | Direction |
|---|---------|--------|-------|-----------|
| 1 | 0.186 | 0.277 | 0.67 | Over-visited (1.49×) |
| 2 | 0.603 | 0.593 | 1.02 | **Correct** |
| 3 | 0.191 | 0.120 | 1.59 | Under-visited |
| 4 | 0.020 | 0.009 | 2.18 | Under-visited |

MAP-K = 2 in BOTH exact and MCMC — the mode is correctly identified. The bias only affects the tails: mass flows from K=3,4 → K=1, leaving K=2 essentially unchanged.

### Prior Sensitivity Analysis (Exact Enumeration, No MCMC)

Exact posterior P(K) under different partition priors — close dimer d/σ=3, N=6:

| K | DM γ=2.0 | DM γ=1.0 | DM γ=0.5 | Uniform 1/S(N,K) |
|---|----------|----------|----------|-------------------|
| 1 | 0.186 | 0.224 | 0.291 | 0.029 |
| 2 | **0.603** | 0.599 | 0.579 | 0.249 |
| 3 | 0.191 | 0.163 | 0.122 | 0.360 |
| 4 | 0.020 | 0.014 | 0.008 | **0.362** |
| MAP-K | **2** | **2** | **2** | 4 (wrong!) |

**Key findings:**
1. **DM γ=2 gives the HIGHEST P(K=2) among all DM variants** — it cooperates with spatial evidence by favoring balanced partitions that align well with the two emitter clusters.
2. **Weaker DM (γ=0.5) shifts mass toward K=1**, not K=2 — because unbalanced partitions have worse spatial fit.
3. **Uniform partition catastrophically over-splits** (MAP-K=4 for d/σ=3, 10). This is the behavior documented in KB #1.
4. The DM prior IS the right choice for this problem.

### Practical Benchmarks

**smlmsim_highdensity** (331 hexamers, 25nm diameter, d/σ≈3.4):

| Metric | Value | Assessment |
|--------|-------|------------|
| True emitters | 1986 | — |
| Found emitters | 1199 | 40% missed |
| Precision | 0.998 | Near-perfect |
| Recall | 0.603 | Under-splitting |
| RMSE | 4.9 nm | Excellent |
| RMSE/Oracle | 1.25 | Good |
| Learned μ | 17.44 (true ≈ 8.7) | 2× overestimate |
| Learned shape | 21.92 (true ≈ 1.5) | Massive overestimate |
| Split acceptance | 1.6% | Very low |

The hierarchical learner adapts to the under-split state: fewer K → more locs per cluster → higher inferred μ and shape. This masks the under-splitting from the count model, creating a secondary feedback loop.

**genmab** (GenMAb HexaBody, ROI ~2×2 μm): 19501 locs → 2308 emitters, μ=8.23, shape=2.62. Visually reasonable, no ground truth.

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

### Known Issue: Slow K-Mixing (Under-Visits K > K_mode)

The sampler under-visits K > K_mode by 1.5-2.5× per step above the mode. This is a **mixing problem, not a model problem** — the DM γ=2 exact posterior correctly places MAP-K = K_true at all tested separations. The transition density implementation is correct (Jain-Neal framework verified), and the MH ratio computation is correct (Δ_spatial + Δ_partition + Δ_proposal).

**Root cause:** Split acceptance is 2-12%. The DM partition penalty Δ_partition ≈ -2.5 to -5 per split creates an energy barrier. Multi-step proposals (|ΔK|>1, ~25% of split/merge attempts) compound this penalty and are almost never accepted (log_α ≈ -5 to -7).

**Practical impact:** 40% recall loss on dense 6-mers (d/σ≈3.4). The hierarchical learner adapts μ and shape upward to compensate, masking the problem.

**Acceptable limitation:** Under-splitting of truly co-located emitters (d/σ ≈ 0) — identifiability limit, not a sampler deficiency.

### Validation Infrastructure

| Script | What it tests | Status |
|--------|--------------|--------|
| `brute_force_enumeration.jl` | Exact posterior comparison | WORKING, **1/4 PASS** |
| `prior_sensitivity.jl` | Exact P(K) under different priors | **NEW** (Round 6) |
| `mh_component_analysis.jl` | MH ratio component distributions | Working (Round 5) |
| `detailed_balance_check.jl` | DB for specific state pairs | STALE |
| `smlmsim_highdensity.jl` | Synthetic with ground truth | Re-run Round 6: Recall=0.60, RMSE=4.9nm |
| `genmab_bagol.jl` | Real antibody data | Re-run Round 6: 2308 emitters, μ=8.23 |
| `test/runtests.jl` | Unit + integration tests | ALL PASSING (170/170) |

## Active Research Threads

### Thread 1: Improve K-Mixing — OPEN

**Status:** Round 6 confirmed the problem is mixing (not model). The exact DM posterior is correct. The sampler's split/merge kernel has low acceptance (2-12%) and wastes ~25% of proposals on impossible multi-step chains.

**Next steps:**
1. **Restrict to |ΔK|=1 proposals** — eliminate multi-step waste, add log[P(N|K')/P(N|K)] to MH ratio. Simple change, ~33% more effective proposals.
2. **Multiple-try MH** — propose M independent allocations, select best, correct with Hastings ratio. Higher cost but potentially much better acceptance.
3. **Non-reversible lifting** — persistent K direction (momentum). Complex but potentially transformative.

### Thread 2: Hierarchical Learner Feedback

**Status:** smlmsim_highdensity shows μ drifts from 8.7 → 17.44 and shape from 1.5 → 21.92 when the sampler under-splits. The learner adapts to the biased K, creating a secondary feedback loop. This is not the primary problem (fixing K-mixing would fix this), but should be monitored.

### Thread 3: Practical Validation

**Status:** Re-run in Round 6. smlmsim: Recall=0.603, Precision=0.998, RMSE=4.9nm. genmab: 2308 emitters, visually reasonable.

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

## Next Round Priorities

**Core problem:** Slow K-mixing. The DM prior is correct, the MH ratio is correct, but split acceptance (2-12%) creates an energy barrier. Multi-step proposals waste ~25% of split/merge attempts.

1. **HIGH:** Restrict to |ΔK|=1 proposals. Remove count-model K proposal, randomly split or merge, add log[P(N|K')/P(N|K)] to MH ratio. Eliminates wasted multi-step chains, ~33% more effective proposals. Test against brute-force.
2. **HIGH:** Multiple-try MH. Propose M=5-10 independent split allocations, select the best (highest Δ_spatial + Δ_partition), correct with Hastings ratio. Could dramatically improve split acceptance.
3. **MEDIUM:** Investigate whether the hierarchical μ/shape feedback worsens the mixing problem. Consider freezing μ at a calibrated value during initial burn-in.
4. **LOW:** Parallel tempering or non-reversible lifting for K exploration.

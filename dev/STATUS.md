# Sampler Research Status

## Current State (2026-03-30, Round 7)

Round 7 replaced the count-model K proposal with |ΔK|=1 random split/merge proposals. This eliminated multi-step waste (~25% of proposals in R6) and added the count-model ratio Δ_count to the MH acceptance. Split acceptance doubled (2.7%→6.7%), ESS improved ~12%, and the close dimer max ratio improved (2.18x→2.01x). The fundamental under-visiting of K > K_mode persists but is slightly reduced.

### Brute-Force Results (Round 7)

| Test | KL | Max ratio | Verdict | R6 KL | R6 ratio | Change |
|------|-----|-----------|---------|-------|----------|--------|
| Well-separated (d/σ=10) | 0.015 | 1.55x | **PASS** | 0.014 | 1.48x | ~same |
| Close dimer (d/σ=3) | 0.036 | 2.01x | FAIL | 0.040 | 2.18x | **improved** |
| Single emitter | 0.054 | 2.81x | FAIL | 0.047 | 3.22x | ratio improved |
| Large dimer (d/σ=8) | 0.038 | 2.56x | FAIL | 0.033 | 2.49x | slightly worse |
| SM-only (well-sep) | 0.019 | 1.76x | FAIL | 0.020 | 1.76x | ~same |

**Split/merge acceptance rates (Round 7):**

| Test | Split acc. | Merge acc. | R6 split | R6 merge |
|------|-----------|-----------|----------|----------|
| Well-sep (full MCMC) | 6.7% | 6.7% | ~3% | ~9% |
| Close dimer | 36.9% | 64.5% | — | — |
| Single emitter | 5.6% | 100% | — | — |
| SM-only (well-sep) | 6.7% | 6.7% | 2.7% | 9.2% |

**Close dimer P(K) detail (d/σ=3) — exact vs MCMC:**

| K | P_exact | P_MCMC | Ratio | Direction |
|---|---------|--------|-------|-----------|
| 1 | 0.186 | 0.272 | 0.68 | Over-visited (1.46×) |
| 2 | 0.603 | 0.594 | 1.02 | **Correct** |
| 3 | 0.191 | 0.124 | 1.54 | Under-visited |
| 4 | 0.020 | 0.010 | 2.01 | Under-visited |

### Practical Benchmarks

**smlmsim_highdensity** (331 hexamers, 25nm diameter, d/σ≈3.4):

| Metric | R7 Value | R6 Value | Assessment |
|--------|----------|----------|------------|
| True emitters | 1986 | 1986 | — |
| Found emitters | 1181 | 1199 | ~same |
| Precision | 0.999 | 0.998 | Near-perfect |
| Recall | 0.594 | 0.603 | ~same |
| RMSE | 4.9 nm | 4.9 nm | Excellent |
| Learned μ | 17.79 | 17.44 | 2× overestimate |
| Learned shape | 21.92 | 21.92 | Massive overestimate |

**genmab** (GenMAb HexaBody, ROI ~2×2 μm): 19501 locs → 2034 emitters, μ=9.43, shape=3.02 (R6: 2308 emitters, μ=8.23, shape=2.62).

### Architecture Summary

```
Move mix: Gibbs allocation (50%) + Split/Merge (50%)
Gibbs:    P(z_i = k | rest) ∝ (n_{-i,k} + γ) × predictive  [DM-weighted]
K proposal: Random |ΔK|=1 (coin flip split/merge, boundary-aware)
Split:    Random seeds + sequential launch + restricted Gibbs scans (Jain-Neal)
          Final scan density enters MH ratio; intermediate scans free.
Merge:    Uniform pair + random seeds + reverse density via restricted Gibbs
          Launch → intermediate scans → transition density to current allocation
MH:       log α = Δ_spatial + Δ_partition + Δ_proposal + Δ_count + Δ_move_type
          Δ_count = log P(N|K') - log P(N|K)  [no longer cancels]
          Δ_move_type = log(d_{K'}/b_K) or log(b_{K'}/d_K)  [boundary correction]
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

The sampler under-visits K > K_mode by 1.5-2× per step above the mode. This is a **mixing problem, not a model problem** — the DM γ=2 exact posterior correctly places MAP-K = K_true at all tested separations.

**Root cause:** Split acceptance is 5-7% (doubled from 2-3% after Round 7's |ΔK|=1 change). The DM partition penalty Δ_partition ≈ -2.5 to -5 per split creates an energy barrier. Multi-step proposals were eliminated in Round 7, improving efficiency but not removing the barrier.

**Practical impact:** ~40% recall loss on dense 6-mers (d/σ≈3.4). The hierarchical learner adapts μ and shape upward to compensate, masking the problem.

**Acceptable limitation:** Under-splitting of truly co-located emitters (d/σ ≈ 0) — identifiability limit, not a sampler deficiency.

### Validation Infrastructure

| Script | What it tests | Status |
|--------|--------------|--------|
| `brute_force_enumeration.jl` | Exact posterior comparison | WORKING, **1/4 PASS** |
| `prior_sensitivity.jl` | Exact P(K) under different priors | Working (Round 6) |
| `mh_component_analysis.jl` | MH ratio component distributions | Working (Round 5) |
| `detailed_balance_check.jl` | DB for specific state pairs | STALE |
| `smlmsim_highdensity.jl` | Synthetic with ground truth | Re-run Round 7: Recall=0.59, RMSE=4.9nm |
| `genmab_bagol.jl` | Real antibody data | Re-run Round 7: 2034 emitters, μ=9.43 |
| `test/runtests.jl` | Unit + integration tests | ALL PASSING (170/170) |

## Active Research Threads

### Thread 1: Improve K-Mixing — OPEN

**Status:** Round 7 implemented |ΔK|=1 proposals (eliminated multi-step waste). Split acceptance doubled (2.7%→6.7%), ESS improved ~12%, close dimer max ratio improved (2.18x→2.01x). But the fundamental DM energy barrier persists (split acceptance still only 5-7%).

**Next steps:**
1. **Multiple-try MH** — propose M=5-10 independent split allocations, select best, correct with Hastings ratio. Higher cost but potentially much better acceptance.
2. **Count-informed ±1 proposal** — choose split vs merge proportional to P(N|K+1)/P(N|K-1) instead of uniform 50/50. Concentrates proposals in the favored direction.
3. **Non-reversible lifting** — persistent K direction (momentum). Complex but potentially transformative.

### Thread 2: Hierarchical Learner Feedback

**Status:** smlmsim_highdensity still shows μ drifts from 8.7 → 17.79 and shape from 1.5 → 21.92 when the sampler under-splits. Unchanged from R6. Not the primary problem (fixing K-mixing would fix this), but should be monitored.

### Thread 3: Practical Validation

**Status:** Re-run in Round 7. smlmsim: Recall=0.594, Precision=0.999, RMSE=4.9nm (~same as R6). genmab: 2034 emitters, μ=9.43 (R6: 2308, μ=8.23).

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

## Next Round: Round 8 — Birth/Death Moves

**Core problem:** Slow K-mixing. Split acceptance is 5-7% due to the DM energy barrier (-2.5 to -5 per split). The monolithic split must propose a full allocation in one shot.

**Plan:** Add birth/death as a third move type (supplementing, not replacing, split/merge). Reviewed and approved by Codex (gpt-5.4).

### Birth Move (K → K+1)

1. Pick a random non-sole-occupant loc i (uniform 1/N_eligible)
2. Remove it from cluster j, create singleton cluster {i}
3. MH acceptance:
   - Δ_spatial = lml(singleton_i) + lml(cluster_j \ i) - lml(cluster_j)
   - Δ_partition = log P_DM(z'|K+1) - log P_DM(z|K)
   - Δ_count = log P(N|K+1) - log P(N|K)
   - Δ_proposal = log(q_death_rev) - log(q_birth_fwd)
4. After acceptance, Gibbs sweeps grow the new cluster by reassigning locs

### Death Move (K → K-1)

1. Pick a singleton cluster (1/n_singletons)
2. Assign its member to another cluster proportional to predictive probability
3. Remove the empty cluster
4. MH acceptance (reverse of birth)

### Proposal Densities

```
q_birth_fwd = p_birth × (1/N_eligible)
q_death_rev = p_death × (1/n_singletons_after) × pred(i|cluster_j_reduced) / Σ_{k≠singleton} pred(i|cluster_k)
```

**Critical:** Death reverse density must be evaluated in the post-birth state x', with singleton excluded from destinations. No Jacobian needed (discrete collapsed move).

### Move Mix

Target: 40% Gibbs / 30% split-merge / 30% birth-death (Codex recommendation). Start with 50% / 25% / 25% conservatively.

### Boundary Handling

- No eligible locs (all sole occupants): skip birth, propose death or Gibbs
- No singletons: skip death, propose birth or Gibbs
- K=1: no death possible

### Expected Benefit

- DM penalty per birth ~-1.2 (vs -2.5 to -5 for split) — half the energy barrier
- Trivial proposal density (no restricted Gibbs, no allocation)
- Decomposes split into incremental steps: birth creates seed → Gibbs grows it
- Won't fully replace split/merge for balanced missed splits, but provides cheap incremental K-mobility

### Codex Feedback (Incorporated)

- Pair birth or targeted birth (biased toward low within-cluster fit) are stronger variants; consider after baseline singleton birth works
- Label symmetry not an issue since we use unlabeled partitions (assignment vector)
- Tune move mix by ESS/sec for K, not raw acceptance rate

## Future Priorities (After Round 8)

1. **MEDIUM:** Multiple-try MH for split/merge
2. **MEDIUM:** Targeted birth (choose loc with lowest within-cluster fit)
3. **MEDIUM:** Count-informed ±1 proposal for split/merge
4. **LOW:** Parallel tempering or non-reversible lifting

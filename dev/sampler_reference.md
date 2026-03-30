# Collapsed BaGoL Sampler Reference

**Authoritative reference for the collapsed Gibbs sampler. Read before modifying. Update after modifying.**

*Matches implementation on `main` branch (Round 7, 2026-03-30). Round 7 changed K proposal from count-model sampling to |ΔK|=1 random split/merge.*

---

## 1. Generative Model

```
K         ~ (implicit from count model)                # number of emitters
s_j       ~ Locmix(data)           j = 1..K            # emitter positions
n_j       ~ NegBin(α, p)           n_j ≥ 1             # locs per emitter
d_i | z_i ~ N₂(s_{z_i}, Σ_i)                           # observed localization
```

The total count `N = Σ n_j ~ NegBin(Kα, p)` by NegBin additivity, where `p = α/(α+μ)`.

**No separate prior on K.** The NegBin count model `P(N|K,α,μ)` regularizes K through its shape parameter Kα. Adding a Poisson prior on K over-penalizes large K.

**Localization mixture prior:** `P(s_j) = (1/N) Σᵢ N(s_j; dᵢ, Σᵢ)`. Each localization contributes a Gaussian component. Evaluated via grid-based bilinear interpolation (O(1), zero allocation).

### Variable Glossary

| Symbol | Code variable | Type | Meaning |
|--------|--------------|------|---------|
| K | `state.n_active` | Int | Number of emitters (clusters) |
| N | `length(locs)` | Int | Total localizations (fixed, observed) |
| z_i | `state.assignments[i]` | Int16 | Cluster label for loc i |
| s_j | `posterior_mean(cs)` | (Float64, Float64) | Emitter j position (integrated out) |
| Σ_i | per-loc covariance | 2×2 matrix | Localization uncertainty from PSF fit |
| μ | `mu` | Float64 | Mean locs per emitter, E[n_j] |
| α | `shape` | Float64 | NegBin shape; CV[n_j] = √(1/α + 1/μ) |
| γ | `shape` (same as α) | Float64 | DM concentration parameter |
| Λ | `cs.Λ_xx, cs.Λ_xy, cs.Λ_yy` | 2×2 sym | Posterior precision = Σᵢ Σᵢ⁻¹ |
| η | `cs.η_x, cs.η_y` | 2-vec | Natural parameter = Σᵢ Σᵢ⁻¹ dᵢ |
| Q | `cs.quad` | Float64 | Quadratic form = Σᵢ dᵢᵀ Σᵢ⁻¹ dᵢ |
| n_j | `cs.n` | Int32 | Locs in cluster j |

### Count Model Parametrization

| Regime | α value | Meaning |
|--------|---------|---------|
| dSTORM | α = 1 | Exponential (geometric discrete) |
| DNA-PAINT | α > 1 | Peaked around μ |
| Fixed blinks | α → ∞ | Delta at μ (Poisson limit) |

The NegBin parametrization: `n_j ~ NegBin(α, p)` where `p = α/(α+μ)`, giving E[n_j] = μ, Var[n_j] = μ(1 + μ/α).

The marginal total count: `N | K ~ NegBin(Kα, p)` with E[N|K] = Kμ, Var[N|K] = Kμ(1 + μ/α).

### Hyperpriors (defaults)

| Parameter | Prior | Default | Prior mean |
|-----------|-------|---------|------------|
| μ | Gamma(a_μ, b_μ) | Gamma(2, 5) | 10 |
| α | Gamma(a_α, b_α) | Gamma(2, 1) | 2 |

---

## 2. Collapsing: Integrating Out Positions

Because the locmix prior `P(s_j)` is evaluated at the posterior mean (saddle-point approximation), we integrate out s_j analytically using the Gaussian conjugacy. For cluster j with locs {d_i : z_i = j}:

### With Locmix Prior

```
log p(data_j) = (1 - n_j) log(2π)
              - ½ Σᵢ log|Σᵢ|
              - ½ (Q - ηᵀ Λ⁻¹ η)
              - ½ log|Λ|
              + log P_locmix(ŝ_j)
```

where `ŝ_j = Λ⁻¹η` is the posterior mean and `P_locmix(ŝ_j) = (1/N) Σₖ N(ŝ_j; dₖ, Σₖ)` is the locmix prior density.

**Code:** `log_marginal_likelihood_locmix(cs, grid)` in `cluster_stats.jl`

The 8 fields of `ClusterStats` (Λ_xx, Λ_xy, Λ_yy, η_x, η_y, quad, n, log_det_sum) are the sufficient statistics. All `add_loc`/`remove_loc` operations are O(1).

Posterior mean: `ŝ_j = Λ⁻¹η` (precision-weighted centroid)
Posterior covariance: `Σ_post = Λ⁻¹`

**Code:** `posterior_mean(cs)`, `posterior_cov(cs)` in `cluster_stats.jl`

### LocmixGrid (O(1) Prior Evaluation)

The locmix prior is precomputed on a 2D grid:
```
grid[i,j] = log[(1/N) Σₖ N(θ_{i,j}; dₖ, Σₖ)]
```

Grid spacing: `dx = dy = max_σ / pixel_per_sigma` (default pixel_per_sigma = 2.0).
Margin: `margin_sigma × max_σ` (default margin_sigma = 3.0).
Evaluation: bilinear interpolation on log-density values, O(1), zero allocation.

**Code:** `build_locmix_grid(loc_precs)`, `evaluate_locmix_grid(grid, x, y)` in `cluster_stats.jl`

### Predictive Distribution

For Gibbs allocation, we need the predictive probability of loc i joining cluster k:

```
log p_pred(d_i | cluster_k) = log p(data_{k∪{i}}) - log p(data_k)
```

For an empty cluster: `p_pred = P_locmix(d_i)` (locmix prior at the localization position).

**Code:** `log_predictive_locmix(cs, lp, grid)` in `cluster_stats.jl`

---

## 3. Move Types

### 3.0 Move Selection

| Move | Probability | Function | K change |
|------|-------------|----------|----------|
| Gibbs allocation sweep | 50% | `gibbs_allocation_sweep!` | Fixed |
| Split/merge (K proposal) | 50% | `propose_split_merge!` | Variable |

**Code:** `run_collapsed_chain` in `collapsed_sampler.jl` — `r < 0.50` → Gibbs, else → split/merge.

### 3.1 Gibbs Allocation Sweep (K fixed)

**What it does:** Reassign each localization among the K existing clusters, with DM-weighted allocation.

**Algorithm:**
1. Random permutation (Fisher-Yates in-place on `state._perm`)
2. For each loc i in random order:
   - Skip sole occupants (maintains K)
   - Remove loc from current cluster: `remove_loc(cs, lp)`
   - For each active cluster k, compute allocation weight
   - Log-sum-exp normalize → sample from categorical
   - `add_loc` to chosen cluster
   - Update `assignments[i]`

**Math:**
```
P(z_i = k | rest) ∝ (n_{-i,k} + γ) × p_pred(d_i | cluster_k)
```

The `(n_{-i,k} + γ)` factor is the Dirichlet-Multinomial partition prior. It provides a "rich get richer" effect that compensates for the combinatorial explosion of partitions at higher K. Without it, the implicit prior is uniform-per-label, which gives S(N,3) >> S(N,1) partitions — strongly favoring higher K.

The concentration parameter γ = α (the count model shape).

**Code:** `gibbs_allocation_sweep!` in `collapsed_moves.jl`

### 3.2 RJMCMC Split/Merge with |ΔK|=1 Proposals

This is the three-stage process for changing K.

#### Stage 1: Random split/merge selection

```
With probability b_K: propose split (K → K+1)
With probability d_K: propose merge (K → K-1)
```

Boundary handling:
- K = 1: b_K = 1.0 (always split, can't merge below 1)
- K ≥ N: d_K = 1.0 (always merge, can't split beyond N)
- Otherwise: b_K = d_K = 0.5

Previous rounds (1-6) used a count-model independence sampler: sample K_new from π_count(K) ∝ P(N|K), which could propose |ΔK| > 1. Multi-step proposals compounded the DM partition penalty and were almost never accepted (~25% of proposals wasted). Round 7 switched to |ΔK|=1 random walk, with the count-model ratio entering the MH acceptance instead.

**Note:** Uses the fixed prior mean μ (not the adaptive μ from hierarchical learning) for the count-model ratio, preventing the μ-K positive feedback loop.

**Code:** `propose_split_merge!` in `collapsed_moves.jl`

#### Stage 2: Execute one split or one merge

**Split (K → K+1):**

1. Select parent cluster uniformly at random: probability 1/K
2. **Random seed selection:** pick two members randomly from the parent cluster (ordered pair, probability 1/(m(m-1)))
   - Seed 1 → sub-cluster A (stays in parent slot)
   - Seed 2 → sub-cluster B (goes to new slot)
3. Sort remaining members by loc index (canonical ordering)
4. **Sequential predictive allocation (launch):** for each remaining member j = 3..m:
   ```
   log w_A = log(n_A + γ) + log p_pred(d_j | sub-A)
   log w_B = log(n_B + γ) + log p_pred(d_j | sub-B)
   p_B = exp(log w_B) / (exp(log w_A) + exp(log w_B))
   ```
   Sample: member j → B with probability p_B, else → A.
5. **Restricted Gibbs scans (Jain-Neal):** if `n_restricted_scans > 0`:
   - Run `n_restricted_scans - 1` intermediate sweeps (no density tracking):
     for each non-seed member in canonical order, remove from current sub-cluster,
     compute DM-weighted predictive for each sub, sample, add back.
   - Run 1 final sweep (sample + track density → `q_alloc`):
     same procedure, but accumulate log P(z_j = chosen) for each member.
   - The final sweep's density REPLACES the sequential allocation density.
   - If `n_restricted_scans == 0`: use sequential allocation density directly.
6. Forward structural density: `q_fwd = (1/K) × q_alloc`
   - Note: seed selection density 1/(m(m-1)) is NOT included (see Stage 3)
7. Reverse structural density: `q_rev = 1/C(K+1, 2)`

**Merge (K → K-1):**

1. Select pair uniformly at random: probability 1/C(K, 2)
2. **Random seed selection for reverse density:** pick two members randomly from the merged set (matching the split's bijection). Seed density cancels.
3. Sort remaining members by loc index. Compute `is_in_b` relative to sub-cluster labels (member[i] is in same original cluster as seed_2).
4. **Compute reverse allocation density:**
   - If `n_restricted_scans > 0` (Jain-Neal):
     a. Sample a launch state via sequential allocation from the merged members
     b. Run `n_restricted_scans - 1` intermediate restricted Gibbs sweeps on launch
     c. Compute transition density from intermediate state to current allocation:
        `q_alloc_rev = _restricted_gibbs_transition_density(intermediate → current)`
        Uses hybrid state: members already processed have target assignments,
        later members have intermediate-state assignments.
   - If `n_restricted_scans == 0`: sequential allocation density of `is_in_b`
5. Forward structural density: `q_fwd = 1/C(K, 2)`
6. Reverse structural density: `q_rev = (1/(K-1)) × q_alloc_rev`
7. Execute: move all locs from slot_b into slot_a, deactivate slot_b.

**Code:** `_do_sequential_split!`, `_restricted_gibbs_sweep!`, `_restricted_gibbs_transition_density`, `_sample_sequential_launch`, `propose_split_merge!` in `collapsed_moves.jl`

#### Stage 3: MH acceptance

```
log α = Δ_spatial + Δ_partition + Δ_proposal + Δ_count + Δ_move_type
```

where:
- `Δ_spatial = Σ_k log p(data_k)_new - Σ_k log p(data_k)_old` (locmix marginal likelihoods)
- `Δ_partition = log P_DM(z_new | K_new) - log P_DM(z_old | K_old)` (Dirichlet-Multinomial)
- `Δ_proposal = log q_rev - log q_fwd` (structural proposal densities)
- `Δ_count = log P(N|K') - log P(N|K)` (count-model ratio — no longer cancels)
- `Δ_move_type = log(d_{K'}/b_K)` for splits, `log(b_{K'}/d_K)` for merges (birth/death rate correction)

**Why Δ_count no longer cancels:** In rounds 1-6, K was proposed from π_count(K) ∝ P(N|K), so P(N|K')/P(N|K) appeared in both the target ratio and the proposal ratio, canceling. With |ΔK|=1 random proposals, the count model is only in the target, not the proposal.

**Δ_move_type:** Zero for interior K (both b and d are 0.5). Non-zero only at boundaries: K=1 split has Δ_move_type = log(0.5/1.0) = -log(2).

**Seed selection cancellation:** The random seed density 1/(m(m-1)) appears as an auxiliary variable in both the split and merge proposals via the RJMCMC bijection framework and cancels in the MH ratio. It must NOT be included explicitly — doing so creates an m(m-1) ≈ 90 factor asymmetry that destroys K estimation (see Knowledge Base entry #12).

**Dirichlet-Multinomial partition prior:**
```
log P_DM(z | K) = log Γ(Kγ) - K log Γ(γ) - log Γ(N + Kγ) + Σ_k log Γ(n_k + γ)
```

On rejection: full rollback from saved state.

**Code:** `_log_dm_partition`, `_total_spatial_lml`, `propose_split_merge!` in `collapsed_moves.jl`

---

## 4. Locmix Prior Invariance (Critical Property)

The locmix prior replaces the uniform spatial prior `P(s_j) = 1/A`. This eliminates the `-log(A)` per-cluster term that appeared in the old uniform-prior formulation.

### Why it matters

Under uniform prior, splitting a cluster from K to K+1 incurs a `-log(A)` penalty per new cluster. For co-located emitters (d ≈ 0), the spatial likelihood improvement from splitting is near zero, but the `-log(A)` penalty always applies. This creates an Occam barrier that prevents correct K estimation at small separations.

Under locmix prior, the prior density `P_locmix(ŝ_j)` at each cluster's posterior mean is data-driven and approximately constant under splits — because the data supports itself. There is no area-dependent penalty per cluster.

### Consequences

| Regime | Split behavior | What decides K |
|--------|---------------|----------------|
| Co-located (d≈0) | Locmix neutral → count model decides | Count model alone (≡ Q-PAINT) |
| Separated (d >> σ) | Locmix supports split positions | Spatial info adds to counts |
| Large ROI | No area penalty | K independent of ROI size |

**Invariant:** BaGoL must NEVER do worse than Q-PAINT at any separation. If it does, the spatial prior has a residual per-cluster penalty.

### Saddle-point approximation

The locmix is evaluated at the posterior mean `ŝ_j = Λ⁻¹η`, not integrated over the posterior. This is excellent for clusters with n ≥ 2 (tight posterior), but may be inaccurate for n = 1 clusters (wide posterior where the prior varies significantly over the posterior).

---

## 5. Why Fixed μ for the Count-Model Ratio

The count-model ratio Δ_count = log P(N|K') - log P(N|K) uses `P(N|K) = NegBin(N; Kα, α/(α+μ))` where μ is the fixed prior mean (set once at initialization from the `mu` kwarg). If the adaptive μ (which tracks data) were used:

```
Feedback loop: K↑ → μ_adaptive tracks smaller clusters → count model shifts → K↑↑
```

Fixed μ breaks this cycle. The count-model ratio is stable regardless of the current chain state.

**Code:** `mu` passed to `propose_split_merge!` in `collapsed_sampler.jl` is the initial value, not the adapted one.

---

## 6. Hierarchical Updates

Updated every `hierarchical_interval` iterations (default 100) via MH with log-normal proposals.

### 6.1 μ Update

**Proposal:** `μ' = μ × exp(ε)`, `ε ~ N(0, 0.3²)`

**Likelihood:** Product of individual NegBin counts per active cluster:
```
log L(μ) = Σ_j log NegBin(n_j; α, α/(α+μ))
```

**Acceptance:**
```
log α = [log L(μ') - log L(μ)]
      + [log P(μ') - log P(μ)]        # Gamma(a_μ, b_μ) prior
      + [log μ' - log μ]              # Jacobian of log-normal proposal
```

**Bounds:** μ ∈ [1, 500]. Proposals outside → reject.

**Code:** `_update_mu_collapsed` in `hierarchical.jl`

### 6.2 α (shape) Update

Same structure as μ. Likelihood evaluated under `NegBin(n_j; α', α'/(α'+μ))`.

**Bounds:** α ∈ [0.5, 50].

**Code:** `_update_shape_collapsed` in `hierarchical.jl`

### 6.3 Global Updates (Partitioned BaGoL)

Pool log-likelihoods across all partition states:
```
log L_global(μ) = Σ_partitions Σ_j log NegBin(n_j; α, α/(α+μ))
```

Single MH step; accepted value broadcast to all chains.

**Code:** `_update_mu_collapsed_global!`, `_update_shape_collapsed_global!` in `hierarchical.jl`

---

## 7. MAP-N Estimation

### 7.1 Dahl Consensus (preferred default)

Find the posterior sample whose assignment vector is closest to the posterior similarity matrix:

```
z_Dahl = argmin_t Σ_{i<j} (1[z_i^t = z_j^t] - PSM_ij)²
```

where `PSM_ij = (1/T) Σ_t 1[z_i^t = z_j^t]` is the co-assignment frequency.

**Code:** `estimate_dahl(samples, locs, psm)` in `mapn.jl`

### 7.2 Overlap MAP-N (used in production pipeline)

Uses Dahl assignments as template, then refines with overlap-based Hungarian matching:

1. K = K_Dahl from Dahl partition labels
2. Filter posterior samples to K = K_Dahl
3. For each matching sample: overlap_hungarian → per-cluster overlap gate (min 50% overlap)
4. Position = mean of well-matched posterior means
5. **Uncertainty via law of total variance:**
   ```
   Σ_total = E[Var(θ|Z)] + Var[E(θ|Z)]
           = Σ_analytic   + Σ_allocation
   ```
   - Term 1: Dahl ClusterStats posterior covariance (analytic, from collapsed model)
   - Term 2: Sample variance of well-matched posterior means across MCMC samples
6. Fallback: if no samples pass filter, use Dahl assignments directly

**Code:** `estimate_mapn_overlap(samples, locs, dahl_assignments)` in `mapn.jl`

### 7.3 Other Methods

| Method | Function | Notes |
|--------|----------|-------|
| Collapsed (histogram + Hungarian) | `estimate_mapn_collapsed` | Mode of K histogram + iterative position matching |
| PSM thresholding | `estimate_mapn_psm` | Union-Find on PSM ≥ 0.5 for K, then position matching |
| VI greedy | `estimate_vi_greedy` | Minimize expected variation of information (Rastelli & Friel 2018) |
| Final-state only | `extract_emitters` | Single sample — no averaging, use only as fallback |

---

## 8. Accumulators

| Accumulator | What it stores | Update cost | Used by |
|-------------|---------------|-------------|---------|
| `EmitterCountHist` | Histogram of K per iteration | O(1) | MAP-N, diagnostics |
| `PartitionSamples` | Thinned assignment vectors | O(N) | MAP-N estimators |
| `PSMAccumulator` | Upper-triangular co-assignment matrix | O(Σ n_k²) | Dahl, PSM, VI |
| `PosteriorImage` | Rao-Blackwellized image (Gaussian blobs) | O(K × patch) | Visualization |
| `NNDistHist` | NN distance histogram | O(K²) | Structure analysis |

---

## 9. Constants and Magic Numbers

| Value | Where | What | Justification |
|-------|-------|------|---------------|
| 50% / 50% | `collapsed_sampler.jl` | Gibbs/split-merge ratio | Empirical; gives K mixing time ~50 iters |
| 50% / 50% | `collapsed_moves.jl` | Split/merge coin flip | Equal opportunity for K±1; boundary-aware |
| γ = α | `collapsed_moves.jl` | DM concentration parameter | Ties partition prior to count model shape |
| 5 | `collapsed_moves.jl` | n_restricted_scans default | 4 intermediate + 1 final Jain-Neal sweep |
| 0.3 | `hierarchical.jl` | Log-normal proposal σ for μ and α | ~25-35% acceptance rate |
| [1, 500] | `hierarchical.jl` | μ bounds | Physical: ≥1 blink, ≤500 is extreme |
| [0.5, 50] | `hierarchical.jl` | α bounds | α<0.5 is very overdispersed, α>50 ≈ Poisson |
| 3.0 | `cluster_stats.jl` | Locmix grid margin (in σ units) | Covers 99.7% of each component |
| 2.0 | `cluster_stats.jl` | Locmix grid pixels per σ | Nyquist-like sampling of Gaussian components |
| 100 | `collapsed_sampler.jl` | `hierarchical_interval` | Enough samples between μ/α updates |
| thin=5 | `rjmcmc.jl` | PartitionSamples thinning | Storage vs. resolution tradeoff |
| 0.5 | `mapn.jl` | PSM threshold for Union-Find | Standard in Bayesian clustering |
| 0.5 | `mapn.jl` | Min overlap fraction for overlap MAP-N | Ensures meaningful cluster correspondence |

---

## 10. Theoretical Limits

### MAP-N Accuracy for Co-located Emitters (Count-Only)

When all emitters are at the same position, only the total count N carries information about K. The MAP-K is:

```
K_MAP = argmax_K  P(N | K, α, μ) = NegBin(N; Kα, p)
```

The accuracy (probability MAP-K = K_true) is bounded by the overlap of neighboring NegBin distributions:

```
P(correct) ≈ μ / (2σ_N)
σ_N = √(K × μ(μ+α)/α)
```

Reference values (α=5, μ=20):

| K_true | Theoretical accuracy |
|--------|---------------------|
| 1 | 84% |
| 2 | 53% |
| 4 | 39% |
| 8 | 28% |

**No algorithm can exceed these limits** for co-located emitters with count data alone. At finite separation, spatial information helps and BaGoL should exceed Q-PAINT.

**Code:** `_log_count_posterior` in `collapsed_moves.jl` implements the K posterior.

---

## 11. Optimality Test Catalog

### Test Isolation Hierarchy

```
                    count model    spatial ML    hier learning
                    ───────────    ──────────    ─────────────
validate_qpaint        ✓              —              —
optimality_nohier      ✓              ✓              —
optimality_hier        ✓              ✓              ✓
nmer_stats             ✓              ✓              —
unified (dev/optimality/)  ✓          ✓              ✓
```

**Diagnostic logic:** If validate_qpaint fails → count model broken. If nohier fails but qpaint passes → spatial ML penalty. If hier fails but nohier passes → hierarchical learning problem.

### Test 1: Q-PAINT Validation (`dev/validate_qpaint.jl`)

**What:** K emitters co-located at (0,0), known μ/α. Pure count-model integration test.

**Setup:** K ∈ {1,2,4,8}, μ=20, α=5, 50 reps per K, 8000 iterations, 2000 burn-in.

**Theoretical bound:** Oracle MAP-K accuracy computed by exhaustive evaluation of π_count(K) for each realized N.

**Pass criterion:** Sampler accuracy ≥ 80% of oracle count-model accuracy for all K.

**Run:** `julia --project=dev dev/validate_qpaint.jl` (~5 min)

### Test 2: Full Optimality Suite (`dev/optimality_tests.jl`)

5 sub-tests with hierarchical learning enabled:

#### 2a. CRLB Limit
- 4-mer at d=250nm (well-separated), 20 trials
- Pass: RMSE < 1.2× oracle, K=4 recovery > 80%

#### 2b. Resolution Curve
- Dimer K=2, d/σ from 0 to 10, 20 trials per point
- Pass: d/σ=0 recovery ≥ 70% of qPAINT, d/σ=10 > 80%

#### 2c. N Sweep
- N ∈ {1,2,3,4,6,8,12,16} at NN=5σ spacing, 15 trials
- Pass: BaGoL ≥ qPAINT at all N, N=1 > 40%, N=4 > 40%

#### 2d. Blink Sweep
- 8-mer 25nm diameter, μ ∈ {3,5,10,20,40}, 15 trials
- Pass: More blinks → better recovery, BaGoL ≥ qPAINT

#### 2e. Calibration
- 6-mer at NN=5σ, 50 trials, Mahalanobis d² with full 2×2 posterior covariance
- Pass: Scale factor c = √(mean(d²)/2) ∈ [0.7, 1.3], d² ~ χ²(2)

**Run:** `julia --threads=auto --project=dev dev/optimality_tests.jl` (~20 min)

### Test 3: No-Hierarchical Variant (`dev/optimality_tests_nohier.jl`)

Identical 5-test suite but μ/α fixed at true values. Uses shape=1000 (Poisson approximation).

**Run:** `julia --threads=auto --project=dev dev/optimality_tests_nohier.jl` (~20 min)

### Test 4: Diameter Sweep (`dev/nmer_optimality_hier.jl`)

Trimer (K=3), diameter 0→80nm, hierarchical learning, μ_true=10, α_true=5.

**Run:** `julia --project=dev dev/nmer_optimality_hier.jl` (~10 min)

### Test 5: Statistical Analysis (`examples/nmer_stats.jl`)

100 trials of single 6-mer, no hierarchical, 20000 iters. Full partition diagnostics.

**Run:** `julia --threads=auto --project=examples examples/nmer_stats.jl` (~1 min)

### Test 6: Unified Sweep (`dev/optimality/run.jl`)

Replaces tests 1-5 with one parametric sweep. Primary axis: d/σ from 0→10 for dimers, 50 trials.

**Run:** `julia --threads=auto --project=dev dev/optimality/run.jl` (~15 min)

### Test 7: In-CI Count-Model Test (`test/runtests.jl`)

`@test`: K ∈ {1,2,4}, well-separated (10σ), hier learning, 25 trials, 4000 iters.

**Pass:** Observed MAP-K accuracy ≥ 80% of oracle (known μ=20, α=5).

---

## 12. Diagnostic Questions Quick Reference

| Question | Which test answers it |
|----------|---------------------|
| Is the count model correctly integrated? | `validate_qpaint.jl` |
| Does the spatial term hurt at d=0? | `optimality_tests.jl` Test 2b at d/σ=0 |
| At what separation does BaGoL beat qPAINT? | Resolution curve crossover |
| Is K systematically under/over-estimated? | `nmer_stats.jl`: VI decomposition |
| Is the chain stuck or is the posterior wrong? | `nmer_stats.jl`: EPL(Dahl) vs EPL(Oracle) |
| Are uncertainties calibrated? | Test 2e: coverage + scale factor |
| Does hierarchical learning help or hurt? | Compare hier vs nohier suites |
| Does more data (higher μ) help? | Test 2d: blink sweep |

---

## 13. Anti-Patterns (Things That Broke Before)

### Using adaptive μ in count-model ratio
**Symptom:** K runs away to high values.
**Root cause:** μ tracks N/K → count model ratio shifts → positive feedback.
**Fix:** Use fixed μ = prior mean for Δ_count. See Section 5.

### Missing DM partition prior (Round 2 root cause)
**Symptom:** Systematic over-splitting. Brute-force detects sampler/target mismatch.
**Root cause:** Without DM prior, implicit partition prior is uniform-per-label. S(N,3) >> S(N,1) creates massive combinatorial bias toward higher K.
**Fix:** Add `(n_{-i,k} + γ)` to Gibbs sweep AND `Δ_partition` to MH ratio.

### Including seed selection density in MH ratio (Round 3 lesson)
**Symptom:** 0% K accuracy. All splits accepted, no merges.
**Root cause:** Adding `-log(m(m-1))` to split forward and merge reverse creates ~90x asymmetry for typical cluster sizes (m≈10).
**Fix:** Seed density cancels via RJMCMC bijection — do NOT include it explicitly. Both split and merge use random seeds as auxiliary variables; the densities cancel.

### DM partition prior in Gibbs only (without MH)
**Symptom:** No change from baseline (K dynamics unchanged).
**Root cause:** DM weighting changes allocation at fixed K but doesn't affect K transitions. Split/merge dynamics dominate.
**Fix:** DM must appear in BOTH Gibbs sweep AND MH acceptance.

### DM in MH without Δ_proposal
**Symptom:** Severe under-splitting (split acceptance drops to <1%).
**Root cause:** Δ_partition and Δ_proposal partially cancel. Including one without the other is worse than neither.
**Fix:** Use computable proposal (sequential allocation) so Δ_proposal can be included.

### Comparing log posteriors across K for MAP-N
**Symptom:** Systematically biased K estimates.
**Root cause:** Normalizing constants differ by K; unnormalized log P(Z|data) is not comparable across K.
**Fix:** Use EPL (expected posterior loss) under VI loss, or histogram-based MAP-N, or PSM methods.

# SMLMBaGoL Mathematical Reference — Current Implementation

*Generated from source code on the `loc-mixture-prior` branch.*

---

## 1. Problem Statement

Given a set of single-molecule localization events

$$\mathcal{D} = \{(x_i, y_i, \sigma_{x,i}, \sigma_{y,i}, \sigma_{xy,i})\}_{i=1}^{N}$$

from an upstream PSF-fitting algorithm, infer:

- The unknown number of emitters $K$
- Their positions $\mathbf{s} = \{(s_{x,j}, s_{y,j})\}_{j=1}^{K}$
- The assignment $z_i \in \{1, \ldots, K\}$ of each localization to an emitter
- Hyperparameters $(\mu, \alpha)$ governing the distribution of localizations per emitter

**Architecture:** Collapsed Gibbs sampler — emitter positions are analytically integrated out via sufficient statistics (`ClusterStats`). The MCMC state is the assignment vector $\mathbf{z}$ only. Partitioned execution with parallel MCMC and synchronized global hyperparameter updates.

---

## 2. Generative Model (per partition)

$$
\begin{aligned}
K &\sim P(K \mid N) \propto P(N \mid K) && \text{(count-model posterior on K)} \\[0.3em]
\mathbf{s}_j &\sim P_{\text{spatial}}(\mathbf{s}) && \text{(emitter positions — see Section 3.2)} \\[0.3em]
n_j &\sim \text{NegBin}(\alpha,\; p) \quad p = \alpha/(\alpha + \mu) && \text{(localizations per emitter)} \\[0.3em]
(x_i, y_i) \mid z_i = j &\sim \mathcal{N}_2\!\left(\mathbf{s}_j,\; \Sigma_i\right) && \text{(observed localization)}
\end{aligned}
$$

where $\Sigma_i = \begin{pmatrix} \sigma_{x,i}^2 & \sigma_{xy,i} \\ \sigma_{xy,i} & \sigma_{y,i}^2 \end{pmatrix}$ is the per-localization covariance from PSF fitting.

**Count model parameters:**
- $\mu$: Mean localizations per emitter, $\mathbb{E}[n_j] = \mu$
- $\alpha$: NegBin shape (called `shape` in code), controls count dispersion
  - $\alpha = 1$: Geometric (dSTORM-like)
  - $\alpha > 1$: Peaked (DNA-PAINT-like)
  - $\alpha \to \infty$: Poisson($\mu$)
  - $\text{Var}[n_j] = \mu(1 + \mu/\alpha)$

**Hyperpriors:**
$$
\mu \sim \text{Gamma}(a_\mu, b_\mu), \quad \alpha \sim \text{Gamma}(a_\alpha, b_\alpha)
$$

Defaults: $a_\mu = 2, b_\mu = 5$ (mean 10), $a_\alpha = 2, b_\alpha = 1$ (mean 2).

---

## 3. Prior Distributions

### 3.1 K Prior (Count-Model Posterior)

There is **no separate prior on K**. Instead, K is regularized through the count-model posterior:

$$P(K \mid N) \propto P(N \mid K, \mu, \alpha) = \text{NegBin}(N;\; K\alpha,\; p)$$

where $p = \alpha / (\alpha + \mu)$. This exploits the NegBin summation property: if $n_j \sim \text{NegBin}(\alpha, p)$ independently, then $N = \sum_{j=1}^K n_j \sim \text{NegBin}(K\alpha, p)$.

- $\mathbb{E}[N \mid K] = K\mu$
- $\text{Var}[N \mid K] = K\mu(1 + \mu/\alpha)$

**Implementation:** `_log_count_posterior(K, N, shape, mu)` in `collapsed_moves.jl`, which calls `logpdf(NegativeBinomial(K * shape, p), N)`.

**Also:** `log_prior_total_count(N, K, mu, shape)` in `priors.jl` provides the same computation for external use.

### 3.2 Spatial Prior — Two Options

#### 3.2a Uniform Prior (default)

$$P(\mathbf{s}_j) = \frac{1}{A}$$

where $A = (x_\max - x_\min)(y_\max - y_\min)$ is the ROI area, computed with automatic padding:
- Percentage padding: 5% of data extent per dimension
- Minimum padding: $3\bar{\sigma}$ (ensures proposals stay in bounds)
- Minimum area floor: $N \pi (3\sigma_\text{med})^2$ prevents per-cluster spatial bonus from dominating when data is compact

**Implementation:** `UniformSpatialPrior` in `priors.jl`. Area computation in `area()`.

#### 3.2b Localization Mixture Prior (new, opt-in via `use_locmix_prior=true`)

$$P(\mathbf{s}) = \frac{1}{N} \sum_{j=1}^{N} \mathcal{N}_2(\mathbf{s};\; \mathbf{d}_j, \Sigma_j)$$

This concentrates prior mass near observed data, dramatically reducing the Occam penalty that penalizes splitting co-located emitters. Each localization contributes a Gaussian component centered at its observed position with its measured covariance.

Under this prior, the marginal likelihood for cluster $k$ becomes:

$$\text{ML}_k = \frac{1}{N} \sum_{j=1}^{N} \text{ML}_\text{flat}(\text{cluster}_k \cup \{\text{virtual}_j\})$$

where $\text{ML}_\text{flat}$ is the marginal likelihood under a flat (improper) prior — identical to `log_marginal_likelihood` but **without** the $-\log(A)$ term.

Each term adds virtual localization $j$ (with its precision) to the cluster and computes the flat-prior ML. This is a sum of $N$ Gaussian integrals, each computable via `ClusterStats` operations.

**Cost:** $O(N)$ per cluster evaluation vs $O(1)$ for uniform prior.

**Implementation:**
- `log_ml_flat(cs)` in `cluster_stats.jl`: ML without $-\log(A)$
- `log_ml_locmix(cs, loc_precs)` in `cluster_stats.jl`: $-\log(N) + \text{logsumexp}_j[\text{log\_ml\_flat}(\text{cs} + \text{virtual}_j)]$
- `log_predictive_locmix(cs, lp, loc_precs)` in `cluster_stats.jl`: ratio of locmix MLs for Gibbs allocation

### 3.3 Individual Count Distribution

$$n_j \sim \text{NegBin}(\alpha, p) \quad \text{where } p = \frac{\alpha}{\alpha + \mu}$$

Using `Distributions.jl` parameterization: `NegativeBinomial(r, p)` with $r = \alpha$.

**Implementation:** `log_negbin_pmf(n, mu, alpha)` in `priors.jl`.

---

## 4. Collapsed Likelihood (ClusterStats)

### 4.1 Sufficient Statistics

The `ClusterStats` struct stores accumulated sufficient statistics for a cluster of localizations, enabling $O(1)$ add/remove and analytic marginal likelihood computation without storing individual data:

| Field | Formula | Description |
|-------|---------|-------------|
| $\Lambda_{xx}, \Lambda_{xy}, \Lambda_{yy}$ | $\sum_i (\Sigma_i^{-1})_{ab}$ | Posterior precision matrix (2x2 symmetric) |
| $\eta_x, \eta_y$ | $\sum_i \Sigma_i^{-1} \mathbf{d}_i$ | Natural parameter vector |
| $q$ | $\sum_i \mathbf{d}_i^T \Sigma_i^{-1} \mathbf{d}_i$ | Quadratic form |
| $n$ | Number of localizations | Count |
| $\ell$ | $\sum_i \log|\Sigma_i|$ | Sum of log-determinants |

**Operations** (all $O(1)$, exact inverses of each other):
- `add_loc(cs, loc)` / `remove_loc(cs, loc)`: add/subtract precision contributions
- `add_loc(cs, lp)` / `remove_loc(cs, lp)`: same using precomputed `LocPrecision` (zero-allocation)

**Implementation:** `ClusterStats` struct and operations in `cluster_stats.jl`. `LocPrecision` holds precomputed per-localization precision contributions cached at initialization via `precompute_loc_precisions(locs)`.

### 4.2 Marginal Likelihood (Uniform Prior)

Emitter position $\mathbf{s}$ integrated out analytically:

$$\log p(\mathcal{D}_k \mid \text{cluster}_k) = (1-n_k)\log(2\pi) - \tfrac{1}{2}\ell_k - \tfrac{1}{2}(q_k - \boldsymbol{\eta}_k^T \Lambda_k^{-1} \boldsymbol{\eta}_k) - \tfrac{1}{2}\log|\Lambda_k| - \log A$$

The $-\log A$ term comes from the uniform spatial prior $P(\mathbf{s}) = 1/A$.

**Implementation:** `log_marginal_likelihood(cs, log_area)` in `cluster_stats.jl`.

### 4.3 Marginal Likelihood (Flat Prior)

Same as Section 4.2 but **without** the $-\log A$ term. Used as a building block for the localization mixture prior:

$$\log p_\text{flat}(\mathcal{D}_k \mid \text{cluster}_k) = (1-n_k)\log(2\pi) - \tfrac{1}{2}\ell_k - \tfrac{1}{2}(q_k - \boldsymbol{\eta}_k^T \Lambda_k^{-1} \boldsymbol{\eta}_k) - \tfrac{1}{2}\log|\Lambda_k|$$

**Implementation:** `log_ml_flat(cs)` in `cluster_stats.jl`.

### 4.4 Posterior Position and Covariance

$$\boldsymbol{\mu}_\text{post} = \Lambda_k^{-1} \boldsymbol{\eta}_k, \qquad \Sigma_\text{post} = \Lambda_k^{-1}$$

**Implementation:** `posterior_mean(cs)` and `posterior_cov(cs)` in `cluster_stats.jl`.

### 4.5 Predictive Probability

For the Gibbs allocation sweep, the predictive probability of assigning localization $i$ to cluster $k$:

**Existing cluster (uniform prior):**
$$\log p(d_i \mid \text{cluster}_k) = \text{log\_ml}(\text{cs}_k + d_i) - \text{log\_ml}(\text{cs}_k)$$

**Empty cluster (uniform prior):** $\log p = -\log A$

**Existing/empty cluster (locmix prior):**
$$\log p(d_i \mid \text{cluster}_k) = \text{log\_ml\_locmix}(\text{cs}_k + d_i) - \text{log\_ml\_locmix}(\text{cs}_k)$$

**Implementation:** `log_predictive(cs, lp, log_area)` and `log_predictive_locmix(cs, lp, loc_precs)` in `cluster_stats.jl`.

---

## 5. Collapsed Gibbs Sampler Moves

### 5.0 State and Move Selection

**State:** The MCMC state is the assignment vector $\mathbf{z} = (z_1, \ldots, z_N)$ only, stored in `CollapsedState.assignments`. Emitter positions are never explicitly sampled — they are derived from `ClusterStats` sufficient statistics as needed.

Each iteration samples one move type:

| Move | Probability | Type |
|------|-------------|------|
| Gibbs allocation sweep | 50% | Fixed-$K$ (reassign all locs) |
| Split/Merge | 50% | Dimension-changing ($K \to K'$) |

**Implementation:** Move selection in `run_collapsed_chain()` in `collapsed_sampler.jl`. The 50/50 split is a simple `rand() < 0.50` branch.

### 5.1 Gibbs Allocation Sweep (Fixed K)

Full sweep over all $N$ localizations in random order. For each loc $i$:

1. Remove $i$ from its current cluster
2. Compute predictive probability for each active cluster $k$:
   - Uniform prior: $\log w_k = \text{log\_predictive}(\text{cs}_k, \text{lp}_i, \log A)$
   - Locmix prior: $\log w_k = \text{log\_predictive\_locmix}(\text{cs}_k, \text{lp}_i, \text{loc\_precs})$
3. Sample $z_i \sim \text{Categorical}(\text{softmax}(\log \mathbf{w}))$
4. Add $i$ to the chosen cluster

**At fixed $K$, the Gibbs conditional is purely spatial** — cluster sizes are determined entirely by spatial evidence. There are no count-based (CRP/Polya) weights.

**Sole occupants are skipped** to maintain $K$ during the sweep. Only split/merge moves change $K$.

Uses Fisher-Yates shuffle on a pre-allocated permutation buffer. Inner loop uses `LocPrecision` for zero-allocation operation.

**Implementation:** `gibbs_allocation_sweep!()` in `collapsed_moves.jl`.

### 5.2 Split/Merge with Count-Model K Proposal

The dimension-changing move has three phases: K proposal, heuristic restructuring, and spatial MH correction.

#### Phase 1: Propose $K'$ from count-model posterior

Sample $K' \sim \pi_\text{count}(K)$ where:

$$\pi_\text{count}(K) \propto P(N \mid K, \mu_0, \alpha) = \text{NegBin}(N;\; K\alpha,\; p)$$

evaluated over $K = 1, \ldots, K_\max$ where $K_\max = \max(2K, \min(N, 30))$.

**Critical: uses fixed $\mu_0$** (the prior mean $a_\mu \cdot b_\mu$, set at initialization) rather than the current adaptive $\mu$. This prevents the $\mu$-$K$ positive feedback loop: if adaptive $\mu$ is used, $\mu \approx N/K$ tracks $K$, making $\mathbb{E}[N \mid K] = K\mu = N$ for any $K$, which renders the count model non-informative for $K$ changes.

If $K' = K$, the move is rejected immediately (no-op).

#### Phase 2: Heuristic restructuring

**If $K' > K$ (split):** Repeatedly split the largest cluster. Each split takes a random half of the cluster's localizations into a new cluster slot. Implemented in `_add_clusters!()`.

**If $K' < K$ (merge):** Repeatedly merge the smallest cluster into its nearest neighbor (by posterior mean distance). Implemented in `_remove_clusters!()`.

#### Phase 3: Mini Gibbs relaxation + Spatial MH correction

After the heuristic restructuring:

1. **5 Gibbs relaxation sweeps** — let the allocation settle spatially before evaluation. Without this, the heuristic creates random allocations with terrible spatial ML, causing MH to reject even correct $K$ changes.

2. **Spatial MH acceptance:**

   **Uniform prior:**
   $$\log \alpha = \underbrace{[\textstyle\sum_k \text{log\_ml}(\text{new}_k) - \sum_k \text{log\_ml}(\text{old}_k)]}_{\text{spatial ML difference}} + \Delta K \cdot \log A$$

   The $+\Delta K \cdot \log A$ cancels the $-\log A$ per cluster in `log_marginal_likelihood`, implementing the area-invariant formulation. This makes the acceptance:
   - Neutral ($\approx 0$) for co-located emitters (pure Q-PAINT behavior)
   - Positive for splits aligned with spatial structure (beats Q-PAINT)

   **Localization mixture prior:**
   $$\log \alpha = \sum_k \text{log\_ml\_locmix}(\text{new}_k) - \sum_k \text{log\_ml\_locmix}(\text{old}_k)$$

   No area term — the locmix ML already has no $-\log A$ artifact.

3. Accept with probability $\min(1, e^{\log \alpha})$. On rejection, restore the full state from pre-allocated rollback buffers.

**Implementation:** `propose_split_merge!()`, `_add_clusters!()`, `_remove_clusters!()`, `_total_spatial_lml()`, `_save_rollback!()`, `_restore_rollback!()` in `collapsed_moves.jl`.

---

## 6. Hierarchical Updates

Updated every `hierarchical_interval` iterations (default 100) via Metropolis-Hastings with log-normal proposals.

### 6.1 $\mu$ Update

**Proposal:** $\mu' = \mu \cdot e^{\epsilon}$, $\epsilon \sim \mathcal{N}(0, 0.3^2)$

**Acceptance:**
$$\log \alpha = \underbrace{\sum_{j=1}^K \left[\log \text{NegBin}(n_j;\; \alpha, p') - \log \text{NegBin}(n_j;\; \alpha, p)\right]}_{\text{per-emitter count likelihood}} + \underbrace{[\log P(\mu') - \log P(\mu)]}_{\text{Gamma prior}} + \underbrace{[\log \mu' - \log \mu]}_{\text{log-normal Jacobian}}$$

where $p = \alpha/(\alpha + \mu)$, $p' = \alpha/(\alpha + \mu')$, and the sum is over **individual active cluster counts** $n_j = |\{i : z_i = j\}|$ from the current collapsed state. The likelihood evaluates `NegativeBinomial(alpha, p)` per cluster — the exact discrete NegBin, not a Gamma approximation.

**Range:** $\mu \in [1, 500]$.

**Implementation:** `_update_mu_collapsed()` in `hierarchical.jl` (per-partition), `_update_mu_collapsed_global!()` (pooled across all partitions).

### 6.2 Shape ($\alpha$) Update

Identical structure to $\mu$ update, with counts evaluated under $\text{NegBin}(n_j;\; \alpha', p')$.

**Range:** $\alpha \in [0.5, 50]$.

**Implementation:** `_update_shape_collapsed()` in `hierarchical.jl` (per-partition), `_update_shape_collapsed_global!()` (pooled across all partitions).

### 6.3 Initialization

- $\mu_\text{init} = a_\mu \cdot b_\mu$ (prior mean, default 10)
- $\mu_0 = \mu_\text{init}$ (fixed, used for split/merge K proposals — never updated)
- $\alpha_\text{init}$: Configurable (default 2.0), or estimated from count CV via `estimate_initial_shape()`

---

## 7. MAP-N Estimation

Five estimators are available, all operating on stored assignment samples from `PartitionSamples` and `PSMAccumulator`. The default pipeline (used in `run_bagol`) is Dahl + overlap.

### 7.1 Posterior Similarity Matrix (PSM)

The PSM $C_{ij}$ is the fraction of post-burn-in samples in which localizations $i$ and $j$ are co-assigned:

$$C_{ij} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{1}[z_i^{(t)} = z_j^{(t)}]$$

**Implementation:** `PSMAccumulator` in `accumulators.jl`.

### 7.2 Dahl's Method (`estimate_dahl`)

Select the MCMC sample whose association matrix is closest to the PSM in squared Frobenius norm:

$$\hat{\mathbf{z}} = \arg\min_{\mathbf{z}^{(t)}} \sum_{i<j} \left(\mathbf{1}[z_i^{(t)} = z_j^{(t)}] - C_{ij}\right)^2$$

This is equivalent to minimizing posterior expected Binder loss restricted to visited partitions.

Returns emitters with ClusterStats posterior positions/covariances, plus per-cluster stability scores (mean within-cluster PSM values).

**Implementation:** `estimate_dahl()` in `mapn.jl`. Helper `_association_psm_distance()` computes the upper-triangle Frobenius distance.

### 7.3 Overlap-Based MAP-N (`estimate_mapn_overlap`) — Default

Uses the Dahl partition as a reference template and matches each MCMC sample via overlap (contingency matrix) Hungarian matching:

1. $K$ from Dahl assignments
2. Filter samples to $K = K_\text{Dahl}$
3. For each matching sample: build contingency matrix $C[i,j] = |\text{ref cluster}_i \cap \text{sample cluster}_j|$ in $O(N)$, then Hungarian matching on $-C$ to maximize overlap
4. **Per-cluster overlap gate:** only include a sample's posterior mean for cluster $i$ if overlap fraction $\geq$ `min_overlap_frac` (default 0.5). This rejects label-switching mismatches.
5. **Position:** mean of well-matched posterior means
6. **Uncertainty via law of total variance:**
   $$\text{Var}(\mathbf{s}) = \underbrace{E[\text{Var}(\mathbf{s} \mid Z)]}_{\text{Term 1: Dahl analytic cov}} + \underbrace{\text{Cov}[E(\mathbf{s} \mid Z)]}_{\text{Term 2: allocation variance}}$$
   - Term 1: `posterior_cov(cs)` from the Dahl partition's ClusterStats ($\Lambda_k^{-1}$)
   - Term 2: sample covariance of posterior means from well-matched samples, including $\Sigma_{xy}$ cross-term
   - This is the marginal posterior variance an RJMCMC sampler would produce
7. Fallback: if no samples match, use Dahl assignments directly (`_emitters_from_assignments`)

**Implementation:** `estimate_mapn_overlap()` in `mapn.jl`. Helper `overlap_hungarian()` builds the contingency matrix and runs Hungarian matching.

### 7.4 Histogram-Mode MAP-N (`estimate_mapn_collapsed`)

1. Build histogram of $K$ across all samples
2. Find MAP-N $K^*$ (most common $K$, with 3-bin smoothing for near-ties)
3. Filter to samples with $K = K^*$
4. Extract posterior mean positions from ClusterStats for each filtered sample
5. Iterative Hungarian matching ($n_\text{refine}$ rounds, default 10): match each sample to reference positions, update reference to component-wise median
6. Final positions: median of matched positions (robust to label switching)
7. Uncertainties: ClusterStats posterior covariance from a reference sample

**Implementation:** `estimate_mapn_collapsed()` in `mapn.jl`. Helper `_smoothed_map_n()` handles K selection with smoothing.

### 7.5 PSM-Corrected MAP-N (`estimate_mapn_psm`)

K determined from PSM block structure (threshold $\geq 0.5$ + union-find connected components), then standard Hungarian pipeline from Section 7.4 on samples with that K.

Fixes the histogram-mode K bias at large K (transient splits inflate per-sample K).

**Implementation:** `estimate_mapn_psm()` in `mapn.jl`. Helper `_psm_cluster_count()` extracts K from the PSM via thresholding.

### 7.6 Greedy VI (`estimate_vi_greedy`)

Greedy search under Variation of Information loss (Rastelli & Friel 2018):

$$\hat{\mathbf{z}} = \arg\min_{\mathbf{a}} \frac{1}{T} \sum_{t=1}^{T} \text{VI}(\mathbf{a}, \mathbf{z}^{(t)})$$

Uses cached contingency tables for $O(1)$ per-sample delta computation. Total cost per sweep: $O(T \times N \times K_\text{up})$.

Algorithm: initialize from Dahl partition, then greedily move each localization to the cluster (or singleton) that minimizes expected VI. Multiple restarts (default 3) with random initialization.

**Implementation:** `estimate_vi_greedy()` in `mapn.jl`. Helper `_vi_from_contingency()` computes VI from contingency tables.

---

## 8. Partitioned Execution

### 8.1 Spatial Partitioning

Precision-weighted DBSCAN clusters localizations using the effective distance:

$$d_\text{eff}(i, j) = \frac{\|\mathbf{p}_i - \mathbf{p}_j\|}{\bar{\sigma}_i + \bar{\sigma}_j}$$

where $\bar{\sigma} = \sqrt{\sigma_x \cdot \sigma_y}$ is the geometric mean uncertainty.

Two localizations are neighbors if $d_\text{eff} < n_\sigma$ (default $n_\sigma = 3$).

Uses `KDTree` for spatial indexing: initial range query at $r_\max = n_\sigma \cdot 2 \cdot \sigma_\max$, then filter by precision-weighted distance.

Oversized clusters (above `max_partition_size`) are split via principal axis bisection at the median. Boundary localizations are flagged within `margin = n_\sigma \cdot \text{median}(\sigma)` of the bounding box edges or split planes.

**Implementation:** `partition_locs()` and `precision_dbscan()` in `partition.jl`. `split_partition()` handles recursive bisection. `mark_boundaries()` flags boundary localizations.

### 8.2 Synchronized Execution

```
for outer in 1:n_outer
    parallel (@sync/@spawn): run sync_interval iterations on each partition
    sequential: write archive samples (if past burn-in)
    global: _update_mu_collapsed_global!(states, ...)
    global: _update_shape_collapsed_global!(states, ...)
end
```

Each partition maintains its own `CollapsedState`. Global updates pool individual cluster counts from **all partitions' current states** (iterating directly over clusters, zero-allocation), then propose a single MH step. The accepted values are broadcast to all chains.

**Fixed $\mu_0$** is shared across all partitions for split/merge moves.

**Implementation:** `_run_bagol_collapsed()` in `rjmcmc.jl`. Iterations dispatched via `run_collapsed_iterations!()` in `collapsed_sampler.jl`.

### 8.3 MAP-N Extraction (Threaded)

After MCMC completes, MAP-N extraction runs in parallel across partitions, sorted largest-first for load balancing:

1. `estimate_dahl()` — find Dahl consensus partition from stored samples + PSM
2. `estimate_mapn_overlap()` — Dahl template + overlap Hungarian matching + per-cluster overlap gating + law of total variance uncertainties

**Implementation:** Threaded extraction loop in `_run_bagol_collapsed()` in `rjmcmc.jl`.

### 8.4 Boundary Deduplication

Emitters near partition boundaries are deduplicated:

1. Flag emitters within $2 \times \text{margin}$ of any boundary localization (`emitter_near_boundary`)
2. Build `KDTree` on boundary emitter positions for spatial indexing
3. Group nearby boundary emitter pairs by (partition_A, partition_B)
4. For each partition pair: Hungarian matching on Euclidean distance
5. Merge matched pairs within $2\sigma_\text{combined}$ via precision-weighted averaging:
   - Position: $\mathbf{s}_\text{merged} = \frac{w_i \mathbf{s}_i + w_j \mathbf{s}_j}{w_i + w_j}$ where $w = 1/\det(\Sigma)$
   - Uncertainty: $\sigma_\text{merged}^{-2} = \sigma_i^{-2} + \sigma_j^{-2}$ (per axis)
   - Cross-covariance: precision-weighted average

**Implementation:** `deduplicate_boundary_emitters()` and `emitter_near_boundary()` in `partitioned.jl`.

---

## 9. Core Types

| Type | File | Purpose |
|------|------|---------|
| `ClusterStats` | `cluster_stats.jl` | Immutable sufficient statistics (precision, natural params, quadratic, count, log-det) |
| `LocPrecision` | `cluster_stats.jl` | Precomputed per-localization precision contributions (cached, zero-alloc ops) |
| `CollapsedState` | `types.jl` | Mutable MCMC state: assignments, cluster stats cache, active bitvector, workspace buffers |
| `CollapsedChainResult` | `types.jl` | Final state + mu/shape + accumulator results + acceptance counts |
| `BaGoLDiagnostics` | `types.jl` | Output diagnostics: n_emitters, posterior_k, acceptance_rates, final params, posterior image |
| `Partition` | `partition.jl` | Spatial cluster with locs, original indices, boundary flags, parent_id |
| `UniformSpatialPrior` | `priors.jl` | Rectangular uniform spatial prior with auto-padding |

---

## 10. Main API

```julia
# Standard workflow — returns (BasicSMLD, BaGoLDiagnostics)
result_smld, diagnostics = run_bagol(smld; n_iterations=10000, burn_in=2000)

# Also accepts Vector{Emitter2DFit} directly (requires camera=):
result_smld, diagnostics = run_bagol(locs; camera=camera, n_iterations=10000)

# Direct chain access — returns CollapsedChainResult
result = run_collapsed_chain(locs;
    n_iterations=10000, burn_in=2000,
    use_locmix_prior=false,
    accumulators=[EmitterCountHist(), PosteriorImage(pixel_size=0.001)]
)
```

**Implementation:** `run_bagol()` in `rjmcmc.jl` dispatches to `_run_bagol_collapsed()`. `run_collapsed_chain()` in `collapsed_sampler.jl` runs a single-partition chain.

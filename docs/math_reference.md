# Mathematical Reference: SMLMBaGoL Collapsed Gibbs Sampler

*Authoritative specification for the current collapsed sampler target
(spatial-model routing update, 2026-04-24).
This file covers the static model. For K-changing moves see:
[split-merge.md](split-merge.md) and [birth-death.md](birth-death.md).*

---

## 1. Problem Context

In single-molecule localization microscopy (SMLM), fluorescent emitters activate stochastically across camera frames. Each activation yields a **localization** $(d_i, \Sigma_i)$: a 2D position with known heteroscedastic uncertainty. A single emitter produces multiple localizations; the number $K$ of physical emitters is unknown.

**Given** $N$ noisy localizations, **determine** $K$, the allocation $z$ of localizations to emitters, and each emitter's position $\theta_k$ with sub-localization precision.

**BaGoL** treats this as a Bayesian mixture model with unknown components. The **collapsed** formulation integrates out emitter positions $\theta_k$ analytically, reducing the state to the discrete allocation vector $z$.

---

## 2. Notation and Glossary

### Data (observed, fixed)

| Symbol | Code | Description |
|--------|------|-------------|
| $N$ | `length(locs)` | Total localizations |
| $d_i = (d_i^x, d_i^y)$ | `loc.x, loc.y` | Position of localization $i$ |
| $\Sigma_i$ | `loc.σ_x, loc.σ_y` | 2$\times$2 covariance of localization $i$ |
| $\Lambda_i = \Sigma_i^{-1}$ | `lp.λ_xx, lp.λ_xy, lp.λ_yy` | Precision matrix of localization $i$ |

### Latent variables

| Symbol | Code | Description |
|--------|------|-------------|
| $K$ | `state.n_active` | Number of emitters |
| $z_i \in \{1,\ldots,K\}$ | `state.assignments[i]` | Allocation of loc $i$ |
| $\theta_k$ | integrated out | True position of emitter $k$ |
| $n_k = |\{i : z_i = k\}|$ | `cs.n` | Localizations in cluster $k$ |

### Sufficient statistics (`ClusterStats`)

| Symbol | Code field | Description |
|--------|------------|-------------|
| $\Lambda_k = \sum_{i: z_i=k} \Lambda_i$ | `Λ_xx, Λ_xy, Λ_yy` | Posterior precision (2$\times$2 symmetric) |
| $\eta_k = \sum_{i: z_i=k} \Lambda_i d_i$ | `η_x, η_y` | Natural parameter |
| $Q_k = \sum_{i: z_i=k} d_i^\top \Lambda_i d_i$ | `quad` | Quadratic form |
| $S_k = \sum_{i: z_i=k} \log|\Sigma_i|$ | `log_det_sum` | Sum of log-determinants |

### Hyperparameters

| Symbol | Code | Default | Description |
|--------|------|---------|-------------|
| $\mu$ | `μ` | 10.0 | Mean localizations per emitter |
| $\alpha$ | `shape` | 2.0 | NegBin shape (1=geometric, $\infty$=Poisson) |
| $\gamma$ | `gamma` or `shape` | 2.0 | DM concentration. Default `nothing` uses $\gamma=\alpha$; explicit `gamma` decouples it. |
| $p$ | derived | $\alpha/(\alpha+\mu)$ | NegBin success probability |
| $\rho$ | `ρ` | 2.0 | Emitter density used only by the flat-spatial legacy Poisson K prior |

### Derived quantities

| Symbol | Description |
|--------|-------------|
| $\hat{\theta}_k = \Lambda_k^{-1}\eta_k$ | Posterior mean position |
| $\Sigma_{\text{post},k} = \Lambda_k^{-1}$ | Posterior covariance |
| $\pi_{\text{locmix}}(\theta) = \frac{1}{N}\sum_j \mathcal{N}(\theta; d_j, \Sigma_j)$ | Locmix spatial prior |

---

## 3. Generative Model

BaGoL supports two spatial targets:

- `spatial_model=:locmix` (default): no explicit prior on $K$; $K$ is regularized by the count model.
- `spatial_model=:flat` (legacy): $K \sim \text{Poisson}(\rho A)$ with a flat spatial prior over area $A$.

$$n_k \mid \mu, \alpha \stackrel{\text{iid}}{\sim} \text{NegBin}(\alpha,\; p), \quad p = \frac{\alpha}{\alpha+\mu}$$

For the default locmix target:

$$\theta_k \sim \pi_{\text{locmix}}(\theta) = \frac{1}{N}\sum_{j=1}^{N}\mathcal{N}(\theta;\, d_j,\, \Sigma_j)$$

For the flat legacy target:

$$\theta_k \sim \text{Uniform}(R), \qquad A = |R|$$

$$d_i \mid z_i, \theta_{z_i} \sim \mathcal{N}(\theta_{z_i},\, \Sigma_i)$$

The total count $N = \sum_k n_k \sim \text{NegBin}(K\alpha, p)$ by the NegBin summation property. The NegBin parameterization gives $E[n_k] = \mu$, $\text{Var}[n_k] = \mu(1 + \mu/\alpha)$.

| Regime | $\alpha$ | Blinking | Distribution |
|--------|----------|----------|--------------|
| dSTORM | 1 | Geometric | Exponential |
| DNA-PAINT | $>1$ | Peaked around $\mu$ | Moderate CV |
| Fixed blinks | $\to\infty$ | Delta at $\mu$ | Poisson limit |

---

## 4. Target Distribution

The sampler targets the collapsed posterior with positions integrated out:

### 4.1 Default locmix target

$$\boxed{P(z, K \mid \text{data}) \;\propto\; P_{\text{count}}(N \mid K) \;\times\; P_{\text{DM}}(z \mid K, N) \;\times\; \prod_{k=1}^{K} \text{ML}_{\text{locmix},k}(z)}$$

where:

- $P_{\text{count}}(N \mid K) = \text{NegBin}(N;\; K\alpha,\; p)$ --- total count model
- $P_{\text{DM}}(z \mid K, N)$ --- Dirichlet-Multinomial partition prior (Section 4.4)
- $\text{ML}_{\text{locmix},k}$ --- collapsed marginal likelihood under locmix prior (Section 5)

There is no separate $P(K)$ term in the locmix target. The previous bug was that
`spatial_model=:locmix` constructed a `LocmixSpatial` object but sampler moves
still evaluated flat likelihoods and included the flat Poisson K prior. The
current sampler routes all spatial marginal likelihoods and predictives through:

- `spatial_ml(cs, state.spatial)`
- `spatial_pred(cs, lp, state.spatial)`

so `LocmixSpatial` uses `log_marginal_likelihood_locmix` and
`log_predictive_locmix`, while `FlatSpatial` uses the uniform-area formulas.

### 4.2 Flat legacy target

For `spatial_model=:flat`, the target is:

$$\boxed{P_{\text{flat}}(z, K \mid \text{data}) \;\propto\; P_K(K \mid \rho,A) \;\times\; P_{\text{count}}(N \mid K) \;\times\; P_{\text{DM}}(z \mid K,N) \;\times\; \prod_{k=1}^{K} \text{ML}_{\text{flat},k}(z)}$$

with:

$$P_K(K \mid \rho,A) = \text{Poisson}(K;\rho A)$$

and $\text{ML}_{\text{flat},k}$ contains one $-\log A$ term per cluster.
The $A^K$ factor in the Poisson K prior cancels the $A^{-K}$ factor from
$K$ flat spatial priors, leaving a target that is area-independent up to
constants when $K$ changes.

**K! occupied-label multiplicity.** The Poisson prior carries a $-\log K!$ (from
$\text{Poisson}(K;\rho A)=e^{-\rho A}(\rho A)^K/K!$). The sampler state is a *labeled* allocation,
so a canonical partition with $K$ occupied clusters has $K!$ equivalent labelings; the coherent
posterior over unlabeled partitions is $T_{\text{fac}} = K!\,T_1$ (one representative $T_1$ times
the multiplicity). To target $T_{\text{fac}}$, the K-changing moves add
$\Delta_{K!} = \log K'! - \log K!$ ($=+\log(K{+}1)$ for split/birth, $-\log K$ for merge/death),
which exactly cancels the prior's $-\log K!$. This term is **gated on the Poisson-K prior**
(`_uses_poisson_k_prior`) and is a **no-op for `:locmix`**. Exact at fixed hyperparameters; the
$e^{\lambda m}$ empty-emitter thinning carry-through only matters when learning
$\mu/\text{shape}/\rho$. See `docs/split-merge.md` / `docs/birth-death.md` (§9.4).

### 4.3 Decoupled allocation variant

With `allocation_model=:decoupled`, the same spatial and count model are used
but $P_{\text{DM}}(z \mid K,N)$ is omitted. Thus the locmix decoupled target is:

$$P(z,K \mid \text{data}) \propto P_{\text{count}}(N \mid K)\prod_k \text{ML}_{\text{locmix},k}$$

and the flat decoupled target additionally includes $P_K(K\mid\rho,A)$ and
uses $\text{ML}_{\text{flat}}$.

### 4.4 Dirichlet-Multinomial Partition Prior

$$P_{\text{DM}}(z \mid K, N) = \frac{\Gamma(K\gamma)}{\Gamma(\gamma)^K \,\Gamma(N + K\gamma)} \prod_{k=1}^{K} \Gamma(n_k + \gamma)$$

with $\gamma = \alpha$ by default. If the user supplies an explicit `gamma`,
that value is used as a deliberate modeling override. With the default
$\gamma=\alpha$, this prior is the correct conditional distribution of
allocations given $K$ and $N$ under the NegBin count model. It is a mathematical
consequence of the generative model
($P_{\text{count}} \times P_{\text{DM}} = \prod_k \text{NegBin}(n_k) / \binom{N}{n_1 \cdots n_K}$).

For the more general derivation starting from an arbitrary iid emitter-count
family `q(n)`, see [count-family-partition-prior.md](count-family-partition-prior.md).

The DM prior provides a "rich-get-richer" effect that compensates for the combinatorial explosion of allocations at higher $K$ (Stirling number $S(N,K)$ grows rapidly). Without it, the implicit allocation prior is uniform-per-label, which overwhelmingly favors high $K$.

**No separate prior on $K$ for locmix.** The NegBin count model alone
regularizes $K$ under `spatial_model=:locmix`. The flat legacy path retains
the Poisson $K$ prior described in Section 4.2.

**Code:** `_log_dm_partition`, `partition_prior`, and `_uses_poisson_k_prior`
in `src/collapsed_moves.jl`.

### 4.5 Joint vs Marginal Mode (Round 11)

**Important:** The joint mode (highest-probability individual allocation) is typically at lower $K$ than the marginal mode $P(K \mid \text{data}) = \sum_z P(z, K \mid \text{data})$. This is because the DM prior assigns very low probability to each specific allocation at high $K$, but the marginal sums over an enormous number of such allocations.

Thermodynamic integration (Round 11) confirmed the marginal peaks near Q-PAINT's MAP $K$, while individual allocations at that $K$ score lower than allocations at lower $K$. This creates a K-mixing challenge for the collapsed sampler (see Section 9.1).

---

## 5. Collapsed Marginal Likelihood

### 5.1 Locmix prior version (grid-based, saddle-point)

$$\log\text{ML}_{\text{locmix},k} = (1 - n_k)\log(2\pi) - \tfrac{1}{2}S_k - \tfrac{1}{2}(Q_k - \eta_k^\top \Lambda_k^{-1}\eta_k) - \tfrac{1}{2}\log|\Lambda_k| + \log\pi_{\text{locmix}}(\hat{\theta}_k)$$

This is the saddle-point (Laplace plug-in) approximation to the exact integral. Round 11 confirmed the saddle-point error is $<0.02$ per cluster --- negligible.

**Code:** `log_marginal_likelihood_locmix(cs, grid)` in `src/cluster_stats.jl`.

### 5.2 Flat prior version

For `spatial_model=:flat`, replace the locmix prior term with the uniform
spatial density:

$$\log\text{ML}_{\text{flat},k} = (1 - n_k)\log(2\pi) - \tfrac{1}{2}S_k - \tfrac{1}{2}(Q_k - \eta_k^\top \Lambda_k^{-1}\eta_k) - \tfrac{1}{2}\log|\Lambda_k| - \log A$$

**Code:** `log_marginal_likelihood(cs, log_area)` in `src/cluster_stats.jl`.

### 5.3 Exact locmix integral (O(N) per cluster)

$$\text{ML}_k^{\text{exact}} = \frac{1}{N}\sum_{j=1}^{N} \text{ML}_{\text{flat}}(\text{cs}_k \cup \{\text{virtual}_j\})$$

Each term adds localization $j$ as a "virtual observation" and computes the flat-prior ML.

The sampler uses the grid saddle-point approximation for speed.

### 5.4 Predictive probabilities

For any spatial model:

$$\log p_{\text{pred}}(d_i \mid D_k) = \log \text{ML}(D_k \cup \{d_i\}) - \log \text{ML}(D_k)$$

Implementation dispatch:

| Spatial model | ML function | Predictive function |
|---------------|-------------|---------------------|
| `LocmixSpatial` | `log_marginal_likelihood_locmix` | `log_predictive_locmix` |
| `FlatSpatial` | `log_marginal_likelihood` | `log_predictive` |

**Code:** `spatial_ml`, `spatial_pred` in `src/cluster_stats.jl`.

### 5.5 ClusterStats operations

All operations are $O(1)$ via additive sufficient statistics:

$$\text{add\_loc}(k, i): \quad \Lambda_k \mathrel{+}= \Lambda_i,\;\; \eta_k \mathrel{+}= \Lambda_i d_i,\;\; Q_k \mathrel{+}= d_i^\top \Lambda_i d_i,\;\; S_k \mathrel{+}= \log|\Sigma_i|,\;\; n_k \mathrel{+}= 1$$

`remove_loc` is the exact inverse. `LocPrecision` caches per-localization contributions.

**Code:** `ClusterStats`, `LocPrecision`, `add_loc`, `remove_loc` in `src/cluster_stats.jl`.

---

## 6. Localization Mixture Prior

### 6.1 Definition

$$\pi_{\text{locmix}}(\theta) = \frac{1}{N}\sum_{j=1}^{N}\mathcal{N}(\theta;\, d_j,\, \Sigma_j)$$

**Key properties:**

- **No area dependence.** The $-\log|R|$ penalty from the uniform prior is eliminated.
- **Split-neutral at $d=0$.** Both sub-clusters see the same concentrated prior mass.
- **Split-positive at $d > 0$.** Each cluster's prior peaks near its own data.
- **Residual Occam effect.** The $-\frac{1}{2}\log|\Lambda_k|$ term creates a per-cluster penalty of $\approx -\log(n_k)$ that the locmix only partially compensates. This is a fundamental property of Bayesian marginal likelihood.

### 6.2 Grid approximation (`LocmixGrid`)

Evaluate $\log\pi_{\text{locmix}}(\theta)$ on a 2D grid ($3\sigma_{\max}$ margin, $\sigma_{\max}/2$ resolution). Bilinear interpolation in log-space gives $O(1)$ evaluation.

**Code:** `LocmixGrid`, `build_locmix_grid`, `log_prior_locmix` in `src/cluster_stats.jl`.

---

## 7. MCMC Moves

### 7.0 Move Selection

| Move | Probability | Function | $K$ change |
|------|-------------|----------|-----------|
| Gibbs allocation sweep | 50% | `gibbs_allocation_sweep!` | Fixed |
| Split/merge | 25% | `propose_split_merge!` | $\pm 1$ |
| Birth/death | 25% $\times$ `n_bd_substeps` | `propose_birth_death!` | $\pm 1$ |

**BD burst:** When birth/death is selected, `n_bd_substeps` (default 5) sequential BD attempts are made.

### 7.1 Gibbs Allocation Sweep ($K$ fixed)

$$P(z_i = k \mid z_{-i}, K, \text{data}) \;\propto\; (n_{-i,k} + \gamma) \;\times\; p_{\text{pred}}(d_i \mid D_k^{-i})$$

The $(n_{-i,k} + \gamma)$ factor is the **DM allocation weight**. The code
proposes from the spatial predictive and applies an MH correction for the DM
factor. Under `allocation_model=:decoupled`, the correction is 1. Sole
occupants are skipped to maintain $K$.

**Code:** `gibbs_allocation_sweep!` in `src/collapsed_moves.jl`.

### 7.2 Split/Merge --- see [split-merge.md](split-merge.md)

The MH log ratio contains:

$$\log\alpha = \Delta_{\text{spatial}} + \Delta_{\text{partition}} + \Delta_{\text{proposal}} + \Delta_{\text{count}} + \Delta_{K\text{-prior}} + \Delta_{\text{move-type}}$$

where $\Delta_{\text{spatial}}$ is computed with `spatial_ml` and
$\Delta_{K\text{-prior}}$ is:

$$\Delta_{K\text{-prior}} =
\begin{cases}
\log P_K(K'\mid\rho,A)-\log P_K(K\mid\rho,A), & \text{FlatSpatial}\\
0, & \text{LocmixSpatial}
\end{cases}$$

### 7.3 Birth/Death --- see [birth-death.md](birth-death.md)

The MH log ratio is the same as split/merge except there is no
$\Delta_{\text{move-type}}$ term. Destination choices use `spatial_pred` so
birth/death follows the selected spatial target.

### 7.4 Hierarchical Updates ($\mu$, $\alpha$, and flat-only $\rho$)

Every `sync_interval` iterations, $\mu$ and $\alpha$ are updated by a short
**multiplicative random-walk MH burst** pooled across all partition states, and
(flat target only) $\rho$ is conjugate-updated. Each burst runs $N$ steps
(default 50), caching the count log-likelihood across steps so only the proposal
is recomputed:

$$\theta' = \theta \cdot e^{\varepsilon}, \quad \varepsilon \sim \mathcal{N}(0, s^2)$$

The conditional posterior is tight when there are many clusters, so a single
fixed-scale step rejects almost everything (the old stepwise/stuck behaviour).
The $N$-step burst plus **Robbins--Monro scale adaptation** —
$s \leftarrow \mathrm{clamp}(s\,e^{0.5(\hat a - 0.3)},\, 0.002,\, 1)$ toward a
~30% acceptance target $\hat a$ — restores mixing. **Adaptation runs during
burn-in only**; the post-burn-in chain is fixed-scale MH, so ergodicity is
preserved.

**The $\alpha$ (shape) likelihood depends on the allocation model.** The per-cluster
count product $\prod_k \text{NB}(n_k;\alpha,p)$ equals
$P(N\mid K)\,P_{\text{DM}}(z\mid\gamma{=}\alpha)$ up to an $\alpha$-constant factor
($N!/\prod_k n_k!$), so it is the correct $\alpha$-likelihood **only when $\gamma$ is
tied to $\alpha$** — the default `allocation_model=:dm` with `gamma=nothing`. When
$\gamma$ is fixed, or under `:decoupled`/`:categorical`, the DM term does not depend on
$\alpha$; then $\alpha$ enters the target **only** through the total-count model, and the
update uses $\log\text{NB}(N; K\alpha, p)$ (= `_log_count_posterior`, identical to the
$\Delta_{\text{count}}$ term of the K-moves). Using the per-cluster product there would
inject a spurious $P_{\text{DM}}(\gamma{=}\alpha)$ factor (a shape-dependent bias,
$\sim1.4$ nats measured). $\mu$ needs no such branch: it enters only the count model
(the DM factor is $\mu$-independent). **Code:** the
`allocation_model===:dm && gamma===nothing` branch in
`_update_shape_collapsed`/`_update_shape_collapsed_global!`.

**Hyperpriors:** $\mu \sim \text{Gamma}(2, 5)$, $\alpha \sim \text{Gamma}(2, 1)$.
**Bounds:** $\mu \in [1, 500]$, $\alpha \in [0.5, 50]$.

The density parameter $\rho$ is conjugate-updated only for `FlatSpatial`,
because only that target contains $P_K(K\mid\rho,A)$. For `LocmixSpatial`,
$\rho$ is not part of the target and remains at its initialized value in
diagnostics.

**Code:** `_update_mu_collapsed_global!`, `_update_shape_collapsed_global!` (the
$N$-step adaptive bursts the chain uses), `_update_rho_collapsed_global!`, and
the single-step per-state `_update_mu_collapsed`/`_update_shape_collapsed` in
`src/hierarchical.jl`.

---

## 8. MAP-N Estimation

### 8.1 Dahl consensus (preferred default)

$$z_{\text{Dahl}} = \arg\min_t \sum_{i<j}\left(\mathbf{1}[z_i^t = z_j^t] - \text{PSM}_{ij}\right)^2$$

**Code:** `estimate_dahl(samples, locs, psm)` in `src/mapn.jl`.

### 8.2 Overlap-Hungarian matching

Uncertainty via law of total variance:

$$\Sigma_{\text{total}} = E[\text{Var}(\theta \mid z)] + \text{Var}[E(\theta \mid z)]$$

**Code:** `estimate_mapn_overlap(samples, locs, dahl_assignments)` in `src/mapn.jl`.

### 8.3 Other methods

| Method | Function | Notes |
|--------|----------|-------|
| Collapsed (histogram + Hungarian) | `estimate_mapn_collapsed` | Mode of $K$ histogram + iterative matching |
| PSM thresholding | `estimate_mapn_psm` | Union-Find on PSM $\geq 0.5$ |
| VI greedy | `estimate_vi_greedy` | Minimize expected variation of information |
| Final-state only | `extract_emitters` | Single sample, fallback only |

---

## 9. Known Limitations

### 9.1 K-mixing at large N (Round 11)

**The collapsed sampler has a fundamental K-mixing limitation at large $N$.** The joint mode is at lower $K$ than the marginal mode because the DM prior assigns very low probability to each specific high-$K$ allocation, but there are exponentially many. Thermodynamic integration confirmed the marginal peaks near Q-PAINT's MAP $K$, but the sampler converges to much lower $K$.

**Why Fazel's approach doesn't suffer:** The original RJMCMC does NOT collapse positions. At good positions, the joint $P(\theta, z, K)$ has high probability at correct $K$. Collapsing creates the Occam factor that penalizes individual high-$K$ allocations.

**Brute-force validation:** At $N = 6$, 4/4 PASS. Target is correct at all $N$; only mixing degrades.

### 9.2 Saddle-point locmix

Error $<0.02$ per cluster (Round 11). Not significant.

### 9.3 Hierarchical learner feedback

Under-splitting $\to$ higher $\mu/\alpha$ $\to$ count model shifts toward lower $K$ $\to$ reinforcing. Secondary effect of K-mixing failure.

### 9.4 Merge DB guard (Fix A) and the peak-K vs Dahl distinction (2026-07-02)

**Fix A --- merge detailed-balance seed-compatibility guard.** A merge draws two reverse-split
seeds with no one-per-cluster constraint; when both land in the same pre-merge cluster
(`is_in_b[1]`), no reverse split can recreate that seed pair, so the true reverse density is $0$.
The Jain-Neal reverse scan never re-scans the seeds and previously fabricated a positive reverse
density $\to$ DB violation $\to$ over-merge $\to$ $K$ biased **down** (grows with cluster size;
invisible at the $N=6$ brute-force). The guard sets $\log q_{\text{alloc,rev}} = -\infty$,
restoring EXACT DB (gate-checkers `dev/detailed_balance_check.jl` + `dev/birth_death_db_check.jl`
report max relative residual $6\times10^{-16}$).

**peak-K vs Dahl.** With Fix A + the $\Delta_{K!}$ term (§4.2), the co-located case is corrected
upward (25 nm hexamer: $K$-mode $5\to6$). On the default `:locmix` config with well-separated
emitters, the **peak-K** estimator (`estimate_mapn_collapsed`, histogram-mode of per-iteration
$K$) shows a 15--20 pt recovery drop --- but this is an **estimator artifact**: the **Dahl**
consensus estimator (`estimate_dahl`, the production MAP-N) is robust, its mean-K staying at
truth (paired $M=300$: Dahl $0.985/0.975/0.965 \to 0.93/0.955/0.955$; mean-K $1.07/2.04/4.05$).
Fix A removed an over-merge that had masked a *diffuse, wandering* upward bias in the locmix
$K$-marginal; peak-K reads the inflated mode, while Dahl's PSM consensus averages the wandering
over-split cut back out (well-separated cuts have no stable fault line; a persistent split at
$d/\sigma\approx0$ would move the PSM and Dahl would follow). The residual locmix upward
$K$-marginal bias (the collapsed co-location Occam defect, §9.1) is an **open target-refinement
thread** --- fix at the target level, not by restoring the DB-violating over-merge. **Recovery
gates use Dahl, not peak-K.**

---

## 10. Partitioned BaGoL

Precision-weighted DBSCAN partitioning, optional bridge refinement, parallel
chains per partition, boundary deduplication via Hungarian matching. See
`src/partition.jl`, `src/partitioned.jl`.

### 10.1 Precision-weighted DBSCAN

Two localizations are neighbors if:

$$\frac{\|d_i-d_j\|}{\bar{\sigma}_i+\bar{\sigma}_j} < \texttt{partition\_sigma}$$

where $\bar{\sigma}_i$ is the geometric mean localization uncertainty.

The public `min_partition_size`/`min_size` parameter uses standard DBSCAN
minimum-points semantics: it counts the point itself. Internally
`precision_neighbors` excludes self, so the implementation passes
`max(min_size - 1, 0)` to the neighbor-count threshold.

### 10.2 Bridge refinement

Precision-weighted DBSCAN can merge separate dense objects through a sparse
transitive bridge. When `bridge_ratio > 0`, each DBSCAN cluster is refined:

1. Compute each point's within-cluster precision-neighbor count.
2. Define core points as those with neighbor count at least
   `ceil(bridge_ratio * median_degree)`.
3. Find connected components of the core-only graph.
4. Keep components with at least `min_split_size` points.
5. Assign pruned bridge points and undersized core fragments to the nearest
   retained component.

If fewer than two retained components remain, the original DBSCAN cluster is
kept unchanged. `bridge_ratio=0.0` disables this pass and preserves the old
plain-DBSCAN behavior.

### 10.3 Oversized partitions and overlap

Clusters larger than `max_partition_size` are split by METIS on a
precision-weighted kNN graph. Optional overlap localizations are added around
sub-partition boundaries so local chains have spatial context. Emitters
primarily supported by overlap localizations are discarded during final
partition merge.

---

## 11. Diagnostic Target Types

The finite-state diagnostics use explicit target marker types:

| Target type | Spatial model | Allocation model | Explicit K prior |
|-------------|---------------|------------------|------------------|
| `DMFlatTarget` | flat | DM | $\text{Poisson}(\rho A)$ |
| `DirectNegBinFlatTarget` | flat | direct NegBin assignment form | $\text{Poisson}(\rho A)$ |
| `DecoupledTarget` | flat | decoupled | $\text{Poisson}(\rho A)$ |
| `DMLocmixTarget` | locmix | DM | none |
| `DirectNegBinLocmixTarget` | locmix | direct NegBin assignment form | none |
| `DecoupledLocmixTarget` | locmix | decoupled | none |

`DM*Target` and `DirectNegBin*Target` are algebraically identical within the
same spatial model; tests verify their exact posterior distributions match.
The diagnostics route sampler runs through `_diagnostic_sampler_kwargs(td)` so
a flat target is tested against `spatial_model=:flat` and a locmix target is
tested against `spatial_model=:locmix`.

---

## 12. References

1. Fazel, M. *et al.* Nature Communications **13**, 7152 (2022).
2. Miller & Harrison. JASA **113**(521):340--356 (2018).
3. Jain & Neal. JCGS **13**(1):158--182 (2004).
4. Green. Biometrika **82**(4):711--732 (1995).
5. Dahl. In *Bayesian Inference for Gene Expression and Proteomics*, 201--218 (2006).
6. Rastelli & Friel. Statistics and Computing **28**(5):1169--1186 (2018).

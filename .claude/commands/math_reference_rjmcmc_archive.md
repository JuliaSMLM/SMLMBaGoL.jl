# SMLMBaGoL Mathematical Reference — Current Implementation

*Generated from source code on the `gamma-count-prior` branch.*

---

## 1. Problem Statement

Given a set of single-molecule localization events

$$\mathcal{D} = \{(x_i, y_i, \sigma_{x,i}, \sigma_{y,i}, \sigma_{xy,i})\}_{i=1}^{N}$$

from an upstream PSF-fitting algorithm, infer:

- The unknown number of emitters $K$
- Their positions $\mathbf{s} = \{(s_{x,j}, s_{y,j})\}_{j=1}^{K}$
- The assignment $z_i \in \{1, \ldots, K\}$ of each localization to an emitter
- Hyperparameters $(\mu, \alpha)$ governing the distribution of localizations per emitter

**Constraints:**
1. Every localization is assigned to exactly one emitter (no noise class)
2. The count distribution is learned hierarchically
3. Explored via RJMCMC within disjoint spatial partitions processed in parallel
4. Global hyperparameters synchronized every $T_\text{sync}$ iterations

---

## 2. Generative Model (per partition)

$$
\begin{aligned}
K &\sim \text{Poisson}(\lambda_K) && \text{(number of emitters)} \\[0.3em]
\mathbf{s}_j &\sim \text{Uniform}(\text{ROI}) \quad j = 1, \ldots, K && \text{(emitter positions)} \\[0.3em]
n_j &\sim \text{Gamma}(\alpha,\; \mu/\alpha) \quad n_j \geq 1 && \text{(localizations per emitter, continuous approx.)} \\[0.3em]
(x_i, y_i) \mid z_i = j &\sim \mathcal{N}_2\!\left(\mathbf{s}_j,\; \Sigma_i\right) && \text{(observed localization)}
\end{aligned}
$$

where $\Sigma_i = \begin{pmatrix} \sigma_{x,i}^2 & \sigma_{xy,i} \\ \sigma_{xy,i} & \sigma_{y,i}^2 \end{pmatrix}$ is the per-localization covariance from PSF fitting.

**Count model parameters:**
- $\mu$: Mean localizations per emitter, $\mathbb{E}[n_j] = \mu$
- $\alpha$: Gamma shape (called `shape` in code), controls count dispersion
  - $\alpha = 1$: Exponential (dSTORM-like)
  - $\alpha > 1$: Peaked (DNA-PAINT-like)
  - $\alpha \to \infty$: Delta function at $\mu$
  - $\text{CV}[n_j] = 1/\sqrt{\alpha}$

**Hyperpriors:**
$$
\mu \sim \text{Gamma}(a_\mu, b_\mu), \quad \alpha \sim \text{Gamma}(a_\alpha, b_\alpha)
$$

Defaults: $a_\mu = 2, b_\mu = 5$ (mean 10), $a_\alpha = 2, b_\alpha = 1$ (mean 2).

---

## 3. Prior Distributions

### 3.1 K Prior (Poisson)

$$P(K) = \text{Poisson}(K; \lambda_K) = \frac{\lambda_K^K e^{-\lambda_K}}{K!}$$

Default: $\lambda_K = N / 5$ where $N$ is the number of localizations in the partition.

**Implementation:** `log_prior_k(k, λ_K)` in `priors.jl:54`.

### 3.2 Spatial Prior (Uniform)

$$P(\mathbf{s}_1, \ldots, \mathbf{s}_K) = \left(\frac{1}{A}\right)^K$$

where $A = (x_\max - x_\min)(y_\max - y_\min)$ is the ROI area, computed with automatic padding:
- Percentage padding: 5% of data extent in each dimension
- Minimum padding: $3 \bar{\sigma}$ (ensures proposals stay in bounds)

**Implementation:** `UniformSpatialPrior` in `priors.jl:6`, `log_spatial_prior()` in `priors.jl:34`.

### 3.3 Count Prior (Marginal Gamma)

Instead of individual count priors $\prod_j P(n_j \mid \mu, \alpha)$, the posterior uses the **marginal** distribution of the total count $N = \sum_j n_j$:

$$P(N \mid K, \mu, \alpha) = \text{Gamma}(N;\; K\alpha,\; \mu/\alpha)$$

This exploits the property that $\sum_{j=1}^K X_j$ where $X_j \sim \text{Gamma}(\alpha, \beta)$ is $\text{Gamma}(K\alpha, \beta)$.

- $\mathbb{E}[N \mid K] = K \mu$
- $\text{Var}[N \mid K] = K \mu^2 / \alpha$

**Implementation:** `log_prior_total_count(N, K, μ, shape)` in `priors.jl:99`.

**Note:** An individual count prior `log_prior_count(n, μ, shape)` exists in `priors.jl:76` but is **not used** in the posterior computation — only the marginal form is used.

### 3.4 Individual Count Prior (used only in hierarchical updates)

$$P(n_j \mid \mu, \alpha) = \text{Gamma}(n_j;\; \alpha,\; \mu/\alpha)$$

This form is used only in the MH updates for $\mu$ and $\alpha$ (Section 6), NOT in the RJMCMC posterior.

---

## 4. Posterior

The log-posterior used for RJMCMC acceptance ratios is:

$$\log P(K, \mathbf{s}, \mathbf{z} \mid \mathcal{D}) = \underbrace{\log P(K)}_{\text{K prior}} + \underbrace{K \cdot (-\log A)}_{\text{spatial prior}} + \underbrace{\log P(N \mid K, \mu, \alpha)}_{\text{marginal count}} + \underbrace{\sum_{j=1}^K \sum_{i: z_i=j} \log \mathcal{N}_2((x_i,y_i); \mathbf{s}_j, \Sigma_i)}_{\text{likelihood}}$$

**Implementation:** `compute_log_posterior()` in `moves.jl:228`.

A variant **without** the spatial prior term is used by split/merge moves: `compute_log_posterior_no_spatial()` in `moves.jl:263`.

---

## 5. RJMCMC Moves

### 5.0 Move Selection

Each iteration samples one move type:

| Move | Probability | Type |
|------|-------------|------|
| Birth | 10% | Dimension-changing ($K \to K+1$) |
| Death | 10% | Dimension-changing ($K \to K-1$) |
| Move | 20% | Fixed-dimension (position update) |
| Allocate | 60% | Fixed-dimension (assignment update) |

**Note:** Split and merge moves are implemented (`propose_split!`, `propose_merge!`) but **never selected** by the move selector. The move selector code at `rjmcmc.jl:26-33` only generates `:birth`, `:death`, `:move`, and `:allocate`.

**Implementation:** `rjmcmc_step!()` in `rjmcmc.jl:19`.

### 5.1 Birth Move ($K \to K+1$)

1. Sample position from mixture proposal:
   $$q_\text{birth}(\mathbf{s}_*) = \frac{1}{N} \sum_{i=1}^N \mathcal{N}_2(\mathbf{s}_*; (x_i, y_i), \text{diag}(\sigma_{x,i}^2, \sigma_{y,i}^2))$$
2. Add new emitter at $\mathbf{s}_*$
3. Gibbs reallocate all localizations (likelihood-only weights)
4. Accept with MH ratio:
   $$\log \alpha = [\log P_\text{new} - \log P_\text{old}] + [-\log(K+1) - \log q_\text{birth}(\mathbf{s}_*)]$$

**Implementation:** `propose_birth!()` in `moves.jl:294`.

### 5.2 Death Move ($K \to K-1$)

1. Select emitter $j$ uniformly: $j \sim \text{Uniform}\{1, \ldots, K\}$
2. Evaluate mixture density at victim position: $\log q_\text{birth}(\mathbf{s}_j)$
3. Remove emitter $j$ and Gibbs reallocate all localizations
4. Accept with MH ratio:
   $$\log \alpha = [\log P_\text{new} - \log P_\text{old}] + [\log K + \log q_\text{birth}(\mathbf{s}_j)]$$

Constraint: $K \geq 1$ always (never removes last emitter).

**Implementation:** `propose_death!()` in `moves.jl:360`.

### 5.3 Move (Gibbs Position Update)

1. Select emitter $j$ uniformly
2. Sample new position from posterior given allocated localizations:
   $$\Lambda_\text{post} = \sum_{i: z_i = j} \Sigma_i^{-1}$$
   $$\boldsymbol{\mu}_\text{post} = \Lambda_\text{post}^{-1} \sum_{i: z_i = j} \Sigma_i^{-1} (x_i, y_i)^T$$
   $$\mathbf{s}_j \sim \mathcal{N}_2(\boldsymbol{\mu}_\text{post}, \Lambda_\text{post}^{-1})$$
3. Reject if outside spatial prior bounds

Full 2D covariance is used: $\Sigma_i = \begin{pmatrix} \sigma_{x,i}^2 & \sigma_{xy,i} \\ \sigma_{xy,i} & \sigma_{y,i}^2 \end{pmatrix}$.

**Implementation:** `propose_move!()` in `moves.jl:613`, `sample_position_gibbs()` in `moves.jl:66`.

### 5.4 Allocate (Gibbs Assignment Update)

1. Pick a random localization $i$
2. Compute log-likelihood for each emitter:
   $$\log w_{ij} = \log \mathcal{N}_2((x_i, y_i); \mathbf{s}_j, \Sigma_i) \quad \forall j$$
3. Sample new assignment from softmax: $z_i \sim \text{Categorical}(\text{softmax}(\log \mathbf{w}_i))$
4. Update positions of affected emitters via Gibbs sampling
5. **Remove emitters with no allocated localizations** (`remove_empty_emitters!`)

**Allocation weights:** Likelihood-only. There are no count-based weights (no Polya/CRP terms).

**Implementation:** `propose_allocate!()` in `moves.jl:525`.

**Note:** The call to `remove_empty_emitters!()` at `moves.jl:601` performs an implicit dimension change ($K \to K-1$) that bypasses the RJMCMC acceptance ratio.

### 5.5 Full Gibbs Reallocation

Used within birth and death moves after adding/removing an emitter. Reassigns ALL localizations:

For each localization $i$:
$$z_i \sim \text{Categorical}\!\left(\frac{\exp(\log w_{ij})}{\sum_r \exp(\log w_{ir})}\right), \quad \log w_{ij} = \log \mathcal{N}_2((x_i, y_i); \mathbf{s}_j, \Sigma_i)$$

**Implementation:** `allocate_gibbs!()` in `moves.jl:167`.

### 5.6 Split Move ($K \to K+1$) — IMPLEMENTED BUT INACTIVE

1. Choose emitter $j$ uniformly, require $n_j \geq 2$
2. Random coin-flip partition of $j$'s localizations into two non-empty sets
3. Create two emitters with Gibbs-sampled positions from their allocations
4. Accept with ratio (no spatial prior):
   $$\log \alpha = [\log P'_\text{no-spatial} - \log P_\text{no-spatial}] + [\log 2 - \log(K+1)]$$

**Implementation:** `propose_split!()` in `moves.jl:696`.

### 5.7 Merge Move ($K \to K-1$) — IMPLEMENTED BUT INACTIVE

1. Require $K \geq 2$. Choose unordered pair $(i, j)$ uniformly from $\binom{K}{2}$ pairs
2. Combine allocations, Gibbs sample merged position
3. Accept with ratio (no spatial prior):
   $$\log \alpha = [\log P'_\text{no-spatial} - \log P_\text{no-spatial}] + [\log K - \log 2]$$

**Implementation:** `propose_merge!()` in `moves.jl:792`.

### 5.8 Uniform Birth/Death — IMPLEMENTED BUT INACTIVE

Alternative birth/death moves using flat spatial proposal $q(\mathbf{s}) = 1/A$ instead of the mixture proposal. Present for testing but never selected.

**Implementation:** `propose_birth_uniform!()` in `moves.jl:418`, `propose_death_uniform!()` in `moves.jl:468`.

---

## 6. Hierarchical Updates

Updated every `hierarchical_interval` iterations (default 100) via Metropolis-Hastings with log-normal proposals.

### 6.1 μ Update

**Proposal:** $\mu' = \mu \cdot e^{\epsilon}$, $\epsilon \sim \mathcal{N}(0, 0.3^2)$

**Acceptance:**
$$\log \alpha = \underbrace{\sum_{j} \left[\log \text{Gamma}(n_j; \alpha, \mu'/\alpha) - \log \text{Gamma}(n_j; \alpha, \mu/\alpha)\right]}_{\text{individual count likelihood}} + \underbrace{[\log P(\mu') - \log P(\mu)]}_{\text{prior}} + \underbrace{[\log \mu' - \log \mu]}_{\text{proposal Jacobian}}$$

where the sum is over **individual emitter counts** $n_j$ from recent samples (last `hierarchical_interval` samples, or current state during burn-in). Counts below 1 are clamped to 0.5.

**Range:** $\mu \in [1, 500]$ (proposals outside this range are rejected).

**Important:** This uses the **product of individual** count priors $\prod_j \text{Gamma}(n_j; \alpha, \mu/\alpha)$, while the posterior (Section 4) uses the **marginal** count prior $\text{Gamma}(N; K\alpha, \mu/\alpha)$.

**Implementation:** `update_mu!()` in `hierarchical.jl:41`.

### 6.2 Shape ($\alpha$) Update

Identical structure to μ update, with counts evaluated under $\text{Gamma}(n_j; \alpha', \mu/\alpha')$.

**Range:** $\alpha \in [0.5, 50]$.

**Implementation:** `update_shape!()` in `hierarchical.jl:97`.

### 6.3 Initialization

- $\mu_\text{init} = a_\mu \cdot b_\mu$ (prior mean, default 10)
- $\alpha_\text{init}$: Configurable (default 2.0), or estimated from count CV via `estimate_initial_shape()`

---

## 7. Partitioned Execution

### 7.1 Spatial Partitioning

Precision-weighted DBSCAN clusters localizations using the effective distance:

$$d_\text{eff}(i, j) = \frac{\|\mathbf{p}_i - \mathbf{p}_j\|}{\bar{\sigma}_i + \bar{\sigma}_j}$$

where $\bar{\sigma} = \sqrt{\sigma_x \cdot \sigma_y}$ is the geometric mean uncertainty.

Two localizations are neighbors if $d_\text{eff} < n_\sigma$ (default $n_\sigma = 3$).

Oversized clusters are split via principal axis bisection at the median.

**Implementation:** `partition_locs()` in `partition.jl:42`, `precision_dbscan()` in `partition.jl:112`.

### 7.2 Synchronized Execution

```
for outer in 1:n_outer
    parallel: run sync_interval iterations on each partition
    global: update_mu_global!(chains)
    global: update_shape_global!(chains)
end
```

Global updates pool individual counts from recent samples across ALL partitions, then propose a single MH step. The accepted value is broadcast to all chains.

**Implementation:** `run_bagol()` in `rjmcmc.jl:279`, global updates in `hierarchical.jl:219` and `hierarchical.jl:275`.

### 7.3 Partition Merging

1. Run `estimate_mapn()` on each partition chain
2. Identify emitters near partition boundaries (within $2 \times$ boundary margin of any boundary localization)
3. Hungarian matching on boundary emitters across adjacent partitions
4. Merge matched pairs closer than $2 \times$ margin via precision-weighted averaging:
   - Position: $\mathbf{s}_\text{merged} = \frac{w_i \mathbf{s}_i + w_j \mathbf{s}_j}{w_i + w_j}$ where $w = 1/\det(\Sigma)$
   - Uncertainty: $\sigma_\text{merged}^{-2} = \sigma_i^{-2} + \sigma_j^{-2}$ (per axis)

**Implementation:** `merge_partition_results()` in `partitioned.jl:152`.

---

## 8. Likelihood

2D Gaussian with full covariance:

$$\log L_i(j) = -\frac{1}{2}\left[\mathbf{d}^T \Sigma_i^{-1} \mathbf{d} + \log\!\left(4\pi^2 \det \Sigma_i\right)\right]$$

where $\mathbf{d} = (x_i - s_{x,j},\; y_i - s_{y,j})^T$.

Falls back to diagonal covariance if $\det \Sigma_i \leq 0$.

**Implementation:** `log_likelihood_single()` in `likelihood.jl:10`.

---

## 9. MAP-N Estimation

Posterior inference via iterative Hungarian matching:

1. **MAP-N selection:** Most frequent $K$ in post-burn-in samples
2. **Filter:** Keep only samples with $K = K_\text{MAP}$
3. **Initialize reference:** Sample whose centroid is closest to the overall centroid
4. **Iterative refinement** ($n_\text{refine}$ iterations, default 10):
   - Hungarian-match each filtered sample to current reference positions
   - Update reference to component-wise median of matched positions
5. **Final estimates:**
   - Position: Median of matched positions (robust)
   - Uncertainty $\sigma_x, \sigma_y$: MAD-based estimate, $\hat{\sigma} = 1.4826 \cdot \text{median}(|x_i - \text{median}|)$
   - Cross-covariance $\sigma_{xy}$: Sample covariance around median

**Implementation:** `estimate_mapn()` in `mapn.jl:123`.

---

## 10. Initialization

1. Create single emitter at centroid of all localizations
2. Assign all localizations to it (nearest-neighbor)
3. Gibbs-sample emitter position from allocations
4. $\mu$ initialized to prior mean ($a_\mu \cdot b_\mu = 10$)
5. $\alpha$ initialized to configured value (default 2.0)

**Implementation:** `run_bagol_chain()` in `rjmcmc.jl:155`, lines 186-198.

---

## 11. Known Issues

### 11.1 μ–K Feedback Loop

The hierarchical update (Section 6) fits μ to **individual allocation counts** using $\prod_j \text{Gamma}(n_j; \alpha, \mu/\alpha)$, while the posterior (Section 4) uses the **marginal** $\text{Gamma}(N; K\alpha, \mu/\alpha)$.

When K is over-estimated, each emitter gets $\sim N/K$ localizations. The hierarchical update then estimates $\mu \approx N/K$. With this μ, the marginal count prior satisfies $\mathbb{E}[N] = K\mu = N$ for any K, providing no penalty for over-counting.

### 11.2 Likelihood-Only Allocation Weights

Allocation (Section 5.4) uses pure likelihood weights $w_{ij} \propto L_{ij}$. There are no count-based (Polya/CRP) weights that would encourage localizations to cluster onto fewer emitters. This allows counts to spread evenly across too many emitters.

### 11.3 Silent Dimension Change in Allocate

`remove_empty_emitters!()` in the allocate move (Section 5.4) performs $K \to K-1$ transitions without computing an acceptance ratio, breaking detailed balance.

### 11.4 Dead Code: Split/Merge Moves

Split and merge moves are fully implemented but never selected by the move scheduler.

### 11.5 Static K Prior

The Poisson prior $P(K; \lambda_K = N/5)$ is fixed at initialization and does not adapt as μ is learned. With true μ = 10, the prior centers K at $N/5 = 2N/\mu$ — roughly double the expected number of emitters.

### 11.6 MAP-N Covariance Inconsistency

In `estimate_mapn()`, $\sigma_x$ and $\sigma_y$ use MAD-based robust estimation, but $\sigma_{xy}$ uses sample covariance around the median. These estimators have different breakdown points and efficiency.

# Mathematical Reference: SMLMBaGoL Collapsed Gibbs Sampler

*Authoritative specification for `main` branch (post-Round 11, 2026-03-31).
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
| $\gamma$ | `shape` (same as $\alpha$) | 2.0 | DM concentration parameter |
| $p$ | derived | $\alpha/(\alpha+\mu)$ | NegBin success probability |

### Derived quantities

| Symbol | Description |
|--------|-------------|
| $\hat{\theta}_k = \Lambda_k^{-1}\eta_k$ | Posterior mean position |
| $\Sigma_{\text{post},k} = \Lambda_k^{-1}$ | Posterior covariance |
| $\pi_{\text{locmix}}(\theta) = \frac{1}{N}\sum_j \mathcal{N}(\theta; d_j, \Sigma_j)$ | Locmix spatial prior |

---

## 3. Generative Model

$$K \sim (\text{no explicit prior --- regularized by count model})$$

$$n_k \mid \mu, \alpha \stackrel{\text{iid}}{\sim} \text{NegBin}(\alpha,\; p), \quad p = \frac{\alpha}{\alpha+\mu}$$

$$\theta_k \sim \pi_{\text{locmix}}(\theta) = \frac{1}{N}\sum_{j=1}^{N}\mathcal{N}(\theta;\, d_j,\, \Sigma_j)$$

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

$$\boxed{P(z, K \mid \text{data}) \;\propto\; P_{\text{count}}(N \mid K) \;\times\; P_{\text{DM}}(z \mid K, N) \;\times\; \prod_{k=1}^{K} \text{ML}_{\text{locmix},k}(z)}$$

where:

- $P_{\text{count}}(N \mid K) = \text{NegBin}(N;\; K\alpha,\; p)$ --- total count model
- $P_{\text{DM}}(z \mid K, N)$ --- Dirichlet-Multinomial partition prior (Section 4.1)
- $\text{ML}_{\text{locmix},k}$ --- collapsed marginal likelihood under locmix prior (Section 5)

### 4.1 Dirichlet-Multinomial Partition Prior

$$P_{\text{DM}}(z \mid K, N) = \frac{\Gamma(K\gamma)}{\Gamma(\gamma)^K \,\Gamma(N + K\gamma)} \prod_{k=1}^{K} \Gamma(n_k + \gamma)$$

with $\gamma = \alpha$ (the NegBin shape). This prior is the correct conditional distribution of allocations given $K$ and $N$ under the NegBin count model. It is NOT a tuning parameter --- it is a mathematical consequence of the generative model ($P_{\text{count}} \times P_{\text{DM}} = \prod_k \text{NegBin}(n_k) / \binom{N}{n_1 \cdots n_K}$).

The DM prior provides a "rich-get-richer" effect that compensates for the combinatorial explosion of allocations at higher $K$ (Stirling number $S(N,K)$ grows rapidly). Without it, the implicit allocation prior is uniform-per-label, which overwhelmingly favors high $K$.

**No separate prior on $K$.** The NegBin count model alone regularizes $K$.

**Code:** `_log_dm_partition` in `src/collapsed_moves.jl`.

### 4.2 Joint vs Marginal Mode (Round 11)

**Important:** The joint mode (highest-probability individual allocation) is typically at lower $K$ than the marginal mode $P(K \mid \text{data}) = \sum_z P(z, K \mid \text{data})$. This is because the DM prior assigns very low probability to each specific allocation at high $K$, but the marginal sums over an enormous number of such allocations.

Thermodynamic integration (Round 11) confirmed the marginal peaks near Q-PAINT's MAP $K$, while individual allocations at that $K$ score lower than allocations at lower $K$. This creates a K-mixing challenge for the collapsed sampler (see Section 9.1).

---

## 5. Collapsed Marginal Likelihood

### 5.1 Locmix prior version (grid-based, saddle-point)

$$\log\text{ML}_{\text{locmix},k} = (1 - n_k)\log(2\pi) - \tfrac{1}{2}S_k - \tfrac{1}{2}(Q_k - \eta_k^\top \Lambda_k^{-1}\eta_k) - \tfrac{1}{2}\log|\Lambda_k| + \log\pi_{\text{locmix}}(\hat{\theta}_k)$$

This is the saddle-point (Laplace plug-in) approximation to the exact integral. Round 11 confirmed the saddle-point error is $<0.02$ per cluster --- negligible.

**Code:** `log_marginal_likelihood_locmix(cs, grid)` in `src/cluster_stats.jl`.

### 5.2 Exact locmix integral (O(N) per cluster)

$$\text{ML}_k^{\text{exact}} = \frac{1}{N}\sum_{j=1}^{N} \text{ML}_{\text{flat}}(\text{cs}_k \cup \{\text{virtual}_j\})$$

Each term adds localization $j$ as a "virtual observation" and computes the flat-prior ML.

### 5.3 ClusterStats operations

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

The $(n_{-i,k} + \gamma)$ factor is the **DM allocation weight**. Sole occupants are skipped to maintain $K$.

**Code:** `gibbs_allocation_sweep!` in `src/collapsed_moves.jl`.

### 7.2 Split/Merge --- see [split-merge.md](split-merge.md)

### 7.3 Birth/Death --- see [birth-death.md](birth-death.md)

### 7.4 Hierarchical Updates ($\mu$ and $\alpha$)

MH with log-normal proposals every `hierarchical_interval` (default 100) iterations.

$$\mu' = \mu \cdot e^{\varepsilon}, \quad \varepsilon \sim \mathcal{N}(0, 0.3^2)$$

**Hyperpriors:** $\mu \sim \text{Gamma}(2, 5)$, $\alpha \sim \text{Gamma}(2, 1)$.
**Bounds:** $\mu \in [1, 500]$, $\alpha \in [0.5, 50]$.

**Code:** `_update_mu_collapsed`, `_update_shape_collapsed` in `src/hierarchical.jl`.

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

---

## 10. Partitioned BaGoL

Precision-weighted DBSCAN partitioning, parallel chains per partition, boundary deduplication via Hungarian matching. See `src/partition.jl`, `src/partitioned.jl`.

---

## 11. References

1. Fazel, M. *et al.* Nature Communications **13**, 7152 (2022).
2. Miller & Harrison. JASA **113**(521):340--356 (2018).
3. Jain & Neal. JCGS **13**(1):158--182 (2004).
4. Green. Biometrika **82**(4):711--732 (1995).
5. Dahl. In *Bayesian Inference for Gene Expression and Proteomics*, 201--218 (2006).
6. Rastelli & Friel. Statistics and Computing **28**(5):1169--1186 (2018).

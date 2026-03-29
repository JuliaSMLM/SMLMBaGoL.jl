# Mathematical Reference: SMLMBaGoL Collapsed Gibbs Sampler

*Authoritative specification for the `locmix-grid` branch.
Update this document when the algorithm changes.*

---

## 1. Problem Context

In single-molecule localization microscopy (SMLM), fluorescent emitters activate stochastically across camera frames. Each activation yields a **localization** $(d_i, \Sigma_i)$: a 2D position with known heteroscedastic uncertainty. A single emitter produces multiple localizations; the number $K$ of physical emitters is unknown.

**Given** $N$ noisy localizations, **determine** $K$, the allocation $z$ of localizations to emitters, and each emitter's position $\theta_k$ with sub-localization precision.

This is hard because: (a) $K$ is unknown, (b) each localization has different precision, (c) emitter separation can be smaller than localization uncertainty, and (d) blinking statistics vary by labeling chemistry (geometric for dSTORM, peaked for DNA-PAINT).

**BaGoL** treats this as a Bayesian mixture model with unknown components. The **collapsed** formulation integrates out emitter positions $\theta_k$ analytically, reducing the state to the discrete allocation vector $z$. Typical precision improvement: 10--50$\times$ over individual localizations.

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
| $p$ | derived | $\alpha/(\alpha+\mu)$ | NegBin success probability |

### Derived quantities

| Symbol | Description |
|--------|-------------|
| $\hat{\theta}_k = \Lambda_k^{-1}\eta_k$ | Posterior mean position |
| $\Sigma_{\text{post},k} = \Lambda_k^{-1}$ | Posterior covariance |
| $\pi_{\text{locmix}}(\theta) = \frac{1}{N}\sum_j \mathcal{N}(\theta; d_j, \Sigma_j)$ | Locmix spatial prior |

---

## 3. Generative Model

$$K \sim (\text{no prior --- regularized by count model})$$

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

$$\boxed{P(z, K \mid \text{data}) \;\propto\; P_{\text{count}}(N \mid K) \;\times\; \prod_{k=1}^{K} \text{ML}_{\text{locmix},k}(z)}$$

where:

- $P_{\text{count}}(N \mid K) = \text{NegBin}(N;\; K\alpha,\; p)$ --- total count model
- $\text{ML}_{\text{locmix},k}$ --- collapsed marginal likelihood under locmix prior (Section 5)

**No partition prior** $P_{\text{partition}}(z \mid K)$ appears in the current formulation. The allocation prior $P(z \mid K)$ is flat: all labelings with $K$ non-empty groups are equally likely. The Gibbs conditional is therefore purely spatial.

**No prior on** $K$. The NegBin likelihood $P(N \mid K)$ alone regularizes $K$; it has a well-defined mode near $K = N/\mu$ and penalizes both under- and over-splitting. There is no Poisson($\lambda_K$) prior because the locmix spatial prior eliminates the spatial Poisson process.

**Code:** `DecoupledTarget` in `src/diagnostics/target.jl`. The `MFMTarget` (with DM partition prior) is retained for diagnostic comparison.

---

## 5. Collapsed Marginal Likelihood

### 5.1 Uniform prior version

Integrating $\theta_k$ against $P(\theta_k) = 1/|R|$ over region $R$ (assuming the posterior is interior to $R$, i.e., $\Phi_R \approx 1$):

$$\log P(D_k \mid R) = (1 - n_k)\log(2\pi) - \tfrac{1}{2}S_k - \tfrac{1}{2}(Q_k - \eta_k^\top \Lambda_k^{-1} \eta_k) - \tfrac{1}{2}\log|\Lambda_k| - \log|R|$$

**Code:** `log_marginal_likelihood(cs, log_area)` in `src/cluster_stats.jl`.

### 5.2 Locmix prior version (grid-based, current branch)

Replace $-\log|R|$ with $\log\pi_{\text{locmix}}(\hat{\theta}_k)$:

$$\log\text{ML}_{\text{locmix},k} = (1 - n_k)\log(2\pi) - \tfrac{1}{2}S_k - \tfrac{1}{2}(Q_k - \eta_k^\top \Lambda_k^{-1}\eta_k) - \tfrac{1}{2}\log|\Lambda_k| + \log\pi_{\text{locmix}}(\hat{\theta}_k)$$

This is the saddle-point (Laplace plug-in) approximation to the exact integral:

$$\text{ML}_k^{\text{exact}} = \int \left[\prod_i \mathcal{N}(d_i;\theta,\Sigma_i)\right] \pi_{\text{locmix}}(\theta)\,d\theta$$

The exact version factors as:

$$\log\text{ML}_k^{\text{exact}} = \log\text{ML}_{\text{flat}}(\text{cs}_k) + \log E_{\text{post}}\!\left[\pi_{\text{locmix}}(\theta)\right]$$

where $\text{ML}_{\text{flat}}$ is the flat-prior marginal and $E_{\text{post}}[\cdot]$ is the expectation under the improper flat posterior $q(\theta) \propto \prod_i \mathcal{N}(d_i;\theta,\Sigma_i)$. The grid approximation replaces $E_{\text{post}}[\pi(\theta)]$ with $\pi(\hat{\theta}_k)$. See Section 10.1 for error analysis.

**Code:** `log_marginal_likelihood_locmix(cs, grid)` in `src/cluster_stats.jl`.

### 5.3 Exact locmix integral (O(N) per cluster)

The locmix prior is a mixture of Gaussians conjugate with the Gaussian likelihood:

$$\text{ML}_k = \frac{1}{N}\sum_{j=1}^{N} \text{ML}_{\text{flat}}(\text{cs}_k \cup \{\text{virtual}_j\})$$

Each term adds localization $j$ as a "virtual observation" (representing the prior component $\mathcal{N}(\theta; d_j, \Sigma_j)$) and computes the flat-prior ML. This is exact, requires no grid, but costs $O(N)$ per cluster instead of $O(1)$.

**Code:** `log_ml_locmix(cs, loc_precs)` in `src/cluster_stats.jl`.

### 5.4 ClusterStats operations

All operations are $O(1)$ via additive sufficient statistics:

$$\text{add\_loc}(k, i): \quad \Lambda_k \mathrel{+}= \Lambda_i,\;\; \eta_k \mathrel{+}= \Lambda_i d_i,\;\; Q_k \mathrel{+}= d_i^\top \Lambda_i d_i,\;\; S_k \mathrel{+}= \log|\Sigma_i|,\;\; n_k \mathrel{+}= 1$$

`remove_loc` is the exact inverse (subtract). `LocPrecision` caches the per-localization contributions, precomputed once at initialization.

**Code:** `ClusterStats`, `LocPrecision`, `add_loc`, `remove_loc` in `src/cluster_stats.jl`.

---

## 6. Localization Mixture Prior

### 6.1 Definition

$$\pi_{\text{locmix}}(\theta) = \frac{1}{N}\sum_{j=1}^{N}\mathcal{N}(\theta;\, d_j,\, \Sigma_j)$$

Each localization contributes a Gaussian component centered at its measured position with its measurement covariance. This concentrates prior mass near the data.

**Key properties:**

- **No area dependence.** The $-\log|R|$ penalty from the uniform prior is eliminated.
- **Automatic scale.** For co-located emitters, the prior concentrates at $\sigma$-scale.
- **Split-neutral at $d=0$.** Both sub-clusters see the same concentrated prior mass.
- **Split-positive at $d > 0$.** Each cluster's prior peaks near its own data.

### 6.2 Grid approximation (`LocmixGrid`)

At initialization, evaluate $\log\pi_{\text{locmix}}(\theta)$ on a 2D grid covering the data bounding box plus $3\sigma_{\max}$ margin, with resolution $\sigma_{\max} / 2$. During MCMC, evaluate at a cluster's posterior mean $\hat{\theta}_k$ via bilinear interpolation in $O(1)$.

The bilinear interpolation operates in **log-space** on the four surrounding grid values.

**Grid construction cost:** $O(N \times n_x \times n_y)$, run once per partition.

**Code:** `LocmixGrid`, `build_locmix_grid`, `log_prior_locmix` in `src/cluster_stats.jl`.

### 6.3 Approximation error

The grid approximation replaces $E_{\text{post}}[\pi(\theta)]$ with $\pi(\hat{\theta})$. The error:

$$\varepsilon = \log E_{\text{post}}[\pi(\theta)] - \log\pi(\hat{\theta})$$

vanishes when (a) the posterior is concentrated (large $n_k$) or (b) the prior is locally flat. It is largest in the close-pair regime ($d \sim \sigma$) where the merged posterior straddles two locmix peaks and $\hat{\theta}$ falls in a saddle. See Section 10.1.

---

## 7. MCMC Moves

### 7.1 Gibbs Allocation Sweep (K fixed)

For each localization $i$ in random order (Fisher-Yates shuffle), reassign among the $K$ active clusters:

$$P(z_i = k \mid z_{-i}, K, \text{data}) \;\propto\; p_{\text{pred}}(d_i \mid D_k^{-i})$$

where the **predictive** is:

$$\log p_{\text{pred}}(d_i \mid D_k^{-i}) = \log\text{ML}_{\text{locmix}}(D_k^{-i} \cup \{d_i\}) - \log\text{ML}_{\text{locmix}}(D_k^{-i})$$

For an **empty cluster**: $p_{\text{pred}}(d_i) = \pi_{\text{locmix}}(d_i)$ (the prior density at $d_i$).

**Sole occupant protection:** If $n_k = 1$ and $z_i = k$, skip $i$ to maintain $K$ during the sweep. Only split/merge changes $K$.

**No count weights.** Under the decoupled model (no partition prior), the conditional depends only on the spatial predictive. No CRP $(n_{k,-i} + \gamma)$ factor appears.

**Cost:** $O(N \cdot K)$ per sweep (each predictive is $O(1)$ with grid locmix).

**Code:** `gibbs_allocation_sweep!` in `src/collapsed_moves.jl`.

### 7.2 Split/Merge (Direct K Sampling + Spatial MH)

A three-phase trans-dimensional move.

#### Phase 1: Propose $K'$ from count-model posterior

$$\pi_{\text{count}}(K) \;\propto\; P(N \mid K, \alpha, \mu) = \text{NegBin}(N;\; K\alpha,\; p)$$

Evaluate for $K = 1, \ldots, K_{\max}$ where $K_{\max} = \max(2K, \min(N, 30))$. Sample $K'$ from this discrete distribution. If $K' = K$: no-op.

**No Poisson prior on $K$.** The NegBin likelihood alone regularizes $K$.

**Code:** `_log_count_posterior(K, N, shape, μ)` in `src/collapsed_moves.jl`.

#### Phase 2: Heuristic restructuring + Gibbs relaxation

**Split** ($K' > K$): Repeat $K' - K$ times: find the largest active cluster, move $\sim$half its locs (random coin-flip per loc) to a new cluster.

**Merge** ($K' < K$): Repeat $K - K'$ times: find the smallest active cluster, merge into its nearest neighbor by posterior mean distance.

**Relaxation:** Run 5 Gibbs allocation sweeps at the new $K'$. This lets localizations migrate to spatially preferred clusters before the MH evaluation. Without relaxation, even correct splits are rejected because the random allocation has poor spatial likelihood.

**Code:** `_add_clusters!`, `_remove_clusters!` in `src/collapsed_moves.jl`.

#### Phase 3: Spatial MH acceptance

$$\Delta_{\text{fit}} = \sum_k \log\text{ML}_{\text{locmix},k}^{\text{new}} - \sum_k \log\text{ML}_{\text{locmix},k}^{\text{old}}$$

Accept if $\Delta_{\text{fit}} \geq 0$ or $u < \exp(\Delta_{\text{fit}})$. On rejection, full rollback from saved state.

**No area correction** needed under the locmix prior (no $-\log|R|$ terms to cancel). The count model is already incorporated in the $K'$ proposal; the MH step corrects only for spatial fit.

**Properties of $\Delta_{\text{fit}}$:**

| Regime | $\Delta_{\text{fit}}$ | What decides $K$ |
|--------|----------------------|------------------|
| Co-located ($d=0$) | $\approx 0$ | Count model only ($\equiv$ Q-PAINT) |
| Separated ($d \gg \sigma$) | $> 0$ for correct splits | Spatial info adds to counts |

**Code:** `propose_split_merge!` in `src/collapsed_moves.jl`.

### 7.3 Move distribution

| Move | Probability | Type | $K$ change |
|------|-------------|------|-----------|
| Gibbs allocation sweep | 50% | Exact (K fixed) | No |
| Split/merge (K proposal) | 50% | Count proposal + spatial MH | Yes |

### 7.4 Hierarchical Updates (mu and shape)

Metropolis-Hastings with log-normal proposals, run every `hierarchical_interval` iterations (default 100).

**mu update:**

$$\mu' = \mu \cdot e^{\varepsilon}, \quad \varepsilon \sim \mathcal{N}(0, 0.3^2)$$

$$\log\alpha_{\text{MH}} = \underbrace{\sum_k \log\text{NegBin}(n_k;\alpha,p') - \sum_k \log\text{NegBin}(n_k;\alpha,p)}_{\text{likelihood ratio}} + \underbrace{\log\text{Gamma}(\mu'; a_\mu, b_\mu) - \log\text{Gamma}(\mu; a_\mu, b_\mu)}_{\text{prior ratio}} + \underbrace{\log\mu' - \log\mu}_{\text{Jacobian}}$$

where $p = \alpha/(\alpha+\mu)$, $p' = \alpha/(\alpha+\mu')$. Bounds: $\mu \in [1, 500]$.

**shape update:** Same structure, evaluating NegBin under $\alpha' = \text{shape}'$. Bounds: $\alpha \in [0.5, 50]$.

**Hyperpriors (defaults):** $\mu \sim \text{Gamma}(2, 5)$, $\alpha \sim \text{Gamma}(2, 1)$.

**learn_distribution options:** `true` (learn both), `false` (fix both), `:mu` (learn $\mu$ only), `:shape` (learn $\alpha$ only).

**Code:** `_update_mu_collapsed`, `_update_shape_collapsed` in `src/hierarchical.jl`.

---

## 8. MAP-N Estimation

### 8.1 Mode selection

Build histogram of $K$ across post-burn-in samples. $\text{MAP-N} = \arg\max$ of histogram, with 3-bin smoothing to break near-ties.

### 8.2 Dahl consensus (preferred default)

Find the stored assignment sample closest to the posterior similarity matrix:

$$z_{\text{Dahl}} = \arg\min_t \sum_{i<j}\left(\mathbf{1}[z_i^t = z_j^t] - \text{PSM}_{ij}\right)^2$$

where $\text{PSM}_{ij} = \frac{1}{T}\sum_t \mathbf{1}[z_i^t = z_j^t]$.

**Code:** `estimate_dahl(samples, locs, psm)` in `src/mapn.jl`.

### 8.3 Overlap-Hungarian matching

Uses Dahl assignments as template, then refines via overlap-based Hungarian:

1. $K = K_{\text{Dahl}}$ from Dahl partition
2. Filter posterior samples to $K = K_{\text{Dahl}}$
3. For each: overlap-Hungarian $\to$ per-cluster overlap gate ($\geq 50\%$)
4. Position: mean of well-matched posterior means
5. Uncertainty via law of total variance:

$$\Sigma_{\text{total}} = \underbrace{E[\text{Var}(\theta \mid z)]}_{\text{analytic (ClusterStats)}} + \underbrace{\text{Var}[E(\theta \mid z)]}_{\text{sample variance across MCMC}}$$

**Code:** `estimate_mapn_overlap(samples, locs, dahl_assignments)` in `src/mapn.jl`.

### 8.4 Other methods

| Method | Function | Notes |
|--------|----------|-------|
| Collapsed (histogram + Hungarian) | `estimate_mapn_collapsed` | Mode of $K$ histogram + iterative position matching |
| PSM thresholding | `estimate_mapn_psm` | Union-Find on PSM $\geq 0.5$ |
| VI greedy | `estimate_vi_greedy` | Minimize expected variation of information |
| Final-state only | `extract_emitters` | Single sample, fallback only |

---

## 9. Partitioned BaGoL

### 9.1 Precision-weighted DBSCAN

Two localizations are neighbors if:

$$\frac{\|d_i - d_j\|}{\sigma_i + \sigma_j} < \text{partition\_sigma}$$

Default `partition_sigma = 3.0`. Clusters below `min_size` are dropped as noise; clusters above `max_size` are sub-split.

**Code:** `partition_locs` in `src/partition.jl`.

### 9.2 Parallel chain execution

Each partition runs an independent collapsed Gibbs chain (via `Threads.@threads`). Hierarchical parameters $\mu$ and $\alpha$ are shared and updated at `sync_interval` (default 500) iterations: pool per-cluster counts across all partition states, run a single global MH step, broadcast accepted values.

**Code:** `_update_mu_collapsed_global!`, `_update_shape_collapsed_global!` in `src/hierarchical.jl`.

### 9.3 Boundary deduplication

After independent processing, emitters near partition boundaries are deduplicated via Hungarian matching based on position proximity. Boundary localizations are flagged using a margin of `partition_sigma * median(σ)`.

**Code:** `src/partitioned.jl`.

---

## 10. Known Approximations

### 10.1 Grid-based locmix (saddle-point vs exact integral)

The grid approximation evaluates $\pi_{\text{locmix}}(\hat{\theta}_k)$ instead of $E_{\text{post}}[\pi(\theta)]$. The error is largest for close pairs ($d \sim \sigma$) where the merged cluster's posterior straddles two locmix peaks and $\hat{\theta}_k$ sits in a saddle between them, systematically underestimating the merged ML relative to the split ML.

For well-separated ($d \gg \sigma$) or truly co-located ($d = 0$) emitters, the approximation is accurate. The exact $O(N)$ version (`log_ml_locmix`) does not suffer from this but is too expensive for the Gibbs hot path.

### 10.2 Heuristic split/merge (asymmetric proposals)

The split/merge is a heuristic: random half-split of the largest cluster (split) or nearest-neighbor merge of the smallest cluster (merge). The 5-sweep Gibbs relaxation mitigates poor initial allocations but does not yield a well-defined proposal density. The MH correction accounts for the spatial fit change but not the proposal asymmetry. This is an **approximate** detailed-balance move.

**Consequence:** On the `smc-split` branch (which uses a proper SMC proposal with MFM partition prior), brute-force enumeration (N=6--8) showed the sampler systematically under-visits $K \geq 2$ partitions by 1.3--25$\times$. Root cause: restricted proposal support from deterministic seeding, not incorrect acceptance arithmetic (spot-checked to machine precision). The current heuristic approach trades formal DB guarantees for unrestricted $K$-reachability and practical mixing.

### 10.3 No partition prior (oversplitting risk at $d=0$)

Without the DM partition prior $P_{\text{partition}}(z \mid K)$, all allocations with the same $K$ are equally likely. This removes the combinatorial penalty for splitting (which stabilizes $K$ at the cost of under-counting --- see Section 11). The resulting target is purely driven by the count model at $d=0$ and by spatial evidence at $d > 0$.

**Risk:** The Stirling number $S(N,K)$ of labeled allocations grows explosively with $K$. Without a partition prior to offset this, the flat allocation prior may admit more total posterior mass at higher $K$ than the data supports. The count model $P(N \mid K)$ must provide sufficient regularization.

### 10.4 Current mu in K proposal

The current branch uses the adaptive $\mu$ (not a fixed $\mu_0$) in the count-model $K$ proposal. Earlier branches used fixed $\mu_0 = \mu_{\text{prior\_shape}} \times \mu_{\text{prior\_scale}}$ to prevent a positive feedback loop ($K\!\uparrow \to \mu\!\downarrow \to \lambda_K\!\uparrow \to K\!\uparrow\!\uparrow$). The current branch accepts this coupling; monitor for runaway $K$ in practice.

---

## 11. Algorithm Variant History

| Branch | Variant | Key feature | DB? | Mixing | Outcome |
|--------|---------|-------------|-----|--------|---------|
| `rjmcmc` | Full RJMCMC | Explicit positions, birth/death | Exact | Slow | Replaced by collapsed |
| `collapsed-gibbs` | Collapsed + uniform | Analytical position integration | $\approx$ | OK | Area sensitivity ($-\log\|R\|$) |
| `loc-mixture-prior` | + locmix prior | $O(N^2 K)$, no area penalty | $\approx$ | OK | Split-neutral at $d=0$, slow |
| `neighbor-locmix` | + KD-tree accel | $O(N \cdot K \cdot |A|)$ filtered | $\approx$ | OK | Performance viable |
| `overlap-mapn` | + better MAP-N | Dahl, overlap-Hungarian | $\approx$ | OK | Better position estimates |
| `smc-split` | + MFM + SMC proposal | Partition prior, proper DB | Exact | Restricted | Reachability problem (Section 10.2) |
| **`locmix-grid`** | **+ grid prior** | **$O(1)$ per cluster** | $\approx$ | **OK** | **Current branch** |
| (PPM variants) | Cancel Occam penalty | $|\Lambda|^{1/2}$ coupling | --- | --- | Runaway splitting |
| (relaxation) | + Gibbs relaxation | Jain-Neal restricted sweeps | Design only | --- | Not implemented |

**Key tradeoff** across branches: the `smc-split` branch had correct DB but restricted reachability (biased $K$ by 1.3--25$\times$ in brute-force tests). The `locmix-grid` branch uses a heuristic proposal with unrestricted reachability and approximate DB. Empirically, the heuristic approach matches or exceeds Q-PAINT performance at all separations.

---

## 12. Open Questions

1. **Should we include the DM partition prior?** The MFM partition prior $P_{\text{partition}}(z \mid K)$ with $\gamma = \alpha$ is the natural companion to the NegBin count model ($P_{\text{count}} \times P_{\text{partition}} = P(z \mid K)$ exactly; see `dev/math_refs/factorization_check.md`). But on `smc-split` it caused under-counting at $d=0$ due to proposal reachability issues.

2. **How to fix split/merge for exact DB without restricting reachability?** The Jain-Neal restricted Gibbs relaxation (sweeps within the split proposal) is the standard fix but has not been implemented. It would let the two seeded sub-clusters exchange members, making all partitions reachable.

3. **Can we get better $K$ proposals than count-model-only?** The current proposal ignores spatial structure when proposing $K'$. A proposal informed by spatial clustering (e.g., using nearest-neighbor distances) could improve acceptance rates.

4. **Grid approximation error at $d \sim \sigma$.** The saddle-point error (Section 10.1) is theoretically the worst in the close-pair regime, but empirically $\Delta_{\text{spatial}}$ is not the dominant blocker. Quantifying the error magnitude and its effect on MAP-N accuracy requires systematic comparison with the exact $O(N)$ version.

5. **Fixed vs adaptive $\mu$ in K proposal.** The current branch uses adaptive $\mu$. The feedback loop risk needs monitoring, especially for large partitions with many clusters.

---

## 13. Diagnostics Module

The `src/diagnostics/` module provides algorithm-agnostic tools for validating sampler correctness:

| File | Purpose |
|------|---------|
| `target.jl` | `AbstractTargetDensity` hierarchy: `DecoupledTarget`, `MFMTarget`, `UniformPriorTarget` |
| `count_model.jl` | Count-model utilities, Q-PAINT MAP-K oracle |
| `partition_metrics.jl` | Variation of information (Meila 2007), expected posterior loss (EPL) |
| `enumeration.jl` | Brute-force enumeration of all labeled partitions for small $N$ |
| `detailed_balance.jl` | Spot-check detailed balance on state pairs |
| `chain_diagnostics.jl` | ESS, $\hat{R}$, autocorrelation |

**Target density abstraction:** Different algorithm variants target different posteriors. The `AbstractTargetDensity` interface lets diagnostic tools work with any variant by parameterizing `log_target(td, z, loc_precs, grid, μ, shape)`.

**Brute-force validation:** For small $N$ (6--8), `enumerate_partitions` computes the exact posterior over all labeled partitions and compares to MCMC visit frequencies. This is the gold standard for detecting DB violations.

**VI decomposition:** `variation_of_information(z_1, z_2)` returns `(vi_total, overseg, underseg)` where `overseg = H(z_1 \mid z_2)` measures splits of true clusters and `underseg = H(z_2 \mid z_1)` measures merges.

---

## 14. References

1. Fazel, M., Greer, M.D., Hsu, H. *et al.* High-Precision Estimation of Emitter Positions using Bayesian Grouping of Localizations. *Nature Communications* **13**, 7152 (2022). [doi:10.1038/s41467-022-34894-2](https://doi.org/10.1038/s41467-022-34894-2)

2. Miller, J.W. & Harrison, M.T. Mixture Models with a Prior on the Number of Components. *JASA* **113**(521):340--356 (2018).

3. Meila, M. Comparing Clusterings---an Information Based Distance. *J. Multivariate Analysis* **98**(5):873--895 (2007).

4. Jain, S. & Neal, R.M. A Split-Merge Markov Chain Monte Carlo Procedure for the Dirichlet Process Mixture Model. *JCGS* **13**(1):158--182 (2004).

5. Green, P.J. Reversible Jump Markov Chain Monte Carlo Computation and Bayesian Model Determination. *Biometrika* **82**(4):711--732 (1995).

6. Dahl, D.B. Model-Based Clustering for Expression Data via a Dirichlet Process Mixture Model. In *Bayesian Inference for Gene Expression and Proteomics*, 201--218 (2006).

7. Rastelli, R. & Friel, N. Optimal Bayesian Estimators for Latent Variable Cluster Models. *Statistics and Computing* **28**(5):1169--1186 (2018).

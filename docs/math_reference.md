# Mathematical Reference: SMLMBaGoL Collapsed Gibbs Sampler

*Authoritative specification for `main` branch (post-Round 11, 2026-03-30).
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

**No separate prior on $K$.** The NegBin count model alone regularizes $K$. No Poisson prior is needed because the locmix spatial prior eliminates the spatial Poisson process.

**Code:** `_log_dm_partition` in `src/collapsed_moves.jl`.

### 4.2 Joint vs Marginal Mode (Round 11)

**Important:** The joint mode (highest-probability individual allocation) is typically at lower $K$ than the marginal mode $P(K \mid \text{data}) = \sum_z P(z, K \mid \text{data})$. This is because the DM prior assigns very low probability to each specific allocation at high $K$, but the marginal sums over an enormous number of such allocations.

Thermodynamic integration (Round 11) confirmed the marginal peaks near Q-PAINT's MAP $K$, while individual allocations at that $K$ score lower than allocations at lower $K$. This creates a K-mixing challenge for the collapsed sampler (see Section 10.1).

---

## 5. Collapsed Marginal Likelihood

### 5.1 Locmix prior version (grid-based, saddle-point)

$$\log\text{ML}_{\text{locmix},k} = (1 - n_k)\log(2\pi) - \tfrac{1}{2}S_k - \tfrac{1}{2}(Q_k - \eta_k^\top \Lambda_k^{-1}\eta_k) - \tfrac{1}{2}\log|\Lambda_k| + \log\pi_{\text{locmix}}(\hat{\theta}_k)$$

This is the saddle-point (Laplace plug-in) approximation to the exact integral:

$$\text{ML}_k^{\text{exact}} = \int \left[\prod_i \mathcal{N}(d_i;\theta,\Sigma_i)\right] \pi_{\text{locmix}}(\theta)\,d\theta$$

Round 11 confirmed the saddle-point error is $<0.02$ per cluster --- negligible.

**Code:** `log_marginal_likelihood_locmix(cs, grid)` in `src/cluster_stats.jl`.

### 5.2 Exact locmix integral (O(N) per cluster)

$$\text{ML}_k^{\text{exact}} = \frac{1}{N}\sum_{j=1}^{N} \text{ML}_{\text{flat}}(\text{cs}_k \cup \{\text{virtual}_j\})$$

Each term adds localization $j$ as a "virtual observation" (representing the prior component $\mathcal{N}(\theta; d_j, \Sigma_j)$) and computes the flat-prior ML.

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
- **Automatic scale.** For co-located emitters, the prior concentrates at $\sigma$-scale.
- **Split-neutral at $d=0$.** Both sub-clusters see the same concentrated prior mass.
- **Split-positive at $d > 0$.** Each cluster's prior peaks near its own data.
- **Residual Occam effect.** The $-\frac{1}{2}\log|\Lambda_k|$ term creates a per-cluster penalty of $\approx -\log(n_k)$ that the locmix only partially compensates. This is a fundamental property of Bayesian marginal likelihood, not an approximation error (Section 10.1).

### 6.2 Grid approximation (`LocmixGrid`)

At initialization, evaluate $\log\pi_{\text{locmix}}(\theta)$ on a 2D grid covering the data bounding box plus $3\sigma_{\max}$ margin, with resolution $\sigma_{\max} / 2$. During MCMC, evaluate at $\hat{\theta}_k$ via bilinear interpolation in $O(1)$.

**Code:** `LocmixGrid`, `build_locmix_grid`, `log_prior_locmix` in `src/cluster_stats.jl`.

---

## 7. MCMC Moves

### 7.0 Move Selection

| Move | Probability | Function | $K$ change |
|------|-------------|----------|-----------|
| Gibbs allocation sweep | 50% | `gibbs_allocation_sweep!` | Fixed |
| Split/merge | 25% | `propose_split_merge!` | $\pm 1$ |
| Birth/death | 25% $\times$ `n_bd_substeps` | `propose_birth_death!` | $\pm 1$ |

**BD burst:** When birth/death is selected, `n_bd_substeps` (default 5) sequential BD attempts are made. Each is independent MH. This increases K-transition throughput at $\sim 2\times$ cost per outer iteration.

### 7.1 Gibbs Allocation Sweep ($K$ fixed)

For each localization $i$ in random order (Fisher-Yates shuffle), reassign among $K$ active clusters:

$$P(z_i = k \mid z_{-i}, K, \text{data}) \;\propto\; (n_{-i,k} + \gamma) \;\times\; p_{\text{pred}}(d_i \mid D_k^{-i})$$

The $(n_{-i,k} + \gamma)$ factor is the **Dirichlet-Multinomial allocation weight**, implementing the partition prior within the Gibbs conditional. Without it, the implicit prior is uniform-per-label, which strongly favors higher $K$ (combinatorial explosion).

The **predictive** is:

$$\log p_{\text{pred}}(d_i \mid D_k^{-i}) = \log\text{ML}_{\text{locmix}}(D_k^{-i} \cup \{d_i\}) - \log\text{ML}_{\text{locmix}}(D_k^{-i})$$

**Sole occupant protection:** If $n_k = 1$ and $z_i = k$, skip to maintain $K$.

**Code:** `gibbs_allocation_sweep!` in `src/collapsed_moves.jl`.

### 7.2 RJMCMC Split/Merge ($|\Delta K| = 1$)

#### Phase 1: Random split/merge selection

$$b_K = P(\text{split at } K), \quad d_K = 1 - b_K$$

Boundary handling: $K = 1 \Rightarrow b_K = 1$; $K \geq N \Rightarrow d_K = 1$; otherwise $b_K = d_K = 0.5$.

#### Phase 2: Execute split or merge

**Split ($K \to K+1$):**

1. Select parent cluster uniformly: $1/K$
2. Random seed selection: pick two members (probability $1/(m(m-1))$, cancels in MH)
3. Canonical ordering of remaining members (sorted by loc index)
4. Sequential predictive allocation (launch):

$$\log w_A = \log(n_A + \gamma) + \log p_{\text{pred}}(d_j \mid \text{sub-A})$$
$$\log w_B = \log(n_B + \gamma) + \log p_{\text{pred}}(d_j \mid \text{sub-B})$$

5. **Restricted Gibbs scans (Jain-Neal):** `n_restricted_scans - 1` intermediate sweeps (no density tracking) + 1 final sweep (density tracked). Only the final sweep's density enters the MH ratio. Default `n_restricted_scans = 5`.

**Merge ($K \to K-1$):**

1. Select pair uniformly: $1/\binom{K}{2}$
2. Random seed selection (matching bijection)
3. Compute reverse allocation density via Jain-Neal: launch $\to$ intermediate sweeps $\to$ transition density from intermediate to current allocation

#### Phase 3: MH acceptance

$$\log \alpha = \Delta_{\text{spatial}} + \Delta_{\text{partition}} + \Delta_{\text{proposal}} + \Delta_{\text{count}} + \Delta_{\text{move\_type}}$$

where:
- $\Delta_{\text{spatial}} = \sum_k \log\text{ML}_k^{\text{new}} - \sum_k \log\text{ML}_k^{\text{old}}$
- $\Delta_{\text{partition}} = \log P_{\text{DM}}(z' \mid K') - \log P_{\text{DM}}(z \mid K)$
- $\Delta_{\text{proposal}} = \log q_{\text{rev}} - \log q_{\text{fwd}}$
- $\Delta_{\text{count}} = \log P(N \mid K') - \log P(N \mid K)$ (uses fixed $\mu_0$, not adaptive)
- $\Delta_{\text{move\_type}} = \log(d_{K'}/b_K)$ for splits

**Fixed $\mu$ in count ratio:** The count-model ratio uses the initial $\mu$ (prior mean), not the adapted value, to prevent the $\mu$-$K$ positive feedback loop.

**Seed selection cancellation:** The seed density $1/(m(m-1))$ appears in both split and merge via the RJMCMC bijection and cancels. It must NOT be included explicitly.

**Code:** `propose_split_merge!` in `src/collapsed_moves.jl`.

### 7.3 Birth/Death Moves ($K \pm 1$, incremental)

Birth/death provides cheaper K-transitions than split/merge. The DM penalty per birth is $\approx -1.2$ (vs $-2.5$ to $-5$ per split).

**Birth ($K \to K+1$):**
1. Pick random non-sole-occupant loc: $1/N_{\text{eligible}}$
2. Detach as singleton cluster

**Death ($K \to K-1$):**
1. Pick random singleton: $1/n_{\text{singletons}}$
2. Absorb into destination via DM-weighted predictive: $w(k) = (n_k + \gamma) \times \text{pred}(i \mid k)$

**Proposal densities:**

$$q_{\text{birth}} = p_{\text{birth}} \times \frac{1}{N_{\text{eligible}}}$$

$$q_{\text{death}} = p_{\text{death}} \times \frac{1}{n_{\text{singletons}}} \times \frac{w(\text{dest})}{\sum_k w(k)}$$

**MH acceptance:**

$$\log \alpha = \Delta_{\text{spatial}} + \Delta_{\text{partition}} + \Delta_{\text{proposal}} + \Delta_{\text{count}}$$

**Code:** `propose_birth_death!` in `src/collapsed_moves.jl`.

### 7.4 Hierarchical Updates ($\mu$ and $\alpha$)

Metropolis-Hastings with log-normal proposals, run every `hierarchical_interval` iterations (default 100).

**$\mu$ update:**

$$\mu' = \mu \cdot e^{\varepsilon}, \quad \varepsilon \sim \mathcal{N}(0, 0.3^2)$$

$$\log\alpha_{\text{MH}} = \sum_k [\log\text{NegBin}(n_k;\alpha,p') - \log\text{NegBin}(n_k;\alpha,p)] + [\log P(\mu') - \log P(\mu)] + [\log\mu' - \log\mu]$$

**Hyperpriors:** $\mu \sim \text{Gamma}(2, 5)$, $\alpha \sim \text{Gamma}(2, 1)$.
**Bounds:** $\mu \in [1, 500]$, $\alpha \in [0.5, 50]$.

**Code:** `_update_mu_collapsed`, `_update_shape_collapsed` in `src/hierarchical.jl`.

---

## 8. MAP-N Estimation

### 8.1 Dahl consensus (preferred default)

$$z_{\text{Dahl}} = \arg\min_t \sum_{i<j}\left(\mathbf{1}[z_i^t = z_j^t] - \text{PSM}_{ij}\right)^2$$

where $\text{PSM}_{ij} = \frac{1}{T}\sum_t \mathbf{1}[z_i^t = z_j^t]$.

**Code:** `estimate_dahl(samples, locs, psm)` in `src/mapn.jl`.

### 8.2 Overlap-Hungarian matching

Uses Dahl assignments as template, then refines via overlap-based Hungarian:

1. $K = K_{\text{Dahl}}$ from Dahl partition
2. Filter posterior samples to $K = K_{\text{Dahl}}$
3. For each: overlap-Hungarian $\to$ per-cluster overlap gate ($\geq 50\%$)
4. Position: mean of well-matched posterior means
5. Uncertainty via law of total variance:

$$\Sigma_{\text{total}} = \underbrace{E[\text{Var}(\theta \mid z)]}_{\text{analytic (ClusterStats)}} + \underbrace{\text{Var}[E(\theta \mid z)]}_{\text{sample variance across MCMC}}$$

**Code:** `estimate_mapn_overlap(samples, locs, dahl_assignments)` in `src/mapn.jl`.

### 8.3 Other methods

| Method | Function | Notes |
|--------|----------|-------|
| Collapsed (histogram + Hungarian) | `estimate_mapn_collapsed` | Mode of $K$ histogram + iterative matching |
| PSM thresholding | `estimate_mapn_psm` | Union-Find on PSM $\geq 0.5$ |
| VI greedy | `estimate_vi_greedy` | Minimize expected variation of information |
| Final-state only | `extract_emitters` | Single sample, fallback only |

---

## 9. Partitioned BaGoL

### 9.1 Precision-weighted DBSCAN

Two localizations are neighbors if:

$$\frac{\|d_i - d_j\|}{\sigma_i + \sigma_j} < \text{partition\_sigma}$$

Default `partition_sigma = 3.0`.

**Code:** `partition_locs` in `src/partition.jl`.

### 9.2 Parallel chain execution

Each partition runs an independent collapsed Gibbs chain (via `Threads.@threads`). Hierarchical parameters $\mu$ and $\alpha$ are shared and updated at `sync_interval` (default 500) iterations.

### 9.3 Boundary deduplication

After independent processing, emitters near partition boundaries are deduplicated via Hungarian matching.

**Code:** `src/partitioned.jl`.

---

## 10. Known Limitations

### 10.1 K-mixing at large N (Round 11)

**The collapsed sampler has a fundamental K-mixing limitation at large $N$.** The joint posterior $P(z, K \mid \text{data})$ has its mode at lower $K$ than the marginal $P(K \mid \text{data})$, because the DM prior assigns very low probability to each specific allocation at high $K$, but there are exponentially many allocations.

Thermodynamic integration (Round 11) confirmed the marginal $P(K)$ peaks near Q-PAINT's MAP $K$ (e.g., $K \approx 7\text{--}10$ for $N=40$ octamers). But the sampler converges to $K \approx 3\text{--}6$ from both high and low initial $K$ --- a severe mixing failure.

**Root cause:** Each K-increasing move (birth, split) faces the per-allocation DM + Occam penalty. Even though the marginal favors high $K$, the chain's transition moves see the per-allocation landscape (which favors low $K$). The mixing time scales exponentially with $N$.

**Why Fazel's approach doesn't suffer:** The original Fazel et al. RJMCMC does NOT collapse positions. The chain state includes explicit emitter positions $\theta_k$. At good positions, the joint $P(\theta, z, K)$ has high probability at the correct $K$, so the chain can stay there. Collapsing creates the Occam factor that penalizes individual high-$K$ allocations.

**Brute-force validation:** At $N = 6$, the mixing barriers are small enough for BD burst to overcome --- 4/4 brute-force PASS. The target distribution is correct at all $N$; only the mixing degrades.

### 10.2 Grid-based locmix (saddle-point vs exact)

The grid approximation replaces $E_{\text{post}}[\pi(\theta)]$ with $\pi(\hat{\theta})$. Round 11 measured the error at $<0.02$ per cluster for octamers at NN$= 1.9\sigma$. **Not a significant source of bias.**

### 10.3 Hierarchical learner feedback

When the sampler under-splits ($K$ too low), each cluster has more locs $\to$ the hierarchical learner infers higher $\mu$ and $\alpha$ $\to$ the count model shifts toward lower $K$ $\to$ reinforcing under-splitting. This is a secondary effect caused by the K-mixing failure.

---

## 11. Constants and Magic Numbers

| Value | Where | What | Justification |
|-------|-------|------|---------------|
| 50/25/25 | `collapsed_sampler.jl` | Gibbs/SM/BD ratio | BD adds cheap K-mobility |
| 5 | `collapsed_sampler.jl` | `n_bd_substeps` | 4/4 brute-force PASS at $N=6$ |
| 50/50 | `collapsed_moves.jl` | Split/merge coin flip | Equal K$\pm$1; boundary-aware |
| $\gamma = \alpha$ | `collapsed_moves.jl` | DM concentration | Derived from NegBin count model |
| 5 | `collapsed_moves.jl` | `n_restricted_scans` | 4 intermediate + 1 final Jain-Neal sweep |
| 0.3 | `hierarchical.jl` | Log-normal proposal $\sigma$ | ~25--35% acceptance |
| [1, 500] | `hierarchical.jl` | $\mu$ bounds | Physical |
| [0.5, 50] | `hierarchical.jl` | $\alpha$ bounds | Physical |

---

## 12. Open Questions

1. **How to improve K-mixing at large $N$?** The fundamental limitation is that the collapsed posterior's joint mode differs from the marginal mode. Possible approaches: (a) un-collapse for K-transitions (return to explicit positions for birth/death), (b) parallel tempering across $K$, (c) SMC-based K proposals, (d) hybrid collapsed/uncollapsed moves.

2. **Should we un-collapse entirely?** Fazel's uncollapsed RJMCMC doesn't have the K-mixing problem because the chain visits $(\theta, z, K)$ states where good positions make high-$K$ states attractive. The cost: slower within-K mixing (must sample positions).

3. **Octamer marginal beyond $K=10$.** TI showed P($K$) increasing through $K=10$ for the octamer at NN$=1.9\sigma$. Does it peak at $K=10$ or continue? Need TI at higher $K$ or SMC estimate.

---

## 13. References

1. Fazel, M., Greer, M.D., Hsu, H. *et al.* High-Precision Estimation of Emitter Positions using Bayesian Grouping of Localizations. *Nature Communications* **13**, 7152 (2022).

2. Miller, J.W. & Harrison, M.T. Mixture Models with a Prior on the Number of Components. *JASA* **113**(521):340--356 (2018).

3. Meila, M. Comparing Clusterings---an Information Based Distance. *J. Multivariate Analysis* **98**(5):873--895 (2007).

4. Jain, S. & Neal, R.M. A Split-Merge Markov Chain Monte Carlo Procedure for the Dirichlet Process Mixture Model. *JCGS* **13**(1):158--182 (2004).

5. Green, P.J. Reversible Jump Markov Chain Monte Carlo Computation and Bayesian Model Determination. *Biometrika* **82**(4):711--732 (1995).

6. Dahl, D.B. Model-Based Clustering for Expression Data via a Dirichlet Process Mixture Model. In *Bayesian Inference for Gene Expression and Proteomics*, 201--218 (2006).

7. Rastelli, R. & Friel, N. Optimal Bayesian Estimators for Latent Variable Cluster Models. *Statistics and Computing* **28**(5):1169--1186 (2018).

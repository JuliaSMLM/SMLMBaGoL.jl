# Hierarchical Bayes RJMCMC for SMLM with Latent Positions — Revised Mathematical Specification

---

## 1. Problem Statement

Given a set of single-molecule localisation events
$$\mathcal{D} = \{(x_i, y_i, \sigma_i)\}_{i=1}^{N}$$
obtained from an upstream PSF-fitting algorithm, we seek to infer simultaneously:

- The unknown number of emitters $k$
- Their mean positions $\mathbf{s} = \{\mathbf{s}_j \in \mathbb{R}^2\}_{j=1}^{k}$  
- The blinking-statistics hyperparameters $(\mu,\kappa)$ governing the distribution of localisations per emitter
- The systematic localization variance $\tau^2$ capturing physical variations

**Key constraints:**
1. Every localisation is assigned to exactly one emitter (no noise class)
2. The localisation/emitter histogram is dataset-specific and learned *in situ*
3. Latent positions are auxiliary variables maintained for computational efficiency
4. The posterior is explored with reversible-jump MCMC (RJMCMC) inside disjoint spatial partitions processed in parallel; global hyperparameters are synchronised every $T_{\mathrm{sync}}$ sweeps

---

## 2. Generative Hierarchy with Latent Positions (per partition)

The hierarchical model introduces latent "true" positions to make $\tau^2$ conjugate:

$$
\begin{aligned}
\lambda_j &\sim \mathrm{Gamma}(\kappa, \kappa/\mu) && \text{(emitter blinking rate)} \\[0.5em]
n_j \mid \lambda_j &\sim \mathrm{Poisson}(\lambda_j), \quad n_j \geq 1 && \text{(observed blinks)} \\[0.5em]
\mathbf{s}_j &\sim \mathrm{Uniform}(\text{ROI}) && \text{(mean molecular position)} \\[0.5em]
\mathbf{r}_i \mid z_i = j &\sim \mathcal{N}_2(\mathbf{s}_j, \tau^2 \mathbf{I}) && \text{(true position at frame)} \\[0.5em]
(x_i, y_i) \mid \mathbf{r}_i &\sim \mathcal{N}_2(\mathbf{r}_i, \mathrm{diag}(\sigma_{x,i}^2, \sigma_{y,i}^2)) && \text{(observed localization)}
\end{aligned}
$$

**Physical interpretation:**
- $\mathbf{s}_j$: Time-averaged molecular position
- $\mathbf{r}_i$: Actual molecular position when photons were emitted (accounts for drift, vibrations, etc.)
- $\tau^2$: Systematic variation (stage drift, molecular motion, thermal effects)

**Marginal likelihood:** Integrating out $\mathbf{r}_i$ recovers the original model:
$$(x_i, y_i) \mid z_i = j \sim \mathcal{N}_2(\mathbf{s}_j, \mathrm{diag}(\sigma_{x,i}^2 + \tau^2, \sigma_{y,i}^2 + \tau^2))$$

**Hyperpriors** (shared across partitions):
$$\mu \sim \mathrm{Gamma}(a_\mu, b_\mu), \quad \kappa \sim \mathrm{Gamma}(a_\kappa, b_\kappa), \quad \tau^2 \sim \mathrm{InverseGamma}(a_\tau, b_\tau)$$

---

## 3. Marginalisation over $\lambda_j$ ⟹ Negative Binomial Prior

Integrating out the latent blinking rates $\lambda_j$ yields:

$$n_j \mid \mu, \kappa \sim \mathrm{NegativeBinomial}\left(\kappa, \frac{\kappa}{\kappa + \mu}\right), \quad \Pr(n_j = 0) = 0$$

The conditional predictive probability of assigning localisation $i$ to emitter $j$ becomes:

$$\boxed{w_{ij} \propto (n_j + \kappa) \cdot L_{ij}}$$

where the spatial likelihood is computed using the marginal form:
$$L_{ij} = \mathcal{N}_2((x_i, y_i); \mathbf{s}_j, \mathrm{diag}(\sigma_{x,i}^2 + \tau^2, \sigma_{y,i}^2 + \tau^2))$$

---

## 4. Integrated MCMC Moves (fixed $k$)

**CRITICAL REVISION:** Latent positions $\mathbf{r}_i$ are auxiliary variables that must remain consistent with the model. They are updated **immediately** whenever their conditioning variables change, not as a separate move.

### 4.1 Allocate Move: Update All Allocations with Integrated Latent Updates

1. **For each localization $i$:**
   - Sample new allocation $z_i$ from categorical distribution with probabilities $w_{ij}/\sum_r w_{ir}$
   
2. **Immediately update latent position for reallocated localization:**
   $$\mathbf{r}_i \sim \mathcal{N}_2\left(\boldsymbol{\mu}_{\mathrm{post}}, \boldsymbol{\Sigma}_{\mathrm{post}}\right)$$
   where:
   $$\boldsymbol{\Sigma}_{\mathrm{post}}^{-1} = \tau^{-2}\mathbf{I} + \mathrm{diag}(\sigma_{x,i}^{-2}, \sigma_{y,i}^{-2})$$
   $$\boldsymbol{\mu}_{\mathrm{post}} = \boldsymbol{\Sigma}_{\mathrm{post}} \left(\tau^{-2}\mathbf{s}_{z_i} + \mathrm{diag}(\sigma_{x,i}^{-2}, \sigma_{y,i}^{-2})(x_i, y_i)^T\right)$$

### 4.2 Move: Update Emitter Position with Integrated Latent Updates

1. **Select emitter $j$ uniformly at random**

2. **Sample new position from posterior:**
   $$\mathbf{s}_j \sim \mathcal{N}_2\left(\bar{\mathbf{r}}_j, \frac{\tau^2}{n_j}\mathbf{I}\right)$$
   where $\bar{\mathbf{r}}_j = \frac{1}{n_j}\sum_{i: z_i = j} \mathbf{r}_i$ and $n_j = |\{i : z_i = j\}|$

3. **Immediately update latent positions for all localizations assigned to moved emitter:**
   For all $i$ where $z_i = j$, sample new $\mathbf{r}_i$ from posterior given new $\mathbf{s}_j$

### 4.3 Hyperparameter Updates (at synchronization points)

$$
\begin{aligned}
\mu &\sim \mathrm{Gamma}\left(a_\mu + \sum_j n_j, \frac{1}{b_\mu + k\kappa}\right) \\[0.5em]
\tau^2 &\sim \mathrm{InverseGamma}\left(a_\tau + \frac{N_{\mathrm{alloc}}}{2}, b_\tau + \frac{1}{2}\sum_{i: z_i \neq 0} \|\mathbf{r}_i - \mathbf{s}_{z_i}\|^2\right) \\[0.5em]
\kappa &\quad \text{slice sampling or Metropolis-Hastings on } \log\kappa
\end{aligned}
$$

where $N_{\mathrm{alloc}} = 2 \times |\{i : z_i \neq 0\}|$ counts allocated coordinate dimensions.

---

## 5. RJMCMC Dimension Moves (per partition)

### 5.1 Birth Move: $k \to k+1$

**Proposal distribution:** Define a mixture of normals centered at all observed localizations:
$$q_{\mathrm{birth}}(\mathbf{s}_*) = \frac{1}{N} \sum_{i=1}^{N} \mathcal{N}_2(\mathbf{s}_*; (x_i, y_i), \mathrm{diag}(\sigma_{x,i}^2, \sigma_{y,i}^2))$$

**Birth procedure:**
1. **Position proposal:** Draw new emitter position $\mathbf{s}_* \sim q_{\mathrm{birth}}(\cdot)$
2. **Cloud size:** Draw $m \sim 1 + \mathrm{NegativeBinomial}(\kappa, \kappa/(\kappa + \mu))$
3. **Allocation proposal:** Select $m$ localisations with probabilities proportional to marginal likelihood $\mathcal{N}_2((x_i, y_i); \mathbf{s}_*, \mathrm{diag}(\sigma_{x,i}^2 + \tau^2, \sigma_{y,i}^2 + \tau^2))$
4. **Latent position initialization:** For each newly allocated localization, sample $\mathbf{r}_i$ from posterior given $\mathbf{s}_*$ and $(x_i, y_i)$
5. **State update:** Form new emitter cluster and increment $k \to k+1$

### 5.2 Death Move: $k \to k-1$

**Death procedure:**
1. **Victim selection:** Choose emitter $r$ uniformly with probability $1/k$
2. **Proposal density:** Evaluate $q_{\mathrm{birth}}(\mathbf{s}_r)$ for the victim's position
3. **Reallocation:** Reassign localisations $\{i : z_i = r\}$ to remaining emitters using weights $w_{ij}$
4. **Latent position update:** Update $\mathbf{r}_i$ for each reallocated localization
5. **Affected emitter update:** For each emitter that received reallocated localizations:
   - Collect all its latent positions $\{\mathbf{r}_i : z_i = j\}$
   - Sample new position: $\mathbf{s}_j \sim \mathcal{N}_2(\bar{\mathbf{r}}_j, \tau^2/n_j\mathbf{I})$
6. **State update:** Remove $\mathbf{s}_r$ and decrement $k \to k-1$

### 5.3 Acceptance Probability

The acceptance ratio incorporates the proposal density ratio:
$$\log \alpha = \Delta \log p_{\mathrm{NB}} + \Delta \log L_{\mathrm{spatial}} + \Delta \log p(k) + \log \frac{q_{\mathrm{death}}}{q_{\mathrm{birth}}}$$

Note: Likelihood calculations use the marginal form (integrating out latent positions).

---

## 6. Algorithm Structure

### 6.1 Within Each Partition

For each RJMCMC iteration:
1. **Select move type** according to weights (e.g., 40% Allocate, 40% Move, 10% Birth, 10% Death)
2. **Execute move** with integrated latent position updates
3. **Accept/reject** according to Metropolis-Hastings ratio
4. **Store sample** if past burn-in and meets thinning criteria

### 6.2 Hierarchical Updates

Every $T_{\mathrm{sync}}$ iterations:
1. **Synchronization barrier:** Wait for all partitions
2. **Pool statistics:** Collect $\{n_j, \sum\|\mathbf{r}_i - \mathbf{s}_{z_i}\|^2\}$ across partitions
3. **Update hyperparameters:** Sample new $(\mu, \kappa, \tau^2)$
4. **Broadcast:** Distribute updated hyperparameters to all partitions

---

## 7. τ² Initialization Strategy

**Data-driven initialization with physical constraints:**

$$
\tau^2_{\mathrm{init}} = \max\left\{
    (0.1 \times \text{median}(\sigma))^2, \quad  
    (0.020)^2  \text{ μm}^2
\right\}
$$

Where:
- $\text{median}(\sigma)$: Median localization precision from data
- $0.020$ μm: Expected systematic error from physical sources
  - Stage drift: 10-50 nm
  - Thermal vibrations: 5-30 nm  
  - Sample movement: 10-20 nm

**Hyperprior:** $\tau^2 \sim \mathrm{InverseGamma}(3, 4 \times \tau^2_{\mathrm{init}})$
- Prior mean: $2 \times \tau^2_{\mathrm{init}}$
- Prior mode: $\tau^2_{\mathrm{init}}$

---

## 8. Key Algorithmic Improvements

- **Integrated updates:** Latent positions updated immediately with their conditioning variables
- **Collapsed $\lambda_j$:** Reduces state space and improves mixing
- **Pólya-weighted allocations:** Prevents pathological shrinkage behaviour  
- **Conjugate $\tau^2$ update:** Eliminates expensive Metropolis-Hastings steps
- **Block updates in Death move:** Maintains detailed balance while ensuring consistency
- **Parallel partitioning:** Near-linear scaling with periodic global synchronisation

---

## 9. Computational Complexity (per partition per iteration)

| Component | Memory | Cost |
|-----------|--------|------|
| Localisations $N$ | $O(N)$ | - |
| Emitters $k$ | $O(k)$ | - |
| Latent positions | $O(N)$ | $O(N)$ |
| Allocate move | $O(1)$ | $O(Nk)$ |
| Move operation | $O(1)$ | $O(n_j)$ |
| Birth/Death | $O(1)$ | $O(N)$ |
| $\tau^2$ update | $O(1)$ | $O(N)$ |
| **Total** | $O(N + k)$ | $O(Nk)$ |

---

## 10. Posterior Inference

**Important:** Final inference uses **emitter positions** $\mathbf{s}_j$, not latent positions $\mathbf{r}_i$:

- **MAPN extraction:** Hungarian algorithm operates on $\{\mathbf{s}_j\}$ across samples
- **Posterior summaries:** Report mean/variance of $\mathbf{s}_j$ positions
- **Uncertainty quantification:** Based on variance in $\mathbf{s}_j$ across MCMC samples

The latent positions $\mathbf{r}_i$ are computational auxiliary variables that:
1. Enable conjugate $\tau^2$ updates
2. Provide physical interpretation
3. Must remain consistent with the model throughout sampling
4. Are marginalized out in final analysis

---

*This revised specification maintains the mathematical elegance of latent positions while ensuring proper Gibbs sampling consistency through immediate updates.*
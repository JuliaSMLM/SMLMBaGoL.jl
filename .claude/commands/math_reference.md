# Hierarchical Bayes RJMCMC for SMLM with Latent Positions — Mathematical Specification

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
3. The posterior is explored with reversible-jump MCMC (RJMCMC) inside disjoint spatial partitions processed in parallel; global hyperparameters are synchronised every $T_{\mathrm{sync}}$ sweeps

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

## 4. Collapsed Gibbs Updates (fixed $k$)

With the blinking rates marginalised out and latent positions included:

**Allocation variables** $z_i$:
Sample from categorical distribution with probabilities $w_{ij}/\sum_r w_{ir}$

**Latent positions** $\mathbf{r}_i \mid \mathbf{s}_{z_i}, (x_i, y_i), \tau^2$:
$$\mathbf{r}_i \sim \mathcal{N}_2\left(\boldsymbol{\mu}_{\mathrm{post}}, \boldsymbol{\Sigma}_{\mathrm{post}}\right)$$
where:
$$\boldsymbol{\Sigma}_{\mathrm{post}}^{-1} = \tau^{-2}\mathbf{I} + \mathrm{diag}(\sigma_{x,i}^{-2}, \sigma_{y,i}^{-2})$$
$$\boldsymbol{\mu}_{\mathrm{post}} = \boldsymbol{\Sigma}_{\mathrm{post}} \left(\tau^{-2}\mathbf{s}_{z_i} + \mathrm{diag}(\sigma_{x,i}^{-2}, \sigma_{y,i}^{-2})(x_i, y_i)^T\right)$$

**Emitter positions** $\mathbf{s}_j \mid \{\mathbf{r}_i : z_i = j\}, \tau^2$:
$$\mathbf{s}_j \sim \mathcal{N}_2\left(\bar{\mathbf{r}}_j, \frac{\tau^2}{n_j}\mathbf{I}\right)$$
where $\bar{\mathbf{r}}_j = \frac{1}{n_j}\sum_{i: z_i = j} \mathbf{r}_i$ and $n_j = |\{i : z_i = j\}|$

**Hyperparameters:**
$$
\begin{aligned}
\mu &\sim \mathrm{Gamma}\left(a_\mu + \sum_j n_j, \frac{1}{b_\mu + k\kappa}\right) \\[0.5em]
\tau^2 &\sim \mathrm{InverseGamma}\left(a_\tau + \frac{N_{\mathrm{alloc}}}{2}, b_\tau + \frac{1}{2}\sum_{i: z_i \neq 0} \|\mathbf{r}_i - \mathbf{s}_{z_i}\|^2\right) && \text{(NOW CONJUGATE!)} \\[0.5em]
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
4. **Latent position initialization:** For newly allocated localizations, sample $\mathbf{r}_i$ from posterior given $\mathbf{s}_*$
5. **State update:** Form new emitter cluster and increment $k \to k+1$

### 5.2 Death Move: $k \to k-1$

**Death procedure:**
1. **Victim selection:** Choose emitter $r$ with probability $1/k$
2. **Proposal density:** Evaluate $q_{\mathrm{birth}}(\mathbf{s}_r)$ for the victim's position
3. **Reallocation:** Reassign localisations $\{i : z_i = r\}$ to remaining emitters using weights $w_{ij}$
4. **Latent position update:** Update $\mathbf{r}_i$ for reallocated localizations
5. **State update:** Remove $\mathbf{s}_r$ and decrement $k \to k-1$

### 5.3 Acceptance Probability

The acceptance ratio incorporates the proposal density ratio:
$$\log \alpha = \Delta \log p_{\mathrm{NB}} + \Delta \log L_{\mathrm{spatial}} + \Delta \log p(k) + \log \frac{q_{\mathrm{death}}}{q_{\mathrm{birth}}}$$

Note: Likelihood calculations use the marginal form (integrating out latent positions).

---

## 6. Parallel Partition Workflow

1. **Spatial decomposition:** Divide ROI into $P$ overlapping tiles; distribute to threads
2. **Local MCMC:** Each thread runs Gibbs+RJMCMC for $T_{\mathrm{sync}}$ iterations  
3. **Synchronisation barrier:** Pool statistics $\{n_j, \sum\|\mathbf{r}_i - \mathbf{s}_{z_i}\|^2\}$ ⟹ global hyperparameter updates
4. **Broadcast:** Distribute updated $(\mu, \kappa, \tau^2)$ to all threads
5. **Merge:** After convergence, consolidate duplicate emitters in overlap regions

---

## 7. Key Algorithmic Advantages

- **Collapsed $\lambda_j$:** Reduces state space and improves mixing
- **Pólya-weighted allocations:** Prevents pathological shrinkage behaviour  
- **Conjugate $\tau^2$ update:** Eliminates expensive Metropolis-Hastings steps
- **Latent positions:** Enable exact Gibbs sampling while maintaining physical interpretability
- **Parallel partitioning:** Near-linear scaling with periodic global synchronisation

---

## 8. Global Hyperparameter Synchronisation

Pooling statistics across all partitions (total emitters $k_{\mathrm{total}}$, localisations $N_{\mathrm{total}}$):

$$
\begin{aligned}
\mu &\sim \mathrm{Gamma}\left(a_\mu + \sum_{\mathrm{all}} n_j, \frac{1}{b_\mu + k_{\mathrm{total}}\kappa}\right) \\[0.8em]
\tau^2 &\sim \mathrm{InverseGamma}\left(a_\tau + \frac{N_{\mathrm{alloc,total}}}{2}, b_\tau + \frac{1}{2}\sum_{\mathrm{all}} \|\mathbf{r}_i - \mathbf{s}_{z_i}\|^2\right)
\end{aligned}
$$

**Conditional density for $\kappa$** (slice sampling):
$$
\begin{aligned}
\log p(\kappa \mid \mathbf{n}, \mu) &= (a_\kappa - 1)\log\kappa - b_\kappa\kappa \\
&\quad - \sum_j \log\Gamma(\kappa) + \sum_j \log\Gamma(n_j + \kappa) \\
&\quad - (N_{\mathrm{total}} + k_{\mathrm{total}}\kappa)\log(\kappa + \mu)
\end{aligned}
$$

---

## 9. Posterior Inference and MAPN

**Important:** Final inference uses **emitter positions** $\mathbf{s}_j$, not latent positions $\mathbf{r}_i$:

- **MAPN extraction:** Hungarian algorithm operates on $\{\mathbf{s}_j\}$ across samples
- **Posterior summaries:** Report mean/variance of $\mathbf{s}_j$ positions
- **Uncertainty quantification:** Based on variance in $\mathbf{s}_j$ across MCMC samples

The latent positions $\mathbf{r}_i$ are computational auxiliary variables that:
1. Enable conjugate $\tau^2$ updates
2. Provide physical interpretation
3. Are marginalized out in final analysis

---

## 10. Computational Complexity (per partition per iteration)

| Component | Memory | Cost |
|-----------|--------|------|
| Localisations $N$ | $O(N)$ | - |
| Emitters $k$ | $O(k)$ | - |
| Latent positions | $O(N)$ | $O(N)$ |
| $\tau^2$ update | $O(1)$ | $O(N)$ |
| **Total** | $O(N + k)$ | $O(N + k)$ |

**Key improvement:** $\tau^2$ update reduced from $O(N^2)$ (Metropolis-Hastings) to $O(N)$ (conjugate Gibbs).

---

## 11. Physical Validation

The model captures real systematic variations:
- **Stage drift**: 10-50 nm over acquisition
- **Molecular flexibility**: Antibody linkages ~10-20 nm
- **Thermal vibrations**: Building/equipment ~5-30 nm
- **Sample movement**: Breathing, thermal expansion

Validation check:
$$\mathbb{E}[\|\mathbf{r}_i - \mathbf{s}_{z_i}\|] = \sqrt{\frac{2\tau^2}{\pi}} \quad \text{(for 2D Gaussian)}$$

---

*This specification extends the original model with latent positions to enable efficient conjugate updates while maintaining the same marginal likelihood and physical interpretability.*
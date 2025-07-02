# Hierarchical Bayes RJMCMC for SMLM — Mathematical Specification

---

## 1. Problem Statement

Given a set of single-molecule localisation events
$$\mathcal{D} = \{(x_i, y_i, \sigma_i)\}_{i=1}^{N}$$
obtained from an upstream PSF-fitting algorithm, we seek to infer simultaneously:

- The unknown number of emitters $k$
- Their positions $\mathbf{s} = \{\mathbf{s}_j \in \mathbb{R}^2\}_{j=1}^{k}$  
- The blinking-statistics hyperparameters $(\mu,\kappa)$ governing the distribution of localisations per emitter

**Key constraints:**
1. Every localisation is assigned to exactly one emitter (no noise class)
2. The localisation/emitter histogram is dataset-specific and learned *in situ*
3. The posterior is explored with reversible-jump MCMC (RJMCMC) inside disjoint spatial partitions processed in parallel; global hyperparameters are synchronised every $T_{\mathrm{sync}}$ sweeps

---

## 2. Generative Hierarchy (per partition)

The hierarchical model for each partition is defined as:

$$
\begin{aligned}
\lambda_j &\sim \mathrm{Gamma}(\kappa, \kappa/\mu) && \text{(emitter blinking rate)} \\[0.5em]
n_j \mid \lambda_j &\sim \mathrm{Poisson}(\lambda_j), \quad n_j \geq 1 && \text{(observed blinks)} \\[0.5em]
\mathbf{s}_j &\sim \mathrm{Uniform}(\text{ROI}) && \text{(flat spatial prior)} \\[0.5em]
(x_i, y_i) \mid z_i = j &\sim \mathcal{N}_2(\mathbf{s}_j, \sigma_i^2 + \tau^2) && \text{(noisy observations)}
\end{aligned}
$$

**Hyperpriors** (shared across partitions):
$$\mu \sim \mathrm{Gamma}(a_\mu, b_\mu), \quad \kappa \sim \mathrm{Gamma}(a_\kappa, b_\kappa), \quad \tau^2 \sim \mathrm{InverseGamma}(a_\tau, b_\tau)$$

---

## 3. Marginalisation over $\lambda_j$ ⟹ Negative Binomial Prior

Integrating out the latent blinking rates $\lambda_j$ yields:

$$n_j \mid \mu, \kappa \sim \mathrm{NegativeBinomial}\left(\kappa, \frac{\kappa}{\kappa + \mu}\right), \quad \Pr(n_j = 0) = 0$$

The conditional predictive probability of assigning localisation $i$ to emitter $j$ becomes:

$$\boxed{w_{ij} \propto (n_j + \kappa) \cdot L_{ij}}$$

where the spatial likelihood is:
$$L_{ij} = \mathcal{N}_2((x_i, y_i); \mathbf{s}_j, \sigma_i^2 + \tau^2)$$

---

## 4. Collapsed Gibbs Updates (fixed $k$)

With the blinking rates marginalised out, the Gibbs updates are:

**Allocation variables** $z_i$:
Sample from categorical distribution with probabilities $w_{ij}/\sum_r w_{ir}$

**Emitter positions** $\mathbf{s}_j \mid \mathcal{L}_j, \tau^2$:
$$\mathbf{s}_j \sim \mathcal{N}_2\left(\bar{\mathbf{r}}_j, W_j^{-1}\mathbf{I}\right)$$
where $W_j = \sum_{i \in \mathcal{L}_j} \frac{1}{\sigma_i^2 + \tau^2}$ and $\bar{\mathbf{r}}_j$ is the precision-weighted mean

**Hyperparameters:**
$$
\begin{aligned}
\mu &\sim \mathrm{Gamma}\left(a_\mu + \sum_j n_j, \frac{1}{b_\mu + k\kappa}\right) \\[0.5em]
\tau^2 &\sim \mathrm{InverseGamma}\left(a_\tau + \frac{N}{2}, b_\tau + \frac{1}{2}\sum_i Q_i\right) \\[0.5em]
\kappa &\quad \text{slice sampling or Metropolis-Hastings on } \log\kappa
\end{aligned}
$$

---

## 5. RJMCMC Dimension Moves (per partition)

### 5.1 Birth Move: $k \to k+1$

**Proposal distribution:** Define a mixture of normals centered at all observed localizations:
$$q_{\mathrm{birth}}(\mathbf{s}_*) = \frac{1}{N} \sum_{i=1}^{N} \mathcal{N}_2(\mathbf{s}_*; (x_i, y_i), \sigma_i^2)$$

**Birth procedure:**
1. **Position proposal:** Draw new emitter position $\mathbf{s}_* \sim q_{\mathrm{birth}}(\cdot)$
2. **Cloud size:** Draw $m \sim 1 + \mathrm{NegativeBinomial}(\kappa, \kappa/(\kappa + \mu))$
3. **Allocation proposal:** Select $m$ localisations with probabilities proportional to $\mathcal{N}_2((x_i, y_i); \mathbf{s}_*, \sigma_i^2 + \tau^2)$
4. **State update:** Form new emitter cluster $\mathcal{L}_*$ and increment $k \to k+1$

### 5.2 Death Move: $k \to k-1$

**Death procedure:**
1. **Victim selection:** Choose emitter $r$ with probability $1/k$
2. **Proposal density:** Evaluate $q_{\mathrm{birth}}(\mathbf{s}_r)$ for the victim's position
3. **Reallocation:** Reassign localisations $\mathcal{L}_r$ to remaining emitters using weights $w_{ij}$
4. **State update:** Remove $\mathbf{s}_r$ and decrement $k \to k-1$

### 5.3 Acceptance Probability

The acceptance ratio incorporates the proposal density ratio:
$$\log \alpha = \Delta \log p_{\mathrm{NB}} + \Delta \log L_{\mathrm{spatial}} + \Delta \log p(k) + \log \frac{q_{\mathrm{death}}}{q_{\mathrm{birth}}}$$

where:
- **Birth:** $q_{\mathrm{fwd}} = q_{\mathrm{birth}}(\mathbf{s}_*) \times \Pr(\text{allocate } \mathcal{L}_*)$
- **Death:** $q_{\mathrm{rev}} = q_{\mathrm{birth}}(\mathbf{s}_r) \times \Pr(\text{select emitter } r)$

**Note:** The sum-of-normals proposal ensures new emitters are positioned near observed data, improving acceptance rates compared to uniform spatial proposals.

---

## 6. Parallel Partition Workflow

1. **Spatial decomposition:** Divide ROI into $P$ overlapping tiles; distribute to threads
2. **Local MCMC:** Each thread runs Gibbs+RJMCMC for $T_{\mathrm{sync}}$ iterations  
3. **Synchronisation barrier:** Pool statistics $\{n_j, Q_i\}$ ⟹ global hyperparameter updates
4. **Broadcast:** Distribute updated $(\mu, \kappa, \tau^2)$ to all threads
5. **Merge:** After convergence, consolidate duplicate emitters in overlap regions

---

## 7. Key Algorithmic Advantages

- **Collapsed $\lambda_j$:** Reduces state space and improves mixing
- **Pólya-weighted allocations:** Prevents pathological shrinkage behaviour  
- **Gibbs positioning:** Eliminates distance penalties in death moves
- **Parallel partitioning:** Near-linear scaling with periodic global synchronisation

---

## 8. Global Hyperparameter Synchronisation

Pooling statistics across all partitions (total emitters $k_{\mathrm{total}}$, localisations $N_{\mathrm{total}}$):

$$
\begin{aligned}
\mu &\sim \mathrm{Gamma}\left(a_\mu + \sum_{\mathrm{all}} n_j, \frac{1}{b_\mu + k_{\mathrm{total}}\kappa}\right) \\[0.8em]
\tau^2 &\sim \mathrm{InverseGamma}\left(a_\tau + \frac{N_{\mathrm{total}}}{2}, b_\tau + \frac{1}{2}\sum_{\mathrm{all}} Q_i\right)
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

## 9. Posterior-Predictive Diagnostics

- **Goodness-of-fit:** Compare empirical histogram of $n_j$ against $\mathrm{NegBin}(\mu, \kappa)$
- **Spatial residuals:** Check $r_i/\sqrt{\sigma_i^2 + \tau^2} \sim \chi^2_2$ distribution
- **Convergence:** Monitor Gelman-Rubin $\hat{R} < 1.05$ for $(k, \mu, \kappa)$ across partitions

---

## 10. Computational Complexity (per partition)

| Component | Memory | Cost per sweep |
|-----------|--------|----------------|
| Localisations $N$ | $O(N)$ | $O(N)$ |
| Emitters $k$ | $O(k)$ | $O(k)$ |
| **Total** | $O(N + k)$ | $O(N + k)$ |

**Parallel efficiency:** $\approx P/(P+1)$ with $O(1)$ synchronisation overhead.

---

## 11. Optional Extensions

- **Robust likelihood:** Replace Gaussian with 95/5% mixture for outlier resistance
- **Anisotropic PSF:** Incorporate per-event covariance matrices $\Sigma_i$  
- **Adaptive split/merge:** Enable advanced moves if effective sample size of $k$ is low

---

*This specification provides a complete mathematical foundation for implementation.*
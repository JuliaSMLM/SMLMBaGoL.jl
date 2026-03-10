# Collapsed Gibbs Sampler — Mathematical Reference

## 1. Generative Model

### Data
| Symbol | Description |
|--------|-------------|
| N | Total number of localizations |
| d_i = (d_i^x, d_i^y) | Position of localization i (2D vector) |
| Sigma_i | 2x2 covariance matrix of localization i |
| Lambda_i = Sigma_i^{-1} | Precision matrix of localization i |

### Latent Variables
| Symbol | Description |
|--------|-------------|
| K | Number of emitters |
| Z = (z_1, ..., z_N) | Allocation vector: z_i in {1,...,K} assigns loc i to emitter z_i |
| theta_k | Position of emitter k (2D vector, integrated out in collapsed formulation) |

### Hyperparameters
| Symbol | Description |
|--------|-------------|
| rho | Emitter spatial intensity (emitters per unit area) |
| |R| | Scalar area of the spatial prior region R (rectangular) |
| mu | Mean localizations per emitter |
| alpha | Gamma shape parameter for count distribution |

**Notation convention:** R denotes the spatial prior region (a rectangle), and
|R| denotes its scalar area. When we write "area A" informally, we mean |R|.

---

## 2. Prior Structure

### 2.1 Spatial Poisson Process Prior on K

The number of emitters follows a Poisson distribution with rate proportional to area:

    K | rho, |R| ~ Poisson(rho * |R|)

    log P(K | rho, |R|) = K * log(rho * |R|) - rho * |R| - log(K!)

**Key property:** The rate lambda_K = rho * |R| scales with area. This is essential
for the area cancellation in Section 4.

### 2.2 Uniform Spatial Prior on Positions

Each emitter position is drawn uniformly over the prior region R:

    theta_k | R ~ Uniform(R)

    P(theta_k | R) = 1/|R|

For K emitters (i.i.d.):

    P(theta_{1:K} | R) = |R|^{-K}

### 2.3 Count Prior (Gamma)

The total number of localizations N, given K emitters, follows a Gamma distribution
(Fazel et al. 2022):

    N | K, mu, alpha ~ Gamma(K * alpha, mu / alpha)

where:
- E[N] = K * mu
- Var[N] = K * mu^2 / alpha
- alpha = 1: exponential (dSTORM)
- alpha > 1: peaked (DNA-PAINT)
- alpha -> infinity: delta function at K * mu

### 2.4 Allocation Prior (CRP Partition Prior)

The allocation vector Z defines a partition of N localizations into K occupied
clusters with sizes n_1, ..., n_K. We place a Chinese Restaurant Process (CRP)
prior with concentration alpha = 1 on this partition.

**Exchangeable Partition Probability Function (EPPF):**

    P(partition) = alpha^K * prod_k Gamma(n_k) / Gamma(N + alpha)

With alpha = 1:

    P(partition) = prod_k (n_k - 1)! / Gamma(N + 1)
                 = prod_k (n_k - 1)! / N!

This is a distribution over partitions (unordered groupings), not over labeled
assignments. K is not fixed — it is determined by the partition.

**Gibbs conditional (CRP predictive rule):** When reassigning localization i
(removed from its current cluster), the conditional probability is:

    P(z_i = j | Z_{-i}) propto n_j^{-i}     (existing cluster j, size excluding loc i)
    P(z_i = new)         propto alpha = 1     (new cluster)

where n_j^{-i} is the number of localizations in cluster j after removing loc i.
If removing loc i empties cluster j (n_j^{-i} = 0), that cluster ceases to exist.

**Why CRP, not finite Dirichlet-Multinomial:** A finite symmetric
Dirichlet-Multinomial with K labeled components and beta = 1 gives the
conditional propto n_j^{-i} + 1, not n_j^{-i}. The "+1" assigns non-zero
weight to empty labeled slots. The CRP has no empty slots — only occupied
clusters participate — which matches the code (empty clusters are deactivated
and get log-probability -Inf).

---

## 3. Collapsed Marginal Likelihood

### 3.1 Per-Cluster Marginal Likelihood

For cluster k with localizations {d_i : z_i = k}, we integrate out theta_k
under the uniform prior over region R.

**Sufficient statistics (ClusterStats):**

    Posterior precision:  Lambda_post = sum_i Lambda_i
    Natural parameter:    eta = sum_i Lambda_i * d_i
    Quadratic form:       Q = sum_i d_i^T Lambda_i d_i
    Log-det sum:          L = sum_i log|Sigma_i|
    Count:                n_k = |{i : z_i = k}|

**Derivation:**

    P(D_k | R) = integral_R P(D_k | theta_k) * P(theta_k | R) d theta_k
               = (1/|R|) * integral_R prod_i N(d_i; theta_k, Sigma_i) d theta_k

Completing the square in the exponent gives a Gaussian in theta_k with
precision Lambda_post and mean Lambda_post^{-1} * eta. The integral over R^2
(all space) evaluates in closed form. For the bounded region R, there is an
additional truncation factor:

    P(D_k | R) = (1/|R|) * C_data * Phi_R

where:
- C_data = (2*pi)^{1-n_k} * |det Sigma_i|^{-1/2} ... (product over i)
           * exp[-1/2 * (Q - eta^T Lambda_post^{-1} eta)]
           * (2*pi) * |det Lambda_post|^{-1/2}
- Phi_R = Pr(theta ~ N(Lambda_post^{-1} eta, Lambda_post^{-1}) lies in R)

**Approximation:** We assume Phi_R ≈ 1, i.e., the posterior on theta_k is
concentrated well inside the prior region R. This is valid when the prior
region is much larger than the posterior uncertainty, which holds whenever
the data cluster is not at the boundary of R. With this approximation:

    log P(D_k | R) = (1 - n_k) * log(2*pi)
                     - (1/2) * L
                     - (1/2) * (Q - eta^T * Lambda_post^{-1} * eta)
                     - (1/2) * log|Lambda_post|
                     - log|R|

The term `-log|R|` arises from the uniform prior normalization 1/|R|.

**When Phi_R ≈ 1 fails:** If the posterior mean falls near the boundary of R
(within a few posterior standard deviations), the approximation overestimates
the marginal likelihood. In practice this is rare because: (a) BaGoL adds
padding around the data extent, and (b) partitioned BaGoL processes local
clusters where data is interior to the partition.

**Implementation:** `log_marginal_likelihood(cs::ClusterStats, log_area)` in `src/cluster_stats.jl`

### 3.2 Full Collapsed Log Posterior

    log P(K, Z | data) = log P(K | rho, |R|)              ... Poisson prior on K
                        + log P(Z | K)                      ... Dirichlet-Multinomial allocation
                        + log P(N | K, mu, alpha)            ... Gamma count prior
                        + sum_{k=1}^{K} log P(D_k | R)      ... marginal likelihoods

The CRP allocation prior contributes:

    log P(partition) = sum_k log Gamma(n_k) - log Gamma(N + 1)
                     = sum_k log((n_k - 1)!) - log(N!)

This term depends on the partition sizes {n_k} but is independent of |R|.

Expanding the |R|-dependent terms from the Poisson prior and marginal likelihoods:

    [K * log(rho * |R|)] + [sum_k (-log|R|)]
    = K * log(rho * |R|) - K * log|R|
    = K * log(rho)

The remaining |R|-dependent term is `-rho * |R|` from the Poisson normalizer.
This is independent of K and does not affect model selection between different
K values (see Section 4).

**Implementation:** `_collapsed_log_posterior(state, N, mu, shape, lambda_K)` in `src/collapsed_moves.jl`

---

## 4. Area Cancellation

### 4.1 The Problem with Fixed lambda_K

If instead of Poisson(rho*|R|) we use a fixed Poisson(lambda_K) independent of |R|:

    log P(K, Z | data) = K * log(lambda_K) - K * log|R| + ...

The `-K*log|R|` persists. For small |R| (compact data), `-log|R| > 0` acts as a
**bonus** per cluster, favoring over-splitting. For |R| = 0.015 um^2, this is
+4.2 nats per cluster.

### 4.2 The Spatial Poisson Process Fix

Setting lambda_K = rho * |R|:

    P(K, theta_{1:K} | rho, R) = [e^{-rho|R|} * (rho|R|)^K / K!] * |R|^{-K}
                                = e^{-rho|R|} * rho^K / K!

The |R|^K from the Poisson rate cancels |R|^{-K} from the uniform position prior.

**What cancels and what remains:**
- **Cancels exactly:** The K-dependent area term `K * log|R|`. This is the term
  that caused over-splitting bias.
- **Remains:** The term `-rho * |R|` from the Poisson normalizer. This is
  K-independent and does not affect the relative posterior probability of
  different K values. It acts as a constant offset in log P(K, Z | data).
- **Remains (negligible):** The truncation factor Phi_R from Section 3.1,
  which depends on |R| but is ≈ 1 under our approximation.

**Physical interpretation:** rho is the emitter density (emitters per um^2).
The expected number of emitters in region R is rho*|R|, which naturally scales
with the observation window.

### 4.3 Connection to RJMCMC

In the RJMCMC sampler, birth moves propose theta* ~ Uniform(R) with density 1/|R|.
The MH acceptance ratio includes:

    P(theta*) / q(theta*) = (1/|R|) / (1/|R|) = 1

The area cancels because the proposal matches the prior. This is the explicit-sampling
analog of the Poisson process cancellation in the collapsed formulation.

Both mechanisms remove the K-dependent area sensitivity. The RJMCMC does it via
prior-proposal cancellation in the MH ratio; the collapsed sampler does it via
the Poisson process prior's rate scaling with area.

### 4.4 Implementation

In the code, `lambda_K` and `log_area` are both computed from the same region R.
The cancellation happens term-by-term:

    From Poisson prior:     +K * log(lambda_K) = +K * log(rho * |R|)
    From marginal likelihoods: -K * log|R|
    Net:                    +K * log(rho)

The default `lambda_K = N / mu` combined with `lambda_K = rho * |R|` gives
`rho = N / (mu * |R|)`.

**Important:** rho is not a free parameter here — it is derived from `lambda_K`
and `|R|` to ensure the cancellation. The user-facing parameter is `lambda_K`
(or equivalently `mu`), and the area dependence is eliminated by construction.

**Scope of invariance:** The K*log|R| cancellation is exact. The sampler is not
fully area-invariant because (a) `-rho*|R|` remains as a K-independent constant,
and (b) the truncation approximation Phi_R ≈ 1 has residual area dependence.
Neither affects relative model selection between K values, which is what matters
for BaGoL's primary output (MAP-N).

---

## 5. Collapsed Gibbs Moves

### 5.1 Gibbs Allocation Sweep

For each localization i (random order), remove from current cluster, then sample
new assignment from the full conditional:

    P(z_i = k | Z_{-i}, data) propto n_k^{-i} * P(d_i | D_k^{-i})   for existing cluster k
    P(z_i = new | Z_{-i}, data) propto P_new * (1/|R|)                for new cluster

where:
- n_k^{-i} = size of cluster k after removing loc i (from the Dirichlet-Multinomial
  conditional, Section 2.4)
- P(d_i | D_k^{-i}) = predictive probability for loc i given cluster k's other data
- P_new = prior probability ratios for creating a new cluster

**Predictive probability:**

    log P(d_i | D_k^{-i}) = log P(D_k^{-i} union {d_i} | R) - log P(D_k^{-i} | R)

This is computed as the difference of marginal likelihoods. The `-log|R|` terms
cancel in the difference (both marginals have exactly one `-log|R|`).

**New cluster probability (fully specified):**

    P_new = alpha * [P(K+1 | rho, |R|) / P(K | rho, |R|)]
            * [P(N | K+1, mu, alpha_count) / P(N | K, mu, alpha_count)]

where alpha = 1 is the CRP concentration parameter. The first ratio is the
Poisson prior change for K -> K+1, the second is the count prior change.
The CRP partition prior change is absorbed into the alpha weight (new table
gets weight alpha in CRP). The `1/|R|` factor outside P_new is the marginal
likelihood of a single-localization cluster.

**Implementation:** `gibbs_allocation_sweep!()` in `src/collapsed_moves.jl`

### 5.2 Block Birth

1. Pick random seed localization
2. Create new cluster with seed
3. Recruit nearby locs (within 5*sigma) via predictive comparison
4. Accept/reject via MH:

    log alpha = [log P(K', Z' | data) - log P(K, Z | data)]
                + [log q_reverse - log q_forward]

where q_forward is the product of individual recruitment decisions (each loc's
probability of moving or staying), and q_reverse = 1/K' (probability of selecting
the new cluster for death).

**Implementation:** `propose_block_birth!()` in `src/collapsed_moves.jl`

### 5.3 Block Death

1. Pick random active cluster uniformly (requires K > 1)
2. Redistribute all its locs to remaining clusters via predictive
3. Accept/reject via MH:

    log alpha = [log P(K', Z' | data) - log P(K, Z | data)]
                + [log q_forward - log q_redistribute]

where q_forward = 1/K (uniform cluster selection) and q_redistribute is the
product of individual reassignment probabilities.

**Implementation:** `propose_block_death!()` in `src/collapsed_moves.jl`

### 5.4 Move Distribution

| Move | Probability | Type |
|------|-------------|------|
| Gibbs sweep | 50% | Exact (intra-model) |
| Block birth | 25% | MH (trans-dimensional) |
| Block death | 25% | MH (trans-dimensional) |

---

## 6. Sufficient Statistics (ClusterStats)

All cluster operations are O(1) via sufficient statistics.

### 6.1 Adding a Localization

    Lambda_post  += Lambda_i
    eta          += Lambda_i * d_i
    Q            += d_i^T * Lambda_i * d_i
    L            += log|Sigma_i|
    n            += 1

### 6.2 Removing a Localization

Exact inverse of adding (subtract instead of add).

### 6.3 Posterior Position

    mu_post = Lambda_post^{-1} * eta

This is the precision-weighted mean of all localizations in the cluster.

### 6.4 Posterior Covariance

    Sigma_post = Lambda_post^{-1}

Uncertainty decreases as 1/n (in precision), giving sqrt(n) improvement in
position estimate (BaGoL's fundamental advantage).

**Implementation:** `ClusterStats`, `add_loc`, `remove_loc`, `posterior_mean`,
`posterior_cov` in `src/cluster_stats.jl`

---

## 7. Hierarchical Bayes Updates

### 7.1 Mean Localizations per Emitter (mu)

Metropolis-Hastings update on mu using Gamma proposal:

    mu' ~ Gamma(alpha_prop, mu / alpha_prop)

Acceptance ratio:

    log alpha = [log P(N | K, mu', alpha) - log P(N | K, mu, alpha)]
                + [log Gamma(mu'; alpha_prior, beta_prior) - log Gamma(mu; alpha_prior, beta_prior)]
                + [log Gamma(mu; alpha_prop, mu'/alpha_prop) - log Gamma(mu'; alpha_prop, mu/alpha_prop)]

where the last line is the proposal ratio (asymmetric Gamma proposal).

### 7.2 Shape Parameter (alpha)

Same MH structure as mu, updating the Gamma shape parameter.

### 7.3 Synchronized Updates (Partitioned Mode)

In partitioned BaGoL, mu and alpha are shared across partitions and updated
at sync intervals. Each partition's cluster statistics contribute to the global
sufficient statistics for the hierarchical update.

---

## 8. MAP-N Estimation

### 8.1 Mode Selection

1. Build histogram of K across post-burn-in samples
2. MAP-N = argmax of histogram (with smoothing for near-ties)
3. Filter to samples with K = MAP-N

### 8.2 Label Matching (Hungarian Algorithm)

Samples with K = MAP-N may have different label orderings. Iterative Hungarian
matching solves label switching:

1. Start with first sample's cluster centroids as reference
2. For each subsequent sample, compute cost matrix (Euclidean distances)
3. Hungarian algorithm finds optimal assignment
4. Update reference centroids as running median

### 8.3 Position and Uncertainty Estimation

- Position: component-wise median across matched samples (robust to outliers)
- Uncertainty: MAD-based (median absolute deviation / 0.6745) per coordinate

**Implementation:** `estimate_mapn_collapsed()` in `src/mapn.jl`

---

## 9. Partitioned BaGoL

### 9.1 Spatial Partitioning

Precision-weighted DBSCAN clusters the data into spatial partitions. Each partition
is processed independently (parallel via Threads.@threads).

### 9.2 Per-Partition Area

Each partition computes its own spatial prior region R_p from its local data extent.
With the Poisson process prior (lambda_K = rho * |R_p|), the K-dependent area
terms cancel in the posterior (Section 4). The K-independent term `-rho * |R_p|`
remains but does not affect model selection within a partition.

### 9.3 Boundary Deduplication

After independent processing, emitters near partition boundaries are deduplicated
via Hungarian matching based on position proximity.

---

## Appendix A: Approximations and Their Validity

| Approximation | Where | Condition for validity |
|---------------|-------|-----------------------|
| Phi_R ≈ 1 (truncation) | Section 3.1 | Posterior mean interior to R by several sigma |
| Continuous Gamma for integer N | Section 2.3 | N > 5 (good for typical BaGoL data) |
| CRP with alpha=1 | Section 2.4 | No strong prior on number of clusters |

---

## References

1. Fazel et al., "High-Precision Estimation of Emitter Positions using Bayesian
   Grouping of Localizations", *Nature Communications* 13, 7152 (2022)
2. Green, P.J., "Reversible Jump MCMC", *Biometrika* 82(4): 711-32 (1995)
3. Richardson, S. and Green, P.J., "Bayesian analysis of mixtures",
   *J.R. Stat. Soc. B* 59: 731-792 (1997)

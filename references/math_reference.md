# Collapsed Gibbs Sampler — Mathematical Reference

## 0. Problem Context

### The Physical Problem

In single-molecule localization microscopy (SMLM), fluorescent emitters are
activated stochastically across many camera frames. Each activation produces a
diffraction-limited spot that is fit to yield a **localization**: a 2D position
estimate (x, y) with an associated uncertainty (sigma_x, sigma_y). A single
physical emitter typically produces multiple localizations across different frames,
each corrupted by independent photon-counting and camera noise.

The fundamental problem: **given N noisy localizations, determine how many
physical emitters K produced them, which localizations belong to which emitter,
and estimate each emitter's true position with sub-localization precision.**

### Why This Is Hard

1. **Unknown K:** The number of emitters is not known a priori. A cluster of 20
   localizations might come from 1 emitter (blinking many times), 2 nearby
   emitters, or even 5 closely-spaced emitters.

2. **Heteroscedastic noise:** Each localization has a different uncertainty,
   determined by photon count and background level. Some localizations are
   precise (bright frames), others are poor (dim frames).

3. **Spatial overlap:** When emitters are separated by less than the localization
   uncertainty, their localization clouds overlap, making assignment ambiguous.

4. **Variable blinking statistics:** Different labeling chemistries (dSTORM vs
   DNA-PAINT) produce very different distributions of localizations per emitter,
   from geometric/exponential (dSTORM, alpha~1) to peaked (DNA-PAINT, alpha>>1).

### The BaGoL Approach

BaGoL (Bayesian Grouping of Localizations) treats this as a Bayesian mixture
model with unknown number of components:

- **Observations:** N localizations {(d_i, Sigma_i)}, each a position with known
  covariance
- **Latent variables:** K (number of emitters), Z (allocation vector assigning
  each localization to an emitter), theta_{1:K} (emitter positions)
- **Inference:** Posterior P(K, Z, theta | data) via MCMC, then extract MAP-N
  (most probable number of emitters) and estimate positions

The **collapsed Gibbs** formulation integrates out emitter positions theta
analytically, leaving only the discrete allocation vector Z as the sampled
variable. This dramatically reduces the state space and improves mixing.

### Key Output

For each spatial cluster of localizations, BaGoL reports:
- **MAP-N:** The most probable number of emitters
- **Emitter positions:** Precision-weighted posterior means with uncertainties
  that improve as ~1/sqrt(n_k) over individual localizations
- **Posterior image:** Rao-Blackwellized probability density of emitter locations

Typical precision improvement: 10-50x over individual localizations, enabling
resolution of emitters separated by ~10 nm from data with ~30 nm localization
precision (Fazel et al. 2022).

---

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

### 2.4 Allocation Prior (Dirichlet-Multinomial)

The allocation vector Z assigns N localizations to K components (where K is
controlled by the Poisson prior in Section 2.1). Conditional on K, we place a
symmetric Dirichlet-Multinomial (DM) prior on the allocations:

    pi | K, beta ~ Dirichlet(beta/K, ..., beta/K)   (K-dimensional)
    z_i | pi     ~ Categorical(pi)

Integrating out the mixing weights pi gives the collapsed DM:

    P(Z | K, beta) = Gamma(beta) / Gamma(N + beta)
                     * prod_k Gamma(n_k + beta/K) / Gamma(beta/K)^K

or equivalently:

    log P(Z | K, beta) = logGamma(beta) - K * logGamma(beta/K)
                        + sum_k logGamma(n_k + beta/K) - logGamma(N + beta)

where n_k = |{i : z_i = k}| is the number of localizations in cluster k.

**Why DM, not CRP:** The CRP is a prior over partitions that induces its own
implicit prior on K (the number of occupied tables). This conflicts with
the explicit Poisson prior on K (Section 2.1) and the Gamma count prior
(Section 2.3), creating an incoherent model with two competing priors on K.
The DM cleanly separates roles:
- P(K): Poisson prior controls how many emitters
- P(Z|K,beta): DM controls allocation balance given K
- P(N|K,mu,alpha): count prior controls blinking statistics

**Gibbs conditional (DM predictive rule):** When reassigning localization i
(removed from its current cluster), K is held fixed and the conditional is:

    P(z_i = k | Z_{-i}, K) propto (n_k^{-i} + beta/K) * P(d_i | D_k^{-i})

for k = 1, ..., K (all K components, including any that are currently empty).
Unlike the CRP, there is no "new cluster" option — K changes only through
birth/death moves (Section 5).

**Default beta = 1:** With beta = 1, the per-component concentration is 1/K.
This gives mild preference for balanced allocations without strongly constraining
cluster sizes. The DM smoothing term beta/K decreases as K grows, naturally
discouraging proliferation of near-empty components.

**Empty clusters during Gibbs sweep:** An empty cluster (n_k = 0) has allocation
weight beta/K in the DM conditional. Its predictive likelihood is the prior
predictive (uniform over R). This makes it unlikely but not impossible for locs
to be assigned to an empty cluster. After each Gibbs sweep, any clusters that
remain empty are deactivated (K decreases), so empty clusters are transient
artifacts that birth/death moves resolve.

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
                        + log P(Z | K, beta)                ... Dirichlet-Multinomial allocation
                        + log P(N | K, mu, alpha)            ... Gamma count prior
                        + sum_{k=1}^{K} log P(D_k | R)      ... marginal likelihoods

The DM allocation prior contributes:

    log P(Z | K, beta) = logGamma(beta) - K * logGamma(beta/K)
                        + sum_k logGamma(n_k + beta/K) - logGamma(N + beta)

This term depends on the partition sizes {n_k} and K but is independent of |R|.

Expanding the |R|-dependent terms from the Poisson prior and marginal likelihoods:

    [K * log(rho * |R|)] + [sum_k (-log|R|)]
    = K * log(rho * |R|) - K * log|R|
    = K * log(rho)

The remaining |R|-dependent term is `-rho * |R|` from the Poisson normalizer.
This is independent of K and does not affect model selection between different
K values (see Section 4).

**Implementation:** `_collapsed_log_posterior(state, N, mu, shape, lambda_K, beta)` in `src/collapsed_moves.jl`

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
new assignment from the Dirichlet-Multinomial conditional (K held fixed):

    P(z_i = k | Z_{-i}, K, data) propto (n_k^{-i} + beta/K) * P(d_i | D_k^{-i})

for k = 1, ..., K (all K active components, including any that are empty).

where:
- n_k^{-i} = size of cluster k after removing loc i
- beta/K = DM smoothing term (per-component concentration)
- P(d_i | D_k^{-i}) = predictive probability for loc i given cluster k's data

There is no "new cluster" option — K changes only through block birth/death moves.
If a cluster empties during the sweep, it persists with allocation weight beta/K
and prior predictive likelihood. After the sweep completes, any empty clusters
are deactivated (K decreases).

**Predictive probability:**

    log P(d_i | D_k^{-i}) = log P(D_k^{-i} union {d_i} | R) - log P(D_k^{-i} | R)

This is computed as the difference of marginal likelihoods. The `-log|R|` terms
cancel in the difference (both marginals have exactly one `-log|R|`).

For an empty cluster (n_k^{-i} = 0), the predictive is the prior predictive:
P(d_i | empty) = 1/|R| (single-loc marginal likelihood under uniform prior).

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
| DM with beta=1 | Section 2.4 | Mild preference for balanced allocations |

---

## References

1. Fazel et al., "High-Precision Estimation of Emitter Positions using Bayesian
   Grouping of Localizations", *Nature Communications* 13, 7152 (2022)
2. Green, P.J., "Reversible Jump MCMC", *Biometrika* 82(4): 711-32 (1995)
3. Richardson, S. and Green, P.J., "Bayesian analysis of mixtures",
   *J.R. Stat. Soc. B* 59: 731-792 (1997)

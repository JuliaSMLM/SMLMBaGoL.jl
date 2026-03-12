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
| alpha | NegBin shape parameter for count distribution |

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

### 2.3 Count Model (Negative Binomial)

**NegBin parameterization:** We use the (r, p) parameterization where:

    NegBin(n; r, p) = Gamma(n + r) / (Gamma(r) * n!) * p^r * (1-p)^n

with E[n] = r(1-p)/p and Var[n] = r(1-p)/p^2.

Each emitter independently generates a random number of localizations. The
physical model:

    lambda_k ~ Gamma(alpha, mu/alpha)      (latent blinking intensity)
    n_k | lambda_k ~ Poisson(lambda_k)     (observed count)

Marginalizing over lambda_k gives:

    n_k | mu, alpha ~ NegBin(alpha, p)     where p = alpha / (alpha + mu)

with:
- E[n_k] = mu
- Var[n_k] = mu * (1 + mu/alpha)
- alpha = 1: geometric (dSTORM)
- alpha > 1: peaked (DNA-PAINT)
- alpha -> infinity: Poisson(mu)

The total count N = sum_k n_k follows NegBin(K*alpha, p), since independent
NegBin random variables with the same p have a NegBin sum.

### 2.4 Model Factorization: Decoupled K and Z

The sampler uses a **decoupled** factorization that separates emitter counting
(K) from spatial assignment (Z|K):

    P(K, Z | D) propto P(K) * P(N | K) * P(Z | K) * prod_k P(D_k | R)

where:
- **P(K)**: Poisson prior (Section 2.1)
- **P(N | K)**: Total count model NegBin(N; K*alpha, p) — determines K
- **P(Z | K)**: Flat allocation prior — all Z with K non-empty groups equally likely
- **P(D_k | R)**: Spatial marginal likelihood (Section 3) — determines Z given K

**Why decoupled?** A fully coupled model would include per-emitter count factors
c(n_k) = NegBin(n_k; alpha, p) in the Gibbs conditional for Z. However, this
creates a count-ratio weight R(n) = c(n+1)/c(n) = (n+alpha)/(n+1) * mu/(alpha+mu)
that interacts problematically with the spatial likelihood:

- For **separated emitters**: spatial evidence already separates clusters correctly;
  the count ratio adds little information but introduces bias.
- For **co-located emitters**: the count ratio is the only signal (spatial evidence
  is identical across clusters). But the count ratio is only weakly informative
  (modest O(1) differences between clusters), while the Occam factor from the
  spatial marginal likelihood creates a ~10.6 nat penalty per split that dominates.

The decoupled approach resolves this by letting the count model operate at the
K level (where it has full N to work with) and letting the spatial model operate
at the Z level (where it has positional information). The two are connected
through the split/merge mechanism that adjusts both K and Z simultaneously.

**Flat allocation prior:** With P(Z|K) flat, the Gibbs conditional is purely
spatial:

    P(z_i = k | Z_{-i}, K, D) propto P(d_i | D_k^{-i})

This is the spatial predictive probability — how well localization i fits the
spatial cluster k. No count weights appear in the sweep.

**Consistency:** The marginal P(N|K) = NegBin(N; K*alpha, p) is the correct
total count distribution. It provides counting information at the K level
without imposing per-cluster count preferences that would conflict with the
spatial evidence.

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

Under the decoupled factorization (Section 2.4):

    log P(K, Z | data) = log P(K | rho, |R|)              ... Poisson prior on K
                        + log P(N | K, mu, alpha)           ... total count model
                        + sum_{k=1}^{K} log P(D_k | R)     ... spatial marginal likelihoods
                        + const                             ... terms independent of K, Z

where P(N | K) = NegBin(N; K*alpha, p) with p = alpha/(alpha+mu).

**Key difference from per-emitter formulation:** The per-emitter model would
include sum_k log c(n_k) instead of log P(N|K). These are related:

    sum_k log c(n_k) = log P(N|K) + log [N! / prod_k n_k!] + log [P(N|K)^{-1} * prod_k c(n_k)]

The per-emitter form carries more information (it distinguishes partitions of N),
but it creates the problems described in Section 2.4 when used in the Gibbs sweep.
The total count model retains the K-dependence while being partition-invariant.

Expanding the |R|-dependent terms from the Poisson prior and marginal likelihoods:

    [K * log(rho * |R|)] + [sum_k (-log|R|)]
    = K * log(rho * |R|) - K * log|R|
    = K * log(rho)

The remaining |R|-dependent term is `-rho * |R|` from the Poisson normalizer.
This is independent of K and does not affect model selection between different
K values (see Section 4).

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

## 5. MCMC Moves

The sampler alternates between two move types: Gibbs allocation sweeps (which
adjust Z at fixed K) and direct K sampling (which changes K using the count model).

### 5.1 Gibbs Allocation Sweep (K fixed)

For each localization i (random order), remove from current cluster, then sample
new assignment from the spatial predictive:

    P(z_i = k | Z_{-i}, K, data) propto P(d_i | D_k^{-i})

for k = 1, ..., K (all K active components).

where:
- n_k^{-i} = size of cluster k after removing loc i
- P(d_i | D_k^{-i}) = predictive probability for loc i given cluster k's data

**No count weights in the Gibbs sweep.** Under the decoupled model (Section 2.4),
the allocation prior P(Z|K) is flat, so the conditional depends only on the
spatial predictive. The count model affects K (via direct K sampling) but not
the within-K allocation.

**Sole occupants are skipped** to maintain K during the sweep. If loc i is the
only member of its cluster, removing it would reduce K. Since K changes are
handled by the direct K sampling move, the sweep preserves K by skipping sole
occupants.

**Predictive probability:**

    log P(d_i | D_k^{-i}) = log P(D_k^{-i} union {d_i} | R) - log P(D_k^{-i} | R)

This is computed as the difference of marginal likelihoods. The `-log|R|` terms
cancel in the difference (both marginals have exactly one `-log|R|`).

For an empty cluster (n_k^{-i} = 0), the predictive is the prior predictive:
P(d_i | empty) = 1/|R| (single-loc marginal likelihood under uniform prior).

**Implementation:** `gibbs_allocation_sweep!()` in `src/collapsed_moves.jl`

### 5.2 Direct K Sampling with Spatial MH Correction (trans-dimensional)

K proposals are drawn from the count-model posterior:

    pi_count(K) propto P(K) * P(N | K, mu_0, alpha)

where P(K) = Poisson(K; lambda_K) and P(N|K) = NegBin(N; K*alpha, p) with
p = alpha/(alpha+mu_0). A spatial Metropolis-Hastings correction then
accepts or rejects based on the spatial fit improvement.

**Algorithm:**
1. Evaluate log pi_count(K) for K = 1, ..., K_max
2. Sample K_new from this distribution
3. If K_new = K_current: no change
4. Execute heuristic split/merge to adjust the allocation
5. Run 5 Gibbs allocation sweeps (mini relaxation)
6. Compute area-invariant spatial MH acceptance (see below)
7. Accept or reject (rollback on rejection)

**Splitting (adding clusters):** The largest active cluster is split by randomly
assigning ~half its locs to a new cluster.

**Merging (removing clusters):** The smallest active cluster is merged into its
nearest neighbor (by posterior mean distance).

**Mini Gibbs relaxation (step 5):** The heuristic split creates a random spatial
allocation with poor spatial LML. Running 5 Gibbs sweeps before the MH
evaluation lets locs migrate to spatially correct clusters. Without this, the
MH step would reject even correct splits because the proposal allocation is
spatially random. The sweeps do not change K.

**Area-invariant spatial MH correction (step 6):**

The spatial marginal likelihood per cluster includes a -log(A) term from the
uniform position prior. Under a spatial Poisson process prior on emitter
positions, P(K) propto (lambda_spatial * A)^K / K!, the +K*log(A) in the
K prior exactly cancels the -K*log(A) from the position integrals, making
the formulation area-invariant (see Section 5.2.1).

The MH acceptance ratio is:

    log alpha = Delta_fit = [sum log ML(new) - sum log ML(old)] + DeltaK * log(A)

where DeltaK = K_new - K_old and +DeltaK*log(A) is the area cancellation.

**Properties of Delta_fit:**
- Co-located emitters (d=0): Delta_fit ~ 0 (neutral). K inference is purely
  count-driven, equivalent to Q-PAINT. The spatial term neither helps nor
  penalizes splits.
- Separated emitters (d >> sigma): Delta_fit > 0 for correct splits. Spatial
  structure provides genuine evidence beyond count data, beating Q-PAINT.
- The spatial MH can only improve K inference relative to count-only: it never
  penalizes splits that Q-PAINT would accept (because the spatial term is
  non-negative for well-separated emitters and neutral for co-located ones).

### 5.2.1 Why Area Cancellation Is Necessary

Each cluster's marginal likelihood includes -log(A) from the uniform position
prior (integrating out the emitter position over area A):

    log ML(cluster) = -log(A) + log integral prod_i N(d_i; theta, Sigma_i) d(theta)

For a split (K -> K+1), the raw spatial ML difference is:

    Delta_spatial = -log(A) + fit_improvement

The -log(A) is a prior volume artifact, not spatial evidence: doubling the FOV
would add -log(2) nats to every split, but the data hasn't changed. For
co-located emitters (d=0), fit_improvement ~ 0, so the raw Delta_spatial
would penalize every split by -log(A) regardless of the count evidence.

The spatial Poisson process prior contributes +log(A) per new emitter,
exactly canceling this artifact. What remains (fit_improvement) is the genuine
spatial signal: zero for co-located emitters, positive for separated ones.

**Fixed mu_0:** The count-model posterior uses a fixed mu_0 = mu_prior_shape *
mu_prior_scale, NOT the adaptive mu that is updated during MCMC. This prevents
a positive feedback loop where adaptive mu tracks K (mu ~ N/K), making the
count model non-informative for K changes. With fixed mu_0, the count model
provides a stable signal for K.

**Implementation:** `propose_split_merge!()` in `src/collapsed_moves.jl`

### 5.3 Move Distribution

| Move | Probability | Type |
|------|-------------|------|
| Gibbs sweep | 50% | Exact (intra-model) |
| K sampling + spatial MH | 50% | Count proposal + spatial correction |

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

**Note:** The hierarchical update modifies the adaptive mu used in the count
model for hierarchical estimation, but the direct K sampling uses the fixed
mu_0. This decouples K mixing from mu adaptation.

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

## 10. Theoretical MAP-N Accuracy Limits

### 10.1 Count-Model Limit for Co-Located Emitters

When all K emitters share the same position, the spatial marginal likelihood
provides no information about K — only the total count N is informative.
The MAP-K estimate is:

    MAP-K = argmax_k P(K=k) * P(N | K=k)

The accuracy P(MAP-K = K_true) is limited by the NegBin count variance:

    sigma_N = sqrt(K * mu * (mu + alpha) / alpha)

For alpha=5, mu=20, the theoretical maximum MAP-N accuracy is:
- K=1: ~84%
- K=2: ~53%
- K=4: ~39%
- K=8: ~28%

**No algorithm can exceed these limits** for co-located emitters using count
information alone. The Q-PAINT validation (dev/validate_qpaint.jl) tests
that the sampler achieves >= 80% of these theoretical limits.

### 10.2 Why Accuracy Decreases with K

The total count N ~ NegBin(K*alpha, p) has variance that grows linearly with K.
The posterior P(K|N) becomes broader as K increases, making it harder to
distinguish K from K+/-1. This is a fundamental property of the count model,
not a sampler limitation.

---

## Appendix A: Approximations and Their Validity

| Approximation | Where | Condition for validity |
|---------------|-------|-----------------------|
| Phi_R ≈ 1 (truncation) | Section 3.1 | Posterior mean interior to R by several sigma |
| Flat allocation prior | Section 2.4 | Count model operates at K level, not per-cluster |
| Heuristic split/merge for Z | Section 5.2 | 5-sweep Gibbs relaxation + spatial MH correction |
| Fixed mu_0 for K sampling | Section 5.2 | Prevents mu-K positive feedback loop |

## Appendix B: Alternative Formulations Considered

### B.1 Per-Emitter Count Ratio in Gibbs Sweep

An alternative formulation uses per-emitter count factors in the Gibbs conditional:

    P(z_i = k | Z_{-i}, K) propto R(n_k^{-i}) * P(d_i | D_k^{-i})

where R(n) = c(n+1)/c(n) = (n+alpha)/(n+1) * mu/(alpha+mu) is the NegBin
count ratio. This is the correct Gibbs conditional for an occupancy-based
model with prior P(n_1,...,n_K | K) = prod_k c(n_k).

**Note on labeled allocations:** A sampler that updates individual z_i operates
on labeled allocations, not occupancy vectors. The correct full conditional for
labeled Z includes a multinomial factor, giving weight (n_k^{-i} + 1) * R(n_k^{-i})
= (n_k^{-i} + alpha) * mu/(alpha+mu), which is increasing in n — a rich-get-richer
dynamic incompatible with the self-regulating behavior intended.

This approach was rejected because the spatial Occam factor (-1/2 log|Lambda_post|)
dominated the count ratio, preventing K changes. This Occam factor is actually
a prior volume artifact from the uniform position prior (-log(A) per cluster),
not genuine spatial evidence against co-location. It is resolved by the
area-invariant formulation (Section 5.2.1), but the decoupled K/Z model
(Section 2.4) with count-only K proposal + spatial MH correction is cleaner.

### B.2 Jain-Neal Split-Merge

Standard Jain-Neal split-merge selects random localization pairs and proposes
splits (if same cluster) or merges (if different clusters). At high K, random
pairs are overwhelmingly from different clusters, so merges are proposed ~K
times more often than splits. This asymmetry prevents the chain from reaching
or sustaining high K values.

---

## References

1. Fazel et al., "High-Precision Estimation of Emitter Positions using Bayesian
   Grouping of Localizations", *Nature Communications* 13, 7152 (2022)
2. Green, P.J., "Reversible Jump MCMC", *Biometrika* 82(4): 711-32 (1995)
3. Richardson, S. and Green, P.J., "Bayesian analysis of mixtures",
   *J.R. Stat. Soc. B* 59: 731-792 (1997)

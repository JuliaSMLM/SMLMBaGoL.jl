```@meta
CurrentModule = SMLMBaGoL
```

# Model & Sampler

This page describes the statistical model and the MCMC sampler as **implemented in the
source**. Where the implementation makes a specific choice (an approximation, a default,
a prior that is active only in one configuration), it is stated here rather than glossed.
Section headings point at the source files that compute each quantity.

Throughout, a localization ``i`` has an observed position ``\mathbf{x}_i \in \mathbb{R}^D``
(``D = 2`` for the standard path), a reported covariance ``\Sigma_i``, and precision
``\Lambda_i = \Sigma_i^{-1}``. All positions and uncertainties are in micrometers (μm).

## Generative model

A field of view contains ``K`` emitters at unknown positions
``\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_K``. Each emitter ``k`` blinks a random
number of times ``n_k``, and each blink produces one localization drawn around the emitter
with its reported uncertainty:

```math
\mathbf{x}_i \mid z_i = k,\ \boldsymbol{\theta}_k \;\sim\; \mathcal{N}(\boldsymbol{\theta}_k,\ \Sigma_i),
```

where the **allocation** ``z_i \in \{1, \dots, K\}`` records which emitter produced
localization ``i``. Both ``K`` and the allocation are unknown; recovering them — and the
pooled positions ``\boldsymbol{\theta}_k`` — is the inference problem.

The per-emitter blink count uses the negative-binomial model of Fazel *et al.* (see
[Count model](#Count-model)), and the number of emitters is regularized by a partition
prior on the allocation (and, in the legacy flat configuration, by an explicit ``K`` prior).

## Collapsed representation

The sampler's **state is the allocation vector alone**. Each emitter position
``\boldsymbol{\theta}_k`` is integrated out analytically, so a cluster is summarized by
additive sufficient statistics (the [`ClusterStats`](@ref) type, `cluster_stats.jl`):

```math
\Lambda = \sum_{i \in c} \Lambda_i, \qquad
\boldsymbol{\eta} = \sum_{i \in c} \Lambda_i \mathbf{x}_i, \qquad
\mathrm{quad} = \sum_{i \in c} \mathbf{x}_i^\top \Lambda_i \mathbf{x}_i, \qquad
n = |c|.
```

The localization uncertainty ``\sigma`` enters **only** here, as precision
``\Lambda_i = \Sigma_i^{-1}`` (`cluster_stats.jl`); for the diagonal case
``\Lambda_i = \mathrm{diag}(1/\sigma_{x,i}^2,\ 1/\sigma_{y,i}^2)``. Adding and removing a
localization are exact ``O(1)`` inverses, which is what makes the Gibbs sweep cheap.

### Posterior position of a cluster

Under a flat prior on ``\boldsymbol{\theta}``, the position posterior is Gaussian with a
precision-weighted-centroid mean and inverse-total-precision covariance
([`ClusterStats`](@ref) `posterior_mean` / `posterior_cov`):

```math
\hat{\boldsymbol{\theta}} = \Lambda^{-1}\boldsymbol{\eta}
   = \Big(\textstyle\sum_i \Lambda_i\Big)^{-1}\Big(\sum_i \Lambda_i \mathbf{x}_i\Big),
\qquad
\widehat{\mathrm{Cov}}(\boldsymbol{\theta}) = \Lambda^{-1}
   = \Big(\textstyle\sum_i \Sigma_i^{-1}\Big)^{-1}.
```

Pooling many localizations shrinks the covariance below that of any single localization —
the super-resolution gain.

### Collapsed marginal likelihood

Integrating ``\boldsymbol{\theta}`` out of a cluster gives its marginal likelihood. With a
flat position prior (`cluster_stats.jl`, `log_ml_flat`):

```math
\log p_{\text{flat}}(\mathbf{x}_{1:n})
= (1-n)\tfrac{D}{2}\log(2\pi)
- \tfrac12 \sum_i \log\det\Sigma_i
- \tfrac12\big(\mathrm{quad} - \boldsymbol{\eta}^\top \Lambda^{-1}\boldsymbol{\eta}\big)
- \tfrac12 \log\det\Lambda .
```

The two spatial models differ only in the **position-prior term** added to this:

- **Flat (uniform ``1/A``)** — adds a constant Occam factor ``-\log A`` per cluster.
- **Localization mixture (`:locmix`, the default)** — adds
  ``\log P_{\text{locmix}}(\hat{\boldsymbol{\theta}})``, the log of a mixture prior
  ``P(\boldsymbol{\theta}) = \tfrac{1}{N}\sum_j \mathcal{N}(\boldsymbol{\theta}; \mathbf{x}_j, \Sigma_j)``
  built from all localizations, evaluated at the cluster posterior mean
  ``\hat{\boldsymbol{\theta}}``. This concentrates prior mass where localizations are dense.

!!! note "Implementation detail: the locmix marginal is a plug-in approximation"
    The default `:locmix` path (`cluster_stats.jl`, `log_marginal_likelihood_locmix`)
    evaluates the mixture prior **at the posterior mean** ``\hat{\boldsymbol{\theta}}`` on a
    precomputed grid — a saddle-point / plug-in approximation, not the exact mixture
    integral. The exact integral (`log_ml_locmix`) is implemented and used by the
    diagnostics, but the live sampler uses the grid plug-in for speed.

### Predictive probability

The Gibbs allocation step needs the predictive for adding one localization
``\mathbf{x}_*`` to a cluster ([`ClusterStats`](@ref), `log_predictive`):

```math
\log\mathrm{pred}(\mathbf{x}_* \mid c) =
\begin{cases}
\text{(prior at } \mathbf{x}_* \text{)}, & n_c = 0,\\[2pt]
\log p(c \cup \mathbf{x}_*) - \log p(c), & n_c > 0.
\end{cases}
```

Under the flat model the ``-\log A`` terms cancel for ``n_c > 0``, so the predictive is
area-independent except when seeding an empty cluster.

## Priors

### Spatial prior

`spatial_model = :locmix` (default) or `:flat`, as above. The default localization-mixture
prior is data-adaptive; the flat prior is the legacy uniform model.

### Allocation (partition) prior

`allocation_model = :dm` (default), `:decoupled`, or `:categorical`. The
Dirichlet-Multinomial (Pólya) prior on a **labeled** allocation ``\mathbf{z}`` with ``K``
clusters of sizes ``n_1, \dots, n_K`` and concentration ``\gamma`` is
(`collapsed_moves.jl`, `_log_dm_partition`):

```math
\log P(\mathbf{z} \mid K) =
\log\Gamma(K\gamma) - K\log\Gamma(\gamma) - \log\Gamma(N + K\gamma)
+ \sum_{k=1}^{K} \log\Gamma(n_k + \gamma).
```

By default ``\gamma = \texttt{shape}`` (the count-distribution shape ``\alpha``), tying the
partition concentration to the count model; passing `gamma` fixes it independently. The
``\Gamma(K\gamma)/\Gamma(N + K\gamma)`` normalizer depends on ``K``, so the DM prior
contributes to ``K``-changing moves. `:decoupled` drops this term entirely (spatial
likelihood alone drives ``\mathbf{z}``); `:categorical` uses ``P(\mathbf{z}\mid K) = K^{-N}``.

### Count model

Each emitter's blink count is negative-binomial, and the total count
``N = \sum_k n_k`` over ``K`` emitters is its ``K``-fold convolution
(`collapsed_moves.jl`, `_log_count_posterior`):

```math
n_k \mid \mu, \alpha \;\sim\; \mathrm{NegBin}\!\Big(r = \alpha,\ p = \tfrac{\alpha}{\alpha + \mu}\Big),
\qquad
N \mid K \;\sim\; \mathrm{NegBin}\!\Big(r = K\alpha,\ p = \tfrac{\alpha}{\alpha + \mu}\Big),
```

with ``\mathbb{E}[n_k] = \mu`` and shape ``\alpha = \texttt{shape}``. This count term is
**always active** and enters every ``K``-changing move as
``\Delta_{\text{count}} = \log P(N \mid K') - \log P(N \mid K)``.

!!! note "The count model is negative-binomial, not Gamma"
    The shorthand ``P(N\mid K) = \mathrm{Gamma}(N; K\alpha, \mu/\alpha)`` describes the
    continuous rate layer of the Gamma–Poisson mixture. The implementation evaluates the
    exact discrete negative-binomial throughout (`priors.jl`, `collapsed_moves.jl`); the
    two share the mean ``K\mu`` but the NegBin carries the extra Poisson sampling variance.

### Emitter-count (``K``) prior

`k_prior = :auto` (default), `:poisson`, or `:none`. Under the default `:locmix` spatial
model, ``K`` is regularized by the count model and the DM partition prior, and there is
**no separate ``K`` prior**. Under `spatial_model = :flat`, an area-cancelled Poisson prior
``K \sim \mathrm{Poisson}(\rho A)`` is added (`priors.jl`, `log_prior_k_poisson`):

```math
\log P(K) = -\rho A + K\log\rho - \log K! \quad (K \ge 1).
```

The ``A^{-K}`` from the uniform per-emitter location priors cancels the ``A^K`` from the
Poisson rate, leaving an area-independent factor. `k_prior = :poisson` is valid **only**
with `spatial_model = :flat` (it throws otherwise); `:auto` selects this policy per spatial
model.

## The sampler

[`run_collapsed_chain`](@ref) runs a collapsed Gibbs / RJMCMC chain. Each iteration picks
one move (`collapsed_sampler.jl`):

- **50% — Gibbs allocation sweep** (``K`` fixed). Each localization is reassigned using a
  predictive-weighted proposal with a Metropolis correction by the DM factor, so the
  effective conditional is
  ``P(z_i = k \mid \mathbf{z}_{-i}) \propto (n_{-i,k} + \gamma)\cdot\mathrm{pred}(\mathbf{x}_i \mid c_k)``.
- **25% — split/merge** (``|\Delta K| = 1``). A reversible-jump proposal that splits one
  cluster into two or merges two into one, refined by Jain–Neal restricted Gibbs scans
  (`n_restricted_scans`, default 5).
- **25% — birth/death** (``n_bd_substeps`` attempts, default 3 in the partitioned path).
  Proposes creating or removing an emitter.

### Acceptance ratios

Split/merge and birth/death are Metropolis–Hastings accepted on a sum of log-terms
(`collapsed_moves.jl`):

```math
\log\alpha = \Delta_{\text{spatial}} + \Delta_{\text{partition}}
           + \Delta_{\text{proposal}} + \Delta_{\text{count}}
           + \Delta_{K\text{-prior}} + \Delta_{\text{move type}},
```

(birth/death omits ``\Delta_{\text{move type}}``), where

- ``\Delta_{\text{spatial}}`` is the change in summed cluster marginal likelihood
  ([above](#Collapsed-marginal-likelihood));
- ``\Delta_{\text{partition}}`` is the change in the DM partition prior (0 for `:decoupled`);
- ``\Delta_{\text{proposal}}`` is the reverse/forward proposal-density ratio (cluster/pair
  selection and the restricted-Gibbs transition density);
- ``\Delta_{\text{count}}`` is the NegBin count-model change ([above](#Count-model));
- ``\Delta_{K\text{-prior}}`` is the Poisson ``K``-prior change — **nonzero only under
  `spatial_model = :flat`**.

### Hierarchical updates

Every `sync_interval` iterations (default 100), the count-distribution hyperparameters are
updated (`hierarchical.jl`), gated by `learn_distribution`:

- **``\mu``, ``\alpha`` (shape)** — **Metropolis–Hastings**, log-normal random walk with the
  multiplicative Jacobian ``\log\mu' - \log\mu``, against the per-cluster NegBin likelihood
  and a Gamma prior (``\mu \sim \mathrm{Gamma}(2, 5)``, mean 10; ``\alpha \sim \mathrm{Gamma}(2, 1)``,
  mean 2). In the partitioned [`run_bagol`](@ref) path the update is **N-step (50) with
  Robbins–Monro step-size adaptation** during burn-in (target acceptance ≈ 0.30), pooling
  counts across all partitions; the step size is frozen after burn-in. The standalone
  [`run_collapsed_chain`](@ref) uses a single-step, fixed-scale version.
- **``\rho``** (flat ``K``-prior density) — **exact conjugate Gibbs**,
  ``\rho \mid K, A \sim \mathrm{Gamma}(a + K,\ \text{rate } b + A)``. This step runs **only**
  under `spatial_model = :flat`; in the default `:locmix` configuration it never fires.

## MAP-N estimation

The chain yields a full posterior over ``(K, \mathbf{z})``. For a point summary, the
**production path** ([`run_bagol`](@ref)) uses, per partition (`mapn.jl`):

1. **Dahl consensus** ([`estimate_dahl`](@ref)) — choose the visited sample whose
   pairwise co-assignment matrix is closest to the posterior similarity matrix (PSM,
   from [`PSMAccumulator`](@ref)) in squared Frobenius distance over the upper triangle:
   ```math
   D(\mathbf{z}) = \sum_{i<j}\big(\mathbf{1}[z_i = z_j] - C_{ij}\big)^2 ,
   ```
   where ``C_{ij}`` is the empirical co-assignment frequency. This fixes the emitter count
   ``K`` and a reference grouping.
2. **Overlap-Hungarian pooling** ([`estimate_mapn_overlap`](@ref)) — restrict to samples
   with exactly ``K`` clusters, match each to the Dahl reference by maximum localization
   overlap (Hungarian on the overlap-count cost), and pool. The emitter position is the
   **mean** of the matched posterior means; the reported covariance follows the law of total
   variance,
   ```math
   \Sigma_{\text{emitter}} = \underbrace{\Sigma^{\text{post}}_{\text{Dahl}}}_{\mathbb{E}[\mathrm{Var}(\theta\mid Z)]}
     + \underbrace{\mathrm{Cov}\big[\overline{\theta}^{(t)}\big]}_{\mathrm{Var}[\mathbb{E}(\theta\mid Z)]},
   ```
   the within-cluster analytic posterior covariance plus the between-sample spread of the
   matched means.

Alternative estimators are exported for direct use: [`estimate_mapn_collapsed`](@ref)
(histogram-mode ``K`` + Hungarian on positions, **median** pooling),
[`estimate_mapn_psm`](@ref) (``K`` from PSM connected components), and
[`estimate_vi_greedy`](@ref) (greedy minimization of summed Variation of Information).

## Large-dataset partitioning

[`run_bagol`](@ref) partitions large data before sampling (`partition.jl`):

- **Precision-weighted DBSCAN** ([`partition_locs`](@ref)) joins two localizations when
  their separation is small relative to their uncertainties,
  ```math
  \frac{\lVert \mathbf{p}_i - \mathbf{p}_j \rVert}{\sigma_i + \sigma_j} < \texttt{partition\_sigma},
  \qquad \sigma_i = \sqrt{\sigma_{x,i}\,\sigma_{y,i}},
  ```
  so `partition_sigma` is a threshold in summed-σ units (default 3). A KD-tree accelerates
  the neighbor query.
- **METIS splitting** divides any partition above `max_partition_size` (default 1000) along
  a precision-weighted kNN affinity graph, keeping each MCMC run bounded.
- **Boundary deduplication** (`partitioned.jl`) matches emitters near shared partition
  boundaries across partition pairs by Hungarian assignment on Euclidean distance, merging a
  matched pair only when it passes a ``2\sigma`` gate. Merged positions use
  determinant-precision weights; photons are summed.

Partitions run in parallel (`Threads.@threads`); the global hierarchical updates synchronize
across them every `sync_interval` iterations.

## Uncertainty correction

BaGoL's accuracy assumes the reported uncertainties ``\sigma`` are correct. When they are
underestimated by a common amount ``\tau``, BaGoL over-splits. [`estimate_se_adjust`](@ref)
recovers ``\tau`` from the data (`se_estimate.jl`):

At the true ``\tau`` and the correct grouping, within-group scaled neighbor distances follow
a Rayleigh(1) law. For a within-group pair with separation ``d``,

```math
z = \frac{d}{\sqrt{\sigma_a^2 + \sigma_b^2 + 2\tau^2}} \sim \mathrm{Rayleigh}(1),
\qquad F(z) = 1 - e^{-z^2/2}.
```

The finder minimizes the Kolmogorov–Smirnov distance of the empirical ``z`` to this CDF over
``\tau``, alternating an **E-step** (regroup the data with a throwaway BaGoL run at the
current ``\tau``, using BaGoL's own Dahl-consensus labels) and an **M-step** (the
KS-minimizing ``\tau`` on that frozen grouping), descending from above until self-consistent.
A spatial-block bootstrap gives a 95% confidence interval. The estimator is isotropic (it
models a single scalar ``\tau``).

[`apply_se_adjust`](@ref) folds ``\tau`` into each localization in quadrature, per axis:

```math
\sigma_x' = \sqrt{\sigma_x^2 + \tau_x^2}, \qquad \sigma_y' = \sqrt{\sigma_y^2 + \tau_y^2}.
```

Setting `se_adjust=:auto` runs the finder and applies the result; passing a number applies a
known ``\tau`` directly. The correction self-guards against double-applying: a localization
set already corrected upstream (metadata `sigma_corrected = true`) is skipped, and the finder
refuses to run on already-corrected data.

## References

- Fazel, M. *et al.* High-Precision Estimation of Emitter Positions using Bayesian Grouping
  of Localizations. *Nature Communications* **13**, 7152 (2022).
  <https://doi.org/10.1038/s41467-022-34894-2>
- Dahl, D. B. Model-Based Clustering for Expression Data via a Dirichlet Process Mixture
  Model. In *Bayesian Inference for Gene Expression and Proteomics* (2006).
- Jain, S. & Neal, R. M. A Split-Merge Markov Chain Monte Carlo Procedure for the Dirichlet
  Process Mixture Model. *J. Comput. Graph. Stat.* **13**, 158–182 (2004).
```

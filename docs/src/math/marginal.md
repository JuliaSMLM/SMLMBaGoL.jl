```@meta
CurrentModule = SMLMBaGoL
```

# Spatial Models & Marginal Likelihood

Integrating the emitter position ``\boldsymbol{\theta}`` out of a cluster gives its
**marginal likelihood** — the score each MCMC move evaluates. The two spatial models
(`spatial_model = :ipp`, `:flat` or `:locmix`) share the Gaussian core and differ only in the
position-prior term.

## The Gaussian core

With ``\mathbf{x}_i \mid \boldsymbol{\theta} \sim \mathcal{N}(\boldsymbol{\theta}, \Sigma_i)``
and a flat (improper) prior on ``\boldsymbol{\theta}``, the exact Gaussian marginal is
(`cluster_stats.jl`, `log_ml_flat`):

```math
\log p_{\text{flat}}(\mathbf{x}_{1:n})
= (1-n)\tfrac{D}{2}\log(2\pi)
- \tfrac12 \sum_i \log\det\Sigma_i
- \tfrac12\big(\mathrm{quad} - \boldsymbol{\eta}^\top \Lambda^{-1}\boldsymbol{\eta}\big)
- \tfrac12 \log\det\Lambda .
```

Every term is built from the sufficient statistics of the [collapsed representation](collapsed.md).
The quadratic ``\mathrm{quad} - \boldsymbol{\eta}^\top\Lambda^{-1}\boldsymbol{\eta}`` measures
how tightly the cluster's localizations agree once their common position is fitted out: a
spatially compact cluster scores high, a diffuse one low.

This "Gaussian core" uses an **improper, flat** prior on ``\boldsymbol{\theta}``
— it is not yet a normalized model. The two spatial models below make it proper, each by
adding one position-prior term; do not confuse this improper core with the `:flat`
**spatial model**, which is the *uniform-over-area* prior of the next section.

## Flat (uniform) spatial prior

The flat model places emitters uniformly over the region of area ``A``, adding a single
Occam factor ``-\log A`` per cluster (`cluster_stats.jl`, `log_marginal_likelihood`):

```math
\log p(\mathbf{x}_{1:n} \mid c) = \log p_{\text{flat}}(\mathbf{x}_{1:n}) - \log A .
```

This ``-\log A`` is what penalizes adding emitters under the flat model — each new cluster
pays one factor of the area.

## Localization-mixture prior (`:locmix`, the default)

The default `:locmix` model replaces the uniform prior with a **localization-mixture** prior that
concentrates emitter-position mass where localizations are dense:

```math
P(\boldsymbol{\theta}) = \frac{1}{N}\sum_{j=1}^{N} \mathcal{N}(\boldsymbol{\theta};\ \mathbf{x}_j,\ \Sigma_j).
```

The marginal then adds ``\log P_{\text{locmix}}`` in place of ``-\log A``. The sampler
evaluates this term as a plug-in: it takes ``\log P_{\text{locmix}}`` at the single point
``\hat{\boldsymbol{\theta}} = \Lambda^{-1}\boldsymbol{\eta}`` (the cluster posterior mean),
not as the full integral over ``\boldsymbol{\theta}`` — a saddle-point approximation:

```math
\log p_{\text{locmix}}(\mathbf{x}_{1:n} \mid c)
= \log p_{\text{flat}}(\mathbf{x}_{1:n}) + \log P_{\text{locmix}}(\hat{\boldsymbol{\theta}}),
\qquad \hat{\boldsymbol{\theta}} = \Lambda^{-1}\boldsymbol{\eta}.
```

![Localization-mixture prior density over a field](../assets/marginal_locmix_grid.png)

*The localization-mixture prior density ``P_{\text{locmix}}(\boldsymbol\theta)`` over a
localization field — prior mass pools on the data. The flat prior would be a constant sheet
by comparison.*

!!! warning "Partition dependence"
    The locmix per-emitter term is built from the localizations of the partition being sampled,
    so the posterior of a field depends on how it is partitioned. The `:ipp` model
    below uses one global map and does not.

!!! note "Exact integral vs. the production path"
    ``\log P_{\text{locmix}}`` is precomputed on a grid (`log_marginal_likelihood_locmix`).
    The plug-in is exact only in the limit where the prior is flat over the posterior width of
    ``\boldsymbol\theta``. The exact mixture integral (`log_ml_locmix`) is implemented and used
    by the diagnostics, but the grid plug-in is the production path.

## Inhomogeneous Poisson process prior (`:ipp`)

`:ipp` is partition-invariant (under the conditions below) and calibrated for dense fields, but under-splits isolated structures (rulers, origami, single-structure fields), where `:locmix` is the better choice.

`:ipp` models emitters as a marked inhomogeneous Poisson process with intensity
``\lambda(\boldsymbol\theta) = \rho\, g(\boldsymbol\theta)``: ``\rho`` is the global emitter
density (learned, see [Hierarchical Learning](hierarchical.md)) and ``g`` a **fixed** pilot
shape built once from all localizations, before partitioning:

```math
g(\boldsymbol\theta) = \frac{A_{\text{field}}}{N_{\text{all}}}\,
\frac{\sum_{j}\mathcal{N}(\boldsymbol\theta;\ \mathbf{x}_j,\ h^2 I)}{m_h(\boldsymbol\theta)},
\qquad \int_{\text{box}} g = A_{\text{field}},
```

with kernel bandwidth ``h`` = `intensity_bandwidth` (default 0.2 μm), ``m_h`` the kernel mass inside
the field's bounding box (edge correction), and the final lookup normalized numerically. Like
locmix the sampler evaluates ``g`` at the cluster posterior mean (saddle point), and it adds the
Poisson``(\rho A_i')`` prior on ``K`` as under `:flat`, so a cluster scores

```math
\log p_{\text{ipp}}(\mathbf{x}_{1:n} \mid c) = \log p_{\text{flat}}(\mathbf{x}_{1:n}) + \log g(\hat{\boldsymbol\theta}),
```

paired with ``K\log(\rho A_i')`` so the area cancels and each emitter contributes ``\rho\, g(\hat{\boldsymbol\theta})``.
Because ``g`` and ``\rho`` are global and the partition exposures ``A_i' = A_{\text{field}} n_i/N_{\text{all}}``
sum to the field, the target of separated groups factorizes: it does not depend on how the field
is partitioned. This is exact only under DM allocation with ``\gamma`` = shape (the default
coupling), the Poisson ``K`` prior on, fixed hyperparameters and a fixed map, and groups
separated beyond the map/likelihood reach with no overlap copies; other allocation settings do
not inherit it. With ``g \equiv 1`` it is exactly `:flat` + Poisson``(\rho A)``. The full statement
(normalization, conventions, limits) is `docs/math_reference.md` §4.6.

## Predictive probability

The [Gibbs sweep](moves.md) needs the predictive for adding one localization
``\mathbf{x}_*`` to a cluster ([`ClusterStats`](@ref), `log_predictive`):

```math
\log\mathrm{pred}(\mathbf{x}_* \mid c) =
\begin{cases}
\text{(prior at } \mathbf{x}_* \text{)}, & n_c = 0,\\[2pt]
\log p(c \cup \mathbf{x}_*) - \log p(c), & n_c > 0.
\end{cases}
```

Under the flat model the ``-\log A`` terms cancel for ``n_c > 0``, so the predictive is
area-independent except when seeding an empty cluster; under locmix the prior terms do not
cancel (they are evaluated at the shifted posterior mean).

![Flat vs localization-mixture spatial prior](../assets/marginal_flat_vs_locmix.png)

*The flat prior (left, uniform) versus the localization-mixture prior (right, concentrated
on the localizations), with the same candidate emitter position (red star) marked in both —
locmix rewards groupings that sit on dense data.*

Next: the [priors](priors.md) that weigh these likelihood scores — on the allocation, the
blink counts, and the emitter number ``K``.

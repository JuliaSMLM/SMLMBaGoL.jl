```@meta
CurrentModule = SMLMBaGoL
```

# Spatial Models & Marginal Likelihood

Integrating the emitter position ``\boldsymbol{\theta}`` out of a cluster gives its
**marginal likelihood** — the score each MCMC move evaluates. The two spatial models
(`spatial_model = :flat` or `:locmix`) share the Gaussian core and differ only in the
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

## Flat (uniform) spatial prior

The flat model places emitters uniformly over the region of area ``A``, adding a single
Occam factor ``-\log A`` per cluster (`cluster_stats.jl`, `log_marginal_likelihood`):

```math
\log p(\mathbf{x}_{1:n} \mid c) = \log p_{\text{flat}}(\mathbf{x}_{1:n}) - \log A .
```

This ``-\log A`` is what penalizes adding emitters under the flat model — each new cluster
pays one factor of the area.

## Localization-mixture prior (`:locmix`, the default)

The default model replaces the uniform prior with a **localization-mixture** prior that
concentrates emitter-position mass where localizations are dense:

```math
P(\boldsymbol{\theta}) = \frac{1}{N}\sum_{j=1}^{N} \mathcal{N}(\boldsymbol{\theta};\ \mathbf{x}_j,\ \Sigma_j).
```

The marginal then adds ``\log P_{\text{locmix}}`` in place of ``-\log A``:

```math
\log p_{\text{locmix}}(\mathbf{x}_{1:n} \mid c)
= \log p_{\text{flat}}(\mathbf{x}_{1:n}) + \log P_{\text{locmix}}(\hat{\boldsymbol{\theta}}),
\qquad \hat{\boldsymbol{\theta}} = \Lambda^{-1}\boldsymbol{\eta}.
```

!!! info "Figure · data-plot · `assets/marginal_locmix_grid.png`"
    The locmix prior density ``P_{\text{locmix}}(\boldsymbol\theta)`` as a heatmap over a
    localization field (built from `build_locmix_grid`), with the localizations overlaid.
    Mass pools on the data; the flat prior would be a constant sheet by comparison.

!!! note "Implementation detail: the locmix marginal is a plug-in approximation"
    The live sampler evaluates ``\log P_{\text{locmix}}`` **at the posterior mean**
    ``\hat{\boldsymbol\theta}`` on a precomputed grid (`log_marginal_likelihood_locmix`) — a
    saddle-point / plug-in approximation, exact only in the limit where the prior is flat
    over the posterior width of ``\boldsymbol\theta``. The exact mixture integral
    (`log_ml_locmix`) is implemented and used by the diagnostics, but the grid plug-in is
    the production path (chosen for speed).

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

!!! info "Figure · data-plot · `assets/marginal_flat_vs_locmix.png`"
    Side-by-side: the per-cluster marginal-likelihood landscape under the flat prior vs.
    the locmix prior for the same localizations, showing how locmix rewards groupings that
    sit on dense data.

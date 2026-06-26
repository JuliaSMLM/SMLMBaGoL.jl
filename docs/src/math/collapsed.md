```@meta
CurrentModule = SMLMBaGoL
```

# Collapsed Representation

The sampler's **state is the allocation vector ``\mathbf{z}`` alone**. Each emitter position
``\boldsymbol{\theta}_k`` is integrated out analytically, so the chain never moves in
continuous position space — only in the discrete space of partitions. This is what makes
the moves cheap and the acceptance ratios free of Jacobian terms.

## Sufficient statistics

A cluster is summarized by additive sufficient statistics (the [`ClusterStats`](@ref) type,
`cluster_stats.jl`). With ``\Lambda_i = \Sigma_i^{-1}`` the precision of localization ``i``:

```math
\Lambda = \sum_{i \in c} \Lambda_i, \qquad
\boldsymbol{\eta} = \sum_{i \in c} \Lambda_i \mathbf{x}_i, \qquad
\mathrm{quad} = \sum_{i \in c} \mathbf{x}_i^\top \Lambda_i \mathbf{x}_i, \qquad
n = |c|.
```

The localization uncertainty ``\sigma`` enters the model **only here**, as precision. For a
2D localization with reported ``\sigma_x, \sigma_y`` and covariance ``\sigma_{xy}``,

```math
\Sigma_i = \begin{pmatrix} \sigma_x^2 & \sigma_{xy} \\ \sigma_{xy} & \sigma_y^2 \end{pmatrix},
\qquad
\Lambda_i = \Sigma_i^{-1} = \frac{1}{\det\Sigma_i}
\begin{pmatrix} \sigma_y^2 & -\sigma_{xy} \\ -\sigma_{xy} & \sigma_x^2 \end{pmatrix},
```

reducing to ``\Lambda_i = \mathrm{diag}(1/\sigma_x^2,\ 1/\sigma_y^2)`` in the diagonal case.
Because the statistics are sums, adding and removing a localization are exact ``O(1)``
inverses (`add_loc` / `remove_loc`) — the operations the Gibbs sweep performs millions of
times.

!!! info "Figure · schematic · `assets/collapsed_precision.png`"
    Precision weighting. Two localizations with small uncertainty discs and one with a
    large disc; the precision-weighted mean ``\hat{\boldsymbol\theta}`` sits near the
    confident pair, not at the geometric centroid. Larger ``\Sigma_i`` ⇒ smaller pull.

## Posterior position of a cluster

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

The covariance is the inverse of the **summed** precision, so pooling ``n`` localizations of
comparable quality shrinks the position uncertainty by roughly ``\sqrt{n}`` — the
super-resolution gain made precise. These two quantities are exactly what the MAP-N output
emitters carry as their position and reported uncertainty.

!!! info "Figure · pipeline · `assets/collapsed_cluster.png`"
    One emitter's localizations (white) with the posterior mean and ``1\sigma`` covariance
    ellipse (red) from [`render_report`](@ref)'s `EllipseRender`. The ellipse is far tighter
    than any single localization's uncertainty.

## Why collapse?

Integrating ``\boldsymbol{\theta}`` out yields the cluster's **marginal likelihood**, the
quantity every move scores against. Because the state is purely discrete, every transition
is a re-partitioning of localizations, and the acceptance ratio is a ratio of marginal
likelihoods and prior factors — no continuous-parameter proposal, no Jacobian. The next
page derives that marginal likelihood for both spatial models.

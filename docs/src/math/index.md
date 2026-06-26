```@meta
CurrentModule = SMLMBaGoL
```

# Mathematics: Overview

This section is a detailed, code-grounded breakdown of the model and sampler. Each page
states the mathematics **as implemented in the source** and cites the file that computes
it. Read it top to bottom for the full derivation, or jump to a single concept:

- [Collapsed Representation](collapsed.md) — state = allocation; positions integrated out.
- [Spatial Models & Marginal Likelihood](marginal.md) — flat vs. locmix; the marginal.
- [Priors](priors.md) — allocation (DM), count (NegBin), and the K prior.
- [The Sampler: Moves](moves.md) — Gibbs sweep, split/merge, birth/death, acceptance.
- [Hierarchical Learning](hierarchical.md) — learning ``\mu``, shape, and ``\rho``.
- [MAP-N Estimation](mapn.md) — Dahl consensus and overlap-Hungarian pooling.
- [Large-Dataset Partitioning](partitioning.md) — precision-weighted DBSCAN, dedup.
- [Uncertainty Correction](se_adjust.md) — the ``\tau`` finder.

## The inference problem

A field of view contains ``K`` emitters at unknown positions
``\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_K``. Each emitter ``k`` blinks a random
number of times ``n_k``, and each blink produces one localization drawn around the emitter
with its reported uncertainty. Writing ``z_i \in \{1, \dots, K\}`` for the **allocation** —
which emitter produced localization ``i`` — the generative model is

```math
K \sim \text{(count prior)}, \qquad
\boldsymbol{\theta}_k \sim \text{(spatial prior)}, \qquad
\mathbf{x}_i \mid z_i = k,\ \boldsymbol{\theta}_k \;\sim\; \mathcal{N}(\boldsymbol{\theta}_k,\ \Sigma_i).
```

Both ``K`` and the allocation ``\mathbf{z}`` are unknown, and so are the pooled positions
``\boldsymbol{\theta}_k``. BaGoL infers the joint posterior
``p(K, \mathbf{z}, \boldsymbol{\theta} \mid \mathbf{x})`` by MCMC, then summarizes it.

![Generative model: one emitter, many localizations](../assets/intro_generative.png)

*A single emitter (orange star) blinks repeatedly, scattering localizations (blue points),
each with its own uncertainty disc ``\Sigma_i``. BaGoL inverts this — the cloud back to the
emitter.*

Rather than committing to one grouping, BaGoL builds a full posterior over **both** the
number of emitters and their positions, and summarizes it with the **MAP-N** estimate (the
most probable count plus a representative grouping). Pooling the localizations of each
emitter yields a position more precise than any single localization — the super-resolution
gain.

![Raw localizations of a simulated hexamer — a blur](../assets/intro_pre.png)
![BaGoL MAP-N result — six resolved emitters](../assets/intro_post.png)

*Top: Gaussian render of the raw localizations of a simulated hexamer (a blur). Bottom: the
BaGoL MAP-N result at the same scale — six resolved emitters. Pipeline: `simulate_nmer` →
[`run_bagol`](@ref) → [`render_report`](@ref).*

## Notation

| Symbol | Meaning |
|---|---|
| ``\mathbf{x}_i \in \mathbb{R}^D`` | observed position of localization ``i`` (``D = 2`` standard) |
| ``\Sigma_i,\ \Lambda_i = \Sigma_i^{-1}`` | reported covariance and precision of localization ``i`` |
| ``z_i`` | allocation: which emitter produced localization ``i`` |
| ``K`` | number of emitters (clusters) |
| ``n_k`` | number of localizations assigned to emitter ``k`` |
| ``N = \sum_k n_k`` | total number of localizations |
| ``\boldsymbol{\theta}_k`` | position of emitter ``k`` (integrated out in the sampler) |
| ``\mu,\ \alpha`` | count-distribution mean and shape (`shape`) |
| ``\gamma`` | DM partition-prior concentration (``= \alpha`` by default) |
| ``A`` | area of the spatial region |

All positions and uncertainties are in micrometers (μm).

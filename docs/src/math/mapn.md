```@meta
CurrentModule = SMLMBaGoL
```

# MAP-N Estimation

The chain yields a full posterior over ``(K, \mathbf{z})``. For a point summary, the
**production path** ([`run_bagol`](@ref)) uses, per partition, Dahl consensus to fix the
count and a reference grouping, then overlap-Hungarian pooling for positions and uncertainty
(`mapn.jl`).

## Dahl consensus

[`estimate_dahl`](@ref) chooses the visited sample whose pairwise co-assignment matrix is
closest to the **posterior similarity matrix** (PSM) in squared Frobenius distance over the
strict upper triangle:

```math
D(\mathbf{z}) = \sum_{i<j}\big(\mathbf{1}[z_i = z_j] - C_{ij}\big)^2 ,
\qquad
C_{ij} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}\big[z_i^{(t)} = z_j^{(t)}\big],
```

where ``C_{ij}`` (from [`PSMAccumulator`](@ref)) is the empirical co-assignment frequency
over the ``T`` post-burn-in samples. This minimizes the posterior expected Binder loss
restricted to the visited partitions, and fixes ``K`` as the cluster count of the winning
sample.

!!! info "Figure · data-plot · `assets/mapn_psm.png`"
    The posterior similarity matrix as a heatmap (localizations ordered by Dahl cluster).
    Sharp blocks = a confident grouping; smeared off-diagonal mass = label ambiguity that
    the reported uncertainty must capture.

## Overlap-Hungarian pooling

[`estimate_mapn_overlap`](@ref) restricts to samples with exactly ``K`` clusters, matches
each to the Dahl reference by **maximum localization overlap** (Hungarian on the
overlap-count cost), and pools. The emitter position is the **mean** of the matched posterior
means, and the reported covariance follows the **law of total variance**:

```math
\Sigma_{\text{emitter}}
= \underbrace{\Sigma^{\text{post}}_{\text{Dahl}}}_{\mathbb{E}[\mathrm{Var}(\theta\mid Z)]}
+ \underbrace{\mathrm{Cov}\big[\overline{\theta}^{(t)}\big]}_{\mathrm{Var}[\mathbb{E}(\theta\mid Z)]} .
```

The first term is the within-cluster analytic [posterior covariance](collapsed.md#Posterior-position-of-a-cluster);
the second adds the between-sample spread of the matched means, so allocation ambiguity
inflates the reported uncertainty rather than being hidden.

!!! info "Figure · pipeline · `assets/mapn_ellipses.png`"
    MAP-N posterior ellipses (red) over the localizations (white) and ground truth (blue),
    from [`render_report`](@ref)'s `circles_groundtruth`. The ellipses sit on the true
    positions and are far tighter than the localization cloud.

## Alternative estimators

Exported for direct use on stored samples (not on the default pipeline path):

- [`estimate_mapn_collapsed`](@ref) — histogram-mode ``K`` + Hungarian on positions, median pooling;
- [`estimate_mapn_psm`](@ref) — ``K`` from PSM connected components;
- [`estimate_vi_greedy`](@ref) — greedy minimization of summed Variation of Information.

All require a [`PSMAccumulator`](@ref) (and [`PartitionSamples`](@ref)) in the chain's
accumulator list.

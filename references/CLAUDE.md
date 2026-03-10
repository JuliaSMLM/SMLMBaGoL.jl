# References Directory

## Files

### From Fazel et al. (2022)
- `bagol_supplement.pdf` — Full supplementary material from the Nature Communications paper
- `bagol_supplement_math.md` — Mathematical specification extracted from the supplement (RJMCMC formulation)
- `bagol_supplement_figures.md` — Supplementary figure descriptions and simulation parameters

**Paper:** Fazel et al., "High-Precision Estimation of Emitter Positions using Bayesian Grouping of Localizations", *Nature Communications* 13, 7152 (2022). [doi:10.1038/s41467-022-34894-2](https://doi.org/10.1038/s41467-022-34894-2)

### Collapsed Gibbs Formulation (this package)
- `math_reference.md` — Mathematical reference for the collapsed Gibbs sampler implemented in `src/collapsed_sampler.jl`, `src/collapsed_moves.jl`, and `src/cluster_stats.jl`. Derives the spatial Poisson process prior that makes the collapsed formulation area-invariant, matching the implicit cancellation in the RJMCMC sampler. This is the authoritative reference for the collapsed sampler's math.

```@meta
CurrentModule = SMLMBaGoL
```

# SMLMBaGoL.jl

SMLMBaGoL performs **Bayesian Grouping of Localizations (BaGoL)** on single-molecule
localization microscopy (SMLM) data. A single fluorophore blinks many times during an
acquisition, scattering its signal across many localizations; BaGoL groups those
localizations back into the individual emitters that produced them, yielding emitter
positions at precision well beyond that of the raw localizations. It operates on the
`SMLMData.SMLD` structures used across the [JuliaSMLM](https://github.com/JuliaSMLM)
ecosystem.

This package is a Julia reimplementation of the BaGoL algorithm of Fazel *et al.*
(*Nature Communications* **13**, 7152, 2022).

## How it works

In an SMLM experiment a single fluorescent emitter blinks repeatedly, so it appears not
as one point but as a scatter of localizations spread around its true location. BaGoL
treats such localizations as measurements from a mixture of emitters whose number and
true positions are both unknown. Given the localizations and their reported
uncertainties, it asks: which arrangements of emitters could plausibly have produced
this data, and how probable is each? To answer, it runs an MCMC sampler whose
reversible-jump (RJMCMC) moves split, merge, create, and remove emitters, exploring
models with different emitter counts as it goes.

Rather than committing to a single grouping, BaGoL builds a full posterior probability
distribution over *both* the number of emitters and their positions: the probability of
each possible emitter count, a super-resolved posterior image, and a position covariance
for each grouped emitter. For downstream analysis we usually summarize this posterior
with the **MAP-N** estimate — the most probable number of emitters, together with a
representative grouping of localizations at that count — which pools the localizations
assigned to each emitter to reach a position more precise than any single localization.

BaGoL's accuracy rests on one assumption: that the reported localization uncertainties
are correct. In practice they are often underestimated, which can lead BaGoL to treat
noise as signal and split one true emitter into several. To guard against this, the
package provides [`estimate_se_adjust`](@ref), which infers the missing uncertainty `τ`
directly from the data; enable it with `se_adjust=:auto` and BaGoL folds `τ` into each
localization in quadrature (`σ² + τ²`) before grouping. The [Model & Sampler](model.md)
page describes the statistics in full.

## Installation

```julia
using Pkg
Pkg.add("SMLMBaGoL")
```

## Quick Start

```julia
using SMLMBaGoL
using SMLMData

# Group localizations into emitters — returns (BasicSMLD, BaGoLDiagnostics)
result_smld, diagnostics = run_bagol(smld)

# Grouped emitter positions (Emitter2DFit, with posterior uncertainties)
result_smld.emitters

# Posterior summary
diagnostics.n_emitters    # MAP-N number of emitters
diagnostics.posterior_k   # emitter-count histogram over the chain (normalize for P(K))
```

[`run_bagol`](@ref) also accepts a `Vector` of localizations directly, given a camera:

```julia
result_smld, diagnostics = run_bagol(locs; camera=camera)
```

## Where to go next

- **[User Guide](guide.md)** — the model and sampler controls, what `run_bagol` returns,
  standard reports, large-dataset partitioning, and direct chain access.
- **[Model & Sampler](model.md)** — the generative model, the collapsed Gibbs / RJMCMC
  sampler, MAP-N estimation, and the uncertainty-correction procedure.
- **[API Reference](api.md)** — every exported type and function.

## Citation

If you use SMLMBaGoL in your research, please cite:

> Fazel, M. *et al.* High-Precision Estimation of Emitter Positions using Bayesian
> Grouping of Localizations. *Nature Communications* **13**, 7152 (2022).
> <https://doi.org/10.1038/s41467-022-34894-2>

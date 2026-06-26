# SMLMBaGoL

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliasmlm.github.io/SMLMBaGoL.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliasmlm.github.io/SMLMBaGoL.jl/dev)
[![Build Status](https://github.com/JuliaSMLM/SMLMBaGoL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaSMLM/SMLMBaGoL.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSMLM/SMLMBaGoL.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSMLM/SMLMBaGoL.jl)

## Overview

SMLMBaGoL performs **Bayesian Grouping of Localizations (BaGoL)** on single-molecule
localization microscopy (SMLM) data. A single fluorophore blinks many times during an
acquisition, scattering its signal across many localizations; BaGoL groups those
localizations back into the individual emitters that produced them, yielding emitter
positions at precision beyond that of the raw localizations. It operates on the
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

BaGoL makes two assumptions about the localizations. First, that each localization is a real
observation of a single emitter; it follows that any spurious or multi-emitter fits from the
upstream analysis pipeline — where one localization stands for two or more emitters — must be
removed by preprocessing before BaGoL. Second, that the reported localization precision `σ` is
correct. This second assumption can be relaxed: where `σ` carries a uniform excess error,
`estimate_se_adjust` estimates it from the data, and `se_adjust=:auto` folds the excess `τ`
into each localization in quadrature (`σ² + τ²`) before grouping.

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

`run_bagol` also accepts a `Vector` of localizations directly, given a camera:

```julia
result_smld, diagnostics = run_bagol(locs; camera=camera)
```

## Key options

`run_bagol(smld; …)` exposes the model and sampler controls (defaults shown):

```julia
result_smld, diagnostics = run_bagol(smld;
    n_iterations       = 4000,    # total MCMC iterations
    burn_in            = 2000,    # iterations discarded before recording
    μ                  = nothing, # mean localizations per emitter (nothing = estimate from data)
    shape              = 2.0,     # count-distribution shape (1 = dSTORM/exponential, >1 = peaked/DNA-PAINT)
    learn_distribution = true,    # learn the count distribution: true | false | :mu | :shape
    se_adjust          = 0.0,     # extra localization uncertainty τ (μm), added in quadrature;
                                  #   use :auto to estimate τ from the data (see "How it works")
    partition_sigma    = 3.0,     # precision-weighted DBSCAN threshold in σ units (Inf = no partitioning)
    sync_interval      = 100,     # iterations between global μ/shape updates
    posterior_pixel_size = 0.002, # Rao-Blackwellized posterior image pixel size in μm (0.0 to disable)
)
```

Advanced model choices — the spatial prior (`spatial_model = :locmix` or `:flat`), the
allocation model (`allocation_model = :dm`, `:decoupled`, or `:categorical`), and the
`K`-prior gating (`k_prior`) — are documented in the
[full documentation](https://juliasmlm.github.io/SMLMBaGoL.jl/stable).

## What you get back

Alongside the grouped `BasicSMLD`, `run_bagol` returns a `BaGoLDiagnostics` with:

- `n_emitters` — MAP-N number of emitters
- `posterior_k` — emitter-count histogram over the chain (`posterior_k[k+1]` = iterations with `K=k`; normalize for the posterior distribution)
- `acceptance_rates` — per-move MCMC acceptance rates
- `final_μ`, `final_shape` — learned count-distribution parameters
- `posterior_image` — Rao-Blackwellized super-resolution posterior image (or `nothing`)
- `se_adjust` — the applied uncertainty correction `(τx, τy)` in μm, or `nothing`
- `convergence_trace` — per-sync `K`/μ/shape trace for burn-in assessment

## Standard reports

`compute_report` collects metrics from a run (optionally against ground truth) and the
companion writers/plotters produce files and figures:

```julia
report = compute_report(result_smld, diagnostics;
    true_positions = sim.true_positions,   # optional ground truth
    locs_smld      = sim.smld)
write_report(report;  output_dir="output")
plot_report(report;   output_dir="output")              # requires CairoMakie
render_report(sim.smld, result_smld; output_dir="output") # requires SMLMRender
```

`plot_report` and `render_report` activate when you load `CairoMakie` and `SMLMRender`,
respectively (package extensions).

## Large datasets

`run_bagol` automatically partitions large datasets with precision-weighted DBSCAN
clustering, runs the partitions in parallel with synchronized global hierarchical
updates, and deduplicates boundary emitters via Hungarian matching. Run with threads
enabled (`julia --threads=auto`) to use the parallelism.

```julia
result_smld, diagnostics = run_bagol(smld;
    partition_sigma    = 3.0,    # DBSCAN neighbor threshold in σ units
    max_partition_size = 1000,   # split larger partitions with METIS
    overlap            = :auto,  # boundary overlap for cross-partition context
    sync_interval      = 100)    # iterations between global μ/shape updates
```

### Advanced: direct chain access

For custom statistics or full control of the sampler, `run_collapsed_chain` runs the
collapsed Gibbs chain directly with a configurable set of accumulators
(`EmitterCountHist`, `PosteriorImage`, `PSMAccumulator`, …) and exposes the MAP-N
estimators (`estimate_dahl`, `estimate_mapn_overlap`, `estimate_mapn_psm`,
`estimate_vi_greedy`). See the documentation for details.

## Examples

See the `examples/` directory for complete workflows. Run with the examples project and
threading enabled:

```bash
julia --threads=auto --project=examples examples/nmer_single_test.jl
```

- `nmer_single_test.jl` — single N-mer with known ground truth
- `nmer_grid_test.jl` — a grid of N-mers
- `smlmsim_test.jl` — realistic SMLM simulation workflow

## Documentation

Full documentation — the statistical model, the configurable spatial and allocation
models and priors, the RJMCMC/collapsed-Gibbs sampler, MAP-N estimation, and the
uncertainty-correction procedure — is available at the
[documentation site](https://juliasmlm.github.io/SMLMBaGoL.jl/stable).

## Citation

If you use SMLMBaGoL in your research, please cite:

> Fazel, M. *et al.* High-Precision Estimation of Emitter Positions using Bayesian
> Grouping of Localizations. *Nature Communications* **13**, 7152 (2022).
> https://doi.org/10.1038/s41467-022-34894-2

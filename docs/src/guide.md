```@meta
CurrentModule = SMLMBaGoL
```

# User Guide

This guide covers the everyday workflow: running [`run_bagol`](@ref), choosing model and
sampler options, reading what comes back, producing standard reports, scaling to large
datasets, and dropping down to the chain directly. For the underlying statistics see
[Model & Sampler](model.md).

## Running BaGoL

The single entry point is [`run_bagol`](@ref). It accepts an `SMLMData.SMLD`:

```julia
using SMLMBaGoL, SMLMData

result_smld, diagnostics = run_bagol(smld)
```

or a `Vector` of localizations together with a camera:

```julia
result_smld, diagnostics = run_bagol(locs; camera=camera)
```

It returns a tuple `(result_smld, diagnostics)`:

- `result_smld` — a `BasicSMLD` whose `emitters` are the grouped emitters
  (`Emitter2DFit`, each carrying a posterior position uncertainty),
- `diagnostics` — a [`BaGoLDiagnostics`](@ref) summarizing the posterior (see
  [What you get back](#What-you-get-back)).

Positions and uncertainties are in **micrometers (μm)** throughout.

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
                                  #   use :auto to estimate τ from the data
    partition_sigma    = 3.0,     # precision-weighted DBSCAN threshold in σ units (Inf = no partitioning)
    sync_interval      = 100,     # iterations between global μ/shape updates
    posterior_pixel_size = 0.002, # Rao-Blackwellized posterior image pixel size in μm (0.0 to disable)
)
```

`learn_distribution` controls hyperparameter learning: `true` learns both `μ` and `shape`,
`false` fixes both, and `:mu` / `:shape` learn only one. Passing anything else throws an
`ArgumentError`.

### Advanced: model choices

Three options select the statistical model itself. All are validated — an unrecognized
value throws an `ArgumentError`.

| Option | Values (default first) | Meaning |
|---|---|---|
| `spatial_model` | `:locmix`, `:flat` | The spatial prior on emitter positions: localization-mixture prior (`:locmix`) or a flat (uniform) prior (`:flat`). |
| `allocation_model` | `:dm`, `:decoupled`, `:categorical` | The partition prior over localization-to-emitter assignments. `:dm` is the Dirichlet-Multinomial partition prior. |
| `k_prior` | `:auto`, `:poisson`, `:none` | Gating of the emitter-count (`K`) prior. `:auto` uses the spatial model's default policy; `:poisson` is only valid with `spatial_model=:flat` and throws otherwise. |

The defaults (`:locmix` + `:dm` + `:auto`) are the recommended configuration. See
[Model & Sampler](model.md) for what each prior is.

### Split/merge and birth/death tuning

```julia
result_smld, diagnostics = run_bagol(smld;
    n_restricted_scans = 5,   # Jain-Neal restricted Gibbs scans per split/merge proposal
    n_bd_substeps      = 3,   # birth/death substeps per iteration
    gamma              = nothing, # DM concentration γ (nothing ties it to `shape`)
)
```

## What you get back

Alongside the grouped `BasicSMLD`, `run_bagol` returns a [`BaGoLDiagnostics`](@ref) with:

- `n_emitters` — MAP-N number of emitters
- `posterior_k` — emitter-count histogram over the chain (`posterior_k[k+1]` = iterations
  with `K=k`; normalize for the posterior distribution over `K`)
- `acceptance_rates` — per-move MCMC acceptance rates
- `final_μ`, `final_shape` — learned count-distribution parameters
- `posterior_image` — Rao-Blackwellized super-resolution posterior image (or `nothing`)
- `se_adjust` — the applied uncertainty correction `(τx, τy)` in μm, or `nothing`
- `convergence_trace` — per-sync `K`/μ/shape trace for burn-in assessment

## Uncertainty correction

BaGoL assumes the reported localization uncertainties are correct. When they are
underestimated, BaGoL can split one true emitter into several. [`estimate_se_adjust`](@ref)
infers the missing uncertainty `τ` from the data, and [`apply_se_adjust`](@ref) folds it
into each localization in quadrature (`σ² + τ²`) before grouping. The simplest path is to
let `run_bagol` do both:

```julia
result_smld, diagnostics = run_bagol(smld; se_adjust=:auto)  # estimate τ, then apply it
diagnostics.se_adjust                                         # the (τx, τy) that was applied
```

You can also pass a known `τ` directly (`se_adjust=0.012`). The correction self-guards
against double-applying to a localization set that was already corrected upstream. See
[Model & Sampler](model.md#Uncertainty-correction) for the estimator.

## Standard reports

[`compute_report`](@ref) collects metrics from a run (optionally against ground truth);
the companion writers and plotters produce files and figures:

```julia
report = compute_report(result_smld, diagnostics;
    true_positions = sim.true_positions,   # optional ground truth
    locs_smld      = sim.smld)
write_report(report;  output_dir="output")
plot_report(report;   output_dir="output")                 # requires CairoMakie
render_report(sim.smld, result_smld; output_dir="output")  # requires SMLMRender
```

[`plot_report`](@ref) and [`render_report`](@ref) are package extensions: they activate
when you load `CairoMakie` and `SMLMRender`, respectively.

## Large datasets

`run_bagol` automatically partitions large datasets with precision-weighted DBSCAN
clustering, runs the partitions in parallel with synchronized global hierarchical updates,
and deduplicates boundary emitters via Hungarian matching. Run with threads enabled
(`julia --threads=auto`) to use the parallelism.

```julia
result_smld, diagnostics = run_bagol(smld;
    partition_sigma    = 3.0,    # DBSCAN neighbor threshold in σ units
    max_partition_size = 1000,   # split larger partitions with METIS
    overlap            = :auto,  # boundary overlap for cross-partition context
    sync_interval      = 100)    # iterations between global μ/shape updates
```

## Direct chain access

For custom statistics or full control of the sampler, [`run_collapsed_chain`](@ref) runs
the collapsed Gibbs chain directly with a configurable set of accumulators:

```julia
result = run_collapsed_chain(locs;
    n_iterations = 10000, burn_in = 2000,
    accumulators = AbstractAccumulator[
        EmitterCountHist(),
        PosteriorImage(pixel_size=0.001),
        PSMAccumulator(),
    ])
```

### Accumulators

Accumulators collect statistics from the chain without storing full samples:

- [`EmitterCountHist`](@ref) — histogram of `K` per iteration
- [`PosteriorImage`](@ref) — Rao-Blackwellized posterior image (Gaussian blobs per cluster)
- [`NNDistHist`](@ref) — nearest-neighbor distance histogram between emitter positions
- [`PartitionSamples`](@ref) — thinned assignment vectors for MAP-N estimation
- [`PSMAccumulator`](@ref) — posterior similarity matrix (co-assignment frequencies) for
  the Dahl / PSM / VI estimators

### MAP-N estimators

All operate on stored assignment samples. PSM-based methods require a [`PSMAccumulator`](@ref):

- [`estimate_dahl`](@ref) — Dahl consensus partition (preferred default)
- [`estimate_mapn_collapsed`](@ref) — histogram-mode `K` + Hungarian matching
- [`estimate_mapn_psm`](@ref) — PSM thresholding with Hungarian refinement
- [`estimate_vi_greedy`](@ref) — variational-inference greedy approximation
- [`estimate_mapn_overlap`](@ref) — overlap-aware Hungarian matching

## Configuration object

[`BaGoLConfig`](@ref) bundles the model and sampler settings into a struct for the
pipeline / config-driven path, with the converged defaults (`4000`/`2000`/`sync100`,
`se_adjust=:auto`). Construct it and pass it to `run_bagol(smld, cfg)`.

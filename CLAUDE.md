# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Run all tests
julia --project=. -e "using Pkg; Pkg.test()"

# Run examples (threads needed for parallel partition processing)
julia --threads=auto --project=examples examples/nmer_single_test.jl

# Interactive development
julia --project=.
using SMLMBaGoL

# Run specific testset interactively (from REPL)
include("test/runtests.jl")  # runs all tests

# Run tests with specific seed for reproducibility
julia --project=. -e "using Random; Random.seed!(123); using Pkg; Pkg.test()"
```

**Examples environment:** `examples/Project.toml` is a separate project with local path deps
to sibling JuliaSMLM repos (`SMLMData`, `SMLMSim`, `SMLMRender`, `SMLMFrameConnection`).
Always use `--project=examples` when running examples.

**Threading:** Partitioned BaGoL uses `Threads.@threads` for parallel partition processing.
Use `julia --threads=auto` for any workload with multiple partitions.

## Code Conventions

- **All `using`/`import` statements must be in `src/SMLMBaGoL.jl` only** - included files have no imports
- Units: positions and uncertainties in micrometers (μm)
- Uncertainty correction should be applied to data before running BaGoL

## Architecture

### Source File Organization
```
src/
├── SMLMBaGoL.jl          # Module entry: all imports + exports
├── spatial.jl             # Coordinate utilities (get_cov_xy, mean_sigma)
├── cluster_stats.jl       # ClusterStats: sufficient statistics for collapsed sampler
├── types.jl               # Types: BaGoLDiagnostics, CollapsedState, BaGoLResult, CollapsedChainResult
├── priors.jl              # UniformSpatialPrior, log_prior_k, log_prior_total_count
├── hierarchical.jl        # Hierarchical Bayes MH updates for μ and shape
├── mapn.jl                # MAP-N estimation: Dahl, PSM, VI-greedy, Hungarian matching
├── accumulators.jl        # Accumulator interface + EmitterCountHist, PosteriorImage, NNDistHist
├── collapsed_moves.jl     # Gibbs sweep, split/merge moves
├── collapsed_sampler.jl   # run_collapsed_chain() - collapsed Gibbs sampler
├── partition.jl           # Precision-weighted DBSCAN clustering
├── partitioned.jl         # Boundary emitter deduplication
├── rjmcmc.jl              # run_bagol() main entry point
├── posterior_image.jl     # Posterior image histogram + PNG writer
├── archive.jl             # Mmap-based binary chain archive
├── simulation.jl          # simulate_localizations, nmer_positions, simulate_nmer, simulate_nmer_grid
├── reports.jl             # compute_report, write_report, match_positions (standard outputs)
└── optimality.jl          # run_optimality_sweep, run_speed_test (Cat 3)

ext/
├── BaGoLMakieExt/         # CairoMakie extension: plot_report, plot_sweep, plot_speed
└── BaGoLRenderExt/        # SMLMRender extension: render_report
```

**Include order matters:** Files are included in dependency order in `SMLMBaGoL.jl`.
`spatial.jl` before `cluster_stats.jl`, `partition.jl` and `partitioned.jl` before `rjmcmc.jl`.

**Package extensions:** CairoMakie and SMLMRender are weak dependencies.
Plotting/rendering functions activate when the user loads these packages.

**Other directories:**
- `examples/` — Complete workflow scripts (separate project environment)
- `dev/` — Validation scripts: brute-force enumeration, detailed balance, experimental data, high-density tests
- `test/` — All tests in `runtests.jl` (single file, all testsets inline). See `test/CLAUDE.md` for testing guidelines.

### Collapsed Gibbs Sampler

- State = allocation vector only (which locs belong to which cluster)
- Emitter positions integrated out analytically via ClusterStats
- Moves: Gibbs allocation sweep (50%), split (25%), merge (25%)
- K proposed from count-model posterior, spatial MH correction with area-invariant formulation
- Rao-Blackwellized posterior image (Gaussian blobs, not point deltas)
- MAP-N estimation via Dahl+Hungarian matching on stored assignment samples

### Core Types

- `ClusterStats` - Immutable sufficient statistics (precision matrix, natural params, quadratic form)
- `CollapsedState` - Assignment vector + ClusterStats cache + active bitvector
- `CollapsedChainResult` - Final state + accumulator results
- `BaGoLDiagnostics` - n_emitters, posterior_k, acceptance_rates, final_μ, final_shape, n_partitions
- `Partition` - Spatial cluster with locs, original_indices, and boundary flags

### Main API

```julia
# 1. Standard workflow
result_smld, diagnostics = run_bagol(smld; n_iterations=10000, burn_in=2000)

# Also accepts Vector{Emitter2DFit} directly:
result_smld, diagnostics = run_bagol(locs; camera=camera, n_iterations=10000)

# Key parameters:
#   μ=10.0                        # Mean locs per emitter
#   shape=2.0                     # Count distribution shape (1=exp, >1=peaked)
#   learn_shape=true              # Update shape during MCMC
#   nsigma=3.0                    # DBSCAN threshold (Inf = no partitioning)
#   sync_interval=500             # Iterations between global μ/shape updates
#   posterior_pixel_size=0.001    # Enable Rao-Blackwellized posterior image

# 2. Standard report (compute metrics + write files + plot + render)
report = compute_report(result_smld, diagnostics;
    true_positions=sim.true_positions, locs_smld=sim.smld)
write_report(report; output_dir="output")
plot_report(report; output_dir="output")     # requires CairoMakie
render_report(sim.smld, result_smld;          # requires SMLMRender
    output_dir="output", true_positions=sim.true_positions)

# 3. Optimality sweep (Category 3)
sweep = run_optimality_sweep(; n_values=[2, 8], mu_values=[5.0, 10.0], n_trials=50)
write_sweep(sweep; output_dir="output")
plot_sweep(sweep; output_dir="output")        # requires CairoMakie

# 4. Speed test
speed = run_speed_test(; n_locs_range=[100, 1000, 10000])
write_speed(speed; output_dir="output")
plot_speed(speed; output_dir="output")        # requires CairoMakie

# 5. Direct chain access
result = run_collapsed_chain(locs;
    n_iterations=10000, burn_in=2000,
    accumulators=AbstractAccumulator[EmitterCountHist(), PosteriorImage(pixel_size=0.001)]
)
```

### Accumulators

Accumulators collect statistics from the chain without storing full samples:

- `EmitterCountHist` - Histogram of K per iteration
- `PosteriorImage` - Rao-Blackwellized posterior image (Gaussian blobs per cluster)
- `NNDistHist` - Nearest-neighbor distance histogram between emitter positions
- `PartitionSamples` - Stores thinned assignment vectors for MAP-N estimation
- `PSMAccumulator` - Posterior similarity matrix (co-assignment frequencies) for Dahl/PSM/VI estimators

### MAP-N Estimation

Multiple estimation methods available, all using stored assignment samples:

- `estimate_mapn_collapsed(samples, locs)` — histogram-mode K + Hungarian matching for label switching. Median positions + posterior covariances.
- `estimate_dahl(samples, locs, psm)` — Dahl consensus partition (sample closest to PSM). Preferred default.
- `estimate_mapn_psm(samples, locs, psm)` — PSM thresholding with Hungarian refinement.
- `estimate_vi_greedy(samples, locs, psm)` — Variational inference greedy approximation.

PSM-based methods require `PSMAccumulator` in the accumulator list to build the co-assignment matrix.

**Fallback:** `extract_emitters(state, locs)` — final chain state only (single sample).

### ClusterStats

Immutable sufficient statistics for a cluster. All operations O(1):
- `add_loc(cs, loc)` / `remove_loc(cs, loc)` — exact inverses
- `log_marginal_likelihood(cs, log_area)` — positions integrated out
- `log_predictive(cs, loc, log_area)` — predictive for Gibbs allocation
- `posterior_mean(cs)` / `posterior_cov(cs)` — posterior position estimates

### Chain Archive

Opt-in via `archive_path` kwarg. Binary format per partition:
- Header (64 bytes) + per-sample: Int16[n_locs] assignments + Float64 μ + Float64 shape
- Post-hoc analysis: `compute_from_archive(AccType, archive, partition_id, locs)`

### Partitioned BaGoL (Large Datasets)
The main `run_bagol` always partitions data using precision-weighted DBSCAN.
Boundary emitters are deduplicated via Hungarian matching after merge.

**Count Prior (from Fazel et al. 2022):** Uses P(N|K) = Gamma(N; K*shape, μ/shape) where N is total localizations.

## Dependencies

Key external packages:
- `SMLMData` - Core SMLM types (`Emitter2DFit`, `BasicSMLD`, `IdealCamera`)
- `Hungarian` - Optimal assignment for MAP-N and boundary deduplication
- `NearestNeighbors` - KDTree for spatial clustering
- `Mmap` - Memory-mapped chain archive (stdlib)

Examples use `CairoMakie` and `SMLMRender` for visualization.

## Reference Implementation (MATLAB)

This package is a Julia reimplementation of BaGoL. The current MATLAB implementation is in [LidkeLab/smite](https://github.com/LidkeLab/smite) at `+smi/@BaGoL`, integrated into the SMITE toolbox with shared SMF/SMD data structures.

**Paper:** Fazel et al., "High-Precision Estimation of Emitter Positions using Bayesian Grouping of Localizations", *Nature Communications* 13, 7152 (2022). [doi:10.1038/s41467-022-34894-2](https://doi.org/10.1038/s41467-022-34894-2)

## Git Workflow

Finding parent branch:
```bash
git show-branch | head -10
```

The `rjmcmc` branch preserves the legacy RJMCMC sampler implementation.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Run all tests
julia --project=. -e "using Pkg; Pkg.test()"

# Run examples (use threads for parallel partition processing)
julia --threads=auto --project=examples examples/02_nmer_demo.jl

# Interactive development
julia --project=.
using SMLMBaGoL

# Run specific testset interactively (from REPL)
include("test/runtests.jl")  # runs all tests

# Run tests with specific seed for reproducibility
julia --project=. -e "using Random; Random.seed!(123); using Pkg; Pkg.test()"
```

## Code Conventions

- **All `using`/`import` statements must be in `src/SMLMBaGoL.jl` only** - included files have no imports
- Units: positions and uncertainties in micrometers (μm)
- Uncertainty correction should be applied to data before running BaGoL
- Legacy RJMCMC types are parametric on coordinate type `T` (preserves Float32/Float64)

## Architecture

### Source File Organization
```
src/
├── SMLMBaGoL.jl          # Module entry: all imports + exports
├── spatial.jl             # Coordinate utilities (get_cov_xy, mean_sigma)
├── cluster_stats.jl       # ClusterStats: sufficient statistics for collapsed sampler
├── types.jl               # All types (RJMCMC + collapsed + diagnostics)
├── priors.jl              # UniformSpatialPrior, log_prior_k, log_prior_total_count
├── likelihood.jl          # Gaussian likelihood (RJMCMC only)
├── moves.jl               # RJMCMC moves: Birth, Death, Move, Allocate, Split, Merge
├── hierarchical.jl        # Hierarchical Bayes updates (both RJMCMC + collapsed)
├── mapn.jl                # estimate_mapn() + estimate_mapn_collapsed() - Hungarian matching
├── accumulators.jl        # Accumulator interface + EmitterCountHist, PosteriorImage, NNDistHist
├── collapsed_moves.jl     # Collapsed moves: Gibbs sweep, block birth/death
├── collapsed_sampler.jl   # run_collapsed_chain() - collapsed Gibbs sampler
├── partition.jl           # Precision-weighted DBSCAN clustering
├── partitioned.jl         # Parallel BaGoL execution + boundary merging
├── rjmcmc.jl              # run_bagol() main entry + run_bagol_chain() + legacy RJMCMC
├── posterior_image.jl     # Posterior image from RJMCMC chain samples + PNG writer
├── archive.jl             # Mmap-based binary chain archive
└── simulation.jl          # simulate_smlm, simulate_grid, simulate_nmers
```

### Two Samplers

**Collapsed Gibbs (default, `sampler=:collapsed`):**
- State = allocation vector only (which locs belong to which cluster)
- Emitter positions integrated out analytically via ClusterStats
- Moves: Gibbs allocation sweep (70%), block birth (15%), block death (15%)
- Rao-Blackwellized posterior image (Gaussian blobs, not point deltas)
- MAP-N estimation via `estimate_mapn_collapsed` on stored assignment samples

**Legacy RJMCMC (`sampler=:rjmcmc`):**
- State = explicit emitter positions + allocations
- Moves: Birth (10%), Death (10%), Move (20%), Allocate (50%)
- MAP-N estimation via iterative Hungarian matching

### Core Types

**Collapsed sampler:**
- `ClusterStats` - Immutable sufficient statistics (precision matrix, natural params, quadratic form)
- `CollapsedState` - Assignment vector + ClusterStats cache + active bitvector
- `CollapsedChainResult` - Final state + accumulator results

**RJMCMC (legacy):**
- `Emitter{T}` - Position (x, y) with allocated localization indices
- `BaGoLState{T}` - Current MCMC state (emitters + log_posterior)
- `RJMCMCChain{T}` - Full chain with samples, config, acceptance stats

**Shared:**
- `BaGoLDiagnostics` - n_emitters, posterior_k, acceptance_rates, final_μ, final_shape, n_partitions
- `Partition` - Spatial cluster with locs, original_indices, and boundary flags

### Main API

```julia
# 1. Standard workflow (collapsed Gibbs, default)
result_smld, diagnostics = run_bagol(smld; n_iterations=10000, burn_in=2000)

# Key parameters:
#   sampler=:collapsed            # or :rjmcmc for legacy
#   nsigma=3.0                    # DBSCAN threshold (Inf = no partitioning)
#   min_partition_size=0          # Keep all clusters
#   max_partition_size=1000       # Split partitions larger than this
#   shape=2.0                     # Gamma shape (1=exponential, >1=peaked)
#   learn_shape=true              # Update shape during MCMC
#   sync_interval=500             # Iterations between global μ/shape updates
#   posterior_pixel_size=0.001    # Enable Rao-Blackwellized posterior image
#   archive_path="path/"          # Enable mmap chain archive

# 2. Direct collapsed chain access
result = run_collapsed_chain(locs;
    n_iterations=10000, burn_in=2000,
    accumulators=AbstractAccumulator[EmitterCountHist(), PosteriorImage(pixel_size=0.001)]
)

# 3. Legacy RJMCMC chain access
chain = run_bagol_chain(locs; n_iterations=10000, burn_in=2000)
emitters, posterior_k = estimate_mapn(chain)
```

### Accumulators (Collapsed Sampler)

Accumulators collect statistics from the chain without storing full samples:

- `EmitterCountHist` - Histogram of K per iteration
- `PosteriorImage` - Rao-Blackwellized posterior image (Gaussian blobs per cluster)
- `NNDistHist` - Nearest-neighbor distance histogram between emitter positions
- `PartitionSamples` - Stores thinned assignment vectors for MAP-N estimation

### MAP-N Estimation

Both samplers use MAP-N (Maximum A Posteriori Number) for emitter extraction:

1. Build histogram of K across post-burn-in samples
2. Find MAP-N = mode of K distribution
3. Filter to samples with K = MAP-N
4. Iterative Hungarian matching to solve label switching
5. Median positions (robust to outliers) + MAD-based uncertainties

**Collapsed:** `estimate_mapn_collapsed(samples, locs)` — positions derived from
ClusterStats posterior means (deterministic given assignments). Uses `PartitionSamples`
accumulator which is automatically included in `run_bagol`.

**RJMCMC:** `estimate_mapn(chain)` — positions from explicit emitter coordinates.

**Fallback:** `extract_emitters(state, locs)` — final chain state only (single sample).

### ClusterStats

Immutable sufficient statistics for a cluster. All operations O(1):
- `add_loc(cs, loc)` / `remove_loc(cs, loc)` — exact inverses
- `log_marginal_likelihood(cs, log_area)` — positions integrated out
- `log_predictive(cs, loc, log_area)` — predictive for Gibbs allocation
- `posterior_mean(cs)` / `posterior_cov(cs)` — posterior position estimates

Math derives from precision-weighted averaging in `moves.jl:sample_position_gibbs`.

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

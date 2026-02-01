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
- Core types are parametric on coordinate type `T` (preserves Float32/Float64)

## Architecture

### Source File Organization
```
src/
├── SMLMBaGoL.jl    # Module entry: all imports + exports
├── types.jl        # Emitter, BaGoLState, BaGoLSample, RJMCMCConfig, RJMCMCChain
├── priors.jl       # UniformSpatialPrior, log_prior_k, log_prior_total_count
├── likelihood.jl   # Gaussian likelihood with systematic uncertainty
├── moves.jl        # RJMCMC moves: Split, Merge, Birth, Death, Move, Allocate
├── hierarchical.jl # Hierarchical Bayes updates for μ and shape
├── rjmcmc.jl       # run_bagol() - main entry point
├── mapn.jl         # estimate_mapn() - Hungarian matching for MAP-N
├── spatial.jl      # Coordinate utilities for partitioning
├── partition.jl    # Precision-weighted DBSCAN clustering
├── partitioned.jl  # Parallel BaGoL execution + boundary merging
└── simulation.jl   # simulate_smlm, simulate_grid, simulate_nmers
```

### Core Types
- `Emitter{T}` - Position (x, y) with allocated localization indices (parametric on Float type)
- `BaGoLState{T}` - Current MCMC state (emitters + log_posterior)
- `BaGoLSample{T}` - Recorded sample with μ and shape values
- `RJMCMCChain{T}` - Full chain with samples, config, acceptance stats
- `BaGoLDiagnostics` - Diagnostics struct with n_emitters, posterior_k, acceptance_rates, final_μ, final_shape, n_partitions
- `Partition` - Spatial cluster with locs, original_indices, and boundary flags

### Main API

Two entry points depending on needs:

```julia
# 1. Standard workflow - returns (BasicSMLD, BaGoLDiagnostics)
#    Auto-partitions data via precision-weighted DBSCAN
result_smld, diagnostics = run_bagol(smld; n_iterations=10000, burn_in=2000)

# Key parameters for run_bagol:
#   nsigma=3.0                    # DBSCAN threshold (Inf = no partitioning)
#   min_partition_size=0          # Keep all clusters (no noise filtering)
#   max_partition_size=1000       # Split partitions larger than this
#   skip_partition_size=typemax   # Skip partitions larger than this
#   shape=2.0                     # Gamma shape param (1=exponential, >1=peaked)
#   learn_shape=true              # Update shape during MCMC
#   λ_K=length(locs)/5.0          # Prior on emitter count
#   sync_interval=500             # Iterations between global μ/shape updates

# 2. Advanced: direct chain access for diagnostics
chain = run_bagol_chain(locs; n_iterations=10000, burn_in=2000)
emitters, posterior_k = estimate_mapn(chain)

# Visualization: include examples/viz_chain_diagnostics.jl
# Then use: plot_bagol(chain, emitters, posterior_k, locs)

# 3. Animation/per-iteration callbacks
records = []
chain = run_bagol_chain(locs;
    callback = (i, move_type, accepted, state, μ, shape) ->
        push!(records, (i, length(state.emitters))),
    callback_interval = 10
)
```

### MAP-N Estimation

`estimate_mapn` extracts emitter positions and uncertainties from the RJMCMC chain using:

1. **Iterative Hungarian matching** with median-based reference positions to handle label switching between nearby emitters
2. **MAD-based uncertainty** (Median Absolute Deviation) which is robust to outliers from residual label switching

The resulting σ values accurately represent the posterior width and are valid for downstream analysis assuming normal distributions.

### RJMCMC Algorithm
- 4 move types: Birth (10%), Death (10%), Move (20%), Allocate (50%)
- Posterior: Poisson prior on K, **marginal Gamma prior** P(N|K) on total count, Gaussian likelihood
- Hierarchical: MH updates for μ, MH updates for shape (when `learn_shape=true`)

**Count Prior (from Fazel et al. 2022):** Uses P(N|K) = Gamma(N; K×shape, μ/shape) where N is total localizations. This marginal formulation correctly accounts for the constraint that individual counts sum to N, avoiding bias toward fewer emitters.

### Partitioned BaGoL (Large Datasets)
The main `run_bagol` always partitions data using precision-weighted DBSCAN.
The `partition_locs` function is exported for advanced control:

```julia
# Manual partitioning
partitions, skipped = partition_locs(locs;
    nsigma=4.0,      # DBSCAN threshold in sigma units
    min_size=10,     # Minimum locs per partition
    max_size=1000,   # Target max locs per partition
    skip_size=Inf    # Skip clusters larger than this
)

# Each partition has:
p.locs              # Vector of Emitter2DFit for this partition
p.original_indices  # Indices back to original locs array
p.is_boundary       # BitVector of which locs are near partition edge
```

Algorithm: Precision-weighted DBSCAN clustering where distance = `||p_i - p_j|| / (σ_i + σ_j)`.
Oversized clusters are recursively bisected along principal axis.
Boundary emitters are deduplicated via Hungarian matching after merge.

## Dependencies

Key external packages:
- `SMLMData` - Core SMLM types (`Emitter2DFit`, `BasicSMLD`, `IdealCamera`)
- `Hungarian` - Optimal assignment for MAP-N estimation
- `NearestNeighbors` - KDTree for spatial clustering

Examples use `CairoMakie` and `SMLMRender` for visualization (see `examples/viz_chain_diagnostics.jl`).

## Git Workflow

Finding parent branch:
```bash
git show-branch | head -10
```
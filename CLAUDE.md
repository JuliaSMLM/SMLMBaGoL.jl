# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Run tests
julia --project=. -e "using Pkg; Pkg.test()"

# Run examples (use threads for parallel partition processing)
julia --threads=auto --project=examples examples/02_nmer_demo.jl

# Interactive development
julia --project=.
using SMLMBaGoL
```

## Code Conventions

- **All `using`/`import` statements must be in `src/SMLMBaGoL.jl` only** - included files have no imports
- Units: positions and uncertainties in micrometers (μm)
- Uncertainty correction should be applied to data before running BaGoL

## Architecture

### Source File Organization
```
src/
├── SMLMBaGoL.jl    # Module entry: all imports + exports
├── types.jl        # Emitter, BaGoLState, BaGoLSample, RJMCMCConfig, RJMCMCChain
├── priors.jl       # UniformSpatialPrior, log_prior_k, log_prior_count
├── likelihood.jl   # Gaussian likelihood with systematic uncertainty
├── moves.jl        # RJMCMC moves: Birth, Death, Move, Allocate
├── hierarchical.jl # Hierarchical Bayes updates for μ and α
├── rjmcmc.jl       # run_bagol() - main entry point
├── mapn.jl         # estimate_mapn() - Hungarian matching for MAP-N
├── spatial.jl      # Coordinate utilities for partitioning
├── partition.jl    # Precision-weighted DBSCAN clustering
├── partitioned.jl  # Parallel BaGoL execution + boundary merging
├── visualization.jl # plot_bagol, plot_mapn, plot_hierarchical_diagnostics
└── simulation.jl   # simulate_smlm, simulate_grid, simulate_nmers
```

### Core Types
- `Emitter{T}` - Position (x, y) with allocated localization indices (parametric on Float type)
- `BaGoLState{T}` - Current MCMC state (emitters + log_posterior)
- `BaGoLSample{T}` - Recorded sample with μ and α values
- `RJMCMCChain{T}` - Full chain with samples, config, acceptance stats
- `BaGoLDiagnostics` - Diagnostics struct with n_emitters, posterior_k, acceptance_rates, final_μ, final_α
- `Partition` - Spatial cluster with locs, indices, and boundary flags

### Main API
```julia
# Primary workflow - returns (BasicSMLD, BaGoLDiagnostics)
result_smld, diagnostics = run_bagol(smld; n_iterations=10000, burn_in=2000)

# Key parameters
run_bagol(smld::SMLD;
    partition_threshold=500,       # Auto-partition if n_locs > threshold (0 = never)
    α::Union{Float64,Symbol}=2.0,  # Shape param or :auto
    learn_α::Bool=false,           # Update α during MCMC
    λ_K=length(locs)/5.0,          # Prior on emitter count
    sync_interval=500)             # Iterations between global μ/α updates (partitioned)

# Advanced: direct chain access
chain = run_bagol_chain(locs; n_iterations=10000, burn_in=2000)
emitters, posterior_k = estimate_mapn(chain)
```

### RJMCMC Algorithm
- 4 move types: Birth (10%), Death (10%), Move (20%), Allocate (60%)
- Posterior: Poisson prior on K, NegBinomial prior on counts, Gaussian likelihood
- Hierarchical: Gibbs updates for μ, optional MH updates for α

### Partitioned BaGoL (Large Datasets)
The main `run_bagol` automatically partitions when `n_locs > partition_threshold`.
For advanced control, use `run_bagol_partitioned` directly:

```julia
result = run_bagol_partitioned(locs;
    nsigma=4.0,              # DBSCAN threshold in sigma units
    min_partition_size=10,   # Minimum locs per partition
    max_partition_size=1000, # Target max locs per partition
    oversized=:split,        # :split or :skip oversized clusters
    boundary_margin=0.0,     # 0 = auto (5×median(σ))
    # ... all run_bagol_chain kwargs
)

# Access results
result.emitters             # Combined Emitter2DFit after boundary deduplication
result.posterior_k          # Combined K histogram
result.chains               # Individual RJMCMCChain per partition
result.partitions           # Partition definitions
result.skipped              # Oversized clusters that were skipped
```

Algorithm: Precision-weighted DBSCAN clustering where distance = `||p_i - p_j|| / (σ_i + σ_j)`.
Oversized clusters are recursively bisected along principal axis.
Boundary emitters are deduplicated via Hungarian matching.

## Git Workflow

Finding parent branch:
```bash
git show-branch | head -10
```
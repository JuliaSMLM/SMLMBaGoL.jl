# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SMLMBaGoL is a high-performance Julia package for Bayesian Grouping of Localizations (BaGoL) on single-molecule localization microscopy (SMLM) data. It uses reversible jump MCMC with hierarchical Bayesian priors to associate raw localizations with individual emitters, yielding higher precision emitter positions. The algorithms are based on https://doi.org/10.1101/752287.

**IMPORTANT**: This codebase underwent a complete architectural redesign. The current implementation is a flat, modern Julia package that replaces the previous nested module structure.

## Development Environment

```julia
# Activate the project
using Pkg
Pkg.activate(".")

# Install dependencies
Pkg.instantiate()
```

## Running Tests

```bash
# Run all tests (recommended)
julia --project=. test/runtests.jl

# Run tests with package development mode
julia --project=. -e "using Pkg; Pkg.test()"
```

## Core Architecture

### Flat Design Philosophy
The package uses a **flat architecture** with no nested modules. Everything is accessible directly from the main `SMLMBaGoL` module. This design prioritizes:
- Type stability and performance
- Direct integration with SMLMData ecosystem
- Generic programming over AbstractEmitter types
- Hierarchical Bayesian analysis as a first-class feature

### File Structure
```
src/
├── SMLMBaGoL.jl      # Main module with exports
├── types.jl          # Core type definitions  
├── bagol.jl          # Main API and clustering
├── rjmcmc.jl         # RJMCMC implementation
└── hierarchical.jl   # Hierarchical Bayesian updates
```

### Core Types
- **`BaGoLResult{T,O}`**: Main result container with chains, posterior, MAP-N emitters, and updated priors
- **`BaGoLChain{T,O}`**: Individual MCMC chain with states, log probabilities, allocations, and observations
- **`HierarchicalPrior{T}`**: Gamma distribution parameters (α, β) with hyperpriors for hierarchical Bayesian analysis
- **`MoveProbs{T}`**: RJMCMC move type probabilities (must sum to 1.0)

## Main API

### Single Entry Point
```julia
result = bagol(emitters; kwargs...)
```

The `bagol()` function is the **only** entry point and works with:
- `Vector{Emitter2DFit}` (from SMLMSim/SMLMData)
- `Vector{Emitter2D}` (basic position + photons)
- Any `Vector{<:AbstractEmitter}` type
- Objects with `.emitters` field (duck typing)

### Key Parameters
- `mcmc_steps::Int = 10_000`: Total MCMC steps per chain
- `burnin::Int = 2_000`: Burn-in steps (must be < mcmc_steps)
- `posterior_resolution::Float = 0.005`: Resolution for posterior image (5nm default)
- `subregion_radius::Float = auto`: DBSCAN clustering radius (auto-detected from σ fields)
- `prior::HierarchicalPrior = auto`: Hierarchical prior parameters
- `n_threads::Int = Threads.nthreads()`: Parallel processing threads

## Algorithm Flow

1. **Clustering**: DBSCAN spatial clustering of emitters into subregions
2. **Parallel RJMCMC**: Independent chains on each subregion with thread-local RNGs
3. **Hierarchical Updates**: Automatic prior parameter updates from MCMC samples
4. **Posterior Computation**: Accumulation of emitter positions into probability image
5. **MAP-N Extraction**: Most probable number of emitters and their positions

## RJMCMC Move Types
- **MOVE_EMITTER**: Adjust emitter position using weighted observation average
- **ADD_EMITTER**: Birth new emitter from observation mixture prior
- **REMOVE_EMITTER**: Death of existing emitter with dimension matching
- **REALLOCATE**: Gibbs sampling of observation-to-emitter assignments
- **SPLIT_EMITTER**: Split one emitter into two (placeholder in current implementation)
- **MERGE_EMITTER**: Merge two emitters into one (placeholder in current implementation)

## Type Adaptation

The code **automatically adapts** to different emitter types:
- **With uncertainty**: Uses `obs.σ_x`, `obs.σ_y` for likelihood calculations
- **Without uncertainty**: Defaults to σ = 0.1 for simple distance-based likelihood
- **Missing fields**: Uses `hasproperty()` checks throughout for robustness

## Performance Considerations

### Type Stability
- All hot paths are type-stable
- Generic programming with proper type parameters
- Zero-allocation RJMCMC steps

### Memory Efficiency
- Pre-allocated output arrays
- Thread-local RNGs to avoid contention
- Minimal copying of large data structures

### Parallelization
- `@threads` for subregion processing
- Each thread gets independent RNG seed
- Linear scaling with thread count

## Hierarchical Bayesian Framework

### Prior Structure
```julia
HierarchicalPrior{T}(α, β, a₀, b₀, c₀, d₀)
```
- `α, β`: Current Gamma parameters for λ (localizations per emitter)
- `a₀, b₀`: Hyperpriors on α (shape, rate)
- `c₀, d₀`: Hyperpriors on β (shape, rate)

### Automatic Updates
The package automatically updates hierarchical priors by:
1. Collecting λ samples (localizations per emitter) from all MCMC chains
2. Computing method-of-moments estimates for α, β
3. Applying shrinkage weighting between prior and data-driven estimates
4. Returning updated prior in `BaGoLResult.updated_prior`

## Integration with SMLMData Ecosystem

### Direct Compatibility
- No data conversion required
- Works with `SMLMSim.simulate()` output directly
- Supports any future `AbstractEmitter` subtypes
- Duck typing for SMLD-like containers

### Example Workflow
```julia
using SMLMSim, SMLMData, SMLMBaGoL

# Generate synthetic data
smld_true, smld_model, smld_noisy = SMLMSim.simulate(params)

# Run BaGoL directly (no conversion needed)
result = bagol(smld_noisy.emitters)

# Access results
mapn_emitters = result.mapn_emitters
posterior_image = result.posterior
updated_prior = result.updated_prior
```

## Key Differences from Legacy Code

### What Changed
- **No nested modules**: Direct access to all functions
- **Single API**: Only `bagol()` function needed
- **Type-generic**: Works with any AbstractEmitter subtype
- **Zero conversion**: Direct SMLMData integration
- **Enhanced hierarchical Bayes**: Automatic prior updates

### What's Preserved
- All core mathematical algorithms
- RJMCMC move types and acceptance probabilities
- Hierarchical Bayesian framework
- Scientific accuracy and validation

### Dependencies
Minimal dependency footprint:
- `SMLMData`: Core emitter types
- `Distributions`: Probability distributions
- `Clustering`: DBSCAN implementation
- `Statistics`, `StatsBase`, `LinearAlgebra`, `Random`: Standard library
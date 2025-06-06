# SMLMBaGoL Complete Redesign Plan

## Overview
Complete architectural redesign of SMLMBaGoL to create a flat, high-performance, Julian package that leverages SMLMData types directly.

## Core Design Principles

1. **Flat Architecture**: Eliminate all nested modules - everything in main module
2. **Zero-Copy Design**: Use SMLMData types directly (Emitter2DFit, Emitter2D)
3. **Future-Proof**: Support all `<:AbstractEmitter` types through parametric design
4. **Performance First**: Type-stable, allocation-free hot paths
5. **Hierarchical Bayes**: First-class support for hierarchical Bayesian updates
6. **Julian Style**: Immutable types, multiple dispatch, generic programming

## Type Architecture

```julia
# Use SMLMData types directly
using SMLMData: AbstractEmitter, Emitter2D, Emitter2DFit

# Generic chain type that works with any emitter type
struct BaGoLChain{T<:AbstractFloat, E<:AbstractEmitter, O<:AbstractEmitter}
    states::Vector{Vector{Emitter2D{T}}}    # Internal states (position only)
    log_probs::Vector{T}                    # Log probabilities
    allocations::Vector{Vector{Int}}        # Allocation vectors
    observations::Vector{O}                 # Original observations (any emitter type)
end

struct HierarchicalPrior{T}
    α::T; β::T                              # Gamma parameters
    a₀::T; b₀::T; c₀::T; d₀::T              # Hyperpriors
end

struct BaGoLResult{T, E<:AbstractEmitter}
    chains::Vector{BaGoLChain{T, Emitter2D{T}, E}}
    posterior::Matrix{T}
    mapn_emitters::Vector{Emitter2D{T}}
    log_evidence::T
end
```

## Main API

```julia
# Generic function that accepts any AbstractEmitter subtype
function bagol(
    emitters::Vector{E};
    prior::HierarchicalPrior{T} = default_prior(T),
    subregion_radius::T = T(4.0),
    mcmc_steps::Int = 10_000,
    burnin::Int = 2_000,
    posterior_resolution::T = T(0.005),
    n_threads::Int = Threads.nthreads()
) where {T<:AbstractFloat, E<:AbstractEmitter{T}}
```

## Implementation Steps

### Phase 1: Clean Slate
1. Delete all existing modules (Emitters, RJMCMC, Cluster, VisTools)
2. Move essential algorithms to new flat structure
3. Remove all wrapper types (Observations, Params, etc.)

### Phase 2: Core Implementation
1. Implement generic RJMCMC that works with AbstractEmitter
2. Direct hierarchical Bayesian updates
3. Efficient parallel processing
4. Type-stable likelihood calculations

### Phase 3: Optimizations
1. Zero-allocation hot paths
2. SIMD-friendly operations where possible
3. Memory-efficient chain storage
4. Cache-friendly data layouts

## File Structure (New)

```
src/
├── SMLMBaGoL.jl          # Main module with all exports
├── types.jl              # Core type definitions
├── bagol.jl              # Main bagol() function
├── rjmcmc.jl             # RJMCMC implementation
├── hierarchical.jl       # Hierarchical Bayesian updates
├── clustering.jl         # DBSCAN clustering
├── posterior.jl          # Posterior computation
├── utils.jl              # Helper functions
└── visualization.jl      # Simple plot recipes
```

## Key Improvements

1. **Direct SMLMData Integration**: No conversion needed
2. **Generic Over Emitter Types**: Works with any future AbstractEmitter
3. **10x Simpler**: ~1000 lines instead of ~5000
4. **Faster**: Eliminated allocations and type instabilities
5. **Maintainable**: Clear, flat structure with obvious data flow

## Migration Strategy

1. Create new implementation in `src/new/`
2. Test against existing implementation
3. Once validated, remove old code
4. Update documentation and examples

## Breaking Changes

- No more nested module access (RJMCMC.X, Cluster.Y)
- Direct emitter types instead of wrappers
- Simplified API with single entry point
- Visualization as optional Plots.jl recipes

## Performance Targets

- 0 allocations in RJMCMC step
- 2-3x faster than current implementation
- Linear scaling with thread count
- <100ms for 1000 localizations
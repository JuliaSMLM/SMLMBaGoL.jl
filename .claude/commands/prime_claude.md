# SMLMBaGoL Package Design and Implementation Plan

Evaluate the plan below and then think deeply about the next piece that can be implimented. Suggest a plan and ask for feedback, but don't code yet. 

## Overview

SMLMBaGoL (Single-Molecule Localization Microscopy Bayesian Grouping of Localizations) is a Julia package implementing the BaGoL algorithm for super-resolution microscopy data analysis. This document outlines the complete design and implementation plan for the package.

## Core Design Philosophy

1. **Leverage Julia's Multiple Dispatch**: The package extensively uses multiple dispatch to handle different emitter types, localization types, and move types elegantly.

2. **Mathematical Correctness by Construction**: The detailed balance requirements of RJMCMC are enforced through careful design of the move acceptance framework.

3. **Extensibility**: New emitter types, localization types, and move types can be easily added without modifying existing code.

4. **Performance**: Type stability and parallel processing are prioritized throughout.

5. **Simplicity**: Flat module structure with organized file hierarchy for clarity.

## File Structure

```
SMLMBaGoL/
├── src/
│   ├── SMLMBaGoL.jl                 # Main module file
│   │
│   ├── core/
│   │   ├── types.jl                 # Abstract types only
│   │   ├── state.jl                 # BaGoLState and chain structures
│   │   ├── priors.jl                # Prior distributions
│   │   └── likelihood.jl            # State-level likelihood calculation only
│   │
│   ├── localizations/
│   │   ├── localization2d.jl        # 2D localization type and methods
│   │   ├── localization3d.jl        # 3D localization type and methods
│   │   └── localization_utils.jl    # Shared localization utilities
│   │
│   ├── emitters/
│   │   ├── emitter2d.jl              # 2D emitter type and methods
│   │   ├── emitter3d.jl              # 3D emitter type and methods
│   │   └── emitter_utils.jl          # Shared emitter utilities
│   │
│   ├── moves/
│   │   ├── move_types.jl            # Move type definitions
│   │   ├── birth_death.jl           # Birth/Death pair implementation (includes acceptance)
│   │   ├── split_merge.jl           # Split/Merge pair implementation (includes acceptance)
│   │   ├── move.jl                  # Position update moves (includes acceptance)
│   │   ├── allocate.jl              # Allocation moves (includes acceptance)
│   │   └── acceptance.jl            # Thin acceptance probability dispatcher
│   │
│   ├── algorithms/
│   │   ├── clustering.jl            # DBSCAN clustering
│   │   ├── hierarchical.jl          # Hierarchical Bayesian updates
│   │   ├── mapn.jl                  # MAPN estimation
│   │   ├── posterior.jl             # Posterior image generation
│   │   └── rjmcmc.jl                # Main RJMCMC loop
│   │
│   └── utils/
│       ├── spatial.jl               # Spatial calculations
│       ├── statistics.jl            # Statistical utilities
│       └── parallel.jl              # Parallel processing utilities
```

### File Structure Rationale

- **Folders for organization, flat module structure**: This provides the best of both worlds - easy navigation during development while maintaining a simple API for users.
- **Separate files for each emitter/localization type**: Enables parallel development and makes it trivial to add new types.
- **Likelihood calculations with localizations**: Since likelihood depends fundamentally on localization uncertainties, each localization type defines how it matches against different emitter types in the same file.
- **Self-contained move files**: Each move file contains its type definition, proposal logic, and acceptance calculation. Birth/Death and Split/Merge are in single files because they are mathematically linked pairs.
- **Core algorithms separated**: The main algorithmic components (RJMCMC, clustering, etc.) are cleanly separated for maintainability.

## Type Hierarchy

### Abstract Types

```julia
# core/types.jl
abstract type AbstractLocalization end
abstract type AbstractEmitter end
abstract type AbstractPrior end
abstract type AbstractRJMCMCMove end
abstract type AbstractChainState end
```

### Design Rationale for Types

- **Abstract prefix naming**: Makes abstract types immediately recognizable
- **Separate hierarchies**: Localizations, emitters, and moves are orthogonal concepts
- **Simple move hierarchy**: All moves inherit directly from AbstractRJMCMCMove

## Mathematical Framework for Detailed Balance

### Core Insight

For reversible move pairs (Birth/Death, Split/Merge), the acceptance probabilities are mathematically related:

```
α(x → x') = min(1, R(x → x'))
α(x' → x) = min(1, R(x' → x))
```

Where `R(x' → x) = 1/R(x → x')` by detailed balance.

### Implementation Strategy

Instead of duplicating calculations, we compute the log acceptance ratio for one direction and negate it for the reverse. Each move file contains its own acceptance logic:

```julia
# moves/acceptance.jl - Just a thin dispatcher
function accept_probability(move_type::Type{<:AbstractRJMCMCMove}, current, proposed)
    # Dispatch to move-specific calculation
    log_ratio = log_acceptance_ratio(move_type, current, proposed)
    return exp(min(0.0, log_ratio))
end

# Fallback for simple Metropolis moves
function log_acceptance_ratio(::Type{<:AbstractRJMCMCMove}, current, proposed)
    log_posterior(proposed) - log_posterior(current)
end
```

Each move file implements its own `log_acceptance_ratio` method. For example, in `birth_death.jl`:

```julia
function log_acceptance_ratio(::Type{Birth}, current, proposed)
    log_acceptance_ratio_birth(current, proposed)
end

function log_acceptance_ratio(::Type{Death}, current, proposed)
    -log_acceptance_ratio_birth(proposed, current)  # Mathematical inverse!
end
```

This ensures detailed balance is maintained by construction and eliminates code duplication.

## Key Type Implementations

### Localizations

```julia
# localizations/localization2d.jl
struct Localization2D{T<:Real} <: AbstractLocalization
    x::T
    y::T
    σx::T
    σy::T
    frame::Int
end

# Each localization type includes its likelihood calculations
function log_likelihood(emitter::Emitter2D, loc::Localization2D)
    # Implementation...
end

function log_likelihood(emitter::Emitter3D, loc::Localization2D)
    # Implementation...
end

# localizations/localization3d.jl
struct Localization3D{T<:Real} <: AbstractLocalization
    x::T
    y::T
    z::T
    σx::T
    σy::T
    σz::T
    frame::Int
end

function log_likelihood(emitter::Emitter2D, loc::Localization3D)
    # Implementation...
end

function log_likelihood(emitter::Emitter3D, loc::Localization3D)
    # Implementation...
end
```

### Emitters

```julia
# emitters/emitter2d.jl
struct Emitter2D{T<:Real} <: AbstractEmitter
    x::T
    y::T
    id::Int
end

# emitters/emitter3d.jl
struct Emitter3D{T<:Real} <: AbstractEmitter
    x::T
    y::T
    z::T
    id::Int
end
```

### Chain State

```julia
# core/state.jl
struct BaGoLState{E<:AbstractEmitter, L<:AbstractLocalization, T<:Real} <: AbstractChainState
    emitters::Vector{E}
    localizations::Vector{L}
    allocations::Vector{Int}  # maps localization i to emitter allocations[i]
    prior::AbstractPrior
    log_likelihood::T
end

mutable struct RJMCMCChain{T<:Real, E<:AbstractEmitter, L<:AbstractLocalization, P<:AbstractPrior}
    localizations::Vector{L}
    current_state::BaGoLState{E,L,T}
    prior::P
    move_weights::Dict{Type{<:AbstractRJMCMCMove}, Float64}
    samples::Vector{BaGoLState{E,L,T}}
    burn_in::Int
    thin::Int
    rng::AbstractRNG
end
```

## Move System Implementation

### Move Types

```julia
# moves/move_types.jl
struct Birth <: AbstractRJMCMCMove end
struct Death <: AbstractRJMCMCMove end
struct Split <: AbstractRJMCMCMove end
struct Merge <: AbstractRJMCMCMove end
struct Move <: AbstractRJMCMCMove end
struct Allocate <: AbstractRJMCMCMove end

# Define inverse relationships
inverse_move(::Type{Birth}) = Death
inverse_move(::Type{Death}) = Birth
inverse_move(::Type{Split}) = Merge
inverse_move(::Type{Merge}) = Split
```

### Birth/Death Implementation

```julia
# moves/birth_death.jl
# Move type definitions
struct Birth <: AbstractRJMCMCMove end
struct Death <: AbstractRJMCMCMove end

inverse_move(::Type{Birth}) = Death
inverse_move(::Type{Death}) = Birth

# Proposal functions
function propose_move(::Type{Birth}, state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    new_state = deepcopy(state)
    
    # Sample new emitter from spatial prior
    new_emitter = sample_emitter_from_prior(state.localizations, E, rng)
    push!(new_state.emitters, new_emitter)
    
    # Reallocate
    new_state.allocations = propose_reallocation(new_state, rng)
    
    return new_state
end

function propose_move(::Type{Death}, state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    length(state.emitters) == 0 && return nothing
    
    new_state = deepcopy(state)
    idx = rand(rng, 1:length(state.emitters))
    deleteat!(new_state.emitters, idx)
    new_state.allocations = reallocate_from_removed(new_state.allocations, idx, rng)
    
    return new_state
end

# Core acceptance ratio calculation for birth (death uses negative)
function log_acceptance_ratio_birth(current::BaGoLState, proposed::BaGoLState)
    @assert length(proposed.emitters) == length(current.emitters) + 1
    
    # Prior ratio: P(K+1)/P(K)
    log_prior_ratio = log_prior_K(length(proposed.emitters), current.prior) - 
                      log_prior_K(length(current.emitters), current.prior)
    
    # Likelihood ratio
    log_likelihood_ratio = log_likelihood(proposed) - log_likelihood(current)
    
    # Proposal ratio: q(death)/q(birth)
    new_emitter = proposed.emitters[end]
    log_q_death = -log(length(proposed.emitters))
    log_q_birth = log_spatial_prior_density(new_emitter, current.localizations)
    
    return log_prior_ratio + log_likelihood_ratio + log_q_death - log_q_birth
end

# Move-specific acceptance calculations
function log_acceptance_ratio(::Type{Birth}, current::BaGoLState, proposed::BaGoLState)
    log_acceptance_ratio_birth(current, proposed)
end

function log_acceptance_ratio(::Type{Death}, current::BaGoLState, proposed::BaGoLState)
    -log_acceptance_ratio_birth(proposed, current)  # Mathematical inverse!
end
```

## Likelihood Calculations

Likelihood calculations are implemented with the localization types, as localizations define how they match against different emitter types. Multiple dispatch enables clean handling of different emitter/localization combinations:

```julia
# localizations/localization2d.jl
struct Localization2D{T<:Real} <: AbstractLocalization
    x::T
    y::T
    σx::T
    σy::T
    frame::Int
end

# 2D localization with 2D emitter
function log_likelihood(emitter::Emitter2D, loc::Localization2D)
    dx = (loc.x - emitter.x) / loc.σx
    dy = (loc.y - emitter.y) / loc.σy
    return -0.5 * (dx^2 + dy^2) - log(2π * loc.σx * loc.σy)
end

# 2D localization with 3D emitter (can't see z)
function log_likelihood(emitter::Emitter3D, loc::Localization2D)
    dx = (loc.x - emitter.x) / loc.σx
    dy = (loc.y - emitter.y) / loc.σy
    return -0.5 * (dx^2 + dy^2) - log(2π * loc.σx * loc.σy)
end

# localizations/localization3d.jl
struct Localization3D{T<:Real} <: AbstractLocalization
    x::T
    y::T
    z::T
    σx::T
    σy::T
    σz::T
    frame::Int
end

# 3D localization with 2D emitter (project to 2D)
function log_likelihood(emitter::Emitter2D, loc::Localization3D)
    dx = (loc.x - emitter.x) / loc.σx
    dy = (loc.y - emitter.y) / loc.σy
    return -0.5 * (dx^2 + dy^2) - log(2π * loc.σx * loc.σy)
end

# 3D localization with 3D emitter
function log_likelihood(emitter::Emitter3D, loc::Localization3D)
    dx = (loc.x - emitter.x) / loc.σx
    dy = (loc.y - emitter.y) / loc.σy
    dz = (loc.z - emitter.z) / loc.σz
    return -0.5 * (dx^2 + dy^2 + dz^2) - log((2π)^(3/2) * loc.σx * loc.σy * loc.σz)
end

# core/likelihood.jl - only contains state-level calculation
function log_likelihood(state::BaGoLState)
    ll = 0.0
    for (i, loc) in enumerate(state.localizations)
        emitter_idx = state.allocations[i]
        if 1 ≤ emitter_idx ≤ length(state.emitters)
            ll += log_likelihood(state.emitters[emitter_idx], loc)
        end
    end
    return ll
end
```

### Design Rationale

Likelihood calculations belong with localization types because:
1. The likelihood fundamentally depends on the localization's uncertainty parameters
2. When adding new localization types, their likelihood models are defined in the same file
3. This follows the principle that "code that changes together stays together"

## Hierarchical Bayesian Integration

```julia
# core/priors.jl
mutable struct HierarchicalGammaPrior <: AbstractPrior
    α::Float64  # shape parameter
    β::Float64  # scale parameter
    α_prior::Tuple{Float64, Float64}  # (a₀, b₀)
    β_prior::Tuple{Float64, Float64}  # (c₀, d₀)
end

# algorithms/hierarchical.jl
function update_hierarchical!(chains::Vector{<:RJMCMCChain}, prior::HierarchicalGammaPrior)
    # Collect allocation counts from all chains
    counts = Int[]
    for chain in chains
        append!(counts, count_allocations(chain.current_state))
    end
    
    # Sample new hyperparameters using Metropolis-Hastings
    α_new, β_new = sample_gamma_hyperparameters(
        counts, prior.α, prior.β, prior.α_prior, prior.β_prior
    )
    
    # Update prior in all chains
    new_prior = HierarchicalGammaPrior(α_new, β_new, prior.α_prior, prior.β_prior)
    for chain in chains
        chain.prior = new_prior
        chain.current_state.prior = new_prior
    end
end
```

## Main Algorithm Flow

```julia
# algorithms/rjmcmc.jl
function run_hierarchical_bagol(clusters::Vector{LocalizationCluster{L}}, 
                               n_iterations::Int, 
                               hierarchical_interval::Int) where L<:AbstractLocalization
    # Initialize chains for each cluster
    chains = [initialize_chain(cluster) for cluster in clusters]
    
    for iter in 1:n_iterations
        # Run RJMCMC in parallel
        Threads.@threads for chain in chains
            rjmcmc_step!(chain)
        end
        
        # Hierarchical update at intervals
        if iter % hierarchical_interval == 0
            update_hierarchical!(chains)
        end
    end
    
    return chains
end

function rjmcmc_step!(chain::RJMCMCChain)
    # Select move type based on weights
    move_type = sample_move_type(chain.move_weights)
    
    # Propose new state
    proposed = propose_move(move_type, chain.current_state, chain.rng)
    proposed === nothing && return  # Move not possible
    
    # Accept/reject
    if rand(chain.rng) < accept_probability(move_type, chain.current_state, proposed)
        chain.current_state = proposed
    end
    
    # Store sample if needed
    if should_store_sample(chain)
        push!(chain.samples, deepcopy(chain.current_state))
    end
end
```

## Extension Points

### Adding New Emitter Types

To add a new emitter type (e.g., rotating dipole):

1. Create `emitters/emitter_rotating2d.jl`:
```julia
struct RotatingEmitter2D{T<:Real} <: AbstractEmitter
    x::T
    y::T
    θ::T  # orientation
    id::Int
end

# Position update method
function update_emitter_position(emitter::RotatingEmitter2D, assigned_locs, rng)
    # Update position and orientation
end
```

2. Add likelihood calculations to existing localization files:
```julia
# In localizations/localization2d.jl
function log_likelihood(emitter::RotatingEmitter2D, loc::Localization2D)
    # Implementation with orientation-dependent PSF
end

# In localizations/localization3d.jl
function log_likelihood(emitter::RotatingEmitter2D, loc::Localization3D)
    # Implementation
end
```

3. Include the new emitter file in `SMLMBaGoL.jl`

### Adding New Localization Types

To add a new localization type (e.g., multi-color):

1. Create `localizations/localization_multicolor.jl`:
```julia
struct LocalizationMulticolor{T<:Real} <: AbstractLocalization
    x::T
    y::T
    σx::T
    σy::T
    frame::Int
    channel::Int
    intensity::T
end

# Define likelihood for all existing emitter types
function log_likelihood(emitter::Emitter2D, loc::LocalizationMulticolor)
    # Standard spatial likelihood
    dx = (loc.x - emitter.x) / loc.σx
    dy = (loc.y - emitter.y) / loc.σy
    return -0.5 * (dx^2 + dy^2) - log(2π * loc.σx * loc.σy)
end

function log_likelihood(emitter::Emitter3D, loc::LocalizationMulticolor)
    # Implementation
end
```

2. Include the new localization file in `SMLMBaGoL.jl`

### Adding New Move Types

To add a new move type (e.g., a Shift move that translates all emitters):

1. Create `moves/shift.jl`:
```julia
struct Shift <: AbstractRJMCMCMove end

# Proposal function
function propose_move(::Type{Shift}, state::BaGoLState{E,L,T}, rng=Random.GLOBAL_RNG) where {E,L,T}
    new_state = deepcopy(state)
    
    # Random small translation
    dx = randn(rng) * 0.1
    dy = randn(rng) * 0.1
    
    # Apply to all emitters
    for i in eachindex(new_state.emitters)
        e = new_state.emitters[i]
        new_state.emitters[i] = E(e.x + dx, e.y + dy, e.id)
    end
    
    return new_state
end

# Acceptance calculation (simple Metropolis)
function log_acceptance_ratio(::Type{Shift}, current::BaGoLState, proposed::BaGoLState)
    # For symmetric proposals, just return likelihood ratio
    log_likelihood(proposed) - log_likelihood(current)
end
```

2. Include in `SMLMBaGoL.jl`:
```julia
include("moves/shift.jl")
```

3. Add to default move weights in chain initialization

This self-contained approach makes it trivial to add new moves without modifying any existing files.

## Implementation Notes

1. **Type Stability**: All functions should be type-stable for performance
2. **Random Number Generation**: Always pass RNG explicitly for reproducibility
3. **Parallel Safety**: Cluster processing is embarrassingly parallel
4. **Memory Efficiency**: Use views where possible, deepcopy only when necessary
5. **Numerical Stability**: Work in log-space for probabilities
6. **Cross-type Compatibility**: When adding new localization types, implement likelihood methods for all existing emitter types, and vice versa

## Testing Strategy

Each component should have corresponding tests:
- Unit tests for each move type
- Property-based tests for detailed balance
- Integration tests for full RJMCMC chains
- Performance benchmarks for key operations

## Future Extensions

The design accommodates future additions:
- Additional localization types (e.g., multi-color, polarization)
- Complex emitter models (e.g., coupled emitters, time-varying positions)
- Alternative clustering algorithms
- GPU acceleration for likelihood calculations
- Advanced visualization and analysis tools
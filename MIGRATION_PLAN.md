# Migration Plan: Localization2D → Emitter2DFit

## Overview
This document outlines the plan to replace the custom `Localization2D` type with SMLMData's `Emitter2DFit` throughout the SMLMBaGoL codebase for better consistency and integration with the SMLM ecosystem.

## Key Differences

### Localization2D (current)
```julia
struct Localization2D{T} <: AbstractObservation
    y::T
    x::T
    σ_y::T
    σ_x::T
end
```

### Emitter2DFit (target)
```julia
# From SMLMData - fields in order:
# x, y, photons, bg, σ_x, σ_y, σ_photons, σ_bg, frame, dataset, track_id, id
```

## Migration Strategy

### Phase 1: Add Compatibility Layer
1. Create conversion functions between types
2. Add constructor overloads for smooth transition
3. Ensure all tests still pass

### Phase 2: Update Core Functions
1. Modify `Observations` type to work with `Emitter2DFit`
2. Update RJMCMC functions to handle the new type
3. Adjust field access patterns (x/y order reversal)

### Phase 3: Update Downstream Code
1. Visualization functions
2. MAP-N extraction
3. Example scripts

## Implementation Steps

### Step 1: Create Compatibility Functions
```julia
# In src/emitters/emitters2D.jl

# Convert Localization2D to Emitter2DFit
function Base.convert(::Type{Emitter2DFit{T}}, loc::Localization2D{T}) where T
    return Emitter2DFit(
        loc.x,           # x
        loc.y,           # y  
        T(1000),         # photons (default)
        T(0),            # bg (default)
        loc.σ_x,         # σ_x
        loc.σ_y,         # σ_y
        T(0),            # σ_photons (default)
        T(0),            # σ_bg (default)
        0,               # frame (default)
        0,               # dataset (default)
        0,               # track_id (default)
        0                # id (default)
    )
end

# Convert Emitter2DFit to Localization2D (for backward compatibility)
function Localization2D(em::Emitter2DFit{T}) where T
    return Localization2D{T}(em.y, em.x, em.σ_y, em.σ_x)
end

# Helper to create minimal Emitter2DFit from position and uncertainty
function Emitter2DFit(x::T, y::T, σ_x::T, σ_y::T) where T
    return Emitter2DFit(x, y, T(1000), T(0), σ_x, σ_y, T(0), T(0), 0, 0, 0, 0)
end
```

### Step 2: Update Type Definitions
```julia
# Option A: Redefine Observations to accept AbstractEmitter
struct Observations{T<:AbstractEmitter}
    ŷ::Vector{T}
end

# Option B: Create type alias for compatibility
const ObservationEmitter = Union{AbstractObservation, Emitter2DFit}
```

### Step 3: Update Core Functions

#### log_p_z_given_y
```julia
# Before
function log_p_z_given_y(loc::Localization2D, emitter::Emitter2D)
    return logpdf(Normal(loc.x, loc.σ_x), emitter.x) +
           logpdf(Normal(loc.y, loc.σ_y), emitter.y)
end

# After - works with both types
function log_p_z_given_y(obs::Emitter2DFit, emitter::Emitter2D)
    return logpdf(Normal(obs.x, obs.σ_x), emitter.x) +
           logpdf(Normal(obs.y, obs.σ_y), emitter.y)
end
```

#### build_prior_y
```julia
# Updated to work with Emitter2DFit
function build_prior_y(obs::Observations{<:Emitter2DFit}) 
    means = [[ob.y, ob.x] for ob in obs.ŷ]
    covs = [[ob.σ_y^2 0.0; 0.0 ob.σ_x^2] for ob in obs.ŷ]
    components = [MvNormal(mean, cov) for (mean, cov) in zip(means, covs)]
    weights = fill(1.0/length(obs.ŷ), length(obs.ŷ))
    return MixtureModel(components, weights)
end
```

#### gen_observations
```julia
# Updated to return Emitter2DFit objects
function gen_observations(emitter_type::Type{<:Emitter2D}, positions, sigmas)
    y = positions[1, :]
    x = positions[2, :]
    σ_y = sigmas[1, :]
    σ_x = sigmas[2, :]
    
    # Create Emitter2DFit objects with minimal required fields
    emitters = [Emitter2DFit(x[i], y[i], σ_x[i], σ_y[i]) for i in 1:length(x)]
    return Observations(emitters)
end
```

#### get_mapn_emitters
```julia
# Return Emitter2DFit instead of Localization2D
function get_mapn_emitters(chain_mapn_sorted::RJMCMC_Chain, obs::Observations)
    n_map = length(chain_mapn_sorted.states[1].emitters)
    n_states = length(chain_mapn_sorted.states)
    
    # Return vector of Emitter2DFit
    mapn_emitters = Vector{Emitter2DFit{Float64}}(undef, n_map)
    
    # ... calculation logic ...
    
    for i in 1:n_map
        # Create Emitter2DFit with position and uncertainty
        mapn_emitters[i] = Emitter2DFit(
            coords[2, i],  # x (note: reversed from Localization2D)
            coords[1, i],  # y
            σs[2, i],      # σ_x
            σs[1, i]       # σ_y
        )
    end
    
    return mapn_emitters
end
```

### Step 4: Update Visualization
```julia
# Plots already work with any object that has x, y, σ_x, σ_y fields
# Just need to ensure proper field access
```

## Benefits of Migration

1. **Consistency**: Use standard SMLMData types throughout
2. **Metadata Preservation**: Can track photons, background, frame info
3. **Ecosystem Integration**: Better compatibility with other SMLM packages
4. **Future Features**: Can leverage photon counts for weighted fitting

## Testing Plan

1. Add tests for conversion functions
2. Verify existing tests pass with compatibility layer
3. Update tests to use new types directly
4. Performance benchmarks to ensure no regression

## Rollback Plan

Keep `Localization2D` type definition and conversion functions for backward compatibility during transition period. Can be deprecated and removed in future version.
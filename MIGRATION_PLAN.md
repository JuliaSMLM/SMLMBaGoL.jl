# Migration Plan: Localization2D → Emitter2DFit (Direct Migration)

## Overview
Direct replacement of the custom `Localization2D` type with SMLMData's `Emitter2DFit` throughout the SMLMBaGoL codebase. No backward compatibility needed.

## Key Differences

### Localization2D (to be removed)
```julia
struct Localization2D{T} <: AbstractObservation
    y::T
    x::T
    σ_y::T
    σ_x::T
end
```

### Emitter2DFit (replacement)
```julia
# From SMLMData - fields in order:
# x, y, photons, bg, σ_x, σ_y, σ_photons, σ_bg, frame, dataset, track_id, id
```

## Direct Migration Steps

### Step 1: Remove Localization2D Type
- Delete `Localization2D` struct definition from `src/emitters/types.jl`
- Remove `AbstractObservation` type if no longer needed

### Step 2: Update Observations Type
```julia
# In src/types.jl
struct Observations{T<:Emitter2DFit}
    ŷ::Vector{T}
end
```

### Step 3: Update Core Functions

#### log_p_z_given_y
```julia
function log_p_z_given_y(obs::Emitter2DFit, emitter::Emitter2D)
    return logpdf(Normal(obs.x, obs.σ_x), emitter.x) +
           logpdf(Normal(obs.y, obs.σ_y), emitter.y)
end
```

#### build_prior_y
```julia
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
# Helper to create minimal Emitter2DFit from position and uncertainty
function create_minimal_emitter2dfit(x::T, y::T, σ_x::T, σ_y::T) where T
    return Emitter2DFit(x, y, T(1000), T(0), σ_x, σ_y, T(0), T(0), 0, 0, 0, 0)
end

function gen_observations(emitter_type::Type{<:Emitter2D}, positions, sigmas)
    y = positions[1, :]
    x = positions[2, :]
    σ_y = sigmas[1, :]
    σ_x = sigmas[2, :]
    
    # Create Emitter2DFit objects with minimal required fields
    emitters = [create_minimal_emitter2dfit(x[i], y[i], σ_x[i], σ_y[i]) for i in 1:length(x)]
    return Observations(emitters)
end
```

#### get_mapn_emitters
```julia
function get_mapn_emitters(chain_mapn_sorted::RJMCMC_Chain, obs::Observations)
    n_map = length(chain_mapn_sorted.states[1].emitters)
    n_states = length(chain_mapn_sorted.states)
    
    # Return vector of Emitter2DFit
    mapn_emitters = Vector{Emitter2DFit{Float64}}(undef, n_map)
    
    # ... calculation logic ...
    
    for i in 1:n_map
        # Create Emitter2DFit with position and uncertainty
        mapn_emitters[i] = create_minimal_emitter2dfit(
            coords[2, i],  # x
            coords[1, i],  # y
            σs[2, i],      # σ_x
            σs[1, i]       # σ_y
        )
    end
    
    return mapn_emitters
end
```

### Step 4: Update interface.jl
- Update `perform_BaGoL_analysis` to work with SMLD2D directly
- Remove any Localization2D creation/conversion

### Step 5: Update Tests
- Replace all Localization2D usage with Emitter2DFit
- Update test data generation

## Benefits of Migration

1. **Consistency**: Use standard SMLMData types throughout
2. **Metadata Preservation**: Can track photons, background, frame info
3. **Ecosystem Integration**: Better compatibility with other SMLM packages
4. **Future Features**: Can leverage photon counts for weighted fitting

## Testing Plan

1. Update tests to use Emitter2DFit directly
2. Verify all tests pass
3. Performance benchmarks to ensure no regression
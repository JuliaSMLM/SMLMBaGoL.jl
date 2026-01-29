# SMLMBaGoL API Overview

Machine-readable API reference for LLM tools and documentation generators.

## Module: SMLMBaGoL

Bayesian Grouping of Localizations (BaGoL) for SMLM data. Groups raw localizations into emitter positions with uncertainties using Reversible Jump MCMC.

### Main Entry Points

```julia
run_bagol(smld::SMLD; kwargs...) -> (BasicSMLD, BaGoLDiagnostics)
```
Run BaGoL analysis on SMLD. Partitions data via precision-weighted DBSCAN, runs parallel MCMC with global hierarchical updates, merges results.

**Partitioning kwargs:**
- `nsigma::Float64=3.0` - DBSCAN threshold in sigma units (Inf=no partitioning)
- `min_partition_size::Int=0` - Minimum locs per partition
- `max_partition_size::Int=1000` - Split partitions larger than this
- `skip_partition_size::Int=typemax(Int)` - Skip partitions larger than this

**MCMC kwargs:**
- `n_iterations::Int=10000` - Total MCMC iterations
- `burn_in::Int=2000` - Burn-in before recording samples
- `sync_interval::Int=500` - Iterations between global μ/shape updates
- `shape::Float64=2.0` - Initial Gamma shape (1=exponential, higher=peaked)
- `learn_shape::Bool=true` - Update shape during MCMC
- `λ_K::Float64=n_locs/5` - Poisson prior mean for K
- `verbose::Bool=true` - Print progress

```julia
run_bagol(locs::Vector{<:AbstractEmitter}; camera, kwargs...) -> (BasicSMLD, BaGoLDiagnostics)
```
Run BaGoL on vector of localizations. Requires camera argument.

```julia
run_bagol_chain(locs::Vector{<:AbstractEmitter}; kwargs...) -> RJMCMCChain
```
Advanced: Run MCMC and return full chain for diagnostics/visualization.

**Additional kwargs:**
- `callback::Function` - Called each iteration: `callback(iter, move_type, accepted, state, μ, shape)`
- `callback_interval::Int=1` - Callback frequency

```julia
estimate_mapn(chain::RJMCMCChain; n_refine=10) -> (Vector{Emitter2DFit}, Vector{Int})
```
Extract MAP-N emitters from chain using iterative Hungarian matching with MAD-based uncertainties.

**Returns:** `(emitters, posterior_k)` where posterior_k[k+1] = count of samples with K=k.

### Partitioning

```julia
partition_locs(locs; nsigma=3.0, min_size=0, max_size=1000, skip_size=typemax(Int), boundary_margin=0.0)
    -> (Vector{Partition}, Vector{Partition})
```
Partition localizations using precision-weighted DBSCAN.

**Returns:** `(partitions, skipped)` - valid partitions and skipped oversized clusters.

### Types

```julia
struct Emitter{T<:AbstractFloat}
    x::T
    y::T
    allocated::Vector{Int}  # Indices of assigned localizations
end
```
Emitter position with allocated localization indices.

```julia
struct BaGoLState{T<:AbstractFloat}
    emitters::Vector{Emitter{T}}
    log_posterior::Float64
end
```
Current MCMC state.

```julia
struct BaGoLSample{T<:AbstractFloat}
    emitters::Vector{Emitter{T}}
    log_posterior::Float64
    μ::Float64      # Mean locs per emitter
    shape::Float64  # Gamma shape parameter
end
```
Recorded sample from chain.

```julia
@kwdef struct RJMCMCConfig
    shape::Float64 = 2.0           # Gamma shape for count distribution
    λ_K::Float64 = 10.0            # Poisson prior mean for K
    μ_prior_shape::Float64 = 2.0   # Gamma hyperprior shape for μ
    μ_prior_scale::Float64 = 5.0   # Gamma hyperprior scale for μ
    shape_prior_shape::Float64 = 2.0
    shape_prior_scale::Float64 = 1.0
    n_iterations::Int = 10000
    burn_in::Int = 2000
    hierarchical_interval::Int = 100
end
```
RJMCMC configuration.

```julia
mutable struct RJMCMCChain{T<:AbstractFloat}
    config::RJMCMCConfig
    samples::Vector{BaGoLSample{T}}
    μ::Float64
    shape::Float64
    learn_shape::Bool
    current_state::BaGoLState{T}
    iteration::Int
    acceptance::Dict{Symbol, Tuple{Int, Int}}
end
```
Full RJMCMC chain with samples and diagnostics.

```julia
struct BaGoLDiagnostics
    n_emitters::Int
    posterior_k::Vector{Int}
    acceptance_rates::Dict{Symbol, Float64}
    final_μ::Float64
    final_shape::Float64
    n_partitions::Int
end
```
Diagnostics from BaGoL analysis.

```julia
struct Partition{E<:AbstractEmitter}
    id::Int
    locs::Vector{E}
    original_indices::Vector{Int}
    is_boundary::BitVector
    parent_id::Int
end
```
Partition of localizations for parallel processing.

```julia
struct UniformSpatialPrior
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64
end
```
Uniform spatial prior over rectangular region.

### Simulation

```julia
simulate_smlm(true_positions; μ=10.0, α=2.0, σ_psf=0.130, mean_photons=500.0,
              photon_shape=2.0, background=10.0, n_frames=1000) -> SimulationResult
```
Generate synthetic SMLM localizations from known emitter positions.

```julia
simulate_grid(; nx=3, ny=3, spacing=0.050, offset=(0.1,0.1), kwargs...) -> SimulationResult
```
Generate emitters on a regular grid.

```julia
simulate_nmers(; n_dimers=4, n_trimers=2, emitter_spacing=0.040,
               cluster_spacing=0.200, kwargs...) -> SimulationResult
```
Generate n-mers (dimers, trimers) for resolution testing.

```julia
print_simulation_summary(result::SimulationResult)
```
Print summary statistics.

```julia
crlb_precision(σ_psf, photons, background) -> Float64
```
Calculate localization precision from CRLB: `σ_loc ≈ σ_psf/√N * √(1 + bg/N)`.

```julia
struct SimulationResult
    localizations::Vector{Emitter2DFit}
    true_positions::Vector{Tuple{Float64, Float64}}
    true_counts::Vector{Int}
    μ::Float64
    α::Float64
    n_frames::Int
end
```

### Priors (Internal, Exported for Advanced Use)

```julia
log_prior_k(k::Int, λ_K::Float64) -> Float64
```
Log Poisson prior on number of emitters K.

```julia
log_prior_total_count(N::Int, K::Int, μ::Float64, shape::Float64) -> Float64
```
Log prior on total count N given K emitters. Uses Gamma(K*shape, μ/shape) marginal.

## Dependencies

- `SMLMData` - Core types (Emitter2DFit, BasicSMLD, IdealCamera, AbstractEmitter)
- `Hungarian` - Optimal assignment for MAP-N estimation
- `NearestNeighbors` - KDTree for spatial clustering
- `Distributions` - Statistical distributions
- `StaticArrays` - Efficient small arrays

## Units Convention

- All positions and uncertainties in **micrometers (μm)**
- Sigma values represent 1-standard-deviation uncertainties

## Count Model

The count distribution n_j ~ Gamma(shape, μ/shape) gives:
- E[n_j] = μ (mean localizations per emitter)
- CV[n_j] = 1/√shape (coefficient of variation)
- shape=1: Exponential (dSTORM)
- shape>1: Peaked (DNA-PAINT-like)

## Typical Workflow

```julia
using SMLMBaGoL, SMLMData

# Load or create data
camera = IdealCamera(64, 64, 0.1)  # 64x64 pixels, 100nm pixel size
smld = BasicSMLD(locs, camera, n_frames, n_datasets)

# Run BaGoL
result_smld, diagnostics = run_bagol(smld;
    n_iterations=10000,
    burn_in=2000,
    shape=2.0,
    learn_shape=true
)

# Access results
result_smld.emitters  # Vector{Emitter2DFit} with grouped positions
diagnostics.n_emitters
diagnostics.posterior_k
diagnostics.acceptance_rates
```

## Advanced: Chain Access

```julia
chain = run_bagol_chain(locs; n_iterations=10000, burn_in=2000)
emitters, posterior_k = estimate_mapn(chain)

# Animation callback
records = []
chain = run_bagol_chain(locs;
    callback = (i, mt, acc, st, μ, shape) -> push!(records, (i, length(st.emitters))),
    callback_interval = 10
)
```

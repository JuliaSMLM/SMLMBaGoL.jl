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
- `τ` (systematic uncertainty) is required with no default - must be set explicitly

## Architecture

### Source File Organization
```
src/
├── SMLMBaGoL.jl   # Module entry: all imports + exports
├── types.jl       # Emitter, BaGoLState, BaGoLSample, RJMCMCConfig, RJMCMCChain
├── priors.jl      # UniformSpatialPrior, log_prior_k, log_prior_count
├── likelihood.jl  # Gaussian likelihood with systematic uncertainty
├── moves.jl       # RJMCMC moves: Birth, Death, Move, Allocate
├── hierarchical.jl # Hierarchical Bayes updates for μ and α
├── rjmcmc.jl      # run_bagol() - main entry point
├── mapn.jl        # estimate_mapn() - Hungarian matching for MAP-N
├── visualization.jl # plot_bagol, plot_mapn, plot_hierarchical_diagnostics
└── simulation.jl  # simulate_smlm, simulate_grid, simulate_nmers
```

### Core Types
- `Emitter` - Position (x, y) with allocated localization indices
- `BaGoLState` - Current MCMC state (emitters + log_posterior)
- `BaGoLSample` - Recorded sample with μ and α values
- `RJMCMCChain` - Full chain with samples, config, acceptance stats

### Main API
```julia
# Primary workflow
chain = run_bagol(locs; τ=0.005, n_iterations=10000, burn_in=2000)
result = estimate_mapn(chain)

# Key parameters
run_bagol(locs;
    τ::Float64,                    # Required: systematic uncertainty
    α::Union{Float64,Symbol}=2.0,  # Shape param or :auto
    learn_α::Bool=false,           # Update α during MCMC
    λ_K=length(locs)/5.0,          # Prior on emitter count
    hierarchical_interval=100)     # Iterations between μ/α updates
```

### RJMCMC Algorithm
- 4 move types: Birth (10%), Death (10%), Move (20%), Allocate (60%)
- Posterior: Poisson prior on K, NegBinomial prior on counts, Gaussian likelihood
- Hierarchical: Gibbs updates for μ, optional MH updates for α

## Git Workflow

Finding parent branch:
```bash
git show-branch | head -10
```
# SMLMBaGoL Examples

This folder contains example scripts demonstrating SMLMBaGoL functionality.

## Setup

The examples folder has its own Julia environment with SMLMBaGoL as a development dependency and additional packages like CairoMakie for visualization.

### First-time setup (already done):
```bash
cd examples
julia --project=. -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'
```

## Running Examples

Each example script includes `Pkg.activate("examples")` at the top, so you can run them directly from the repository root without specifying the project environment:

```bash
# From SMLMBaGoL repository root:
julia examples/basic_demo.jl
```

This is much simpler than:
```bash
# Alternative (not needed):
cd examples && julia --project=. basic_demo.jl
```

## Available Examples

### `basic_demo.jl`
Complete simulation and analysis workflow:
- Generates synthetic 6-mer complex data
- Runs BaGoL analysis with realistic priors
- Compares estimated vs true emitter positions
- Demonstrates the full pipeline from simulation to results

**Key features demonstrated:**
- `simulate_n_mer()` with customizable parameters
- `run_bagol()` with simulation-generated priors
- Ground truth comparison and success metrics
- Realistic photon noise modeling

**Expected output:**
- ~150 localizations from 6-mer simulation
- BaGoL analysis with RJMCMC sampling
- Position recovery within ~50 nm of true locations
- Emitter count estimation (may vary due to stochasticity)

## Adding New Examples

1. Create new `.jl` file in this folder
2. Add `using Pkg; Pkg.activate("examples")` at the top
3. Add any new dependencies to `Project.toml` if needed
4. Document the example here in this file

## Notes

- Examples use the development version of SMLMBaGoL from `../`
- Changes to the main package are immediately available in examples
- Each example is self-contained and can be run independently
- All examples include detailed comments explaining the workflow
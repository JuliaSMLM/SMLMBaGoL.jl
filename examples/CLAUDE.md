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
julia examples/smlm_demo.jl
```

This is much simpler than:
```bash
# Alternative (not needed):
cd examples && julia --project=. smlm_demo.jl
```

## Available Examples

### `smlm_demo.jl`
Complete SMLMBaGoL workflow with simulation, analysis, and visualization:
- **User-configurable parameters** at the top of the script
- Generates synthetic n-mer complex data with realistic photon noise
- Runs BaGoL analysis with RJMCMC sampling
- Computes MAPN estimates with uncertainty quantification
- **Assesses chain quality** with burn-in and mixing diagnostics
- Generates super-resolution images (localizations, emitters, posterior)
- Compares results to ground truth with quantitative metrics

**Key features demonstrated:**
- `simulate_n_mer()` with customizable n-mer geometry
- `run_bagol()` for Bayesian emitter detection
- `estimate_mapn()` for optimal emitter positioning
- `diagnose_chains()` for chain quality assessment
- `gen_sr_image()` for multiple visualization types
- Ground truth comparison with position error analysis
- Comprehensive summary statistics and recovery metrics

**Configurable parameters:**
- N-mer size, diameter, photon count
- Localizations per emitter (mean/variance)
- RJMCMC iterations and burn-in
- Visualization pixel size and output options

**Expected output:**
- Synthetic data generation with realistic noise
- RJMCMC analysis discovering emitter count from data
- MAPN estimates with nm-scale uncertainties
- Chain quality diagnostics (burn-in and mixing assessment)
- Three types of super-resolution images (PNG files)
- Quantitative position accuracy assessment

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
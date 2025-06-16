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

### `smlmsim_demo.jl`
Complete SMLMSim integration workflow with realistic SMLM simulation:
- **SMLMSim simulation** with realistic photophysical noise and blinking
- **Spatial partitioning** for efficient analysis of large datasets
- **Threading support** for parallel partition processing
- **Comprehensive diagnostics** with burn-in and convergence assessment
- **Multiple visualization types** including uncertainty circle plots
- **Camera-aware imaging** with proper field-of-view bounds

**Key features demonstrated:**
- `simulate_static_smlm()` with realistic noise models
- `run_bagol()` with spatial partitioning and threading
- `estimate_mapn()` for optimal emitter positioning
- `diagnose_chains()` for chain quality assessment
- `gen_sr_image()` for super-resolution visualization
- `sr_circles()` for uncertainty visualization

**Configurable parameters:**
- SMLMSim density, PSF width, photon thresholds
- Blinking rates and localization statistics
- RJMCMC iterations, burn-in, and partitioning
- Threading and hierarchical update settings
- Visualization pixel sizes and output options

### `smlmsim_advanced_examples.jl`
Advanced SMLMSim integration patterns and specialized use cases:
- **High-density simulations** with custom field sizes
- **High-photon precision studies** for ultra-precise localization
- **Direct SMLMSim API usage** with custom fluorophore parameters
- **Large-scale simulation workflows** and scalability testing
- **Custom analysis pipelines** with adaptive workflows

### `smlm_demo.jl`
Original SMLMBaGoL workflow with native simulation:
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
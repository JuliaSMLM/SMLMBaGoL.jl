# CLAUDE.md - Development Environment Guide

This guide provides context for AI assistants working with the SMLMBaGoL development environment.

## Project Overview
SMLMBaGoL (Single Molecule Localization Microscopy Bayesian Gaussian-Ordered Localization) is a Julia package for analyzing super-resolution microscopy data using Bayesian methods and Reversible Jump Markov Chain Monte Carlo (RJMCMC).

## Development Environment Setup

### Key Dependencies
- **Julia**: 1.10+ (currently using 1.11.5)
- **SMLMData**: v0.3.1 (for data structures and I/O)
- **SMLMSim**: v0.3.3 (for simulation)
- **CairoMakie**: v0.13.10 (for plotting)
- **Revise.jl**: For hot code reloading during development

### Activation
```julia
using Pkg
Pkg.activate("dev")
using Revise  # Always load first for hot reloading
```

## Development Scripts

### Core Workflow Scripts
1. **`bagol_interface.jl`** - Main interface testing for the bagol algorithm
   - Generates synthetic n-mer data
   - Tests observation and super-resolution image generation
   - Runs the complete BaGoL pipeline
   - Saves visualization outputs

2. **`gen_nmers.jl`** - Synthetic data generation
   - Creates n-mer patterns using SMLMSim
   - Default: 10 datasets, 1000 frames, 50 Hz framerate
   - Generates true, model, and noisy localization data

3. **`basic2Dworkflow.jl`** - Basic 2D analysis workflow
   - Simple RJMCMC testing
   - Direct algorithm validation

### Simulation Scripts
4. **`gen_nmers_big.jl`** - Large-scale simulation
5. **`scatter_animate.jl`** - Animation utilities
6. **`animate_chain.jl`** - MCMC chain visualization

### Interface Testing
7. **`roi_interface.jl`** - Region of interest analysis
8. **`roi_profile.jl`** - ROI profiling and performance testing
9. **`smite.jl`** - SMITE format compatibility testing

### Setup
10. **`setup_dev.jl`** - Development environment initialization

## API Updates (Recent)
The project has been updated to use the new SMLMData v0.3.1 API:

### Key Changes
- `SMLD2D` → `SMLD` (unified container type)
- Data access via `smld.emitters` array instead of direct fields
- Camera information in `smld.camera` with pixel edges
- Updated coordinate system handling

### SMLMData API Reference
```julia
# Get API documentation
using SMLMData
SMLMData.api()  # Comprehensive API overview
```

## Common Development Patterns

### Data Generation
```julia
include("gen_nmers.jl")  # Creates smld_true, smld_model, smld_noisy
println("Total Localizations: ", length(smld_noisy.emitters))
```

### Visualization
```julia
# Generate observation images
obs = BGL.gen_observations(emitter_type, positions, sigmas)
bglim = BGL.gen_obs_image(obs, pixel_size)
display(bglim.data)

# Generate super-resolution images
srim = BGL.gen_sr_image(obs, sr_pixel_size)
srim_color = BGL.VisTools.gen_color_image(srim; max_quantile=0.99)
save("output.png", srim_color)
```

### Running BaGoL
```julia
# Setup prior
prior_λ = Gamma(α, θ)  # Based on expected density

# Run analysis
srs, post = bagol(smld_noisy; prior_λ, pixelsize=1.0)
```

## Testing and Validation

### Performance Monitoring
- Use `@time` macros for benchmarking
- Progress bars with `@showprogress` for long operations
- Thread count: `Threads.nthreads()`

### Output Validation
- Save intermediate images for visual inspection
- Check localization counts and distributions
- Validate coordinate transformations

## Troubleshooting

### Common Issues
1. **Point2f0 errors**: Update to `Point2f` for newer CairoMakie
2. **SMLD2D not found**: Use `SMLD` instead
3. **Field access errors**: Access emitter data via `smld.emitters[i].field`
4. **Camera size issues**: Use `cam.pixel_edges_x/y` for dimensions

### Performance Tips
- Use `Revise.jl` for hot code reloading
- Activate development environment with `Pkg.activate("dev")`
- Use appropriate pixel sizes (1.0 for analysis, 1/50 for super-resolution)
- Monitor memory usage for large datasets

## File Outputs
- **`observations_nmer.png`** - Raw observation images
- **`sr_nmer.png`** - Super-resolution reconstructions
- **`posterior_nmer.png`** - Bayesian posterior maps

## Version Compatibility
- Julia 1.10+ required
- SMLMData v0.3.1+ for new API
- SMLMSim v0.3.3+ for latest simulation features
- Ensure MicroscopePSFs compatibility (v0.5.1+)

## Development Workflow
1. Activate development environment
2. Load Revise for hot reloading
3. Generate or load test data
4. Run analysis scripts
5. Validate outputs and performance
6. Iterate on algorithm improvements
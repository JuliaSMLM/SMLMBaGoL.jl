# CLAUDE.md - Test Environment Guide

This guide provides context for AI assistants working with the SMLMBaGoL test environment.

## Test Environment Overview

The test environment has its own Project.toml to include additional dependencies not needed in the main package, particularly:
- **SMLMSim**: For generating synthetic test data
- **CairoMakie**: For visualization tests
- Any other testing-specific dependencies

## Test Structure

### Main Test File
- `runtests.jl`: Entry point that includes all test files within `@testset` blocks

### Test Files
- `workflow.jl`: End-to-end workflow test based on the development scripts
  - Simulates SMLM data using SMLMSim
  - Runs BaGoL analysis
  - Verifies outputs without requiring visual inspection

## Setting Up Test Environment

**IMPORTANT**: You do NOT need to manually add the package being tested with `Pkg.develop()`. 

### How Julia Testing Works:
When you run `Pkg.test()` or `]test`, Julia automatically:
1. Activates the test environment
2. Makes your package available as if it were developed
3. Runs your tests with access to both your package and test dependencies

### Manual Setup (if needed):
```julia
# Only add test-specific dependencies
using Pkg
Pkg.activate("test")
Pkg.add("Distributions")  # Only external test dependencies
Pkg.status()
```

## Running Tests

```bash
# PREFERRED: From project root, Pkg.test() handles everything automatically
julia --project=. -e 'using Pkg; Pkg.test()'

# Alternative: Manual run (but you lose automatic package availability)
julia --project=test test/runtests.jl
```

### Key Points:
- `Pkg.test()` is the recommended way to run tests
- It automatically handles the test environment and package availability
- The test Project.toml should only contain test-specific dependencies
- Standard library packages (Test, Random, Statistics) are automatically available

## Test Philosophy

1. **Integration Tests**: Focus on complete workflows rather than unit tests
2. **No Visual Output**: Tests should verify results programmatically, not save images
3. **Fast Execution**: Use smaller datasets than development scripts
4. **Reproducibility**: Set random seeds for consistent results

## Common Test Patterns

### Data Generation
```julia
# Generate small synthetic dataset for testing
smld_true, smld_model, smld_noisy = SMLMSim.simulate(
    SMLMSim.StaticSMLMParams(
        density=5.0,  # Lower density for faster tests
        σ_psf=0.13,
        minphotons=100,
        ndatasets=2,  # Fewer datasets
        nframes=100,  # Fewer frames
        framerate=50.0
    );
    pattern=SMLMSim.Nmer2D(d=0.05),
    molecule=SMLMSim.GenericFluor(photons=1e5, k_off=50.0, k_on=0.05),
    camera=SMLMData.IdealCamera(32, 32, 0.1)
)
```

### Verification
```julia
# Verify basic properties
@test length(smld_noisy.emitters) > 0
@test all(e -> e.photons > 0, smld_noisy.emitters)

# Verify BaGoL results
@test length(srs) > 0
@test all(sr -> length(sr.chains) > 0, srs)
```

## Dependencies

The test Project.toml should include:
- SMLMBaGoL (the package being tested)
- Test (standard library)
- SMLMSim (for data generation)
- SMLMData (for data structures)
- Distributions (for priors)
- Statistics (for analysis)
- Random (for reproducibility)

## Notes

- Tests should complete in reasonable time (< 1 minute total)
- Use smaller parameters than development scripts
- Set random seeds for reproducibility
- Avoid file I/O unless testing that functionality specifically
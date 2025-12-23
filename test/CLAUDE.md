# Testing Guidelines for test/

This directory contains all tests for the package. Follow these conventions when writing or modifying tests.

## Test Structure

### runtests.jl Organization
- **Only contains**: `using` statements and the overall test structure
- **No test logic**: All actual tests are included from other files
- **All imports here**: Any packages needed for testing must be imported at the top of runtests.jl

### Test File Organization
1. **User-facing API tests** (e.g., `test_api.jl`)
   - Tests all exported functions that users interact with
   - Tests various keyword arguments and options
   - Focuses on expected use cases and behavior

2. **Internal function tests** (organized by module/concept)
   - Separate files for different modules or logical groupings
   - Tests internal functions and implementation details
   - Clear naming scheme (e.g., `test_utils.jl`, `test_parser.jl`)

### Important Rules
- **No using statements in included files** - All imports must be in runtests.jl
- **Aim for simplicity** - Good coverage without bloating tests
- **Avoid pedantic edge cases** - Focus on meaningful tests that aid development
- **Maintainability first** - Tests should be easy to update as code evolves

## Running Tests

### From Julia REPL
```julia
# Activate the project (from package root)
using Pkg
Pkg.activate(".")

# Run all tests
Pkg.test()

# Or with package name
Pkg.test("SMLMBaGoL")
```

### During Development
```julia
# Run specific test file directly
include("test/runtests.jl")

# Or run a specific test file
include("test/test_specific.jl")  # Only works if no using statements needed
```

## Writing New Tests
- Group related tests in `@testset` blocks with descriptive names
- Use meaningful test descriptions
- Test both success cases and expected failures
- Keep tests focused and independent

## Current Test Structure

### Existing Test Files
- `test_emitter_utils.jl` - Tests for emitter utility functions
- `test_hierarchical_fitting.jl` - Tests for hierarchical Bayesian fitting
- `test_hierarchical_k_math.jl` - Mathematical validation of hierarchical κ parameters
- `test_latent_positions.jl` - Tests for latent position handling

### Key Testing Areas
- **Core RJMCMC moves**: Birth, Death, Move, Allocate operations
- **Likelihood calculations**: Standard and consistency likelihood implementations
- **Prior distributions**: Spatial, count, and hierarchical priors
- **Hierarchical updates**: Parameter updates and initialization
- **Utility functions**: Emitter manipulation, data conversion
- **Integration tests**: End-to-end workflow validation

## SMLMBaGoL-Specific Testing Notes

### Multi-threading Tests
- Use `julia --threads=auto` when testing parallel functionality
- Test both single-threaded and multi-threaded code paths
- Ensure reproducibility with proper random seeds

### Statistical Tests
- Use appropriate tolerances for floating-point comparisons
- Test statistical properties over multiple runs where needed
- Validate MCMC chain properties (acceptance rates, convergence)

### Integration with SMLMSim
- Test simulation integration for reproducible test data
- Validate against known ground truth scenarios
- Test various localization precision scenarios
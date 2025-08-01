# Development and Exploratory Tests

This directory contains development tests, mathematical validations, and exploratory analyses that are used during development but are not part of the formal test suite.

## Structure

- `math/` - Mathematical concept validation and visualization
- `concepts/` - Algorithm concept testing and analysis
- `benchmarks/` - Performance profiling and benchmarking

## Key Differences from Formal Tests

### Development Tests (`dev/`)
- Can produce plots and visualizations
- May have longer runtime
- Output human-readable results and analysis
- Used for debugging and understanding
- Not required to pass for package functionality
- Can use additional dependencies (plotting, etc.)

### Formal Tests (`test/`)
- Must return pass/fail status
- Should run quickly
- Minimal output (only on failure)
- Required for CI/CD
- Test correctness and regressions
- Limited dependencies

## Running Development Tests

```julia
# Activate the dev environment
cd("dev")
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Run a specific validation
include("math/validate_hierarchical_k.jl")
```

## Creating New Development Tests

1. Choose appropriate subdirectory (`math/`, `concepts/`, or `benchmarks/`)
2. Create descriptive filename (e.g., `explore_*.jl`, `validate_*.jl`, `analyze_*.jl`)
3. Include clear documentation at the top of the file
4. Feel free to use plotting, printing, and interactive analysis

## Example Structure

```julia
#=
# Slice Sampling Convergence Analysis
# 
# This script explores the convergence properties of the adaptive slice sampler
# for the hierarchical κ parameter under different data conditions.
#
# Outputs:
# - Convergence plots
# - Acceptance rate statistics
# - Parameter trace plots
=#

using SMLMBaGoL
using Plots
using Statistics

# Your exploratory code here...
```
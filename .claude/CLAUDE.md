# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SMLMBaGoL is a Julia package that performs Bayesian Grouping of Localizations (BaGoL) on single-molecule localization microscopy (SMLM) data. It processes raw localizations to associate them with individual emitters, yielding higher precision localizations of single emitters. The algorithms implemented are presented in a bioRxiv paper (https://doi.org/10.1101/752287).

## Development Environment

```julia
# Activate the project
using Pkg
Pkg.activate(".")

# Install dependencies
Pkg.instantiate()

# For development, you can also activate and use the dev environment
Pkg.activate("dev")
Pkg.instantiate()
```

## Running Tests

```bash
# Run all tests
julia --project=. test/runtests.jl

# Run tests with Revise for development
julia --project=. -e 'using Revise; include("test/runtests.jl")'
```

## Running Examples

```bash
# Run the example script
julia --project=examples examples/smlmbagol_demo.jl
```

## Building Documentation

```bash
# Build the documentation
julia --project=docs docs/make.jl
```

## Code Structure

The package is organized into several modules:

1. **Main Module**: `SMLMBaGoL` - The entry point for the package

2. **Submodules**:
   - `Emitters` - Handles emitter-related functionality
   - `RJMCMC` - Implements Reversible Jump Markov Chain Monte Carlo algorithms
   - `Cluster` - Provides clustering functionality
   - `VisTools` - Visualization tools for displaying results

3. **Main Functions**:
   - `perform_BaGoL_analysis()` - High-level interface for running the BaGoL algorithm
   - `bagol()` - Lower-level interface with more flexibility
   - `rjmcmc()` - Core function for running the RJMCMC algorithm

4. **Data Structure Requirements**:
   - Input data must be in an `SMLMData.SMLD2D` structure
   - Required fields: `x`, `y`, `σ_x`, `σ_y`, and `framenum`
   - Other fields can be placeholders if not available

## Workflow

The typical workflow involves:

1. Prepare SMLM data in an `SMLMData.SMLD2D` structure
2. Run `perform_BaGoL_analysis()` with appropriate parameters
3. Process the output: MAPN emitter positions, posterior distribution, and parameters

## Parameter Configuration

The package uses several parameter structures to control behavior:
- `HBParams2D` - Parameters for hierarchical Bayesian analysis
- `SubregionParams2D` - Parameters for subregion processing
- `PreThreshParams2D` - Parameters for pre-thresholding
- `PreclusterParams2D` - Parameters for pre-clustering
- `MCParams2D` - Parameters for Monte Carlo simulation

## Current Development

The current branch (refactor-rjmcmc) is focused on refactoring the RJMCMC module, suggesting active development in this area.
"""
SMLMBaGoL - Bayesian Grouping of Localizations for Single Molecule Localization Microscopy

A high-performance Julia package for analyzing SMLM data using reversible jump MCMC
with hierarchical Bayesian priors.
"""
module SMLMBaGoL

using SMLMData
using Distributions
using Random
using LinearAlgebra
using Statistics
using StatsBase
using Clustering
using Images
using CairoMakie
using SpecialFunctions

# Re-export key types from SMLMData
export AbstractEmitter, Emitter2D, Emitter2DFit

# Main API
export bagol, BaGoLResult, HierarchicalPrior

# Image functions
export save_posterior_image

# Plotting functions
export circles!, plot_circles

# Types from refactor-rjmcmc integration
export BaGoLChain, MoveType, MoveProbs
export Params, Observations, Allocations, Localization2D
export log_p_z_given_y, Posterior2D

# Include all components
include("types.jl")
include("rjmcmc_likelihood.jl")    # Basic likelihood functions (no dependencies)
include("rjmcmc_core.jl")          # Core infrastructure (uses likelihood)
include("rjmcmc_moves.jl")         # Move functions (uses core + likelihood)
include("rjmcmc_add_remove.jl")    # Add/remove (uses core + likelihood)  
include("rjmcmc_split_merge.jl")   # Split/merge (uses core + likelihood)
include("rjmcmc_reallocate.jl")   # Reallocation (uses core + likelihood)
include("hierarchical.jl")
include("bagol.jl")
include("images.jl")
include("plots.jl")

end # module
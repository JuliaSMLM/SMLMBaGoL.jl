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

# Re-export key types from SMLMData
export AbstractEmitter, Emitter2D, Emitter2DFit

# Main API
export bagol, BaGoLResult, HierarchicalPrior

# Image functions
export save_posterior_image

# Plotting functions
export circles!, plot_circles

# Types
export BaGoLChain, MoveType, MoveProbs

# Include all components
include("types.jl")
include("rjmcmc.jl")
include("hierarchical.jl")
include("bagol.jl")
include("images.jl")
include("plots.jl")

end # module
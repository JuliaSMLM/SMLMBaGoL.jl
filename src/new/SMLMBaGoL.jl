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

# Re-export key types from SMLMData
export AbstractEmitter, Emitter2D, Emitter2DFit

# Main API
export bagol, BaGoLResult, HierarchicalPrior

# Types
export BaGoLChain, MoveType, MoveProbs

# Include all components
include("types.jl")
include("rjmcmc.jl")
include("hierarchical.jl")
include("bagol.jl")

# Optional visualization (to be added later)
# include("visualization.jl")

end # module
module SMLMBaGoL

using CairoMakie
using Distributions
using Hungarian
using LinearAlgebra
using Random
using SMLMData
using SpecialFunctions: logfactorial, logabsbinomial
using Statistics

# Export main API
export run_bagol
export estimate_mapn
export RJMCMCConfig, RJMCMCChain
export MAPNResult

# Export types
export Emitter, BaGoLState, BaGoLSample

# Export priors
export UniformSpatialPrior

# Export visualization
export gen_sr_image, plot_localizations, plot_mapn

# Include source files
include("types.jl")
include("priors.jl")
include("likelihood.jl")
include("moves.jl")
include("hierarchical.jl")
include("rjmcmc.jl")
include("mapn.jl")
include("visualization.jl")

end # module

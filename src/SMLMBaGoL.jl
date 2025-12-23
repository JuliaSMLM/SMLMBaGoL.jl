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
export plot_bagol, plot_mapn, plot_hierarchical_diagnostics

# Export simulation
export SimulationResult
export simulate_smlm, simulate_grid, simulate_nmers
export print_simulation_summary, crlb_precision

# Include source files
include("types.jl")
include("priors.jl")
include("likelihood.jl")
include("moves.jl")
include("hierarchical.jl")
include("rjmcmc.jl")
include("mapn.jl")
include("visualization.jl")
include("simulation.jl")

end # module

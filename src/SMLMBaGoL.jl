module SMLMBaGoL

using Distributions
using Hungarian
using LinearAlgebra
using NearestNeighbors
using Random
using SMLMData
using SpecialFunctions: logfactorial, logabsbinomial
using StaticArrays
using Statistics

# Export main API
export run_bagol, run_bagol_chain
export estimate_mapn
export RJMCMCConfig, RJMCMCChain
export BaGoLDiagnostics

# Export types
export Emitter, BaGoLState, BaGoLSample

# Export partitioning (internal use, partition_locs exposed for advanced users)
export Partition
export partition_locs

# Export priors
export UniformSpatialPrior

# Export simulation
export SimulationResult
export simulate_smlm, simulate_grid, simulate_nmers
export print_simulation_summary, crlb_precision

# Include source files
include("types.jl")
include("spatial.jl")      # Must come before likelihood (provides get_cov_xy)
include("priors.jl")
include("likelihood.jl")
include("moves.jl")
include("hierarchical.jl")
include("mapn.jl")
include("partition.jl")    # Must come before rjmcmc (provides Partition)
include("partitioned.jl")  # Must come before rjmcmc (provides merge_partition_results)
include("rjmcmc.jl")
include("simulation.jl")

end # module

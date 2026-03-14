module SMLMBaGoL

using Distributions
using Hungarian
using LinearAlgebra
using Mmap
using NearestNeighbors
using Random
using SMLMData
using SpecialFunctions: logfactorial, logabsbinomial, loggamma
using StaticArrays
using Statistics

# Export main API
export run_bagol
export run_collapsed_chain
export estimate_mapn_collapsed, estimate_mapn_overlap, estimate_mapn_psm, estimate_dahl, estimate_vi_greedy
export save_posterior_png
export BaGoLDiagnostics

# Export collapsed types
export CollapsedState, CollapsedChainResult, BaGoLResult
export ClusterStats
export AbstractAccumulator, EmitterCountHist, PosteriorImage, NNDistHist
export PartitionSamples, PSMAccumulator

# Export partitioning (internal use, partition_locs exposed for advanced users)
export Partition
export partition_locs

# Export priors
export UniformSpatialPrior

# Export archive
export BaGoLArchive

# Export simulation
export SimulationResult
export simulate_smlm, simulate_grid, simulate_nmers
export print_simulation_summary, crlb_precision

# Include source files
include("spatial.jl")      # Must come before cluster_stats (provides get_cov_xy)
include("cluster_stats.jl") # Must come before types.jl (CollapsedState uses ClusterStats)
include("types.jl")
include("priors.jl")
include("hierarchical.jl")
include("mapn.jl")
include("accumulators.jl")   # Accumulator interface for collapsed sampler
include("collapsed_moves.jl")
include("collapsed_sampler.jl")
include("partition.jl")    # Must come before rjmcmc (provides Partition)
include("partitioned.jl")  # Must come before rjmcmc (provides deduplicate_boundary_emitters)
include("rjmcmc.jl")
include("posterior_image.jl")
include("archive.jl")       # Mmap chain archive
include("simulation.jl")

end # module

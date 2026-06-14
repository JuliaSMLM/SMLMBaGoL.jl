module SMLMBaGoL

using Distributions
using Hungarian
using LinearAlgebra
using Metis
using Mmap
using NearestNeighbors
using Random
using SMLMData
using SparseArrays
using SpecialFunctions: logfactorial, logabsbinomial, loggamma
using StaticArrays
using Statistics

# Export config
export BaGoLConfig

# Export main API
export run_bagol
export run_collapsed_chain, initialize_from_assignments
export estimate_mapn_collapsed, estimate_mapn_overlap, estimate_mapn_psm, estimate_dahl, estimate_vi_greedy
export save_posterior_png
export BaGoLDiagnostics

# Export collapsed types
export CollapsedState, CollapsedChainResult, BaGoLResult
export ClusterStats
export MultiClusterStats, FeatureSet, run_multicue_chain, build_multicue_precisions, extract_multicue, gaussian_contribution
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
export nmer_positions, nmer_grid_positions, nmer_random_positions
export simulate_localizations, simulate_nmer, simulate_nmer_grid
export make_camera, print_simulation_summary

# Export reports
export compute_report, write_report, match_positions, compute_fov
export plot_report, render_report  # extension stubs

# Export optimality
export run_optimality_sweep, run_speed_test, count_model_map_k
export write_sweep, write_speed
export plot_sweep, plot_speed  # extension stubs

# Export diagnostics — target density
export AbstractTargetDensity, DecoupledTarget, DecoupledLocmixTarget
export DMFlatTarget, DMLocmixTarget, DirectNegBinFlatTarget, DirectNegBinLocmixTarget
export FazelFlatTarget
export log_target, evaluate_target

# Export diagnostics — finite-state validation
export AbstractSpatialModel, FlatSpatial, LocmixSpatial, spatial_ml, spatial_pred
export AbstractAllocationModel, DMAllocation, DecoupledAllocation, CategoricalAllocation

export enumerate_canonical_partitions, canonicalize, exact_posterior
export enumerate_labeled_partitions
export run_kernel_invariance_test, run_labeled_invariance_test

# Export diagnostics — detailed balance
export DetailedBalanceResult, check_detailed_balance

# Export diagnostics — mixing
export ChainDiagnosticAccumulator
export effective_sample_size, autocorrelation, split_gelman_rubin
export indicator_ess, run_mixing_test

# Include source files
include("spatial.jl")
include("cluster_stats.jl")
include("types.jl")
include("config.jl")
include("priors.jl")
include("hierarchical.jl")
include("mapn.jl")
include("accumulators.jl")
include("collapsed_moves.jl")
include("collapsed_sampler.jl")
include("multicue.jl")
include("partition.jl")
include("partitioned.jl")
include("rjmcmc.jl")
include("posterior_image.jl")
include("archive.jl")
include("simulation.jl")
include("reports.jl")
include("optimality.jl")

# Diagnostics (3 files)
include("diagnostics/target.jl")
include("diagnostics/finite_state.jl")
include("diagnostics/mixing.jl")
include("diagnostics/detailed_balance.jl")

end # module

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

# Export diagnostics — target density abstraction
export AbstractTargetDensity, DecoupledTarget, MFMTarget, UniformPriorTarget
export log_target, evaluate_target, log_mfm_partition_ratio

# Export diagnostics — enumeration
export enumerate_canonical_partitions, canonicalize
export exact_posterior, compare_mcmc_to_exact, run_enumeration_test

# Export diagnostics — detailed balance
export DetailedBalanceResult
export check_detailed_balance, run_detailed_balance_suite, print_db_summary

# Export diagnostics — partition metrics
export variation_of_information, expected_posterior_loss, partition_diagnostics

# Export diagnostics — chain diagnostics
export ChainDiagnosticAccumulator
export effective_sample_size, autocorrelation, gelman_rubin
export run_mixing_test, run_length_test

# Export diagnostics — count model utilities
export count_model_posterior, qpaint_recovery_rate
export count_model_confusion_matrix, theoretical_accuracy_bound

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
include("reports.jl")       # Standard report computation + write
include("optimality.jl")    # Optimality sweep + speed test

# Diagnostics module (algorithm-agnostic evaluation infrastructure)
include("diagnostics/target.jl")            # Target density abstraction (must come first)
include("diagnostics/count_model.jl")       # Count model analysis
include("diagnostics/partition_metrics.jl")  # VI, EPL, partition comparison
include("diagnostics/chain_diagnostics.jl") # ESS, R-hat, autocorrelation, mixing tests
include("diagnostics/enumeration.jl")       # Brute-force partition enumeration
include("diagnostics/detailed_balance.jl")  # Detailed balance verification

end # module

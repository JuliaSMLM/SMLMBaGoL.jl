module SMLMBaGoL

using Random
using LinearAlgebra
using Statistics
using Distributions
using SpecialFunctions
using StatsBase
using Hungarian
using Clustering
using HypothesisTests
using SMLMSim
using SMLMData
using SMLMData: AbstractEmitter, Emitter2D, Emitter2DFit
using CairoMakie
using Images
using Images: Gray, RGB, N0f8
using Colors
using ColorSchemes

# Core types
include("core/types.jl")

# Concrete types
# Emitter types come from SMLMData
include("localizations/localization2d.jl")

# Core functionality
include("core/state.jl")
include("core/priors.jl")
include("core/likelihood.jl")

# Utilities
include("utils/emitter_utils.jl")
include("utils/display.jl")

# Moves
include("moves/move_types.jl")
include("moves/birth_death.jl")
include("moves/move.jl")
include("moves/allocate.jl")
include("moves/acceptance.jl")

# Algorithms
include("algorithms/partitioning.jl")
include("algorithms/rjmcmc.jl")
include("algorithms/mapn.jl")

# Hierarchical
include("hierarchical/updates.jl")

# Simulation
include("sim/n_mer.jl")
include("sim/smlmsim_integration.jl")

# Visualization
include("visualization/sr_image.jl")
include("visualization/plotting.jl")
include("visualization/hierarchical_viz.jl")

# Diagnostics
include("diagnostics/chain_diagnostics.jl")

# Exports
export AbstractLocalization, AbstractPrior, AbstractRJMCMCMove, AbstractChainState
# Note: AbstractEmitter, Emitter2D, Emitter2DFit come from SMLMData
export Localization2D, BaGoLState, RJMCMCChain
export UniformSpatialPrior, GammaPrior, HierarchicalGammaPrior, CompoundPrior
export Birth, Death, Move, Allocate, inverse_move
export log_likelihood, log_prior, log_prior_spatial, log_prior_K, log_posterior
export sample_spatial_prior, create_spatial_prior_from_localizations, create_default_prior
export propose_move, accept_probability, log_acceptance_ratio
export run_bagol, initialize_chain, run_rjmcmc!, rjmcmc_step!
export initialize_chains_from_data, handle_chain_continuation
export partition_localizations, estimate_partitioning_radius, update_hierarchical!
export estimate_mapn
export simulate_n_mer, simulate_n_mer_with_prior
export smld_to_localizations, simulate_static_smlm, simulate_nmer_smlmsim
export gen_sr_image
export circle!, sr_circles, sr_circles!, sr_circles_combined, sr_circles_combined!
export diagnose_chains, quick_diagnose, assess_burn_in, assess_chain_length
export plot_hierarchical_evolution, plot_gamma_distributions, plot_emitter_count_histogram
export analyze_hierarchical_convergence, get_hierarchical_summary

end # module SMLMBaGoL
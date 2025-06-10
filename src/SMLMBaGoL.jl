module SMLMBaGoL

using Random
using LinearAlgebra
using Statistics
using Distributions
using SpecialFunctions
using StatsBase

# Core types
include("core/types.jl")

# Concrete types
include("emitters/emitter2d.jl")
include("localizations/localization2d.jl")

# Core functionality
include("core/state.jl")
include("core/priors.jl")
include("core/likelihood.jl")

# Utilities
include("utils/emitter_utils.jl")

# Moves
include("moves/move_types.jl")
include("moves/birth_death.jl")
include("moves/split_merge.jl")
include("moves/move.jl")
include("moves/allocate.jl")
include("moves/acceptance.jl")

# Algorithms
include("algorithms/clustering.jl")
include("algorithms/rjmcmc.jl")

# Hierarchical
include("hierarchical/updates.jl")

# Exports
export AbstractLocalization, AbstractEmitter, AbstractPrior, AbstractRJMCMCMove, AbstractChainState
export Emitter2D, Localization2D, BaGoLState, RJMCMCChain
export UniformSpatialPrior, GammaPrior, HierarchicalGammaPrior, CompoundPrior
export Birth, Death, Split, Merge, Move, Allocate, inverse_move
export log_likelihood, log_prior, log_prior_spatial, log_prior_K, log_posterior
export sample_spatial_prior, create_spatial_prior_from_localizations, create_default_prior
export propose_move, accept_probability, log_acceptance_ratio
export run_bagol, initialize_chain, run_rjmcmc!, rjmcmc_step!
export cluster_localizations, update_hierarchical!

end # module SMLMBaGoL
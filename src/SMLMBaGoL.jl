module SMLMBaGoL

using Random
using LinearAlgebra
using Statistics
using Distributions
using SpecialFunctions

# Core types
include("core/types.jl")

# Concrete types
include("emitters/emitter2d.jl")
include("localizations/localization2d.jl")

# Core functionality
include("core/state.jl")
include("core/priors.jl")
include("core/likelihood.jl")

# Exports
export AbstractLocalization, AbstractEmitter, AbstractPrior, AbstractRJMCMCMove, AbstractChainState
export Emitter2D, Localization2D, BaGoLState, RJMCMCChain
export UniformSpatialPrior, GammaPrior, HierarchicalGammaPrior, CompoundPrior
export log_likelihood, log_prior, log_prior_spatial, log_prior_K
export sample_spatial_prior, create_spatial_prior_from_localizations

end # module SMLMBaGoL
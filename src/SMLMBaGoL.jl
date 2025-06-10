module SMLMBaGoL

using Random
using LinearAlgebra
using Statistics
using Distributions

# Core types
include("core/types.jl")

# Concrete types
include("emitters/emitter2d.jl")
include("localizations/localization2d.jl")

# Core functionality
include("core/state.jl")
include("core/likelihood.jl")

# Exports
export AbstractLocalization, AbstractEmitter, AbstractPrior, AbstractRJMCMCMove, AbstractChainState
export Emitter2D, Localization2D, BaGoLState, RJMCMCChain
export log_likelihood

end # module SMLMBaGoL
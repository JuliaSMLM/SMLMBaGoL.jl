module RJMCMC

using Distributions
using StatsBase 
using SpecialFunctions
using Hungarian

include("types.jl")
include("move.jl")
include("allocate.jl")
include("add-remove.jl")
include("split-merge.jl")
include("interface.jl")
include("emitters2D.jl")
include("buildchain.jl")
include("mapn.jl")

end

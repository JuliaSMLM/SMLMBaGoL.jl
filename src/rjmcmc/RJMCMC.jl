module RJMCMC

using Distributions
using StatsBase 
using SpecialFunctions

include("types.jl")
include("move.jl")
include("allocate.jl")
include("add-remove.jl")
include("split-merge.jl")
include("interface.jl")
include("emitters2D.jl")
include("buildchain.jl")


end

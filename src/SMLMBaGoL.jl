module SMLMBaGoL

using SMLMData
using Distributions
using NearestNeighbors
using StatsBase
using Clustering
using LinearAlgebra

include("types.jl")
include("methods.jl")
# These are exported for the submodules
export AbstractObservation
export Observations
export Allocations

include("emitters/Emitters.jl")
include("rjmcmc/RJMCMC.jl")

# To get exports   
using .RJMCMC
using .Emitters

end
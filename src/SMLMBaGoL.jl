module SMLMBaGoL

using SMLMData
using Distributions
using NearestNeighbors
using StatsBase
using Clustering
using LinearAlgebra


include("types.jl")
export AbstractObservation
export Observations

include("emitters/Emitters.jl")
# include("rjmcmc/RJMCMC.jl")


# export RJMCMC
export Emitters

end
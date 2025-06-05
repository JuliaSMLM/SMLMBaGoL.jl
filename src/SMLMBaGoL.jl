module SMLMBaGoL

using SMLMData
using Distributions
using StatsBase
using Clustering
using LinearAlgebra
using ProgressMeter

include("types.jl")
include("methods.jl")
# These are exported for the submodules
export AbstractObservation
export AbstractEmitter
export Observations
export Allocations
export Params

include("emitters/Emitters.jl")
using .Emitters

include("rjmcmc/RJMCMC.jl")
using .RJMCMC

include("cluster/Cluster.jl")  
using .Cluster

include("posterior.jl")
include("mapn.jl")
include("interface.jl")

include("vis_tools/VisTools.jl")
using .VisTools

export bagol

end
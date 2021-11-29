module SMLMBaGoL

using Base
using SMLMData
using Distributions
using NearestNeighbors
using StatsBase
using Clustering
using LinearAlgebra

include("typedefinitions.jl")
include("mathhelpers.jl")
include("chainoperations.jl")
include("priordistributions.jl")
include("imagedistribution.jl")
include("gensubregions.jl")
include("removeoutliers.jl")
include("precluster.jl")
include("jumps.jl")
include("allocatelocs.jl")
include("moveemitters.jl")
include("runRJMCMC.jl")
include("runbagol.jl")
include("removeoverlap.jl")
include("map.jl")

end
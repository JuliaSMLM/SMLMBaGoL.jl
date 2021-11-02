module SMLMBaGoL

include("typedefinitions.jl")
include("mathhelpers.jl")
include("priordistributions.jl")
include("imagedistribution.jl")
include("gensubregions.jl")
include("removeoutliers.jl")
include("precluster.jl")
include("probjump.jl")
include("proposals.jl")
include("allocatelocs.jl")
include("moveemitters.jl")
include("runRJMCMC.jl")
include("runbagol.jl")

end
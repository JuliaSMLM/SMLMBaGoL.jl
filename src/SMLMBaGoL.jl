module SMLMBaGoL

include("typedefinitions.jl")
include("mathhelpers.jl")
include("priorpositions.jl")
include("priorallocations.jl")
include("priorkemitters.jl")
include("priordrift.jl")
include("gensubregions.jl")
include("removeoutliers.jl")
include("precluster.jl")
include("makegaussim.jl")
include("probjump.jl")
include("proposebirth.jl")
include("proposedeath.jl")
include("allocatelocs.jl")
include("moveemitters.jl")
include("runRJMCMC.jl")
include("runbagol.jl")

end
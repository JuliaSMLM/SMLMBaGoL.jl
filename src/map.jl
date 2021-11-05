using SMLMData
using StatsBase
using Clustering

# This file contains functions/methods related to computing maximum a 
# posteriori estimates (MAP) of quantities related to BaGoL analysis.


function mapn(smld::SMLMData.SMLD2D,
              chain::SMLMBaGoL.BaGoLChain2D)
    # Determine the mode number of emitters in this chain.
    μ, a, k = SMLMBaGoL.catfields(chain)
    n = StatsBase.mode(k)

    # Extract all states with `n` emitters.
    mapnbool = k .== n
    mapnstates = deepcopy(chain.states[mapnbool])
    μmapn, amapn, _ = SMLMBaGoL.catfields(mapnstates)

    # Perform k-means clustering on the states with `n` emitters.
    # Clustering.kmeans(transpose(μmapn)
end

function mapn(smld::SMLMData.SMLD2D,
              chain::Matrix{Vector{SMLMBaGoL.BaGoLChain2D}})
    
end
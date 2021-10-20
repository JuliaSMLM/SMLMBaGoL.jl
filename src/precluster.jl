using SMLMData
using Clustering

"""
    smld_preclustered = precluster_hierarchical(smld::SMLMData.SMLD2D,
                                                maxdist::Float64)

Pre-cluster localizations in `smld` and define a Gamma prior from the results.

# Description
Localizations in the input structure `smld` are clustered using hierarchical
using hierarchical clustering and stored in the output SMLD2D structure 
`smld_preclustered`, with the field connectID updated to reflect cluster
assignments.  The parameter `maxdist` defines the maximum distance allowed
between emitters placed in the same precluster.
"""
function precluster_hierarchical(smld::SMLMData.SMLD2D, maxdist::Float64=0.15)
    # Compute the separations between all localizations in `smld`.
    dist = SMLMBaGoL.pairwise_dist([smld.x smld.y])

    # Perform hierarchical clustering based on `dist`.
    htree = Clustering.hclust(dist; uplo=:U)
    smld_preclustered = deepcopy(smld)
    smld_preclustered.connectID = Clustering.cutree(htree; h=maxdist)

    return smld_preclustered
end

function pairwise_dist(data::Matrix{Float64})
    # Compute the distance between rows of `data`.
    npoints = size(data)[1]
    dist = fill!(Matrix{Float64}(undef, size(data)[1], size(data)[1]), Inf64)
    for ii = 1:npoints, jj = (ii+1):npoints
        dist[ii, jj] = sqrt(sum((data[ii, :]-data[jj, :]).^2))
    end

    return dist
end
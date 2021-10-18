using FrameConnection
using SMLMData
using SpecialFunctions
using Optim

"""
    k, θ = constructprior_lambda(smld_preclustered::SMLD)

Pre-cluster localizations in `smld` and define a Gamma prior from the results.

# Description
Localizations in the input structure `smld` are clustered together based on
their spatiotemporal separations and then assumed to be a single emitter.  A
Gamma prior is then constructed by fitting the distribution of localizations
per emitter.
"""
function constructprior_lambda(smld_preclustered::SMLMData.SMLD)
    # Determine the number of localizations per pre-cluster.
    clusterdata = FrameConnection.organizeclusters(smld_preclustered)
    _, nobservations = FrameConnection.computeclusterinfo(clusterdata)

    # Perform an MLE to estimate parameters for a Gamma prior.
    nclusters = Float32(size(clusterdata)[1])
    negloglikelihood(k) = 
        -log((SpecialFunctions.gamma(k[1])*(k[2]^k[1]))^(-nclusters) *
        prod(nobservations.^(k[1]-1.0) .* exp.(-nobservations/k[2])))
    optimizer = Optim.NelderMead()
    optimresults = Optim.optimize(negloglikelihood, [0, 0], [Inf, Inf], [1.0, 2.0], 
        Fminbox(optimizer))
    k = optimresults.minimizer[1]
    theta = optimresults.minimizer[2]

    return k, theta
end
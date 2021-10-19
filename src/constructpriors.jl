using FrameConnection
using SMLMData
using SpecialFunctions
using Statistics
using Optim

"""
    α, β, clusterdata, nobservations = 
        constructprior_lambda(smld_preclustered::SMLD, forceexp::Bool)

Pre-cluster localizations in `smld` and define a Gamma prior from the results.

# Description
Localizations in the input structure `smld` are clustered together based on
their spatiotemporal separations and then assumed to be a single emitter.  A
Gamma prior is then constructed by fitting the distribution of localizations
per emitter.  The Gamma prior is defined using the convention with a shape 
parameter `α` and a rate parameter `β`.  The flag `forcexp` forces the shape
parameter `α` to be 1 (corresponding to the exponential distribution). 
"""
function constructprior_lambda(smld_preclustered::SMLMData.SMLD, forceexp::Bool=false)
    # Determine the number of localizations per pre-cluster.
    clusterdata = FrameConnection.organizeclusters(smld_preclustered)
    _, nobservations = FrameConnection.computeclusterinfo(clusterdata)

    # Compute the MLE for our prior.
    if forceexp
        alpha, beta = expMLE(clusterdata, nobservations)
    else
        alpha, beta = gammaMLE(clusterdata, nobservations)
    end

    return alpha, beta, clusterdata, nobservations
end

function gammaMLE(clusterdata, nobservations)
    # Define initial guesses for our Gamma parameters.
    # NOTE: These estimates were taken from the Gamma distribution wikipedia 
    #       page https://en.wikipedia.org/wiki/Gamma_distribution
    nclusters = Float32(size(clusterdata)[1])
    s = log((1/nclusters)*sum(nobservations)) - 
        (1/nclusters)*sum(log.(nobservations))
    alphainit = 3.0 - s + sqrt((s-3.0)^2+24.0*s)/(12.0*s)
    betainit = (1/(alphainit*nclusters)) * sum(nobservations)
    initguess = [alphainit, betainit]

    # Perform an MLE to estimate parameters for a Gamma prior.
    negloglikelihood(k) = 
        -log(((k[2]^k[1])/SpecialFunctions.gamma(k[1]))^(nclusters) *
        prod(nobservations.^(k[1]-1.0) .* exp.(-k[2]*nobservations)))
    optimizer = Optim.NelderMead()
    optimresults = Optim.optimize(negloglikelihood, [0, 0], [Inf, Inf], initguess, 
        Fminbox(optimizer))
    alpha = optimresults.minimizer[1]
    beta = optimresults.minimizer[2]

    return alpha, beta
end

function expMLE(clusterdata, nobservations)
    alpha = 1.0
    beta = 1/Statistics.mean(nobservations)

    return alpha, beta
end
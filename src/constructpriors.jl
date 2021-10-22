using Distributions
using FrameConnection
using SMLMData
using SpecialFunctions
using Statistics
using Optim

"""
    prior = prior_kemitters(nloc::Int, λ::Int)

Generate a prior on the number of emitters assuming Poisson loc. per emitter.

# Description
This method generates a prior on the total number of emitters `k` assuming that
the number of localizations per emitter `λ` is Poisson distributed.

# Inputs
-`nloc`: Total number of localizations.
-`λ`: Number of localizations per emitter.
"""
function prior_kemitters(nloc::Int, λ::Int)
    # Initialize the SMLMBaGoL.PriorStruct1D structure.
    prior = SMLMBaGoL.PriorStruct1D(k_start, k_step, nsteps, "Emitters", [])

    # Define the prior on the number of emitters assuming Poisson `λ`.
    kemitters = collect(1:nloc)
    poissdist = Vector{Float64}(undef, nloc)
    for ii = 1:nloc
        # Distributions.Poisson() seems to do a smarter calculation than what
        # I tried when typing in the Poisson pdf here manually (i.e., my naive
        # version was not numerically stable).
        pdfcurrent = Distributions.Poisson(kemitters[ii] * λ)
        poissdist[ii] = Distributions.pdf(pdfcurrent, nloc)
    end
    prior.pdf = poissdist ./ sum(poissdist)

    return prior
end

"""
    prior = prior_kemitters(nloc::Int, α::Float64, β::Float64)

Generate a prior on the number of emitters assuming gamma loc. per emitter.

# Description
This method generates a prior on the total number of emitters `k` assuming that
the number of localizations per emitter is gamma distributed.

# Inputs
-`nloc`: Total number of localizations.
-`α`: Shape parameter of the gamma distribution.
-`β`: Scale parameter of the gamma distribution.
"""
function prior_kemitters(nloc::Int, α::Float64, β::Float64)
    # Initialize the SMLMBaGoL.PriorStruct1D structure.
    prior = SMLMBaGoL.PriorStruct1D(k_start, k_step, nsteps, "Emitters", [])

    # Define the prior on the number of emitters assuming the localizations per
    # emitter is gamma distributed.
    kemitters = collect(1:nloc)
    gammadist = Vector{Float64}(undef, nloc)
    for ii = 1:nloc
        # Distributions.Poisson() seems to do a smarter calculation than what
        # I tried when typing in the gamma pdf here manually (i.e., my naive
        # version was not numerically stable).
        pdfcurrent = Distributions.Gamma(α, kemitters[ii] / β)
        gammadist[ii] = Distributions.pdf(pdfcurrent, nloc)
    end
    prior.pdf = gammadist ./ sum(gammadist)

    return prior
end

"""
    prior = prior_allocations(nloc::Int, kemitters::Int)

Generate a prior on the allocations of localizations to emitters.

# Description
This method constructs an SMLMBaGoL.PriorStruct1D for the categorical prior
on the allocations per emitter, where the probability of allocation to each
emitter is assumed to be equal.

# Inputs
-`nloc`: Total number of localizations.
-`kemitters`: Total number of emitters.
"""
function prior_allocations(nloc::Int, kemitters::Int)
    # Prepare the SMLMBaGoL.PriorStruct1D structure.
    prior = SMLMBaGoL.PriorStruct1D(1, 1, nloc, "Allocations", [])
    prior.pdf = ((1/kemitters) ^ nloc) * ones(nloc)

    return prior
end

"""
    prior = prior_positions(datarange::Float64)

Generate a prior on the positions of emitters.

# Description
This method constructs an SMLMBaGoL.PriorStruct1D for the uniform prior on the
1D positions of localizations.

# Inputs
-`datarange`: Range of positions spanned by the localizations.
"""
function prior_positions(datarange::Float64, kemitters::Int)
    # Prepare the SMLMBaGoL.PriorStruct1D structure.
    prior = SMLMBaGoL.PriorStruct1D(1, 1, kemitters, "Positions", [])
    prior.pdf = (1/datarange) * ones(kemitters)

    return prior
end

"""
    prior = prior_drift()

Generate a prior on the positions of emitters.

# Description
This method constructs an SMLMBaGoL.PriorStruct1D for the Normal prior on the
1D drift velocities of emitters.
"""
function prior_drift()
    # Prepare the SMLMBaGoL.PriorStruct1D structure.
    prior = SMLMBaGoL.PriorStruct1D()

    return prior
end

"""
    gammaprior = construct_gamma(α::Float64, β::Float64, 
                                 θ_start::Float64, θ_step::Float64, 
                                 nsteps::Int)

Generate a prior structure from the gamma distribution.

# Description
The distribution `gamma(α, β)` is evaluated at `nsteps` of size `θ_step` 
starting at `θ_start`.  An SMLMBaGoL.PriorStruct1D is output storing the
resulting distribution.

# Inputs
-`α`: Shape parameter of the gamma distribution.
-`β`: Scale parameter of the gamma distribution.
-`θ_start`: Start of range at which we evaluate the gamma pdf.
-`θ_step`: Step size of steps made starting from `θ_start`.
-`nsteps`: Total number of steps (evaluations) made of the gamma pdf.
"""
function construct_gamma(α::Float64, β::Float64, 
                         θ_start::Float64, θ_step::Float64, 
                         nsteps::Int)
    # Initialize the SMLMBaGoL.PriorStruct1D structure.
    gammaprior = SMLMBaGoL.PriorStruct1D(θ_start, θ_step, nsteps, "Gamma(α, β)", [])

    # Evaluate the Gamma distribution at the points of interest.
    θ = range(θ_start, step = θ_step, length = nsteps)
    gammadist = Distributions.Gamma(α, 1 / β)
    gammaprior.pdf = Distributions.pdf(gammadist, θ)

    return gammaprior
end

"""
    poissonprior = construct_poisson(λ::Float64,
                                     θ_start::Float64, θ_step::Float64, 
                                     nsteps::Int)

Generate a prior structure from the Poisson distribution.

# Description
The distribution `Poisson(λ)` is evaluated at `nsteps` of size `θ_step` 
starting at `θ_start`.  An SMLMBaGoL.PriorStruct1D is output storing the
resulting distribution.

# Inputs
-`λ`: Mean value of the Poisson distribution.
-`θ_start`: Start of range at which we evaluate the Poisson pdf.
-`θ_step`: Step size of steps made starting from `θ_start`.
-`nsteps`: Total number of steps (evaluations) made of the Poisson pdf.
"""
function construct_poisson(λ::Float64, 
                           θ_start::Float64, θ_step::Float64, 
                           nsteps::Int)
    # Initialize the SMLMBaGoL.PriorStruct1D structure.
    poissonprior = SMLMBaGoL.PriorStruct1D(θ_start, θ_step, nsteps, 
                                           "Poisson(λ)", [])

    # Evaluate the Poisson distribution at the points of interest.
    θ = range(θ_start, step = θ_step, length = nsteps)
    poissondist = Distributions.Poisson(λ)
    poissonprior.pdf = Distributions.pdf(poissondist, θ)

    return poissonprior
end

# """
#     α, β, clusterdata, nobservations = 
#         constructprior_lambda(smld_preclustered::SMLD, forceexp::Bool)

# Fit a Gamma distribution to preclusters in smld_preclustered.

# # Description
# Localizations in the input structure `smld` are clustered together based on
# their spatiotemporal separations and then assumed to be a single emitter.  A
# Gamma prior is then constructed by fitting the distribution of localizations
# per emitter.  The Gamma prior is defined using the convention with a shape 
# parameter `α` and a rate parameter `β`.  The flag `forcexp` forces the shape
# parameter `α` to be 1 (corresponding to the exponential distribution). 
# """
# function constructprior_lambda(smld_preclustered::SMLMData.SMLD2D, 
#                                forceexp::Bool=false)
#     # Determine the number of localizations per pre-cluster.
#     clusterdata = FrameConnection.organizeclusters(smld_preclustered)
#     _, nobservations = FrameConnection.computeclusterinfo(clusterdata)

#     # Compute the MLE for our prior.
#     if forceexp
#         alpha, beta = expMLE(clusterdata, nobservations)
#     else
#         alpha, beta = gammaMLE(clusterdata, nobservations)
#     end

#     return alpha, beta, clusterdata, nobservations
# end

# function gammaMLE(clusterdata, nobservations)
#     # Define initial guesses for our Gamma parameters.
#     # NOTE: These estimates were taken from the Gamma distribution wikipedia 
#     #       page https://en.wikipedia.org/wiki/Gamma_distribution
#     nclusters = Float32(size(clusterdata)[1])
#     s = log((1/nclusters)*sum(nobservations)) - 
#         (1/nclusters)*sum(log.(nobservations))
#     alphainit = 3.0 - s + sqrt((s-3.0)^2+24.0*s)/(12.0*s)
#     betainit = (1/(alphainit*nclusters)) * sum(nobservations)
#     initguess = [alphainit, betainit]

#     # Perform an MLE to estimate parameters for a Gamma prior.
#     negloglikelihood(k) = 
#         -log(((k[2]^k[1])/SpecialFunctions.gamma(k[1]))^(nclusters) *
#         prod(nobservations.^(k[1]-1.0) .* exp.(-k[2]*nobservations)))
#     optimizer = Optim.NelderMead()
#     optimresults = Optim.optimize(negloglikelihood, [0, 0], [Inf, Inf], initguess, 
#         Fminbox(optimizer))
#     alpha = optimresults.minimizer[1]
#     beta = optimresults.minimizer[2]

#     return alpha, beta
# end

# function expMLE(clusterdata, nobservations)
#     alpha = 1.0
#     beta = 1 / Statistics.mean(nobservations)

#     return alpha, beta
# end
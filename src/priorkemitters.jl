using Distributions

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
    pmf = poissdist ./ sum(poissdist)

    # Create a distribution using the Distributions package using our pmf.
    prior = Distributions.DiscreteNonParametric(kemitters, pmf)

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
    pmf = gammadist ./ sum(gammadist)

    # Create a distribution using the Distributions package using our pmf.
    prior = Distributions.DiscreteNonParametric(kemitters, pmf)

    return prior
end
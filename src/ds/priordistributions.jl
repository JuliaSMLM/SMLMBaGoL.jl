using Distributions
using Base

# This file contains functions defining the BaGoL priors.

"""
    prior = prior_kemitters(nloc::Int, η::Float64, γ::Float64)

Generate a prior on the number of emitters assuming gamma loc. per emitter.

# Description
This method generates a prior on the total number of emitters `k` assuming that
the number of localizations per emitter is gamma distributed.

# Inputs
- `nloc`: Total number of localizations.
- `η`: Shape parameter of the gamma distribution.
- `γ`: Scale parameter of the gamma distribution.

# Outputs
- `prior`: Distributions.Distribution defined by assuming a gamma prior for the
           number of blinks per emitter.
"""
function prior_kemitters(nloc::Int, η::Float64, γ::Float64)
    # Define the prior on the number of emitters assuming the localizations per
    # emitter is gamma distributed.
    kemitters = collect(1:nloc)
    if nloc > 1
        gammadist = Vector{Float64}(undef, nloc)
        for ii = 1:nloc
            pdfcurrent = Distributions.Gamma(kemitters[ii] * η, γ)
            gammadist[ii] = Distributions.pdf(pdfcurrent, nloc)
        end
        pmf = gammadist ./ sum(gammadist)
    else
        pmf = [1.0]
    end

    # Create a distribution using the Distributions package using our pmf.
    return Distributions.DiscreteNonParametric(kemitters, pmf)
end

"""
    prior = prior_kemitters(nloc::Int, λ::Int)

Generate a prior on the number of emitters assuming Poisson loc. per emitter.

# Description
This method generates a prior on the total number of emitters `k` assuming that
the number of localizations per emitter `λ` is Poisson distributed.

# Inputs
- `nloc`: Total number of localizations.
- `λ`: Number of localizations per emitter.

# Outputs
- `prior`: Distributions.Distribution defined by assuming a Poisson prior for 
           the number of blinks per emitter.
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
    return Distributions.DiscreteNonParametric(kemitters, pmf)
end

"""
    prob = prior_allocations(nloc::Int, kemitters::Int)

Generate a prior on the allocations of localizations to emitters.

# Description
This method returns the probability of any given allocation assuming the
weights (of the categorical prior being used) are all equal.  Preparing the
full distribution as done with the other priors is not feasible in this case.

# Inputs
- `nloc`: Total number of localizations.
- `kemitters`: Total number of emitters.

# Outputs
- `prob`: Probability of any given allocation set, assuming equal weighting to 
          all emitters.
"""
function prior_allocations(nloc::Int, kemitters::Int)
    # Return the probability of allocation.
    return 1 / (kemitters^nloc)
end

"""
    prior = prior_positions(roisize::Vector{Float64})

Generate prior distributions on the positions of emitters.

# Description
This method constructs prior distributions for the positions of emitters along
multiple dimensions.  The prior on emitter positions is given in each dimension
as a uniform distribution over the range of possible emitter positions (i.e., 
the range [0.5, roisize+0.5]).

# Inputs
- `roisize`: Size of the region of interest supporting the emitter positions. 
             (ndimensionsx1)

# Outputs
- `prior`: Vector of Distributions.Distribution for the position prior along 
           each dimension.
"""
function prior_positions(roisize::Vector{Int})
    # Prepare the distributions using the Distributions package.
    ndim = Base.length(roisize)
    prior = Vector{Distributions.Distribution}(undef, ndim)
    for ii = 1:ndim
        prior[ii] = prior_positions(roisize[ii])
    end

    return prior 
end

"""
    prior = prior_positions(roisize::Int)

Generate a prior distribution on the positions of emitters.

# Description
This method constructs a prior distribution for the positions of emitters.  The
prior on emitter positions is given as a uniform distribution over the range
of possible emitter positions (i.e., the range [0.5, roisize+0.5])

# Inputs
- `roisize`: Size of the region of interest supporting the emitter positions.

# Outputs
- `prior`: Uniform distribution prior over the emitter positions.
"""
function prior_positions(roisize::Int)
    # Prepare a distribution using the Distributions package.
    return Distributions.Uniform(0.5, 0.5 + roisize)
end

"""
    sample = rand(distrib::Vector{Distributions.Distribution}, nsamples::Int)

Sample the `distrib` distributions `nsamples` times.

# Description
This method makes random samples from the distributions in `distrib` and stores
the results in a matrix.  The intention is that `distrib` is a vector of 
distributions each representing a different dimension, so our output matrix 
can represent, e.g., random spatial coordinates.

# Inputs
- `distrib`: Vector of Distributions.Distribution types.
- `nsamples`: Number of samples to be made from each of `distrib`.

# Outputs
- `sample`: Samples from the vector of distributions in `distrib`.
"""
function rand(distrib::Vector{Distributions.Distribution}, nsamples::Int)
    # Sample the n distributions in distrib.
    ndistrib = Base.length(distrib)
    sample = Matrix{Float64}(undef, nsamples, ndistrib)
    for ii = 1:ndistrib
        sample[:, ii] = Distributions.rand(distrib[ii], nsamples)
    end

    return sample
end

"""
    prior = prior_drift(σ_a::Vector{Float64})

Generate priors on the drift velocities of emitters.

# Description
This method constructs normal priors on the 1D drift velocities of emitters.

# Inputs
- `σ_a`: Standard deviation of the drift velocity prior. (2x1)([x; y])
"""
function prior_drift(σ_a::Vector{Float64})
    # Prepare the distributions.
    ndim = Base.length(σ_a)
    prior = Vector{Distributions.Distribution}(undef, ndim)
    for ii = 1:ndim
        prior[ii] = SMLMBaGoL.prior_drift(σ_a[ii])
    end

    return prior
end

"""
    prior = prior_drift(σ_a::Float64)

Generate a prior on the drift velocities of emitters.

# Description
This method constructs a Normal prior on the 1D drift velocities of emitters.

# Inputs
- `σ_a`: Standard deviation of the drift velocity prior.

# Outputs
- `prior`: Normal prior on a 1D drift velocity, given as a 
           Distributions.Distribution.
"""
function prior_drift(σ_a::Float64)
    # Prepare the normal distribution.
    return Distributions.Normal(0, σ_a)
end
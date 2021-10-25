using Distributions

"""
    prior = prior_positions(roisize::Float64)

Generate a prior distribution on the positions of emitters.

# Description
This method constructs a prior distribution for the positions of emitters.  The
prior on emitter positions is given as a uniform distribution over the range
of possible emitter positions (i.e., the range [0.5, roisize+0.5])

# Inputs
-`roisize`: Size of the region of interest supporting the emitter positions.
"""
function prior_positions(roisize::Float64)
    # Prepare a distribution using the Distributions package.
    prior = Distributions.Uniform(0.5, 0.5 + roisize)

    return prior
end
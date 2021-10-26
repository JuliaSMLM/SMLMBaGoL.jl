using Distributions

"""
    prior = prior_drift(σ_a::Float64)

Generate a prior on the drift velocities of emitters.

# Description
This method constructs a Normal prior on the 1D drift velocities of emitters.

# Inputs
-`σ_a`: Standard deviation of the drift velocity prior.
"""
function prior_drift(σ_a::Float64)
    # Prepare the Normal distribution.
    return Distributions.Normal(0, σ_a)
end
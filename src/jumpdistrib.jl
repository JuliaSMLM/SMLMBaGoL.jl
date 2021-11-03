using Distributions
using Base

"""
    distribution = jumpdistrib(jumppmf::Vector{Float64})

Generate a distribution of jump choices from the provided `jumppmf`.

# Description
This function generates a distribution of the jump probabilities that can be 
sampled to propose a jump labeled by an index in the range [1, length(jumppmf)]
using rand(distribution).
"""
function jumpdistrib(jumppmf::Vector{Float64})
    # Create a distribution for the jumps using the Distributions package.
    return Distributions.DiscreteNonParametric(1:Base.length(jumppmf), jumppmf)
end
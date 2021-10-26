using Distributions

"""
    prior = prior_allocations(nloc::Int, kemitters::Int)

Generate a prior on the allocations of localizations to emitters.

# Description
This method constructs a categorical prior on the allocations per emitter,
where the probability of allocation to each emitter is assumed to be equal. As
such, we can just use the discrete uniform distribution (since each category
has the same probability).

# Inputs
-`nloc`: Total number of localizations.
-`kemitters`: Total number of emitters.
"""
function prior_allocations(nloc::Int, kemitters::Int)
    # Prepare the distribution over the support [1, kemitters^nloc].
    return Distributions.DiscreteUniform(1, kemitters^nloc)
end
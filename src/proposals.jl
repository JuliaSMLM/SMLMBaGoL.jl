using SMLMData
using Distributions
using Base

# This file contains some functions useful for jump proposals.
# NOTE: Not all required functions for birth/death proposals are contained in
#       this file. Some are contained in related files, e.g., allocatelocs.jl
#       for allocation probabilities.

"""
    proposemove(smld::SMLMData.SMLD2D,
                currentstate::SMLMBaGoL.BaGoLState2D,
                mcparams::SMLMBaGoL.MCParams)
        
Propose a move of emitters.

# Description
This method proposes a new SMLMBaGoL.BaGoLState2D in which the emitter
positions have been randomly sampled from the posterior distribution of emitter
positions.

# Inputs
-`smld`: SMLD2D structure containing localization coordinates.
-`currentstate`: Current state of the Markov chain defining the emitter
                 positions and localization allocations.
-`mcparams`: Structure of MCMC parameters/distributions.

# Outputs
-`proposal`: Proposal state containing the proposed emitter moves.
"""
function proposemove(smld::SMLMData.SMLD2D,
                     currentstate::SMLMBaGoL.BaGoLState2D,
                     mcparams::SMLMBaGoL.MCParams)
    # Propose a move of the emitters.
    proposal = deepcopy(currentstate)
    proposal.μ, proposal.a = SMLMBaGoL.moveemitters(smld, 
        currentstate.z, mcparams.σ_a, currentstate.k)

    return proposal
end

"""
    proposeallocation(smld::SMLMData.SMLD2D,
                      currentstate::SMLMBaGoL.BaGoLState2D)
        
Propose a reallocation of localizations to emitters.

# Description
This method proposes a new SMLMBaGoL.BaGoLState2D in which the localizations
in `smld` are reallocated to emitters in `currentstate`.

# Inputs
-`smld`: SMLD2D structure containing localization coordinates.
-`currentstate`: Current state of the Markov chain defining the emitter
                 positions and localization allocations.

# Outputs
-`proposal`: Proposal state containing the proposed emitter allocations.
"""
function proposeallocation(smld::SMLMData.SMLD2D,
                           currentstate::SMLMBaGoL.BaGoLState2D)
    # Propose a reallocation of localizations to emitters.
    proposal = deepcopy(currentstate)
    proposal.z = SMLMBaGoL.allocatelocs(smld, currentstate.μ, currentstate.a)

    return proposal
end

"""
    proposebirth(smld::SMLMData.SMLD2D,
                 currentstate::SMLMBaGoL.BaGoLState2D,
                 mcparams::SMLMBaGoL.MCParams)
        
Propose a new emitter.

# Description
This method proposes a new SMLMBaGoL.BaGoLState2D in which a new emitter
is proposed, followed by a reallocation of localizations to emitters.

# Inputs
-`smld`: SMLD2D structure containing localization coordinates.
-`currentstate`: Current state of the Markov chain defining the emitter
                 positions and localization allocations.
-`mcparams`: Structure of MCMC parameters/distributions.

# Outputs
-`proposal`: Proposal state containing the newly proposed emitter and
             reallocations of localizations to emitters.
-`p_im`: Probability of an emitter existing at the proposed emitter location.
"""
function proposebirth(smld::SMLMData.SMLD2D,
                      currentstate::SMLMBaGoL.BaGoLState2D,
                      mcparams::SMLMBaGoL.MCParams)
    # Propose a new emitter by treating a Gaussian SR image of the (raw) 
    # localizations as a density distribution.
    coords, sampleind = SMLMBaGoL.samplecoords2D(mcparams.imdistrib, 
                                                 mcparams.srimsize[1])
    coords ./= mcparams.srmag
    coords .+= mcparams.roi[1:2] .- 1.0
    proposal = SMLMBaGoL.BaGoLState2D()
    proposal.k = currentstate.k + 1
    proposal.μ = [currentstate.μ; transpose(coords)]
    proposal.a = [currentstate.a; transpose(mcparams.σ_a * Base.randn(2))]

    # Perform Gibbs sampling for the allocations.
    proposal.z = SMLMBaGoL.allocatelocs(smld, proposal.μ, proposal.a)

    return proposal, mcparams.imdistrib.p[sampleind]
end

"""
    proposedeath(smld::SMLMData.SMLD2D,
                 currentstate::SMLMBaGoL.BaGoLState2D,
                 mcparams::SMLMBaGoL.MCParams)
        
Propose the death of an existing emitter.

# Description
This method proposes a new SMLMBaGoL.BaGoLState2D in which one of the emitters
in `currentstate` is removed, with the `smld` localizations being reallocated
to the new emitter set.

# Inputs
-`smld`: SMLD2D structure containing localization coordinates.
-`currentstate`: Current state of the Markov chain defining the emitter
                 positions and localization allocations.
-`mcparams`: Structure of MCMC parameters/distributions.

# Outputs
-`proposal`: Proposal state reflecting the death of one of the emitters.
-`p_im`: Probability of an emitter existing at the removed emitter location.
"""
function proposedeath(smld::SMLMData.SMLD2D,
                      currentstate::SMLMBaGoL.BaGoLState2D,
                      mcparams::SMLMBaGoL.MCParams)
    # Randomly remove an emitter.
    proposal = deepcopy(currentstate)
    removeind = Base.rand(1:proposal.k)
    SMLMBaGoL.removeemitter!(proposal, removeind)

    # Perform Gibbs sampling for the allocations.
    proposal.z = SMLMBaGoL.allocatelocs(smld, proposal.μ, proposal.a)

    # Determine the probability of an emitter being at the location that was
    # removed.
    coords_mag = mcparams.srmag .* (currentstate.μ[removeind, :].-0.5)
    inds = Int.(round.(max.(min.(1.0, coords_mag), mcparams.srimsize)))
    ind = (inds[2]-1)*mcparams.srimsize[1] + inds[1]
    
    return proposal, mcparams.imdistrib.p[ind]
end

"""
    acceptbirth(smld::SMLMData.SMLD2D,
                proposal::SMLMBaGoL.BaGoLState2D,
                p_im::Float64,
                currentstate::SMLMBaGoL.BaGoLState2D,
                mcparams::SMLMBaGoL.MCParams)
        
Compute the acceptance probability of accepting the proposed emitter birth.

# Description
This method computes the acceptance probability of accepting the proposed
emitter birth defined by `proposal` with respect to the current set of emitters
in `currentstate`.

# Inputs
-`smld`: SMLD2D structure containing localization coordinates.
-`proposal`: Proposal state reflecting the birth of an emitter.
-`p_im`: Probability of an emitter existing at the proposed location.
         (see mcparams.imdistrib)
-`currentstate`: Current state of the Markov chain defining the emitter
                 positions and localization allocations.
-`mcparams`: Structure of MCMC parameters/distributions.

# Outputs
-`accept`: Acceptance probability of accepting the proposed state change.
"""
function acceptbirth(smld::SMLMData.SMLD2D,
                     proposal::SMLMBaGoL.BaGoLState2D,
                     p_im::Float64,
                     currentstate::SMLMBaGoL.BaGoLState2D,
                     mcparams::SMLMBaGoL.MCParams)
    # Compute the probability ratio for the allocations.
    t = Float64.(smld.framenum)
    logLallocprime = SMLMBaGoL.emitterlogL2D(
        [smld.y smld.x], [smld.σ_y smld.σ_x], t, 
        proposal.μ, proposal.a, proposal.z)
    logLalloc = SMLMBaGoL.emitterlogL2D(
        [smld.y smld.x], [smld.σ_y smld.σ_x], t, 
        currentstate.μ, currentstate.a, currentstate.z)
    pallocratio = exp(logLallocprime - logLalloc)
         
    # Compute the probability ratio for the number of emitters.
    # NOTE: The death proposal uses this same function, so the proposed `k`
    #       can be smaller than the current value (hence the k=min(...) below).
    nloc = SMLMData.length(smld)
    k = min(currentstate.k, proposal.k)
    mcparams.priork = SMLMBaGoL.prior_kemitters(nloc, mcparams.α, mcparams.β)
    pkratio = mcparams.priork.p[k+1] / mcparams.priork.p[k]

    # Compute the complete proposal ratio.
    pjumpratio = mcparams.p_jump[3] / mcparams.p_jump[4]
    return pallocratio * pkratio * ((k/(k+1))^nloc) * pjumpratio / 
        (p_im*mcparams.area)
end

"""
    acceptdeath(smld::SMLMData.SMLD2D,
                proposal::SMLMBaGoL.BaGoLState2D,
                p_im::Float64,
                currentstate::SMLMBaGoL.BaGoLState2D,
                mcparams::SMLMBaGoL.MCParams)
        
Compute the acceptance probability of accepting the proposed emitter death.

# Description
This method computes the acceptance probability of accepting the proposed
emitter death defined by `proposal` with respect to the current set of emitters
in `currentstate`.

# Inputs
-`smld`: SMLD2D structure containing localization coordinates.
-`proposal`: Proposal state reflecting the death of one of the emitters.
-`p_im`: Probability of an emitter existing at the proposed location.
         (see mcparams.imdistrib)
-`currentstate`: Current state of the Markov chain defining the emitter
                 positions and localization allocations.
-`mcparams`: Structure of MCMC parameters/distributions.

# Outputs
-`accept`: Acceptance probability of accepting the proposed state change.
"""
function acceptdeath(smld::SMLMData.SMLD2D,
                     proposal::SMLMBaGoL.BaGoLState2D,
                     p_im::Float64,
                     currentstate::SMLMBaGoL.BaGoLState2D,
                     mcparams::SMLMBaGoL.MCParams)
    return 1.0 / SMLMBaGoL.acceptbirth(smld, 
        proposal, p_im, currentstate, mcparams)
end

"""
    addstate!(chain::SMLMBaGoL.BaGoLChain2D, 
              state::SMLMBaGoL.BaGoLState2D, 
              accepted::Bool)
        
Add `state` to the end of `chain`.

# Inputs
-`chain`: Chain of states.
-`state`: State to be added at the end of `chain`.
-`accepted`: Boolean indicating acceptance of the `state` being added to
             `chain`.
"""
function addstate!(chain::SMLMBaGoL.BaGoLChain2D, 
                   state::SMLMBaGoL.BaGoLState2D, 
                   accepted::Bool)
    # Update the chain to include `state`.
    push!(chain.states, state)
    push!(chain.accept, accepted)
    chain.n += 1
end

"""
    removestate!(chain::SMLMBaGoL.BaGoLChain2D, remove)
        
Remove the state directed to by `remove` from the `chain`.

# Inputs
-`chain`: Chain of states.
-`remove`: State to be removed from `chain`, defined in any way allowed by the
           input `inds` in the Julia method deleteat!().
"""
function removestate!(chain::SMLMBaGoL.BaGoLChain2D, remove)
    # Update the chain remove the state `remove`.
    deleteat!(chain.states, remove)
    deleteat!(chain.accept, remove)
    chain.n = Base.length(chain.states)
end

"""
    removeemitter!(state::SMLMBaGoL.BaGoLState2D, k::Int)
        
Remove the emitter indexed as `k` from the given `state`.

# Inputs
-`state`: State of a Markov chain.
-`k`: Emitter index of the emitter to be removed from `state`.
"""
function removeemitter!(state::SMLMBaGoL.BaGoLState2D, k::Int)
    # Remove the k-th emitter and ensure `z` consists of integers 1:k_emitters.
    keepind = setdiff(1:state.k, k)
    state.k -= 1
    state.μ = state.μ[keepind, :]
    state.a = state.a[keepind, :]
    state.z[state.z .> state.k] .-= 1
end
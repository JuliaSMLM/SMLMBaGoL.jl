using SMLMData
using Distributions
using Base

# This file contains some functions useful for jump proposals/acceptance.
# NOTE: Not all required functions for birth/death proposals are contained in
#       this file. Some are contained in related files, e.g., allocatelocs.jl
#       for allocation probabilities.

"""
    distribution = jumpdistrib(jumppmf::Vector{Float64})

Generate a distribution of jump choices from the provided `jumppmf`.

# Description
This function generates a distribution of the jump probabilities that can be 
sampled to propose a jump labeled by an index in the range [1, length(jumppmf)]
using rand(distribution).

# Inputs
-`jumppmf`: Probability mass function defining the jump distribution.

# Outputs
-`distribution`: Distributions.Distribution defined by input `jumppmf`.
"""
function jumpdistrib(jumppmf::Vector{Float64})
    # Create a distribution for the jumps using the Distributions package.
    return Distributions.DiscreteNonParametric(1:Base.length(jumppmf), jumppmf)
end

"""
    proposal = proposemove(smld::SMLMData.SMLD2D,
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
    proposal = proposeallocation(smld::SMLMData.SMLD2D,
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
    proposal, p_im = proposebirth(smld::SMLMData.SMLD2D,
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
    proposal, p_im = proposedeath(smld::SMLMData.SMLD2D,
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
    α = acceptbirth(smld::SMLMData.SMLD2D,
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
-`α`: Acceptance probability of accepting the proposed state change.
      NOTE: I'm not enforcing `α`<=1.0 in this output, so care must be taken
            when using it elsewhere!
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
    nloc = Base.length(smld)
    k = min(currentstate.k, proposal.k)
    mcparams.priork = SMLMBaGoL.prior_kemitters(nloc, mcparams.α, mcparams.β)
    pkratio = mcparams.priork.p[k+1] / mcparams.priork.p[k]

    # Compute the complete proposal ratio.
    pjumpratio = mcparams.p_jump[3] / mcparams.p_jump[4]
    # return pallocratio * pkratio * pjumpratio / (p_im*mcparams.area)
    return pallocratio * pkratio * ((k/(k+1))^nloc) * pjumpratio / 
        (p_im*mcparams.area)
end

"""
    α = acceptdeath(smld::SMLMData.SMLD2D,
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
    state, accepted = move(smld::SMLMData.SMLD2D, 
                           state::SMLMBaGoL.BaGoLState2D,
                           mcparams::SMLMBaGoL.MCParams2D)

Return a state with the emitters in `state` moved to new positions.

# Description
This function moves emitters in `state` to new positions sampled from the 
position posterior distribution.

# Inputs
-`smld`: SMLD2D structure containing the localizations.
-`state`: Current state of the Markov chain.
-`mcparams`: Structure of MCMC parameters.

# Outputs
-`state`: A proposed state with the moved emitters.
-`accepted`: Boolean indicating whether or not the move was accepted, which is 
             always true since moves use Gibbs sampling.
"""
function move(smld::SMLMData.SMLD2D, 
              state::SMLMBaGoL.BaGoLState2D,
              mcparams::SMLMBaGoL.MCParams2D)
    # Propose a move and accept it (emitter moves are always accepted).
    return SMLMBaGoL.proposemove(smld, state, mcparams), true
end

"""
    state, accepted = reallocate(smld::SMLMData.SMLD2D, 
                                 state::SMLMBaGoL.BaGoLState2D)

Return a state with `smld` localizations reallocated to emitters in `state`.

# Description
This function reallocates localizations in `smld` to emitters in `state`.

# Inputs
-`smld`: SMLD2D structure containing the localizations.
-`state`: Current state of the Markov chain.

# Outputs
-`state`: A proposed state with the (potentially) redefined allocations.
-`accepted`: Boolean indicating whether or not the move was accepted, which is 
             always true since allocations use Gibbs sampling.
"""
function reallocate(smld::SMLMData.SMLD2D, 
                    state::SMLMBaGoL.BaGoLState2D)
    # Propose an allocation and accept it (allocations are always accepted).
    return SMLMBaGoL.proposeallocation(smld, state), true
end

"""
    state, accepted = birth(smld::SMLMData.SMLD2D, 
                            state::SMLMBaGoL.BaGoLState2D,
                            mcparams::SMLMBaGoL.MCParams2D)

Propose and determine acceptance of a birth move.

# Description
This function proposes the birth of an emitter and then determines if it should
be accepted into the Markov chain.

# Inputs
-`smld`: SMLD2D structure containing the localizations.
-`state`: Current state of the Markov chain.
-`mcparams`: Structure of MCMC parameters.

# Outputs
-`state`: A proposed state with one more emitter (if the move was accepted) or
          the input state `state`.
-`accepted`: Boolean indicating whether or not the proposal was accepted.
"""
function birth(smld::SMLMData.SMLD2D, 
               state::SMLMBaGoL.BaGoLState2D,
               mcparams::SMLMBaGoL.MCParams2D)
    # Propose a birth of a new emitter (unless there are as many emitters as
    # localizations, in which case we'll return the input `state`).
    if state.k < Base.length(smld)
        proposal, p_im = SMLMBaGoL.proposebirth(smld, state, mcparams)
        acceptance = SMLMBaGoL.acceptbirth(smld, 
                                           proposal, 
                                           p_im, 
                                           state, 
                                           mcparams)
    else
        proposal = deepcopy(state)
        acceptance = 1.0
    end

    # Determine whether or not we should accept the new emitter.
    if Base.rand() <= acceptance
        return proposal, true
    else
        return state, false
    end
end

"""
    state, accepted = death(smld::SMLMData.SMLD2D, 
                            state::SMLMBaGoL.BaGoLState2D,
                            mcparams::SMLMBaGoL.MCParams2D)

Propose and determine acceptance of a death move.

# Description
This function proposes the death of an emitter and then determines if it should
be accepted into the Markov chain.

# Inputs
-`smld`: SMLD2D structure containing the localizations.
-`state`: Current state of the Markov chain.
-`mcparams`: Structure of MCMC parameters.

# Outputs
-`state`: A proposed state with one less emitter (if the move was accepted) or
          the input state `state`.
-`accepted`: Boolean indicating whether or not the proposal was accepted.
"""
function death(smld::SMLMData.SMLD2D, 
               state::SMLMBaGoL.BaGoLState2D,
               mcparams::SMLMBaGoL.MCParams2D)
    # Propose the death of a random emitter (unless there is only 1 emitter 
    # left, in which case we should return the current state).
    if state.k == 1
        proposal = deepcopy(state)
        acceptance = 1.0
    else
        proposal, p_im = SMLMBaGoL.proposedeath(smld, state, mcparams)
        acceptance = SMLMBaGoL.acceptdeath(smld, 
                                           proposal, 
                                           p_im, 
                                           state, 
                                           mcparams)
    end

    # Determine whether or not we should accept emitter death.
    if Base.rand() <= acceptance
        return proposal, true
    else
        return state, false
    end
end
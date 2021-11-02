using SMLMData
using Distributions
using Base

# This file contains some functions useful for birth and death jump proposals.
# NOTE: Not all required functions for birth/death proposals are contained in
#       this file. Some are contained in related files, e.g., allocatelocs.jl
#       for allocation probabilities.

function proposemove(smld::SMLMData.SMLD2D,
                     currentstate::SMLMBaGoL.BaGoLState2D,
                     mcparams::SMLMBaGoL.MCParams)
    # Propose a move of the emitters.
    proposal = deepcopy(currentstate)
    proposal.μ, proposal.a = SMLMBaGoL.moveemitters(smld, 
        currentstate.z, mcparams.σ_a, currentstate.k)

    return proposal
end

function proposeallocation(smld::SMLMData.SMLD2D,
                           currentstate::SMLMBaGoL.BaGoLState2D)
    # Propose a reallocation of localizations to emitters.
    proposal = deepcopy(currentstate)
    proposal.z = SMLMBaGoL.allocatelocs(smld, currentstate.μ, currentstate.a)

    return proposal
end

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

    return proposal, sampleind
end

function proposedeath(smld::SMLMData.SMLD2D,
                      currentstate::SMLMBaGoL.BaGoLState2D)
    # Randomly remove an emitter.
    proposal = deepcopy(currentstate)
    removeind = Base.rand(1:proposal.k)
    SMLMBaGoL.removeemitter!(proposal, removeind)

    # Perform Gibbs sampling for the allocations.
    proposal.z = SMLMBaGoL.allocatelocs(smld, proposal.μ, proposal.a)

    return proposal, removeind
end

function acceptbirth(smld::SMLMData.SMLD2D,
                     proposal::SMLMBaGoL.BaGoLState2D,
                     positionind::Int,
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
        (mcparams.imdistrib.p[positionind] * mcparams.area)
end

function acceptdeath(smld::SMLMData.SMLD2D,
                     proposal::SMLMBaGoL.BaGoLState2D,
                     positionind::Int,
                     currentstate::SMLMBaGoL.BaGoLState2D,
                     mcparams::SMLMBaGoL.MCParams)
    return 1.0 / SMLMBaGoL.acceptbirth(smld, 
        proposal, positionind, currentstate, mcparams)
end

function removeemitter!(chain::SMLMBaGoL.BaGoLChain2D, k::Int)
    # Remove the k-th emitter from the end of the chain, ensuring we update
    # the allocations array `z` to consist of integers 1:k
    keepind = setdiff(1:chain.k[end], k)
    chain.k[end] -= 1
    chain.μ[end] = chain.μ[end][keepind, :]
    chain.a[end] = chain.a[end][keepind, :]
    chain.z[end][chain.z[end] .> chain.k[end]] .-= 1

    return
end

function removeemitter!(state::SMLMBaGoL.BaGoLState2D, k::Int)
    # Remove the k-th emitter and ensure `z` consists of integers 1:k_emitters.
    keepind = setdiff(1:state.k, k)
    state.k -= 1
    state.μ = state.μ[keepind, :]
    state.a = state.a[keepind, :]
    state.z[state.z .> state.k] .-= 1

    return
end

















# function proposemove(smld::SMLMData.SMLD2D,
#                      chain::SMLMBaGoL.BaGoLChain2D,
#                      mcparams::SMLMBaGoL.MCParams)
#     # Propose a move of the emitters.
#     proposal = SMLMBaGoL.getstate(chain)
#     proposal.μ[1], proposal.a[1] = SMLMBaGoL.moveemitters(smld, 
#         proposal.z[end], mcparams.σ_a, proposal.k[1])

#     return proposal
# end

# function proposeallocation(smld::SMLMData.SMLD2D,
#                            chain::SMLMBaGoL.BaGoLChain2D)
#     # Propose a reallocation of localizations to emitters.
#     proposal = SMLMBaGoL.getstate(chain)
#     proposal.z[1] = SMLMBaGoL.allocatelocs(smld, proposal.μ[1], proposal.a[1])

#     return proposal
# end

# function proposebirth(smld::SMLMData.SMLD2D,
#                       chain::SMLMBaGoL.BaGoLChain2D,
#                       mcparams::SMLMBaGoL.MCParams)
#     # Propose a new emitter by treating a Gaussian SR image of the (raw) 
#     # localizations as a density distribution.
#     coords, sampleind = SMLMBaGoL.samplecoords2D(mcparams.imdistrib, 
#                                                  mcparams.srimsize[1])
#     coords ./= mcparams.srmag
#     coords .+= mcparams.roi[1:2] .- 1.0
#     proposal = SMLMBaGoL.BaGoLChain(1)
#     # println(chain.k[end])
#     proposal.k[1] = chain.k[end] + 1
#     # println(chain.k[end])
#     proposal.μ[1] = [chain.μ[end]; transpose(coords)]
#     proposal.a[1] = [chain.a[end]; transpose(mcparams.σ_a * Base.randn(2))]

#     # Perform Gibbs sampling for the allocations.
#     proposal.z[1] = SMLMBaGoL.allocatelocs(smld, proposal.μ[1], proposal.a[1])

#     return proposal, sampleind
# end

# function proposedeath(smld::SMLMData.SMLD2D,
#                       chain::SMLMBaGoL.BaGoLChain2D)
#     # Randomly remove an emitter.
#     proposal = SMLMBaGoL.getstate(chain)
#     removeind = Base.rand(1:proposal.k[1])
#     removeemitter!(proposal, removeind)

#     # Perform Gibbs sampling for the allocations.
#     proposal.z[1] = SMLMBaGoL.allocatelocs(smld, proposal.μ[1], proposal.a[1])

#     return proposal, removeind
# end

# function removeemitter!(chain::SMLMBaGoL.BaGoLChain2D, k::Int)
#     # Remove the k-th emitter from the end of the chain, ensuring we update
#     # the allocations array `z` to consist of integers 1:k
#     keepind = setdiff(1:chain.k[end], k)
#     chain.k[end] -= 1
#     chain.μ[end] = chain.μ[end][keepind, :]
#     chain.a[end] = chain.a[end][keepind, :]
#     chain.z[end][chain.z[end] .> chain.k[end]] .-= 1

#     return
# end

# function removeemitter!(state::SMLMBaGoL.BaGoLState2D, k::Int)
#     # Remove the k-th emitter and ensure `z` consists of integers 1:k_emitters.
#     keepind = setdiff(1:state.k, k)
#     state.k -= 1
#     state.μ = state.μ[keepind, :]
#     state.a = state.a[keepind, :]
#     state.z[state.z .> state.k] .-= 1

#     return
# end

# function acceptbirth(smld::SMLMData.SMLD2D,
#                      proposal::SMLMBaGoL.BaGoLChain2D,
#                      positionind::Int,
#                      chain::SMLMBaGoL.BaGoLChain2D,
#                      mcparams::SMLMBaGoL.MCParams)
#     # Compute the probability ratio for the allocations.
#     t = Float64.(smld.framenum)
#     logLallocprime = SMLMBaGoL.emitterlogL2D(
#         [smld.y smld.x], [smld.σ_y smld.σ_x], t, 
#         proposal.μ[1], proposal.a[1], proposal.z[1])
#     logLalloc = SMLMBaGoL.emitterlogL2D(
#         [smld.y smld.x], [smld.σ_y smld.σ_x], t, 
#         chain.μ[end], chain.a[end], chain.z[end])
#     pallocratio = exp(logLallocprime - logLalloc)
         
#     # Compute the probability ratio for the number of emitters.
#     # NOTE: The death proposal uses this same function, so the proposed `k`
#     #       can be smaller than the current value (hence the k=min(...) below).
#     nloc = SMLMData.length(smld)
#     k = min(chain.k[end], proposal.k[end])
#     mcparams.priork = SMLMBaGoL.prior_kemitters(nloc, mcparams.α, mcparams.β)
#     pkratio = mcparams.priork.p[k+1] / mcparams.priork.p[k]

#     # Compute the complete proposal ratio.
#     pjumpratio = mcparams.p_jump[3] / mcparams.p_jump[4]
#     return pallocratio * pkratio * ((k/(k+1))^nloc) * pjumpratio / 
#         (mcparams.imdistrib.p[positionind] * mcparams.area)
# end

# function acceptdeath(smld::SMLMData.SMLD2D,
#                      proposal::SMLMBaGoL.BaGoLChain2D,
#                      positionind::Int,
#                      chain::SMLMBaGoL.BaGoLChain2D,
#                      mcparams::SMLMBaGoL.MCParams)
#     return 1.0 / SMLMBaGoL.acceptbirth(smld, proposal, positionind, chain, mcparams)
# end
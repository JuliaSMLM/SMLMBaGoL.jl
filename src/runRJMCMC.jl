using SMLMData
using Distributions
using Base

"""
    runRJMCMC!(smld::Matrix{SMLMData.SMLD2D}, 
               mcparams::MCParams2D)

Perform reversible jump Markov chain monte carle (RJMCMC).

# Description
This function performs RJMCMC analysis by preparing some distributions and
parameters and then building the RJMCMC chain.  This function dispatches on
the vector `smld` version of runRJMCMC!() on each entry of the matrix input 
`smld`.

# Inputs
-`smld`: Matrix of SMLD2D structures.
-`mcparams`: Structure of MCMC parameters.

# Outputs
-`chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
"""
function runRJMCMC!(smld::Matrix{SMLMData.SMLD2D}, 
                    rois::Matrix{Vector{Float64}}, 
                    mcparams::MCParams2D)
    # Loop over subregions in `smld` and perform RJMCMC.
    smldsize = size(smld)
    chain = Matrix{Vector{SMLMBaGoL.BaGoLChain2D}}(undef, smldsize)
    for ii = 1:smldsize[1], jj = 1:smldsize[2]
        # Generate a distinct SMLD2D for each precluster.
        smldclusters, _ = SMLMData.isolateconnected(smld[ii, jj])

        # Run RJMCMC on each precluster.
        mcparams.roi = rois[ii, jj]
        chain[ii, jj] = runRJMCMC!(smldclusters, mcparams)
    end

    return chain
end

"""
    runRJMCMC!(smld::Vector{SMLMData.SMLD2D}, 
               mcparams::MCParams2D)

Perform reversible jump Markov chain monte carle (RJMCMC).

# Description
This function performs RJMCMC analysis by preparing some distributions and
parameters and then building the RJMCMC chain.  This function dispatches on
the single `smld` version of runRJMCMC!() on each entry of the vector input 
`smld`.

# Inputs
-`smld`: Vector of SMLD2D structures.
-`mcparams`: Structure of MCMC parameters.

# Outputs
-`chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
"""
function runRJMCMC!(smld::Vector{SMLMData.SMLD2D}, 
                    mcparams::MCParams2D)
    # Loop over preclusters and perform RJMCMC on each of them.
    nclusters = Base.length(smld)
    chain = Vector{SMLMBaGoL.BaGoLChain2D}(undef, nclusters)
    for nn = 1:nclusters
        chain[nn] = runRJMCMC!(smld[nn], mcparams)
    end
    
    return chain
end

"""
    runRJMCMC!(smld::SMLMData.SMLD2D, 
               mcparams::MCParams2D)

Perform reversible jump Markov chain monte carle (RJMCMC).

# Description
This function performs RJMCMC analysis by preparing some distributions and
parameters and then building the RJMCMC chain.

# Inputs
-`smld`: SMLD2D structure containing the localizations.
-`mcparams`: Structure of MCMC parameters.

# Outputs
-`chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
"""
function runRJMCMC!(smld::SMLMData.SMLD2D, 
                    mcparams::MCParams2D)
    # Prepare an emitter distribution from the provided localizations 
    # (approximated as the normalized Gaussian image of localizations).
    mcparams.imdistrib, mcparams.srimsize = SMLMBaGoL.imagedistribution(smld, 
        mcparams.srmag, mcparams.nsigma, mcparams.roi)
    mcparams.area = Float64(prod(smld.datasize[1:2]))
    
    # Prepare some distributions (e.g., priors) and define initial states.
    mcparams.jumpdistrib = SMLMBaGoL.jumpdistrib(mcparams.p_jump)
    nloc = SMLMData.length(smld)
    mcparams.priork = SMLMBaGoL.prior_kemitters(nloc, mcparams.α, mcparams.β)
    k = Int(ceil(nloc / (mcparams.α*mcparams.β)))
    # mcparams.priorμ = SMLMBaGoL.prior_positions(smld.datasize)
    # μ = SMLMBaGoL.rand(mcparams.priorμ, k) .+ repeat(transpose(roi[1:2]), k) .- 1.0
    μ, _ = SMLMBaGoL.samplecoords2D(mcparams.imdistrib, mcparams.srimsize[1], k)
    μ ./= mcparams.srmag
    μ .+= repeat(transpose(mcparams.roi[1:2]), k) .- 1.0
    mcparams.priora = SMLMBaGoL.prior_drift(mcparams.σ_a * [1.0; 1.0])
    a = zeros(Float64, k, 2)
    mcparams.priorz = SMLMBaGoL.prior_allocations(nloc, k)
    z = SMLMBaGoL.allocatelocs(smld, μ, a)

    # Run the chain for the burn-in iterations.
    initstate = SMLMBaGoL.BaGoLState2D(k, z, μ, a)
    initchain = SMLMBaGoL.buildchain(smld, initstate, mcparams, true)
    
    # Run the chain for the remaining true iterations.
    chain = SMLMBaGoL.buildchain(smld, initchain, mcparams)

    return chain
end

"""
    buildchain(smld::SMLMData.SMLD2D, 
               initstate::SMLMBaGoL.BaGoLState2D,
               mcparams::SMLMBaGoL.MCParams2D,
               burnin::Bool = false) 

Build an RJMCMC chain starting with state `initstate`.

# Description
This function constructs an RJMCMC chain starting with the provided initial
state in `initstate`.

# Inputs
-`smld`: SMLD2D structure containing the localizations.
-`initstate`: Initializing state of the chain.
-`mcparams`: Structure of MCMC parameters.
-`burnin`: Boolean indicating whether or not a burn-in phase is being
           requested.  When true, only the final state of the chain is
           retained.

# Outputs
-`chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
"""
function buildchain(smld::SMLMData.SMLD2D, 
                    initstate::SMLMBaGoL.BaGoLState2D,
                    mcparams::SMLMBaGoL.MCParams2D,
                    burnin::Bool = false)
    # Run the chain for the number of iterations specified in `mcparams`. If
    # this is a burn-in run (`burnin=true`), we'll only keep the most recent
    # state after each iteration.
    niter = burnin ? mcparams.n_burnin : mcparams.n_chain
    state = deepcopy(initstate)
    accepted = true
    chain = SMLMBaGoL.BaGoLChain2D(state, accepted)
    for ii = 1:niter
        # If there is only one emitter, allocate all localizations to that 
        # emitter.
        if state.k == 1
            state.z = ones(Int, Base.length(state.z))
        end

        # If any emitters in the chain have no allocations, remove them before
        # proceeding.
        for kk in state.k:-1:1
            if !any(state.z .== kk)
                SMLMBaGoL.removeemitter!(state, kk)
            end
        end

        # Select a jump.
        jumptype = Distributions.rand(mcparams.jumpdistrib)

        # Update the state based on the proposed jump.
        state, accepted = SMLMBaGoL.updatestate(smld, state, mcparams, jumptype)

        # If needed, store this state in the output chain.
        if !burnin
            SMLMBaGoL.addstate!(chain, state, accepted)
        end
    end

    # Determine what to return based on `burnin`.
    if burnin
        return SMLMBaGoL.BaGoLChain2D(state)
    else
        return chain
    end
end

"""
    buildchain(smld::SMLMData.SMLD2D, 
               initchain::SMLMBaGoL.BaGoLChain2D,
               mcparams::SMLMBaGoL.MCParams2D,
               burnin::Bool = false) 

Build an RJMCMC chain starting with chain `initchain`.

# Description
This function constructs an RJMCMC chain starting with the provided initial
state in at the end of `initchain`.  This method is just used to dispatch on
the version of buildchain() with an initial state input.

# Inputs
-`smld`: SMLD2D structure containing the localizations.
-`initchain`: Chain whose last state is used to initialize the chain.
-`mcparams`: Structure of MCMC parameters.
-`burnin`: Boolean indicating whether or not a burn-in phase is being
           requested.  When true, only the final state of the chain is
           retained.

# Outputs
-`chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
"""
function buildchain(smld::SMLMData.SMLD2D, 
                    initchain::SMLMBaGoL.BaGoLChain2D,
                    mcparams::SMLMBaGoL.MCParams2D,
                    burnin::Bool = false)
    # Call the version of buildchain() with a state input.
    return buildchain(smld, initchain.states[end], mcparams, burnin)
end

"""
    updatestate(smld::SMLMData.SMLD2D, 
                state::SMLMBaGoL.BaGoLState2D,
                mcparams::SMLMBaGoL.MCParams2D, 
                jumptype::Int)  

Propose and accept/reject a jump of type `jumptype`.

# Description
This function moves proposes a jump of type `jumptype` and then either returns
an updated state (jump accepted) or the input `state` (jump rejected).

# Inputs
-`smld`: SMLD2D structure containing the localizations.
-`state`: Current state of the Markov chain.
-`mcparams`: Structure of MCMC parameters.
-`jumptype`: Index of the jump to be proposed.
             (1=move, 2=reallocate, 3=birth, 4=death)

# Outputs
-`state`: A proposed state with the moved emitters.
-`accepted`: Boolean indicating whether or not the move was accepted, which is 
             always true since moves use Gibbs sampling.
"""
function updatestate(smld::SMLMData.SMLD2D, 
                     state::SMLMBaGoL.BaGoLState2D,
                     mcparams::SMLMBaGoL.MCParams2D, 
                     jumptype::Int)  
    # Update the state based on the specified jump.
    if jumptype == 1
        # Jump type 1 is a move of the existing emitters.
        # Move jumps are done by Gibbs sampling so are always accepted. 
        state, accepted = SMLMBaGoL.move(smld, state, mcparams)
    elseif jumptype == 2
        # Jump type 2 is a reallocation of localizations to emitters.
        # Reallocation jumps are done by Gibbs sampling so are always accepted.
        state, accepted = SMLMBaGoL.reallocate(smld, state)
    elseif jumptype == 3
        # Jump type 3 is a birth of new emitter.  If there are already 
        # as many emitters as localizations, we'll just return the current
        # state.
        state, accepted = SMLMBaGoL.birth(smld, state, mcparams)
    elseif jumptype == 4
        # Jump type 4 is a death of an existing emitter.  If there is only 1
        # emitter, we don't want to remove it so we'll return the current
        # state.
        state, accepted = SMLMBaGoL.death(smld, state, mcparams)
    else
        error("Unknown jump type!")
    end

    return state, accepted
end

"""
    move(smld::SMLMData.SMLD2D, 
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
    reallocate(smld::SMLMData.SMLD2D, state::SMLMBaGoL.BaGoLState2D)

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
    birth(smld::SMLMData.SMLD2D, 
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
    if state.k < SMLMData.length(smld)
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
    death(smld::SMLMData.SMLD2D, 
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
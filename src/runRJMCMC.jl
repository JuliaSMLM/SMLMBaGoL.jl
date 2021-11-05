using SMLMData
using Distributions
using Base

"""
    chain = runRJMCMC(smld::SMLMData.SMLD2D, 
                      roi::Vector{Float64}, 
                      mcparams::MCParams2D)

Perform reversible jump Markov chain monte carlo (RJMCMC).

# Description
This function performs RJMCMC analysis by preparing some distributions and
parameters and then building the RJMCMC chain.

# Inputs
-`smld`: SMLD2D structure containing the localizations.
-`roi`: Region of interest corresponding to `smld` localizations. 
        (pixels)([ystart; xstart; yend; xend])
-`mcparams`: Structure of MCMC parameters.

# Outputs
-`chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
"""
function runRJMCMC(smld::SMLMData.SMLD2D,
                   roi::Vector{Float64},
                   mcparams::SMLMBaGoL.MCParams2D)
    # Prepare an emitter distribution from the provided localizations 
    # (approximated as the normalized Gaussian image of localizations).
    internals = SMLMBaGoL.Internals2D()
    internals.roi = roi
    internals.imdistrib, internals.srimsize = SMLMBaGoL.imagedistribution(smld, 
        mcparams.srmag, mcparams.nsigma, internals.roi)
    internals.area = Float64(prod(smld.datasize[1:2]))
    
    # Prepare some distributions (e.g., priors) and define initial states.
    internals.jumpdistrib = SMLMBaGoL.jumpdistrib(mcparams.p_jump)
    nloc = Base.length(smld)
    internals.priork = SMLMBaGoL.prior_kemitters(nloc, mcparams.α, mcparams.β)
    k = Int(ceil(nloc / (mcparams.α*mcparams.β)))
    μ, _ = SMLMBaGoL.samplecoords2D(internals.imdistrib, internals.srimsize[1], k)
    μ ./= mcparams.srmag
    μ .+= repeat(transpose(internals.roi[1:2]), k) .- 1.0
    internals.priora = SMLMBaGoL.prior_drift(mcparams.σ_a * [1.0; 1.0])
    a = zeros(Float64, k, 2)
    internals.priorz = SMLMBaGoL.prior_allocations(nloc, k)
    z = SMLMBaGoL.allocatelocs(smld, μ, a)

    # Run the chain for the burn-in iterations.
    initstate = SMLMBaGoL.BaGoLState2D(k, z, μ, a)
    initchain = SMLMBaGoL.buildchain(smld, initstate, mcparams, internals, true)
    
    # Run the chain for the remaining true iterations.
    chain = SMLMBaGoL.buildchain(smld, initchain, mcparams, internals)

    return chain
end

"""
    chain = runRJMCMC(smld::Vector{SMLMData.SMLD2D}, 
                      roi::Vector{Float64}, 
                      mcparams::MCParams2D)

Perform reversible jump Markov chain monte carle (RJMCMC).

# Description
This function performs RJMCMC analysis by preparing some distributions and
parameters and then building the RJMCMC chain.  This function dispatches on
the single `smld` version of runRJMCMC!() on each entry of the vector input 
`smld`.

# Inputs
-`smld`: Vector of SMLD2D structures.
-`roi`: Region of interest corresponding to `smld` localizations. 
        (pixels)([ystart; xstart; yend; xend])
-`mcparams`: Structure of MCMC parameters.

# Outputs
-`chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
"""
function runRJMCMC(smld::Vector{SMLMData.SMLD2D}, 
                   roi::Vector{Float64},
                   mcparams::MCParams2D)
    # Loop over preclusters and perform RJMCMC on each of them.
    nclusters = Base.length(smld)
    chain = Vector{SMLMBaGoL.BaGoLChain2D}(undef, nclusters)
    for nn = 1:nclusters
        chain[nn] = runRJMCMC(smld[nn], roi, mcparams)
    end
    
    return chain
end

"""
    chain = runRJMCMC!(smld::Matrix{SMLMData.SMLD2D}, 
                       rois::Matrix{Vector{Float64}}
                       mcparams::MCParams2D)

Perform reversible jump Markov chain monte carle (RJMCMC).

# Description
This function performs RJMCMC analysis by preparing some distributions and
parameters and then building the RJMCMC chain.  This function dispatches on
the vector `smld` version of runRJMCMC!() on each entry of the matrix input 
`smld`.

# Inputs
-`smld`: Matrix of SMLD2D structures.
-`rois`: Region of interest of each entry in `smld`.
-`mcparams`: Structure of MCMC parameters.

# Outputs
-`chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
"""
function runRJMCMC(smld::Matrix{SMLMData.SMLD2D}, 
                   rois::Matrix{Vector{Float64}}, 
                   mcparams::MCParams2D)
    # Loop over subregions in `smld` and perform RJMCMC.
    smldsize = size(smld)
    chain = Matrix{Vector{SMLMBaGoL.BaGoLChain2D}}(undef, smldsize)
    for ii = 1:smldsize[1], jj = 1:smldsize[2]
        # Generate a distinct SMLD2D for each precluster.
        smldclusters, _ = SMLMData.isolateconnected(smld[ii, jj])

        # Run RJMCMC on each precluster.
        chain[ii, jj] = runRJMCMC(smldclusters, rois[ii, jj], mcparams)
    end

    return chain
end

"""
    chain = buildchain(smld::SMLMData.SMLD2D, 
                       initstate::SMLMBaGoL.BaGoLState2D,
                       mcparams::SMLMBaGoL.MCParams2D,
                       internals::SMLMBaGoL.Internals2D,
                       burnin::Bool = false) 

Build an RJMCMC chain starting with state `initstate`.

# Description
This function constructs an RJMCMC chain starting with the provided initial
state in `initstate`.

# Inputs
-`smld`: SMLD2D structure containing the localizations.
-`initstate`: Initializing state of the chain.
-`mcparams`: Structure of MCMC parameters.
-`internals`: Structure of distributions/parameters (e.g., priors).
-`burnin`: Boolean indicating whether or not a burn-in phase is being
           requested.  When true, only the final state of the chain is
           retained.

# Outputs
-`chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
"""
function buildchain(smld::SMLMData.SMLD2D, 
                    initstate::SMLMBaGoL.BaGoLState2D,
                    mcparams::SMLMBaGoL.MCParams2D,
                    internals::SMLMBaGoL.Internals2D,
                    burnin::Bool = false)
    # Run the chain for the number of iterations specified in `mcparams`. If
    # this is a burn-in run (`burnin=true`), we'll only keep the most recent
    # state after each iteration.
    niter = burnin ? mcparams.n_burnin : mcparams.n_chain
    state = deepcopy(initstate)
    if !burnin
        chain = SMLMBaGoL.BaGoLChain2D(niter)
        chain.states[1] = state
        chain.accepted[1] = true
    end
    for ii = 2:niter
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
        jumptype = Distributions.rand(internals.jumpdistrib)

        # Update the state based on the proposed jump.
        state, accepted = SMLMBaGoL.updatestate(smld, state, 
            mcparams, internals, jumptype)

        # If needed, store this state in the output chain.
        if !burnin
            chain.states[ii] = state
            chain.accepted[ii] = accepted
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
    chain = buildchain(smld::SMLMData.SMLD2D, 
                       initchain::SMLMBaGoL.BaGoLChain2D,
                       mcparams::SMLMBaGoL.MCParams2D,
                       internals::SMLMBaGoL.Internals2D,
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
-`internals`: Structure of distributions/parameters (e.g., priors).
-`burnin`: Boolean indicating whether or not a burn-in phase is being
           requested.  When true, only the final state of the chain is
           retained.

# Outputs
-`chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
"""
function buildchain(smld::SMLMData.SMLD2D, 
                    initchain::SMLMBaGoL.BaGoLChain2D,
                    mcparams::SMLMBaGoL.MCParams2D,
                    internals::SMLMBaGoL.Internals2D,
                    burnin::Bool = false)
    # Call the version of buildchain() with a state input.
    return buildchain(smld, initchain.states[end], mcparams, internals, burnin)
end

"""
    state, accepted = updatestate(smld::SMLMData.SMLD2D, 
                                  state::SMLMBaGoL.BaGoLState2D,
                                  mcparams::SMLMBaGoL.MCParams2D,
                                  internals::SMLMBaGoL.Internals2D,
                                  jumptype::Int)  

Propose and accept/reject a jump of type `jumptype`.

# Description
This function moves proposes a jump of type `jumptype` and then either returns
an updated state (jump accepted) or the input `state` (jump rejected).

# Inputs
-`smld`: SMLD2D structure containing the localizations.
-`state`: Current state of the Markov chain.
-`mcparams`: Structure of MCMC parameters.
-`internals`: Structure of distributions/parameters (e.g., priors).
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
                     internals::SMLMBaGoL.Internals2D,
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
        state, accepted = SMLMBaGoL.birth(smld, state, mcparams, internals)
    elseif jumptype == 4
        # Jump type 4 is a death of an existing emitter.  If there is only 1
        # emitter, we don't want to remove it so we'll return the current
        # state.
        state, accepted = SMLMBaGoL.death(smld, state, mcparams, internals)
    else
        error("Unknown jump type!")
    end

    return state, accepted
end

function updatemcparams!(mcparams::SMLMBaGoL.MCParams2D)
end
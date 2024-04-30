using SMLMData
using Distributions
using Base

"""
    chain = runRJMCMC(smld::SMLMData.SMLD2D, 
                      roi::Vector{<:Real}, 
                      mcparams::MCParams2D)

Perform reversible jump Markov chain monte carlo (RJMCMC).

# Description
This function performs RJMCMC analysis by preparing some distributions and
parameters and then building the RJMCMC chain.

# Inputs
- `smld`: SMLD2D structure containing the localizations.
- `roi`: Region of interest corresponding to `smld` localizations. 
         (pixels)([ystart; xstart; yend; xend])
- `mcparams`: Structure of MCMC parameters.

# Outputs
- `chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
"""
function runRJMCMC(smld::SMLMData.SMLD2D,
    roi::Vector{<:Real},
    mcparams::SMLMBaGoL.MCParams2D)

    # Prepare an emitter distribution from the provided localizations 
    # (approximated as the normalized Gaussian image of localizations).
    internals = SMLMBaGoL.Internals2D()
    internals.roi = roi
    internals.imdistrib, internals.srimsize = SMLMBaGoL.imagedistribution(smld;
        mag = mcparams.imdistrib_mag, nsigma = mcparams.nsigma, roi = internals.roi)
    internals.area = (roi[4] - roi[2] + 1.0) * (roi[3] - roi[1] + 1.0)

    # Prepare some distributions (e.g., priors) and define initial states.
    internals.jumpdistrib = SMLMBaGoL.jumpdistrib(mcparams.p_jump)
    internals.priora = SMLMBaGoL.prior_drift(mcparams.σ_a * [1.0; 1.0])
    nloc = Base.length(smld)
    internals.priork = SMLMBaGoL.prior_kemitters(nloc, mcparams.η, mcparams.γ)
    k = Int(ceil(nloc / (mcparams.η * mcparams.γ)))
    μ_SR, _ = SMLMBaGoL.samplecoords2D(internals.imdistrib, internals.srimsize[1], k)
    μ = ((μ_SR .- 0.5) ./ mcparams.imdistrib_mag) .+ 0.5
    μ .+= repeat(transpose(internals.roi[1:2]), k) .- 1.0
    a = zeros(Float32, k, 2)
    z = SMLMBaGoL.allocatelocs(smld, μ, a)

    # Run the chain for the burn-in iterations.
    initstate = SMLMBaGoL.BaGoLState2D(k, z, μ, a)
    SMLMBaGoL.isolateuseful!(initstate)
    if mcparams.n_burnin > 0
        initchain = SMLMBaGoL.buildchain(smld, initstate, mcparams, internals, true)
    else
        initchain = SMLMBaGoL.BaGoLChain2D(initstate, true)
    end

    # Run the chain for the remaining true iterations.
    if mcparams.n_chain > 0
        chain = SMLMBaGoL.buildchain(smld, initchain, mcparams, internals)
    else
        chain = deepcopy(initchain)
    end

    return chain
end

"""
    chain = runRJMCMC(smld::Vector{SMLMData.SMLD2D}, 
                      roi::Vector{<:Real}, 
                      mcparams::MCParams2D)

Perform reversible jump Markov chain monte carle (RJMCMC).

# Description
This function performs RJMCMC analysis by preparing some distributions and
parameters and then building the RJMCMC chain.  This function dispatches on
the single `smld` version of runRJMCMC() on each entry of the vector input 
`smld`.

# Inputs
- `smld`: Vector of SMLD2D structures.
- `roi`: Region of interest corresponding to `smld` localizations. 
         (pixels)([ystart; xstart; yend; xend])
- `mcparams`: Structure of MCMC parameters.

# Outputs
- `chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
"""
function runRJMCMC(smld::Vector{SMLMData.SMLD2D},
    roi::Vector{<:Real},
    mcparams::MCParams2D)

    # Loop over preclusters and perform RJMCMC on each of them.
    nclusters = Base.length(smld)
    chain = Vector{SMLMBaGoL.BaGoLChain2D}(undef, nclusters)
    for nn = 1:nclusters
        chain[nn] = SMLMBaGoL.runRJMCMC(smld[nn], roi, mcparams)
    end

    return chain
end

"""
    chain = runRJMCMC(smld::Matrix{SMLMData.SMLD2D}, 
                      rois::Matrix{Vector{<:Real}}
                      mcparams::MCParams2D)

Perform reversible jump Markov chain monte carle (RJMCMC).

# Description
This function performs RJMCMC analysis by preparing some distributions and
parameters and then building the RJMCMC chain.  This function dispatches on
the vector `smld` version of runRJMCMC() on each entry of the matrix input 
`smld`.

# Inputs
- `smld`: Matrix of SMLD2D structures.
- `rois`: Region of interest of each entry in `smld`.
- `mcparams`: Structure of MCMC parameters.

# Outputs
- `chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
"""
function runRJMCMC(smld::Matrix{SMLMData.SMLD2D},
    rois::Matrix{Vector{T}} where T<:Real,
    mcparams::MCParams2D)

    # Loop over subregions in `smld` and perform RJMCMC.
    smldsize = size(smld)
    chain = Matrix{Vector{SMLMBaGoL.BaGoLChain2D}}(undef, smldsize)
    for ii = 1:smldsize[1], jj = 1:smldsize[2]
        # Generate a distinct SMLD2D for each precluster.
        smldclusters, _ = SMLMData.isolateconnected(smld[ii, jj])

        # Run RJMCMC on each precluster.
        chain[ii, jj] = SMLMBaGoL.runRJMCMC(smldclusters, rois[ii, jj], mcparams)
    end

    return chain
end

"""
    chain, λchain = runRJMCMC(smld::Matrix{SMLMData.SMLD2D}, 
                      rois::Matrix{Vector{<:Real}}
                      mcparams::MCParams2D,
                      hbparams::HBParams2D)

Perform reversible jump Markov chain monte carle (RJMCMC).

# Description
This function performs RJMCMC analysis by preparing some distributions and
parameters and then building the RJMCMC chain.  This function dispatches on
the vector `smld` version of runRJMCMC() on each entry of the matrix input 
`smld`.

# Inputs
- `smld`: Matrix of SMLD2D structures.
- `rois`: Region of interest of each entry in `smld`.
- `mcparams`: Structure of MCMC parameters.
- `hbparams`: Structure of hierarchical Bayes parameters.

# Outputs
- `chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
- `λchain`: An array of the λ parameters that were used for each hierarchical 
            sample.
"""
function runRJMCMC(smld::Matrix{SMLMData.SMLD2D},
    rois::Matrix{Vector{T}} where T<:Real,
    roioverlap::Real,
    mcparams::MCParams2D,
    hbparams::HBParams2D)

    # If hierarchical Bayes is turned off, reset some parameters.
    if !hbparams.on
        hbparams.nsamples = maximum([mcparams.n_burnin; mcparams.n_chain])
        hbparams.nthinning = 0
    end

    # Perform the initial burn-in of the chain.
    mcparams_hb = deepcopy(mcparams)
    mcparams_hb.n_chain = 0
    mcparams_hb.n_burnin = hbparams.nsamples
    nsamples_λ = maximum([1; floor(Int, mcparams.n_burnin / hbparams.nsamples)])
    smldsize = size(smld)
    smldclusters = Matrix{Vector{SMLMData.SMLD2D}}(undef, smldsize)
    chain = Matrix{Vector{SMLMBaGoL.BaGoLChain2D}}(undef, smldsize)
    for nn = 1:nsamples_λ
        # Run the chain with the current set of hyperparameters.
        for ii = 1:smldsize[1], jj = 1:smldsize[2]
            # Generate a distinct SMLD2D for each precluster.
            smldclusters[ii, jj], _ = SMLMData.isolateconnected(smld[ii, jj])

            # Run RJMCMC on each precluster.
            chain[ii, jj] = SMLMBaGoL.runRJMCMC(smldclusters[ii, jj], rois[ii, jj], mcparams_hb)
        end

        # Remove emitters in the overlapping regions and count the number of
        # localizations and number of emitters.
        validchain = SMLMBaGoL.removeoverlap(chain, smld[1].datasize, rois, roioverlap)
        nloc = Int[]
        k = Int[]
        for nn = 1:length(validchain)
            for kk = 1:length(validchain[nn])
                if !isempty(validchain[nn][kk].states)
                    push!(nloc, length(smldclusters[nn][kk]))
                    push!(k, validchain[nn][kk].states[end].k)
                end
            end
        end

        # Sample new hyperparameters.
        for mm = 1:hbparams.nthinning
            mcparams_hb, _ = SMLMBaGoL.updatepriorλ(nloc, k, mcparams_hb, hbparams)
        end
    end

    # Loop over subregions in `smld` and perform RJMCMC, updating the 
    # hierarchical parameters every hbparams.nsamples samples in the chain.
    mcparams_hb.n_chain = hbparams.nsamples
    mcparams_hb.n_burnin = 0
    nsamples_λ = maximum([1; floor(Int, mcparams.n_chain / hbparams.nsamples)])
    smldclusters = Matrix{Vector{SMLMData.SMLD2D}}(undef, smldsize)
    chain = Matrix{Vector{SMLMBaGoL.BaGoLChain2D}}(undef, smldsize)
    λchain = Matrix{Float32}(undef, nsamples_λ, 2)
    for nn = 1:nsamples_λ
        # For each hierarchical sample, we'll run the chain for mcparams_hb.n_chain 
        # samples, ensuring that no additional burn-in is made.
        for ii = 1:smldsize[1], jj = 1:smldsize[2]
            # Generate a distinct SMLD2D for each precluster.
            smldclusters[ii, jj], _ = SMLMData.isolateconnected(smld[ii, jj])
        
            # Run RJMCMC on each precluster.
            if nn > 1
                chain[ii, jj] = Base.cat(chain[ii, jj],
                    SMLMBaGoL.runRJMCMC(smldclusters[ii, jj], rois[ii, jj], mcparams_hb))
            else
                chain[ii, jj] = SMLMBaGoL.runRJMCMC(smldclusters[ii, jj], rois[ii, jj], mcparams_hb)
            end
        end

        # Remove emitters in the overlapping regions and count the number of
        # localizations and number of emitters.
        validchain = SMLMBaGoL.removeoverlap(chain, smld[1].datasize, rois, roioverlap)
        nloc = Int[]
        k = Int[]
        for nn = 1:length(validchain)
            for kk = 1:length(validchain[nn])
                if !isempty(validchain[nn][kk].states)
                    push!(nloc, length(smldclusters[nn][kk]))
                    push!(k, validchain[nn][kk].states[end].k)
                end
            end
        end

        # Sample new hyperparameters.
        for mm = 1:hbparams.nthinning
            mcparams_hb, _ = SMLMBaGoL.updatepriorλ(nloc, k, mcparams_hb, hbparams)
        end
        λchain[nn, :] = [mcparams_hb.η mcparams_hb.γ]
    end

    return chain, λchain
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
- `smld`: SMLD2D structure containing the localizations.
- `initstate`: Initializing state of the chain.
- `mcparams`: Structure of MCMC parameters.
- `internals`: Structure of distributions/parameters (e.g., priors).
- `burnin`: Boolean indicating whether or not a burn-in phase is being
            requested.  When true, only the final state of the chain is
            retained.

# Outputs
- `chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
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
- `smld`: SMLD2D structure containing the localizations.
- `initchain`: Chain whose last state is used to initialize the chain.
- `mcparams`: Structure of MCMC parameters.
- `internals`: Structure of distributions/parameters (e.g., priors).
- `burnin`: Boolean indicating whether or not a burn-in phase is being
            requested.  When true, only the final state of the chain is
            retained.

# Outputs
- `chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
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
                                  currentstate::SMLMBaGoL.BaGoLState2D,
                                  mcparams::SMLMBaGoL.MCParams2D,
                                  internals::SMLMBaGoL.Internals2D,
                                  jumptype::Int)  

Propose and accept/reject a jump of type `jumptype`.

# Description
This function proposes a jump of type `jumptype` and then either returns
an updated state (jump accepted) or the input `state` (jump rejected).

# Inputs
- `smld`: SMLD2D structure containing the localizations.
- `currentstate`: Current state of the Markov chain.
- `mcparams`: Structure of MCMC parameters.
- `internals`: Structure of distributions/parameters (e.g., priors).
- `jumptype`: Index of the jump to be proposed.
              (1=move, 2=reallocate, 3=birth, 4=death)

# Outputs
- `state`: A proposed state with the moved emitters.
- `accepted`: Boolean indicating whether or not the move was accepted.
"""
function updatestate(smld::SMLMData.SMLD2D,
    currentstate::SMLMBaGoL.BaGoLState2D,
    mcparams::SMLMBaGoL.MCParams2D,
    internals::SMLMBaGoL.Internals2D,
    jumptype::Int)

    # Update the state based on the specified jump.
    if jumptype == 1
        # Jump type 1 is a move of the existing emitters.
        # Move jumps are done by Gibbs sampling so are always accepted. 
        state, accepted = SMLMBaGoL.move(smld, currentstate, mcparams)
    elseif jumptype == 2
        # Jump type 2 is a reallocation of localizations to emitters.
        # Reallocation jumps are done by Gibbs sampling so are always accepted.
        state, accepted = SMLMBaGoL.reallocate(smld, currentstate)
    elseif jumptype == 3
        # Jump type 3 is a birth of new emitter.  If there are already 
        # as many emitters as localizations, we'll just return the current
        # state.
        state, accepted = SMLMBaGoL.birth(smld, currentstate, mcparams, internals)
    elseif jumptype == 4
        # Jump type 4 is a death of an existing emitter.  If there is only 1
        # emitter, we don't want to remove it so we'll return the current
        # state.
        state, accepted = SMLMBaGoL.death(smld, currentstate, mcparams, internals)
    else
        error("Unknown jump type!")
    end

    # Make sure the updated state contains useful emitters (i.e., all emitters 
    # in the state have localizations allocated to them).
    SMLMBaGoL.isolateuseful!(state)

    return state, accepted
end

"""
    mcparams, accepted = SMLMBaGoL.updatepriorλ(nloc::Vector{Int}, 
                                                k::Vector{Int}, 
                                                mcparams::SMLMBaGoL.MCParams2D,
                                                hbparams::SMLMBaGoL.HBParams2D)

Propose and accept/reject an update of the prior on the blinks per emitter.

# Description
This function proposes an update for the parameters defining the distribution
of the number of blinks per emitter λ.

# Inputs
- `nloc`: Number of localizations.
- `k`: Number of emitters.
- `mcparams`: Structure of parameters (see SMLMBaGoL.MCParams2D)
- `hbparams`: Structure of parameters (see SMLMBaGoL.HBParams2D)

# Outputs
- `mcparams`: Copy of input `mcparams` with parameters related to λ updated.
- `accepted`: Boolean indicating which updates were accepted (the prior on λ
              can have multiple parameters, so this might be an array).
"""
function updatepriorλ(nloc::Vector{Int}, k::Vector{Int},
    mcparams::SMLMBaGoL.MCParams2D,
    hbparams::SMLMBaGoL.HBParams2D)

    # Proposed and accept/reject an update to the prior on λ.
    mcparams = deepcopy(mcparams)
    accepted = [false; false]
    η_prop, accepted[1] = SMLMBaGoL.updateη(nloc, k, mcparams, hbparams)
    mcparams.η = accepted[1] ? η_prop : mcparams.η
    γ_prop, accepted[2] = SMLMBaGoL.updateγ(nloc, k, mcparams, hbparams)
    mcparams.γ = accepted[2] ? γ_prop : mcparams.γ

    return mcparams, accepted
end
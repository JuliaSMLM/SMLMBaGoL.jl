using SMLMData
using Distributions
using Base

function runRJMCMC!(smld::Matrix{SMLMData.SMLD2D}, 
                    rois::Matrix{Vector{Float64}}, 
                    mcparams::MCParams2D)
    # Loop over subregions in `smld` and perform RJMCMC.
    smldsize = size(smld)
    for ii = 1:smldsize[1], jj = 1:smldsize[2]
        # Generate a distinct SMLD2D for each precluster.
        smldclusters, _ = SMLMData.isolateconnected(smld[ii, jj])

        # Run RJMCMC on each precluster.
        mcparams.roi = rois[ii, jj]
        runRJMCMC!(smldclusters, mcparams)
    end

end

function runRJMCMC!(smld::Vector{SMLMData.SMLD2D}, 
                    mcparams::MCParams2D)
    # Loop over preclusters and perform RJMCMC on each of them.
    nclusters = Base.length(smld)
    for nn = 1:nclusters
        runRJMCMC!(smld[nn], mcparams)
    end
    
end

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
    initchain = SMLMBaGoL.BaGoLChain([k], [μ], [a], [z])
    chain = SMLMBaGoL.burnin(smld, initchain, mcparams)
    # chain = SMLMBaGoL.BaGoLChain(mcparams.n_chain)
end

function burnin(smld::SMLMData.SMLD2D, 
                initchain::SMLMBaGoL.BaGoLChain,
                mcparams::SMLMBaGoL.MCParams2D)
    # Run the chain for the burn-in iterations.
    for ii = 1:mcparams.n_burnin
        # If there is only one emitter, allocate all localizations to that 
        # emitter.
        if initchain.k[end] == 1
            initchain.z[end] = ones(Int, Base.length(initchain.z[end]))
        end

        # If any emitters in the chain have no allocations, remove them before
        # proceeding.
        for kk in initchain.k[end]:-1:1
            if !any(initchain.z[end] .== kk)
                SMLMBaGoL.removeemitter!(initchain, kk)
            end
        end

        # Select a jump.
        jumptype = Distributions.rand(mcparams.jumpdistrib)

        # Update the chain based on the jump.
        SMLMBaGoL.updatechain!(smld, initchain, mcparams, jumptype)
    end
end

function updatechain!(smld::SMLMData.SMLD2D, 
                      chain::SMLMBaGoL.BaGoLChain,
                      mcparams::SMLMBaGoL.MCParams2D, 
                      jumptype::Int)  
    # Update the chain based on the specified jump.
    if jumptype == 1
        # Jump type 1 is a move of the existing emitters.
        # Move jumps are done by Gibbs sampling so are always accepted. 
        proposal = SMLMBaGoL.proposemove(smld, chain, mcparams)
        acceptance = 1.0
    elseif jumptype == 2
        # Jump type 2 is a reallocation of localizations to emitters.
        # Reallocation jumps are done by Gibbs sampling so are always accepted.
        proposal = SMLMBaGoL.proposeallocation(smld, chain)
        acceptance = 1.0
    elseif jumptype == 3
        # Jump type 3 is a birth of new emitter.  If there are already 
        # as many emitters as localizations, we'll just return the current
        # state of the chain.
        if chain.k[end] < SMLMData.length(smld)
            proposal, positionind = SMLMBaGoL.proposebirth(smld, chain, mcparams)
            acceptance = SMLMBaGoL.acceptbirth(smld, 
                                               proposal, 
                                               positionind, 
                                               chain, 
                                               mcparams)
        else
            proposal = getstate(chain)
            acceptance = 1.0
        end
    elseif jumptype == 4
        # Jump type 4 is a death of an existing emitter.  If there is only 1
        # emitter, we don't want to remove it so we'll return the current
        # state of the chain.
        if chain.k[end] == 1
            proposal = getstate(chain)
            acceptance = 1.0
        else
            proposal, positionind = SMLMBaGoL.proposedeath(smld, chain)
            acceptance = 1 / SMLMBaGoL.acceptdeath(smld, 
                                                   proposal, 
                                                   positionind, 
                                                   chain, 
                                                   mcparams)
        end
    else
        error("Unknown jump type!")
    end

    # Test the jump and store it if it's accepted.
    if Base.rand() <= acceptance
        SMLMBaGoL.catchain!(chain, proposal)
    else
        SMLMBaGoL.keepstate!(chain)
    end
end

function catchain!(chain1::SMLMBaGoL.BaGoLChain, chain2::SMLMBaGoL.BaGoLChain)
    fields = fieldnames(SMLMBaGoL.BaGoLChain)
    for ff in fields
        setfield!(chain1, ff, [getfield(chain1, ff); getfield(chain2, ff)])
    end
end
function catchain(chain1::SMLMBaGoL.BaGoLChain, chain2::SMLMBaGoL.BaGoLChain)
    fields = fieldnames(SMLMBaGoL.BaGoLChain)
    chain = SMLMBaGoL.BaGoLChain(SMLMBaGoL.length(chain1) 
        + SMLMBaGoL.length(chain2))
    for ff in fields
        setfield!(chain, ff, [getfield(chain1, ff); getfield(chain2, ff)])
    end

    return chain
end

function keepstate!(chain::SMLMBaGoL.BaGoLChain)
    fields = fieldnames(SMLMBaGoL.BaGoLChain)
    for ff in fields
        setfield!(chain, ff, [getfield(chain, ff); [getfield(chain, ff)[end]]])
    end
end

function getstate(chain::SMLMBaGoL.BaGoLChain)
    return SMLMBaGoL.BaGoLChain(chain.k[end], chain.μ[end], chain.a[end], chain.z[end])
end
function getstate(chain::SMLMBaGoL.BaGoLChain, ind::Vector{Int})
    return SMLMBaGoL.BaGoLChain(chain.k[ind], chain.μ[ind], chain.a[ind], chain.z[ind])
end
function getstate(chain::SMLMBaGoL.BaGoLChain, ind::Int)
    return SMLMBaGoL.BaGoLChain(chain.k[ind], chain.μ[ind], chain.a[ind], chain.z[ind])
end
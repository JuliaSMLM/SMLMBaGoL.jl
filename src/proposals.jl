using SMLMData
using Distributions

# This file contains some functions useful for birth and death jump proposals.
# NOTE: Not all required functions for birth/death proposals are contained in
#       this file. Some are contained in related files, e.g., allocatelocs.jl
#       for allocation probabilities.


function proposebirth(smld::SMLMData.SMLD2D,
                      chain::BaGoLParams,
                      mcparams::MCParams)
    # Propose a new emitter by treating a Gaussian SR image of the (raw) 
    # localizations as a density distribution.
    srimsize = round.(smld.datasize * mcparams.srmag)
    coords, sampleind = samplecoords2D(mcparams.imdistrib, srimsize[1])
    μprime = [chain.μ[end]; coords]
    aprime = [chain.a[end]; mcparams.σ_a * randn(2)]
    k = chain.k[end]
    wprime = ones(k+1) / (k+1)

    # Perform Gibbs sampling for the allocations.
    zprime = allocatelocs(smld, μprime, aprime)

    # Compute the probability ratio for the allocations.
    t = Float64.(smld.framenum)
    pallocprime = emitterlikelihood2D([smld.x smld.y], [smld.σ_x smld.σ_y], t,
                                      μprime, aprime, zprime)
    palloc = emitterlikelihood2D([smld.x smld.y], [smld.σ_x smld.σ_y], t,
                                 chain.μ[end], chain.a[end], chain.z[end])
    pallocratio = pallocprime / palloc                                 
    # logLalloc_prime = logLalloc_kernel([smld.x smld.y], [smld.σ_x smld.σ_y], t,
    #                                    μprime, aprime, wprime)
    # logLalloc = logLalloc_kernel([smld.x smld.y], [smld.σ_x smld.σ_y], t,
    #                              chain.μ[end], chain.a[end], chain.w[end])
    # pallocratio = exp(logLalloc_prime - logLalloc)

    # Compute the probability ratio for the number of emitters.
    nloc = SMLMData.length(smld)
    pkdistrib = priorkemitters(nloc, mcparams.α, mcparams.β)
    pkratio = pkdistrib.pdf(k+1) / pkdistrib.pdf(k)

    # Compute the complete proposal ratio.
    pjumpratio = mcparams.p_jump[3] / mcparams.p_jump[4]
    acceptancep = pallocratio * pkratio * ((k/(k+1))^nloc) * pjumpratio / 
                  (mcparams.imdistrib.pdf[sampleind] * mcparams.area)

    # Determine if we should save this proposal to the chain.
    if rand() < acceptancep
        chain_out = deepcopy(chain)
        push!(chain_out.k, k + 1)
        push!(chain_out.w, wprime)
        push!(chain_out.μ, μprime)
        push!(chain_out.a, aprime)
        push!(chain_out.z, zprime)
    end

    return chain_out
end

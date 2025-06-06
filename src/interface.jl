
function bagol(smld;
    prior_λ::Distributions.Distribution=Gamma(1.0, 1.0),
    posterior_pixel_size::Float64=0.02)

    # create subregions - pass emitters directly
    min_pts = 1
    ϵ = mean([em.σ_x for em in smld.emitters]) * 4 
    subregions = gen_subregions(smld.emitters, min_pts, ϵ)
    @info "Number of subregions: $(length(subregions))"
    # create posterior image 
    posterior = Posterior2D(smld; pixelsize = posterior_pixel_size)

    # setup chain 
    # run chain on all subregions 
    @info "bagol: running rjmcmc on subregions"
    
    @showprogress Threads.@threads for i in eachindex(subregions)
        subregion = subregions[i]
        chain, mapn_coords = rjmcmc(subregion.obs, prior_λ)
        subregion.chains[1] = chain
    end

    # merge results
    @info "bagol: making posterior image"
    for subregion in subregions
        add!(posterior, subregion.chains[1])
    end
   
    return subregions, posterior
end


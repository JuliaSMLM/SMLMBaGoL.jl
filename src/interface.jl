
function bagol(smld2D;
    prior_λ::Distributions.Distribution=Gamma(1.0, 1.0),
    pixelsize::Float64=1.0,
    posterior_pixel_size::Float64=pixelsize/50)

    # create subregions
    positions = Transpose(hcat(smld2D.y, smld2D.x))
    sigmas = Transpose(hcat(smld2D.σ_y, smld2D.σ_x))
    emitter_type = Emitter2D
    min_pts = 1
    ϵ = mean(smld2D.σ_x) * 4 
    subregions = gen_subregions(positions, sigmas, emitter_type, min_pts, ϵ)
    @info "Number of subregions: $(length(subregions))"
    # create posterior image 
    posterior = Posterior2D(smld2D; pixelsize = posterior_pixel_size)

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


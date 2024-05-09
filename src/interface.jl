
function bagol(smld2D;
    prior_λ::Distributions.Distribution=Gamma(1.0, 1.0),
    pixel_size::Float64=1.0)

    # create subregions
    positions = Transpose(hcat(smld2D.x, smld2D.y))
    sigmas = Transpose(hcat(smld2D.σ_x, smld2D.σ_y))
    emitter_type = Emitter2D
    min_pts = 1
    ϵ = mean(smld2D.σ_x) * 4 
    subregions = gen_subregions(positions, sigmas, emitter_type, min_pts, ϵ)
    @info "Number of subregions: $(length(subregions))"
    # create posterior image 
    posterior_pixel_size = pixel_size/20
    posterior = Posterior2D(smld2D; pixel_size = posterior_pixel_size)

    # setup chain 
    # run chain on all subregions 
    for (i, subregion) in enumerate(subregions)
        @info "Running chain on subregion: $(i)"
        chain,  = rjmcmc(subregion.obs, prior_λ)
        subregion.chains[1] = chain
    end

    # merge results
    for subregion in subregions
        add!(posterior, subregion.chains[1])
    end

    return posterior
end


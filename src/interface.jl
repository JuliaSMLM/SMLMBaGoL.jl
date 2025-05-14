
function bagol(smld2D;
    prior_λ::Distributions.Distribution=Gamma(1.0, 1.0),
    pixel_size::Float64=1.0)

    # create subregions
    positions = Transpose(hcat(smld2D.y, smld2D.x))
    sigmas = Transpose(hcat(smld2D.σ_y, smld2D.σ_x))
    emitter_type = Emitter2D
    min_pts = 1
    ϵ = mean(smld2D.σ_x) * 4 
    subregions = gen_subregions(positions, sigmas, emitter_type, min_pts, ϵ)
    @info "Number of subregions: $(length(subregions))"
    # create posterior image 
    posterior_pixel_size = pixel_size/20
    posterior = Posterior2D(smld2D; pixelsize = posterior_pixel_size)

    # setup chain 
    # run chain on all subregions 
    Threads.@threads for i in eachindex(subregions)
        subregion = subregions[i]
        chain, mapn_coords = rjmcmc(subregion.obs, prior_λ)
        subregion.chains[1] = chain
    end

    # merge results
    for subregion in subregions
        add!(posterior, subregion.chains[1])
    end

    # find maximum x value in all subregions 
    max_x = 0
    for subregion in subregions
        max_x = min(max_x, maximum([emitter.x for state in subregion.chains[1].states for emitter in state.emitters]))
    end
    println("Max x: ", max_x)

    # find maximum y value in all subregions 
    max_y = 0
    for subregion in subregions
        max_y = min(max_y, maximum([emitter.y for state in subregion.chains[1].states for emitter in state.emitters]))
    end
    println("Max y: ", max_y)
    
   
    return subregions, posterior
end



function gen_posterior(chain, emitters, obs; pixelsize = 1.0)
    # plot just the posterior distribution of the emitters with square pixels
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect())
    # Collate coordinates for chain
    chain_x = Float64[]
    chain_y = Float64[]
    for state in chain.states
        for emitter in state.emitters
            push!(chain_x, emitter.x)
            push!(chain_y, emitter.y)
        end
    end
    #calc bins from data range and pixelsize
    # outerbounds should caculated as 2xsigma
    xmin = Inf
    for obs in obs.ŷ
        xmin = min(xmin, minimum(obs.x - 2 * obs.σ_x))
    end
    xmax = -Inf
    for obs in obs.ŷ
        xmax = max(xmax, maximum(obs.x + 2 * obs.σ_x))
    end
    ymin = Inf
    for obs in obs.ŷ
        ymin = min(ymin, minimum(obs.y - 2 * obs.σ_y))
    end
    ymax = -Inf
    for obs in obs.ŷ
        ymax = max(ymax, maximum(obs.y + 2 * obs.σ_y))
    end

    xrange = xmax - xmin
    yrange = ymax - ymin
    nbins_x = Int(ceil(xrange / pixelsize))
    nbins_y = Int(ceil(yrange / pixelsize))

    # use xmin, xmax, ymin, ymax to set the range of the histogram
    edges_x = range(xmin, stop=xmax, length=nbins_x+1)
    edges_y = range(ymin, stop=ymax, length=nbins_y+1)
    hist_data = fit(Histogram, (chain_x, chain_y), (edges_x, edges_y))
    heatmap!(ax, hist_data.edges[1], hist_data.edges[2], hist_data.weights, colormap=:inferno)

    #save 
    save("posterior.png", fig)

    # add the true values as green X markers
    true_x = [emitter.x for emitter in emitters.emitters]
    true_y = [emitter.y for emitter in emitters.emitters]
    scatter!(ax, true_x, true_y, color=:green, markersize=1, marker=:x)
    display(fig)

    #save 
    save("posterior_emitters.png", fig)

    # draw circles from observations
    for obs in obs.ŷ
        draw_circle!(ax, Point2f0(obs.x, obs.y), obs.σ_x; 
        color=:white, linewidth=0.1, linealpha=0.1)
    end
    display(fig)
    save("posterior_emitters_circles.png", fig)
    return fig
end

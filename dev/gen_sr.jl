
function gen_sr(prior_y, obs)
    # Plot prior distribution image for y
    vals = rand(prior_y, 10000)

    pixelsize = 1.0
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

    # generate x,y grid 
    xygrid = [pdf(prior_y, [y, x]) for x in range(xmin, stop=xmax, length=nbins_x), y in range(ymin, stop=ymax, length=nbins_y)]

    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect())
    heatmap!(ax, range(xmin, stop=xmax, length=nbins_x), range(ymin, stop=ymax, length=nbins_y), xygrid, colormap=:inferno)
    display(fig)
    save("sr_image.png", fig)
end


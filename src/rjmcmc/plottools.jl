
function draw_circle!(axis, center::Point2f0, radius::Float64; points::Int=100, color=:black,
    linewidth=1.0, linestyle=:solid, linealpha=1.0, linecolor=:black, fillalpha=0.0, fillcolor=:black)
    θ = LinRange(0, 2π, points)
    x = center[1] .+ radius * cos.(θ)
    y = center[2] .+ radius * sin.(θ)
    lines!(axis, x, y, color=color, linewidth=linewidth, linestyle=linestyle, linealpha=linealpha, linecolor=linecolor)
end

function get_data_range(obs; sigma_factor=2.0)
    xmin = Inf
    for obs in obs.ŷ
        xmin = min(xmin, minimum(obs.x - sigma_factor * obs.σ_x))
    end
    xmax = -Inf
    for obs in obs.ŷ
        xmax = max(xmax, maximum(obs.x + sigma_factor * obs.σ_x))
    end
    ymin = Inf
    for obs in obs.ŷ
        ymin = min(ymin, minimum(obs.y - sigma_factor * obs.σ_y))
    end
    ymax = -Inf
    for obs in obs.ŷ
        ymax = max(ymax, maximum(obs.y + sigma_factor * obs.σ_y))
    end
    return xmin, xmax, ymin, ymax
end

function plot_posterior(chain, obs; pixelsize=1.0, title = "Posterior")
    # Collate coordinates for chain
    chain_x = Float64[]
    chain_y = Float64[]
    for state in chain.states
        for emitter in state.emitters
            push!(chain_x, emitter.x)
            push!(chain_y, emitter.y)
        end
    end

    xmin, xmax, ymin, ymax = get_data_range(obs)
    xrange = xmax - xmin
    yrange = ymax - ymin
    nbins_x = Int(ceil(xrange / pixelsize))
    nbins_y = Int(ceil(yrange / pixelsize))

    # use xmin, xmax, ymin, ymax to set the range of the histogram
    edges_x = range(xmin, stop=xmax, length=nbins_x + 1)
    edges_y = range(ymin, stop=ymax, length=nbins_y + 1)
    hist_data = fit(Histogram, (chain_x, chain_y), (edges_x, edges_y))

    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect(), title=title)
    heatmap!(ax, hist_data.edges[1], hist_data.edges[2], hist_data.weights, colormap=:inferno)
    display(fig)
    return fig, ax
end

function plot_observations!(ax, obs::Observations; color=:black, linewidth=1, linealpha=1)
    for obs in obs.ŷ
        draw_circle!(ax, Point2f0(obs.x, obs.y), obs.σ_x;
            color, linewidth, linealpha)
    end
end

function plot_observations(obs::Observations; color=:black, linewidth=1, linealpha=1)
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect())
    plot_observations!(ax, obs; color=color, linewidth=linewidth, linealpha=linealpha)
    display(fig)
    return fig, ax
end


function plot_sr(obs, prior_y; pixelsize=1.0)
    # get data range
    xmin, xmax, ymin, ymax = get_data_range(obs)

    # build range for pdf grid
    xrange = xmax - xmin
    yrange = ymax - ymin
    nbins_x = Int(ceil(xrange / pixelsize))
    nbins_y = Int(ceil(yrange / pixelsize))

    # generate x,y grid
    xygrid = [pdf(prior_y, [y, x]) for x in range(xmin, stop=xmax, length=nbins_x), y in range(ymin, stop=ymax, length=nbins_y)]

    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect(), title="SR Image")
    heatmap!(ax, range(xmin, stop=xmax, length=nbins_x), range(ymin, stop=ymax, length=nbins_y), xygrid, colormap=:inferno)
    display(fig)
    return fig, ax
end

function plot_true_values!(ax, emitters; color=:green, markersize=1, marker=:x)
    true_x = [emitter.x for emitter in emitters.emitters]
    true_y = [emitter.y for emitter in emitters.emitters]
    scatter!(ax, true_x, true_y, color=color, markersize=markersize, marker=marker)
end

function plot_prior_k(prior_k, obs; color=:blue)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="k", ylabel="pdf", title="Prior distribution for n. emitters")
    k_vec = 0:length(obs.ŷ)
    barplot!(ax, k_vec, pdf.(prior_k, k_vec), label="prior_k")
    display(fig)
end

function plot_prior_λ(prior_λ, obs; color=:blue)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="λ", ylabel="pdf", title="Prior distribution for loc/emitter")
    k_vec = 0:length(obs.ŷ)
    lines!(ax, k_vec, pdf.(prior_λ, k_vec), label="prior_λ")
    display(fig)
end

function plot_state_length(chain; color=:blue)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Jump number", ylabel="Number of emitters", title="Chain length over run")
    lines!(ax, 1:length(chain.states), length.(chain.states))
    display(fig)
end

function plot_sld(chain)
    # state length distribution
    n_map, n_vec = find_mapn(chain)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Number of emitters", ylabel="Frequency", title="Number of emitters over run")
    hist!(ax, n_vec, bins=0.5:1:maximum(n_vec)+0.5)
    display(fig)
end

function animate_chain(chain, z_chain, obs, emitters; filename="scatter_animation.mp4")
    # Create an observable for the frame index
    frame_index = Observable(1)

    # Create the figure and axis
    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y")

    # Emitters
    for emitter in emitters.emitters
        scatter!(ax, [emitter.x], [emitter.y], color=:red)
    end

    # Observations color coded by allocation
    color_iterator = [:red, :green, :blue, :yellow, :cyan, :magenta, :black]

    function calc_idx(loc_id::Int)
        idx = (1 + (loc_id - 1) % length(color_iterator))
        return idx
    end

    colors = @lift(
        [color_iterator[calc_idx(z_chain[$(frame_index)].idx[i])]
         for i in eachindex(obs.ŷ)]
    )

    circles = [draw_circle!(ax, Point2f0(obs.ŷ[i].x, obs.ŷ[i].y), obs.ŷ[i].σ_x, color=colors[][i]) for i in eachindex(obs.ŷ)]

    # Function to extract coordinates for a given state
    coords = @lift(
        [Point2f0.(chain.states[$(frame_index)].emitters[j].x, chain.states[$(frame_index)].emitters[j].y)
         for j in 1:length(chain.states[$(frame_index)].emitters)],
    )

    scatter!(ax, coords,
        color=:blue,
        marker=:x,
        markersize=10)

    # Record the animation
    n_frames = length(chain.states)
    record(fig, filename, 1:n_frames; framerate=24) do i
        frame_index[] = i  # Update the frame index observable
        [circle.color = colors[][j] for (j, circle) in enumerate(circles)]
    end

    # Show final figure
    display(fig)
end


# Visualization for BaGoL

"""
Generate super-resolution image from localizations using Gaussian rendering.
"""
function gen_sr_image(
    locs::Vector{<:SMLMData.AbstractEmitter};
    pixel_size::Float64 = 0.010,  # 10 nm pixels
    n_sigma::Float64 = 3.0
)
    if isempty(locs)
        return zeros(1, 1), (0.0, 0.0, 0.0, 0.0)
    end

    xs = [loc.x for loc in locs]
    ys = [loc.y for loc in locs]
    σs = [mean([loc.σ_x, loc.σ_y]) for loc in locs]

    x_min, x_max = extrema(xs)
    y_min, y_max = extrema(ys)

    # Add padding
    pad = maximum(σs) * n_sigma
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad

    nx = ceil(Int, (x_max - x_min) / pixel_size)
    ny = ceil(Int, (y_max - y_min) / pixel_size)

    img = zeros(ny, nx)

    for (loc, σ) in zip(locs, σs)
        # Convert to pixel coordinates
        px = (loc.x - x_min) / pixel_size
        py = (loc.y - y_min) / pixel_size
        pσ = σ / pixel_size

        # Render Gaussian
        r = ceil(Int, n_sigma * pσ)
        px_int = round(Int, px)
        py_int = round(Int, py)

        for dy in -r:r
            for dx in -r:r
                ix = px_int + dx
                iy = py_int + dy
                if 1 <= ix <= nx && 1 <= iy <= ny
                    d2 = ((ix - px)^2 + (iy - py)^2)
                    img[iy, ix] += exp(-d2 / (2 * pσ^2))
                end
            end
        end
    end

    return img, (x_min, x_max, y_min, y_max)
end

"""
Plot localizations with uncertainty circles.
"""
function plot_localizations(
    locs::Vector{<:SMLMData.AbstractEmitter};
    ax = nothing,
    color = :blue,
    alpha = 0.3,
    show_sigma::Bool = true
)
    if ax === nothing
        fig = CairoMakie.Figure(size = (600, 600))
        ax = CairoMakie.Axis(fig[1, 1], aspect = CairoMakie.DataAspect())
    end

    xs = [loc.x for loc in locs]
    ys = [loc.y for loc in locs]

    CairoMakie.scatter!(ax, xs, ys, color = color, markersize = 3)

    if show_sigma
        for loc in locs
            σ = mean([loc.σ_x, loc.σ_y])
            θ = range(0, 2π, length = 50)
            cx = loc.x .+ σ .* cos.(θ)
            cy = loc.y .+ σ .* sin.(θ)
            CairoMakie.lines!(ax, cx, cy, color = (color, alpha), linewidth = 0.5)
        end
    end

    return ax
end

"""
Plot MAP-N results with emitter positions and uncertainties.
"""
function plot_mapn(
    result::MAPNResult,
    locs::Vector{<:SMLMData.AbstractEmitter};
    save_path::Union{String, Nothing} = nothing
)
    fig = CairoMakie.Figure(size = (1200, 500))

    # Left: SR image with emitter circles
    ax1 = CairoMakie.Axis(fig[1, 1], title = "Localizations + Emitters",
                          aspect = CairoMakie.DataAspect())

    # Plot localizations
    xs = [loc.x for loc in locs]
    ys = [loc.y for loc in locs]
    CairoMakie.scatter!(ax1, xs, ys, color = :gray, markersize = 3, alpha = 0.5)

    # Plot emitters
    for (i, ((ex, ey), (σx, σy))) in enumerate(zip(result.emitters, result.uncertainties))
        CairoMakie.scatter!(ax1, [ex], [ey], color = :red, markersize = 10, marker = :star5)

        # Uncertainty circle (mean of x and y uncertainty)
        σ = mean([σx, σy])
        if σ > 0
            θ = range(0, 2π, length = 50)
            cx = ex .+ σ .* cos.(θ)
            cy = ey .+ σ .* sin.(θ)
            CairoMakie.lines!(ax1, cx, cy, color = :red, linewidth = 1.5)
        end
    end

    # Right: Posterior on K
    ax2 = CairoMakie.Axis(fig[1, 2], title = "Posterior P(K)",
                          xlabel = "Number of emitters", ylabel = "Probability")

    k_vals = 0:(length(result.posterior_k) - 1)
    probs = result.posterior_k ./ sum(result.posterior_k)
    CairoMakie.barplot!(ax2, k_vals, probs, color = :steelblue)

    # Mark MAP-N
    CairoMakie.vlines!(ax2, [result.n_emitters], color = :red, linestyle = :dash,
                       label = "MAP-N = $(result.n_emitters)")
    CairoMakie.axislegend(ax2, position = :rt)

    if save_path !== nothing
        CairoMakie.save(save_path, fig)
    end

    return fig
end

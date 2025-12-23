# Visualization for BaGoL

"""
Draw a circle at (x, y) with radius r.
"""
function draw_circle!(ax, x, y, r; color=:black, linewidth=1.0, alpha=1.0)
    θ = range(0, 2π, length=50)
    cx = x .+ r .* cos.(θ)
    cy = y .+ r .* sin.(θ)
    CairoMakie.lines!(ax, cx, cy, color=(color, alpha), linewidth=linewidth)
end

"""
Plot BaGoL results with proper visualization:
- Localizations as 1σ circles (gray)
- Chain emitter positions as scatter (light red)
- MAP-N emitters as 1σ circles (red)
- True positions as X markers (blue) if provided
"""
function plot_bagol(
    chain::RJMCMCChain,
    result::MAPNResult,
    locs::Vector{<:SMLMData.AbstractEmitter};
    true_positions::Vector{Tuple{Float64, Float64}} = Tuple{Float64, Float64}[],
    save_path::Union{String, Nothing} = nothing
)
    fig = CairoMakie.Figure(size=(1200, 500))

    # Left: Main visualization
    ax1 = CairoMakie.Axis(fig[1, 1], title="BaGoL Results",
                          xlabel="x (μm)", ylabel="y (μm)",
                          aspect=CairoMakie.DataAspect())

    # 1. Localizations as 1σ circles (gray)
    for loc in locs
        σ = mean([loc.σ_x, loc.σ_y])
        draw_circle!(ax1, loc.x, loc.y, σ; color=:gray, linewidth=0.5, alpha=0.4)
    end

    # 2. Chain emitter positions as scatter (all samples)
    chain_xs = Float64[]
    chain_ys = Float64[]
    for sample in chain.samples
        for emitter in sample.emitters
            push!(chain_xs, emitter.x)
            push!(chain_ys, emitter.y)
        end
    end
    if !isempty(chain_xs)
        CairoMakie.scatter!(ax1, chain_xs, chain_ys,
            color=(:red, 0.05), markersize=3, label="Chain samples")
    end

    # 3. MAP-N emitters as 1σ circles (red)
    for ((ex, ey), (σx, σy)) in zip(result.emitters, result.uncertainties)
        σ = mean([σx, σy])
        if σ > 0
            draw_circle!(ax1, ex, ey, σ; color=:red, linewidth=2.0)
        end
        CairoMakie.scatter!(ax1, [ex], [ey], color=:red, markersize=8)
    end

    # 4. True positions as X (blue)
    if !isempty(true_positions)
        true_xs = [p[1] for p in true_positions]
        true_ys = [p[2] for p in true_positions]
        CairoMakie.scatter!(ax1, true_xs, true_ys,
            color=:blue, marker=:xcross, markersize=15, strokewidth=3,
            label="True ($(length(true_positions)))")
    end

    # Right: Posterior on K
    ax2 = CairoMakie.Axis(fig[1, 2], title="Posterior P(K)",
                          xlabel="Number of emitters", ylabel="Probability")

    k_vals = 0:(length(result.posterior_k) - 1)
    probs = result.posterior_k ./ sum(result.posterior_k)
    CairoMakie.barplot!(ax2, k_vals, probs, color=:steelblue)

    # Mark true K and MAP-N
    if !isempty(true_positions)
        CairoMakie.vlines!(ax2, [length(true_positions)], color=:blue,
            linestyle=:dash, linewidth=2, label="True K")
    end
    CairoMakie.vlines!(ax2, [result.n_emitters], color=:red,
        linestyle=:solid, linewidth=2, label="MAP-N = $(result.n_emitters)")
    CairoMakie.axislegend(ax2, position=:rt)

    if save_path !== nothing
        CairoMakie.save(save_path, fig)
    end

    return fig
end

"""
Simple plot without chain samples (for quick visualization).
"""
function plot_mapn(
    result::MAPNResult,
    locs::Vector{<:SMLMData.AbstractEmitter};
    true_positions::Vector{Tuple{Float64, Float64}} = Tuple{Float64, Float64}[],
    save_path::Union{String, Nothing} = nothing
)
    fig = CairoMakie.Figure(size=(1200, 500))

    ax1 = CairoMakie.Axis(fig[1, 1], title="Localizations + MAP-N Emitters",
                          xlabel="x (μm)", ylabel="y (μm)",
                          aspect=CairoMakie.DataAspect())

    # Localizations as 1σ circles
    for loc in locs
        σ = mean([loc.σ_x, loc.σ_y])
        draw_circle!(ax1, loc.x, loc.y, σ; color=:gray, linewidth=0.5, alpha=0.4)
    end

    # MAP-N emitters as 1σ circles
    for ((ex, ey), (σx, σy)) in zip(result.emitters, result.uncertainties)
        σ = mean([σx, σy])
        if σ > 0
            draw_circle!(ax1, ex, ey, σ; color=:red, linewidth=2.0)
        end
        CairoMakie.scatter!(ax1, [ex], [ey], color=:red, markersize=8)
    end

    # True positions as X
    if !isempty(true_positions)
        true_xs = [p[1] for p in true_positions]
        true_ys = [p[2] for p in true_positions]
        CairoMakie.scatter!(ax1, true_xs, true_ys,
            color=:blue, marker=:xcross, markersize=15, strokewidth=3)
    end

    # Posterior on K
    ax2 = CairoMakie.Axis(fig[1, 2], title="Posterior P(K)",
                          xlabel="Number of emitters", ylabel="Probability")

    k_vals = 0:(length(result.posterior_k) - 1)
    probs = result.posterior_k ./ sum(result.posterior_k)
    CairoMakie.barplot!(ax2, k_vals, probs, color=:steelblue)

    if !isempty(true_positions)
        CairoMakie.vlines!(ax2, [length(true_positions)], color=:blue,
            linestyle=:dash, linewidth=2, label="True K")
    end
    CairoMakie.vlines!(ax2, [result.n_emitters], color=:red,
        linestyle=:solid, linewidth=2, label="MAP-N")
    CairoMakie.axislegend(ax2, position=:rt)

    if save_path !== nothing
        CairoMakie.save(save_path, fig)
    end

    return fig
end

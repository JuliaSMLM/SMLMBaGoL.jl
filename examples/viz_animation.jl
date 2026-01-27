# BaGoL Chain Animation
# =====================
# Animate RJMCMC chain evolution for diagnostics and visualization.
#
# Usage:
#   include("viz_animation.jl")
#   collector = AnimationCollector()
#   chain = run_bagol_chain(locs; callback=make_animation_callback(collector, 10))
#   animate_chain(collector, locs; filename="chain.mp4")

using CairoMakie
using Statistics
using SMLMData
using SMLMBaGoL: BaGoLState, Emitter

"""
Collects per-iteration data for animation.
"""
mutable struct AnimationCollector
    records::Vector{NamedTuple{(:iter, :K, :move_type, :accepted, :emitter_positions, :μ, :α),
                               Tuple{Int, Int, Symbol, Bool, Vector{Tuple{Float64, Float64}}, Float64, Float64}}}
end

AnimationCollector() = AnimationCollector([])

"""
Create a callback function for run_bagol_chain that collects animation data.

# Arguments
- `collector`: AnimationCollector to store records
- `interval`: Record every N iterations (default 10)
"""
function make_animation_callback(collector::AnimationCollector, interval::Int=10)
    return (iter, move_type, accepted, state, μ, α) -> begin
        if iter % interval == 0
            # Deep copy emitter positions
            positions = [(Float64(e.x), Float64(e.y)) for e in state.emitters]
            push!(collector.records, (
                iter = iter,
                K = length(state.emitters),
                move_type = move_type,
                accepted = accepted,
                emitter_positions = positions,
                μ = μ,
                α = α
            ))
        end
    end
end

"""
Draw a circle at (x, y) with radius r.
"""
function draw_circle_anim!(ax, x, y, r; color=:black, linewidth=1.0, alpha=1.0)
    θ = range(0, 2π, length=50)
    cx = x .+ r .* cos.(θ)
    cy = y .+ r .* sin.(θ)
    lines!(ax, cx, cy, color=(color, alpha), linewidth=linewidth)
end

"""
Animate the RJMCMC chain evolution.

Creates an MP4 showing:
- Left: Emitter positions evolving over time with localizations
- Right top: K (emitter count) over iterations
- Right bottom: Acceptance rate evolution

# Arguments
- `collector`: AnimationCollector with recorded data
- `locs`: Input localizations
- `filename`: Output filename (default "chain_animation.mp4")
- `fps`: Frames per second (default 30)
- `true_positions`: Optional ground truth positions for comparison
"""
function animate_chain(
    collector::AnimationCollector,
    locs::Vector{<:SMLMData.AbstractEmitter};
    filename::String = "chain_animation.mp4",
    fps::Int = 30,
    true_positions::Vector{Tuple{Float64, Float64}} = Tuple{Float64, Float64}[]
)
    records = collector.records

    if isempty(records)
        @warn "No records in collector - animation not created"
        return nothing
    end

    # Compute bounds from localizations
    xs_loc = [loc.x for loc in locs]
    ys_loc = [loc.y for loc in locs]
    margin = 0.05
    x_range = (minimum(xs_loc) - margin, maximum(xs_loc) + margin)
    y_range = (minimum(ys_loc) - margin, maximum(ys_loc) + margin)

    # Median sigma for circle sizes
    median_σ = median([mean([loc.σ_x, loc.σ_y]) for loc in locs])

    # Prepare K evolution data
    iters = [r.iter for r in records]
    ks = [r.K for r in records]

    # Create figure
    fig = Figure(size=(1200, 600))

    # Left panel: spatial view
    ax_spatial = Axis(fig[1:2, 1], title="Emitter Evolution",
                      xlabel="x (μm)", ylabel="y (μm)",
                      aspect=DataAspect())
    xlims!(ax_spatial, x_range)
    ylims!(ax_spatial, y_range)

    # Right top: K trace
    ax_k = Axis(fig[1, 2], title="Emitter Count (K)",
                xlabel="Iteration", ylabel="K")
    xlims!(ax_k, 0, maximum(iters))
    ylims!(ax_k, 0, maximum(ks) + 2)

    # Right bottom: iteration info
    ax_info = Axis(fig[2, 2], title="Iteration Info")
    hidedecorations!(ax_info)
    hidespines!(ax_info)

    # Record animation
    record(fig, filename, eachindex(records); framerate=fps) do frame_idx
        empty!(ax_spatial)
        empty!(ax_k)
        empty!(ax_info)

        r = records[frame_idx]

        # Draw localizations (static background)
        for loc in locs
            σ = mean([loc.σ_x, loc.σ_y])
            draw_circle_anim!(ax_spatial, loc.x, loc.y, σ;
                color=:gray, linewidth=0.3, alpha=0.3)
        end

        # Draw true positions if provided
        if !isempty(true_positions)
            for (tx, ty) in true_positions
                scatter!(ax_spatial, [tx], [ty], marker=:xcross,
                    color=:blue, markersize=12)
            end
        end

        # Draw current emitters
        for (ex, ey) in r.emitter_positions
            scatter!(ax_spatial, [ex], [ey], color=:red, markersize=10)
            draw_circle_anim!(ax_spatial, ex, ey, median_σ;
                color=:red, linewidth=2.0, alpha=0.8)
        end

        # K trace up to current frame
        lines!(ax_k, iters[1:frame_idx], ks[1:frame_idx], color=:steelblue, linewidth=2)
        scatter!(ax_k, [r.iter], [r.K], color=:red, markersize=10)

        # Info panel
        move_color = r.accepted ? :seagreen : :red
        acc_text = r.accepted ? "Accepted" : "Rejected"

        text!(ax_info, 0.5, 0.8, text="Iteration: $(r.iter)",
            align=(:center, :center), fontsize=16)
        text!(ax_info, 0.5, 0.6, text="K = $(r.K) emitters",
            align=(:center, :center), fontsize=14)
        text!(ax_info, 0.5, 0.4, text="Move: $(r.move_type) ($acc_text)",
            align=(:center, :center), fontsize=14, color=move_color)
        text!(ax_info, 0.5, 0.2, text="μ = $(round(r.μ, digits=2)), α = $(round(r.α, digits=2))",
            align=(:center, :center), fontsize=12, color=:gray)
    end

    println("Animation saved: $filename")
    return filename
end

"""
Create a static summary figure from animation data.

Shows snapshots at burn-in, middle, and end of chain.
"""
function plot_chain_snapshots(
    collector::AnimationCollector,
    locs::Vector{<:SMLMData.AbstractEmitter};
    burn_in::Int = 2000,
    save_path::Union{String, Nothing} = nothing
)
    records = collector.records

    if isempty(records)
        @warn "No records in collector"
        return Figure()
    end

    # Find records at key points
    all_iters = [r.iter for r in records]
    max_iter = maximum(all_iters)

    # Find closest records to burn-in, middle, end
    target_iters = [burn_in, div(max_iter, 2), max_iter]
    snapshot_indices = [argmin(abs.(all_iters .- t)) for t in target_iters]

    fig = Figure(size=(1200, 400))

    titles = ["At Burn-in ($(records[snapshot_indices[1]].iter))",
              "Middle ($(records[snapshot_indices[2]].iter))",
              "Final ($(records[snapshot_indices[3]].iter))"]

    median_σ = median([mean([loc.σ_x, loc.σ_y]) for loc in locs])

    for (col, (idx, title)) in enumerate(zip(snapshot_indices, titles))
        ax = Axis(fig[1, col], title=title,
                  xlabel="x (μm)", ylabel="y (μm)",
                  aspect=DataAspect())

        # Draw localizations
        for loc in locs
            σ = mean([loc.σ_x, loc.σ_y])
            draw_circle_anim!(ax, loc.x, loc.y, σ;
                color=:gray, linewidth=0.3, alpha=0.3)
        end

        # Draw emitters at this snapshot
        r = records[idx]
        for (ex, ey) in r.emitter_positions
            scatter!(ax, [ex], [ey], color=:red, markersize=8)
            draw_circle_anim!(ax, ex, ey, median_σ;
                color=:red, linewidth=2.0, alpha=0.8)
        end

        # Add K label
        text!(ax, 0.02, 0.98, text="K = $(r.K)",
            align=(:left, :top), fontsize=12, space=:relative)
    end

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

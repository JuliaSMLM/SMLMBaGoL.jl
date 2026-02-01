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

# Color palette for emitter assignments (up to 12 distinct colors)
const EMITTER_COLORS = [
    colorant"#e41a1c",  # red
    colorant"#377eb8",  # blue
    colorant"#4daf4a",  # green
    colorant"#984ea3",  # purple
    colorant"#ff7f00",  # orange
    colorant"#ffff33",  # yellow
    colorant"#a65628",  # brown
    colorant"#f781bf",  # pink
    colorant"#999999",  # gray
    colorant"#66c2a5",  # teal
    colorant"#fc8d62",  # salmon
    colorant"#8da0cb",  # periwinkle
]

"""
Animation record storing state at one iteration.
"""
struct AnimationRecord
    iter::Int
    K::Int
    move_type::Symbol
    accepted::Bool
    emitter_positions::Vector{Tuple{Float64, Float64}}
    allocations::Vector{Vector{Int}}  # allocations[i] = loc indices for emitter i
    μ::Float64
    shape::Float64
end

"""
Collects per-iteration data for animation.
"""
mutable struct AnimationCollector
    records::Vector{AnimationRecord}
end

AnimationCollector() = AnimationCollector(AnimationRecord[])

"""
Create a callback function for run_bagol_chain that collects animation data.

# Arguments
- `collector`: AnimationCollector to store records
- `interval`: Record every N iterations (default 10)
"""
function make_animation_callback(collector::AnimationCollector, interval::Int=10)
    return (iter, move_type, accepted, state, μ, shape) -> begin
        if iter % interval == 0
            # Deep copy emitter positions and allocations
            positions = [(Float64(e.x), Float64(e.y)) for e in state.emitters]
            allocations = [copy(e.allocated) for e in state.emitters]
            push!(collector.records, AnimationRecord(
                iter,
                length(state.emitters),
                move_type,
                accepted,
                positions,
                allocations,
                μ,
                shape
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
Get color for emitter index (cycles through palette).
"""
function emitter_color(idx::Int)
    return EMITTER_COLORS[mod1(idx, length(EMITTER_COLORS))]
end

"""
Build a mapping from localization index to emitter index.
Returns a vector where result[loc_idx] = emitter_idx (0 if unassigned).
"""
function build_loc_to_emitter_map(allocations::Vector{Vector{Int}}, n_locs::Int)
    loc_to_emitter = zeros(Int, n_locs)
    for (emitter_idx, loc_indices) in enumerate(allocations)
        for loc_idx in loc_indices
            if 1 <= loc_idx <= n_locs
                loc_to_emitter[loc_idx] = emitter_idx
            end
        end
    end
    return loc_to_emitter
end

"""
Animate the RJMCMC chain evolution.

Creates an MP4 showing:
- Left: Emitter positions and localizations colored by assignment
- Right top: K (emitter count) over iterations
- Right bottom: Iteration info

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

    n_locs = length(locs)

    # Compute bounds from localizations
    xs_loc = [loc.x for loc in locs]
    ys_loc = [loc.y for loc in locs]
    margin = 0.05
    x_range = (minimum(xs_loc) - margin, maximum(xs_loc) + margin)
    y_range = (minimum(ys_loc) - margin, maximum(ys_loc) + margin)

    # Median sigma for emitter circle sizes
    median_σ = median([mean([loc.σ_x, loc.σ_y]) for loc in locs])

    # Prepare K evolution data
    iters = [r.iter for r in records]
    ks = [r.K for r in records]

    # Create figure
    fig = Figure(size=(1200, 600))

    # Left panel: spatial view
    ax_spatial = Axis(fig[1:2, 1], title="Emitter Evolution",
                      xlabel="x (μm)", ylabel="y (μm)",
                      aspect=DataAspect(), yreversed=true)
    xlims!(ax_spatial, x_range)
    ylims!(ax_spatial, y_range[2], y_range[1])  # reversed for microscopy convention

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

        # Build localization -> emitter mapping for this frame
        loc_to_emitter = build_loc_to_emitter_map(r.allocations, n_locs)

        # Draw localizations colored by emitter assignment
        for (loc_idx, loc) in enumerate(locs)
            σ = mean([loc.σ_x, loc.σ_y])
            emitter_idx = loc_to_emitter[loc_idx]

            if emitter_idx > 0
                # Assigned to an emitter - use emitter's color
                color = emitter_color(emitter_idx)
                draw_circle_anim!(ax_spatial, loc.x, loc.y, σ;
                    color=color, linewidth=1.0, alpha=0.7)
            else
                # Unassigned - gray
                draw_circle_anim!(ax_spatial, loc.x, loc.y, σ;
                    color=:gray, linewidth=0.3, alpha=0.3)
            end
        end

        # Draw true positions if provided
        if !isempty(true_positions)
            for (tx, ty) in true_positions
                scatter!(ax_spatial, [tx], [ty], marker=:xcross,
                    color=:black, markersize=12, strokewidth=2)
            end
        end

        # Draw current emitters (larger circles with matching colors)
        for (emitter_idx, (ex, ey)) in enumerate(r.emitter_positions)
            color = emitter_color(emitter_idx)
            scatter!(ax_spatial, [ex], [ey], color=color, markersize=12,
                strokecolor=:black, strokewidth=1)
            draw_circle_anim!(ax_spatial, ex, ey, median_σ;
                color=color, linewidth=2.5, alpha=0.9)
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
        text!(ax_info, 0.5, 0.2, text="μ = $(round(r.μ, digits=2)), shape = $(round(r.shape, digits=2))",
            align=(:center, :center), fontsize=12, color=:gray)
    end

    println("Animation saved: $filename")
    return filename
end

"""
Create a static summary figure from animation data.

Shows snapshots at burn-in, middle, and end of chain with colored allocations.
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

    n_locs = length(locs)

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
                  aspect=DataAspect(), yreversed=true)

        r = records[idx]
        loc_to_emitter = build_loc_to_emitter_map(r.allocations, n_locs)

        # Draw localizations colored by assignment
        for (loc_idx, loc) in enumerate(locs)
            σ = mean([loc.σ_x, loc.σ_y])
            emitter_idx = loc_to_emitter[loc_idx]

            if emitter_idx > 0
                color = emitter_color(emitter_idx)
                draw_circle_anim!(ax, loc.x, loc.y, σ;
                    color=color, linewidth=1.0, alpha=0.7)
            else
                draw_circle_anim!(ax, loc.x, loc.y, σ;
                    color=:gray, linewidth=0.3, alpha=0.3)
            end
        end

        # Draw emitters at this snapshot
        for (emitter_idx, (ex, ey)) in enumerate(r.emitter_positions)
            color = emitter_color(emitter_idx)
            scatter!(ax, [ex], [ey], color=color, markersize=10,
                strokecolor=:black, strokewidth=1)
            draw_circle_anim!(ax, ex, ey, median_σ;
                color=color, linewidth=2.5, alpha=0.9)
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

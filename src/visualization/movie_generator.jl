"""
    generate_chain_movie(chain::RJMCMCChain, output_path::String; kwargs...)

Generate an MP4 movie showing the evolution of an MCMC chain.

Visualizes:
- Localizations as colored circles (color indicates allocated emitter)
- Emitters as colored X markers
- Metrics display showing iteration, K, and log-likelihood

# Arguments
- `chain::RJMCMCChain`: The MCMC chain to visualize
- `output_path::String`: Path for output MP4 file (default: "examples/output/chain_evolution.mp4")

# Keyword Arguments
- `fps::Int=10`: Frames per second for the output video
- `figsize=(800, 600)`: Figure size in pixels
- `xlims=nothing`: X-axis limits (auto-detected if nothing)
- `ylims=nothing`: Y-axis limits (auto-detected if nothing)
- `color_palette=:Set1_9`: ColorScheme for emitter colors
- `localization_alpha=0.5`: Transparency for localization circles
- `localization_radius_scale=1.0`: Scale factor for localization uncertainty circles
- `emitter_markersize=15`: Size of emitter X markers
- `show_metrics=true`: Whether to display metrics overlay
- `unallocated_color=:gray`: Color for unallocated localizations
- `sample_range=nothing`: Range of samples to animate (default: all samples)

# Example
```julia
# Basic usage
generate_chain_movie(chain, "my_analysis.mp4")

# Custom settings
generate_chain_movie(chain, "detailed.mp4", fps=30, color_palette=:Dark2_8)
```
"""
function generate_chain_movie(
    chain::RJMCMCChain,
    output_path::String = "examples/output/chain_evolution.mp4";
    fps::Int = 10,
    figsize = (800, 600),
    xlims = nothing,
    ylims = nothing,
    color_palette = :Set1_9,
    localization_alpha = 0.5,
    localization_radius_scale = 1.0,
    emitter_markersize = 15,
    show_metrics = true,
    unallocated_color = :gray,
    sample_range = nothing
)
    # Ensure output directory exists
    output_dir = dirname(output_path)
    if !isempty(output_dir) && !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Determine which samples to use
    samples_to_use = isnothing(sample_range) ? chain.samples : chain.samples[sample_range]
    n_frames = length(samples_to_use)
    
    if n_frames == 0
        error("No samples to animate. Check that the chain has been run and samples collected.")
    end
    
    # Setup figure
    fig = CairoMakie.Figure(size = figsize)
    ax = CairoMakie.Axis(fig[1:3, 1:3], aspect = CairoMakie.DataAspect())
    
    # Determine coordinate bounds if not specified
    if isnothing(xlims) || isnothing(ylims)
        computed_xlims, computed_ylims = compute_bounds(samples_to_use, chain.localizations)
        xlims = isnothing(xlims) ? computed_xlims : xlims
        ylims = isnothing(ylims) ? computed_ylims : ylims
    end
    
    CairoMakie.xlims!(ax, xlims)
    CairoMakie.ylims!(ax, ylims)
    
    # Get color palette
    colors = ColorSchemes.colorschemes[color_palette].colors
    n_colors = length(colors)
    
    # Initialize color tracking with position history
    emitter_history = Vector{Tuple{Float64, Float64}}()  # Store previous positions
    color_history = Vector{Int}()  # Store corresponding colors
    next_color_idx = [1]
    
    # Add metrics display if requested
    if show_metrics
        metrics_text = CairoMakie.Label(fig[0, 1:3], "", fontsize = 14, halign = :center)
    end
    
    # Record animation
    CairoMakie.record(fig, output_path, 1:n_frames; framerate = fps) do frame_idx
        # Clear axis
        CairoMakie.empty!(ax)
        
        # Get current state
        state = samples_to_use[frame_idx]
        
        # Update color assignments for emitters
        emitter_colors = assign_emitter_colors_stable(
            state.emitters, colors, emitter_history, color_history, next_color_idx, n_colors
        )
        
        # Plot localizations with colors based on allocation
        plot_colored_localizations!(
            ax, state.localizations, state.allocations, 
            emitter_colors, unallocated_color, 
            localization_alpha, localization_radius_scale
        )
        
        # Plot emitters as X markers
        plot_emitters!(ax, state.emitters, emitter_colors, emitter_markersize)
        
        # Update metrics display
        if show_metrics
            update_metrics_display!(
                metrics_text, state, frame_idx, n_frames, 
                isnothing(sample_range) ? frame_idx : sample_range[frame_idx]
            )
        end
    end
    
    return nothing
end

"""
    assign_emitter_colors_stable(emitters, colors, emitter_history, color_history, next_color_idx, n_colors)

Assign consistent colors to emitters using spatial matching with previous frames.
Maintains stable colors across birth/death events by matching emitters to their closest
previous positions within a tolerance.
"""
function assign_emitter_colors_stable(emitters, colors, emitter_history, color_history, next_color_idx, n_colors)
    emitter_colors = Dict{Int, Any}()
    used_colors = Set{Int}()
    matched_history_indices = Set{Int}()
    
    # Match current emitters to previous emitters by proximity
    for (idx, emitter) in enumerate(emitters)
        best_match_idx = nothing
        best_distance = Inf
        
        # Find closest emitter from previous frame within tolerance
        for (hist_idx, prev_pos) in enumerate(emitter_history)
            if hist_idx in matched_history_indices
                continue  # Already matched
            end
            
            distance = sqrt((emitter.x - prev_pos[1])^2 + (emitter.y - prev_pos[2])^2)
            if distance < 0.05 && distance < best_distance  # 50nm tolerance for matching
                best_distance = distance
                best_match_idx = hist_idx
            end
        end
        
        if best_match_idx !== nothing
            # Reuse color from matched previous emitter
            color_idx = color_history[best_match_idx]
            emitter_colors[idx] = colors[color_idx]
            used_colors = union(used_colors, [color_idx])
            matched_history_indices = union(matched_history_indices, [best_match_idx])
        else
            # Assign new color for new emitter
            color_idx = find_next_available_color(used_colors, n_colors, next_color_idx[1])
            emitter_colors[idx] = colors[color_idx]
            used_colors = union(used_colors, [color_idx])
            
            # Update next color index
            next_color_idx[1] = mod1(color_idx + 1, n_colors)
        end
    end
    
    # Update history with current frame
    empty!(emitter_history)
    empty!(color_history)
    for (idx, emitter) in enumerate(emitters)
        push!(emitter_history, (emitter.x, emitter.y))
        # Find the color index from emitter_colors
        current_color = emitter_colors[idx]
        color_idx = findfirst(c -> c == current_color, colors)
        push!(color_history, color_idx)
    end
    
    return emitter_colors
end

"""
    find_next_available_color(used_colors, n_colors, start_idx)

Find the next available color index that hasn't been used in this frame.
"""
function find_next_available_color(used_colors, n_colors, start_idx)
    for offset in 0:(n_colors-1)
        candidate = mod1(start_idx + offset, n_colors)
        if !(candidate in used_colors)
            return candidate
        end
    end
    # Fallback: reuse colors if all are taken
    return start_idx
end

"""
    plot_colored_localizations!(ax, localizations, allocations, emitter_colors, unallocated_color, alpha, radius_scale)

Plot localizations as circles colored by their allocated emitter.
"""
function plot_colored_localizations!(
    ax, localizations, allocations, emitter_colors, unallocated_color, alpha, radius_scale
)
    # Group localizations by allocation for efficient plotting
    allocation_groups = Dict{Int, Vector{Int}}()
    
    for (i, alloc) in enumerate(allocations)
        if !haskey(allocation_groups, alloc)
            allocation_groups[alloc] = Int[]
        end
        push!(allocation_groups[alloc], i)
    end
    
    # Plot each group with its color
    for (alloc, indices) in allocation_groups
        color = alloc > 0 ? emitter_colors[alloc] : unallocated_color
        
        for idx in indices
            loc = localizations[idx]
            plot_localization_circle!(ax, loc, color, alpha, radius_scale)
        end
    end
end

"""
    plot_localization_circle!(ax, localization, color, alpha, radius_scale)

Plot a single localization as a circle with uncertainty radius.
"""
function plot_localization_circle!(ax, localization, color, alpha, radius_scale)
    # Generate circle points
    θ = range(0, 2π, length=50)
    # Use average of σx and σy for circular approximation
    radius = radius_scale * (localization.σx + localization.σy) / 2
    x_circle = localization.x .+ radius .* cos.(θ)
    y_circle = localization.y .+ radius .* sin.(θ)
    
    # Plot circle
    CairoMakie.lines!(ax, x_circle, y_circle, color = (color, alpha), linewidth = 1.5)
end

"""
    plot_emitters!(ax, emitters, emitter_colors, markersize)

Plot emitters as X markers with their assigned colors.
"""
function plot_emitters!(ax, emitters, emitter_colors, markersize)
    for (idx, emitter) in enumerate(emitters)
        CairoMakie.scatter!(
            ax, [emitter.x], [emitter.y], 
            marker = :x, 
            color = emitter_colors[idx], 
            markersize = markersize,
            strokewidth = 2
        )
    end
end

"""
    update_metrics_display!(text_label, state, frame_idx, n_frames, iteration)

Update the metrics display with current state information.
"""
function update_metrics_display!(text_label, state, frame_idx, n_frames, iteration)
    K = length(state.emitters)
    log_lik = state.log_likelihood
    
    metrics_str = "Frame: $frame_idx/$n_frames | Iteration: $iteration | K: $K | Log-likelihood: $(round(log_lik, digits=2))"
    
    text_label.text = metrics_str
end

"""
    compute_bounds(samples, localizations)

Compute appropriate axis bounds from samples and localizations.
"""
function compute_bounds(samples, localizations)
    # Get bounds from localizations
    loc_xs = [loc.x for loc in localizations]
    loc_ys = [loc.y for loc in localizations]
    
    # Get bounds from all emitters across samples
    emitter_xs = Float64[]
    emitter_ys = Float64[]
    
    for state in samples
        for emitter in state.emitters
            push!(emitter_xs, emitter.x)
            push!(emitter_ys, emitter.y)
        end
    end
    
    # Combine and add padding
    all_xs = vcat(loc_xs, emitter_xs)
    all_ys = vcat(loc_ys, emitter_ys)
    
    x_min, x_max = extrema(all_xs)
    y_min, y_max = extrema(all_ys)
    
    x_padding = 0.1 * (x_max - x_min)
    y_padding = 0.1 * (y_max - y_min)
    
    xlims = (x_min - x_padding, x_max + x_padding)
    ylims = (y_min - y_padding, y_max + y_padding)
    
    return xlims, ylims
end
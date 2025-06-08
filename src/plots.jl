"""
    plots.jl

Interactive plotting utilities for SMLMBaGoL using CairoMakie.
Provides functions for visualizing SMLM localizations and BaGoL results.
"""

using CairoMakie

"""
    circles!(ax, centers, radii; kwargs...)

Draw circles on a Makie axis using lines!() internally.

# Arguments
- `ax`: Makie axis
- `centers`: Vector of (x,y) tuples or 2-column matrix of circle centers
- `radii`: Vector of radii or single radius for all circles
- `kwargs...`: Additional styling arguments passed to lines!()

# Example
```julia
fig = Figure()
ax = Axis(fig[1,1])
centers = [(0.0, 0.0), (1.0, 1.0)]
radii = [0.5, 0.3]
circles!(ax, centers, radii; color=:red, linewidth=2)
```
"""
function circles!(ax, centers, radii; kwargs...)
    # Convert centers to consistent format
    if centers isa Matrix
        centers = [(centers[i,1], centers[i,2]) for i in 1:size(centers,1)]
    end
    
    # Handle single radius case
    if radii isa Number
        radii = fill(radii, length(centers))
    end
    
    # Draw each circle (unfilled outlines only)
    for (center, radius) in zip(centers, radii)
        θ = range(0, 2π, length=50)
        x_circle = center[1] .+ radius .* cos.(θ)
        y_circle = center[2] .+ radius .* sin.(θ)
        # Close the circle by connecting back to start
        x_circle = vcat(x_circle, x_circle[1])
        y_circle = vcat(y_circle, y_circle[1])
        lines!(ax, x_circle, y_circle; kwargs...)
    end
end

"""
    plot_circles(localizations; mapn_results=nothing, true_values=nothing, 
                 localization_radius=0.01, mapn_radius=0.05, true_radius=0.03,
                 figsize=(800, 600))

Create a circle plot showing SMLM localizations and optionally MAP-N results and true emitter positions.

# Arguments
- `localizations`: Vector of emitters (observations/localizations)
- `mapn_results`: Optional vector of MAP-N emitters 
- `true_values`: Optional vector of true emitter positions
- `localization_radius`: Radius for localization circles (default: 0.01 μm = 10 nm)
- `mapn_radius`: Radius for MAP-N emitter circles (default: 0.05 μm = 50 nm)  
- `true_radius`: Radius for true emitter circles (default: 0.03 μm = 30 nm)
- `figsize`: Figure size as (width, height) tuple

# Returns
- `fig`: Makie Figure object

# Example
```julia
# Basic usage with just localizations
fig = plot_circles(smld_noisy.emitters)

# Full comparison plot
fig = plot_circles(smld_noisy.emitters; 
                   mapn_results=result.mapn_emitters,
                   true_values=smld_true.emitters)
```
"""
function plot_circles(localizations; mapn_results=nothing, true_values=nothing,
                     localization_radius=0.01, mapn_radius=0.05, true_radius=0.03,
                     figsize=(800, 600))
    
    fig = Figure(size=figsize)
    ax = CairoMakie.Axis(fig[1,1], 
              xlabel="X position (μm)", 
              ylabel="Y position (μm)",
              title="SMLM Localization Results",
              aspect=DataAspect())
    
    # Plot localizations (smallest circles, light gray)
    if !isempty(localizations)
        loc_centers = [(e.x, e.y) for e in localizations]
        circles!(ax, loc_centers, localization_radius; 
                color=:lightgray, linewidth=1)
    end
    
    # Plot true values (medium circles, green)
    if !isnothing(true_values) && !isempty(true_values)
        true_centers = [(e.x, e.y) for e in true_values]
        circles!(ax, true_centers, true_radius; 
                color=:green, linewidth=2)
    end
    
    # Plot MAP-N results (larger circles, red)
    if !isnothing(mapn_results) && !isempty(mapn_results)
        mapn_centers = [(e.x, e.y) for e in mapn_results]
        circles!(ax, mapn_centers, mapn_radius; 
                color=:red, linewidth=3)
    end
    
    # Add legend
    legend_elements = []
    legend_labels = []
    
    if !isempty(localizations)
        push!(legend_elements, CairoMakie.LineElement(color=:lightgray, linewidth=1))
        push!(legend_labels, "Localizations ($(length(localizations)))")
    end
    if !isnothing(true_values) && !isempty(true_values)
        push!(legend_elements, CairoMakie.LineElement(color=:green, linewidth=2))
        push!(legend_labels, "True emitters ($(length(true_values)))")
    end
    if !isnothing(mapn_results) && !isempty(mapn_results)
        push!(legend_elements, CairoMakie.LineElement(color=:red, linewidth=3))
        push!(legend_labels, "MAP-N emitters ($(length(mapn_results)))")
    end
    
    if !isempty(legend_elements)
        CairoMakie.Legend(fig[1,2], legend_elements, legend_labels)
    end
    
    return fig
end
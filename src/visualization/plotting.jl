# CairoMakie plotting utilities for SMLMBaGoL
# Provides circle plotting primitives and localization visualization

"""
    circle!(x, y, radius; n_points=100, kwargs...)

Plot a circle centered at (x, y) with given radius using lines!

This function creates a circle by generating points along the parametric circle
equation and connecting them with lines!. All keyword arguments are passed 
through to lines! for styling.

# Arguments
- `x::Real`: x-coordinate of circle center
- `y::Real`: y-coordinate of circle center  
- `radius::Real`: radius of the circle
- `n_points::Int = 100`: number of points to generate (higher = smoother)
- `kwargs...`: additional keyword arguments passed to lines! (color, linewidth, etc.)

# Examples
```julia
using CairoMakie
fig = Figure()
ax = Axis(fig[1,1])

# Basic circle
circle!(ax, 0.0, 0.0, 1.0)

# Styled circle
circle!(ax, 2.0, 2.0, 0.5, color=:red, linewidth=2)

# Multiple circles with different styles
circle!(ax, -1.0, 1.0, 0.3, color=:blue, linestyle=:dash)
```
"""
function circle!(x::Real, y::Real, radius::Real; n_points::Int=100, kwargs...)
    # Generate parametric circle points
    θ = range(0, 2π, length=n_points+1)  # +1 to close the circle
    circle_x = x .+ radius .* cos.(θ)
    circle_y = y .+ radius .* sin.(θ)
    
    # Plot using lines!
    lines!(circle_x, circle_y; kwargs...)
end

"""
    circle!(ax, x, y, radius; kwargs...)

Plot a circle on the specified axis.

# Arguments
- `ax`: The axis to plot on
- `x::Real`: x-coordinate of circle center
- `y::Real`: y-coordinate of circle center
- `radius::Real`: radius of the circle
- `kwargs...`: additional keyword arguments passed to lines!
"""
function circle!(ax, x::Real, y::Real, radius::Real; kwargs...)
    # Generate parametric circle points  
    θ = range(0, 2π, length=101)  # 100 segments + 1 to close
    circle_x = x .+ radius .* cos.(θ)
    circle_y = y .+ radius .* sin.(θ)
    
    # Plot using lines!
    lines!(ax, circle_x, circle_y; kwargs...)
end

"""
    sr_circles!(ax, localizations; uncertainty_field=:σx, scale_factor=1.0, kwargs...)

Plot uncertainty circles for localizations on the given axis.

This function plots circles centered at each localization position with radius
proportional to the localization uncertainty. The axis is configured to match
SMLMData coordinate conventions (origin at top-left, equal aspect ratio).

# Arguments
- `ax`: The axis to plot on
- `localizations`: Vector of localization objects with x, y, and uncertainty fields
- `uncertainty_field::Symbol = :σx`: Field name for uncertainty (σx, σy, etc.)
- `scale_factor::Real = 1.0`: Scale factor for circle radii (1.0 = 1σ, 2.0 = 2σ, etc.)
- `kwargs...`: additional styling arguments passed to circle! (color, linewidth, etc.)

# Examples
```julia
# Plot 1σ uncertainty circles
sr_circles!(ax, localizations)

# Plot 2σ uncertainty circles in red
sr_circles!(ax, localizations, scale_factor=2.0, color=:red)

# Use y-uncertainty instead of x-uncertainty
sr_circles!(ax, localizations, uncertainty_field=:σy, color=:blue)
```
"""
function sr_circles!(ax, localizations; uncertainty_field::Symbol=:σx, scale_factor::Real=1.0, kwargs...)
    for loc in localizations
        # Get position and uncertainty
        x_pos = loc.x
        y_pos = loc.y
        uncertainty = getfield(loc, uncertainty_field)
        
        # Plot circle with scaled uncertainty as radius
        circle!(ax, x_pos, y_pos, uncertainty * scale_factor; kwargs...)
    end
end

"""
    sr_circles(localizations; camera=nothing, uncertainty_field=:σx, scale_factor=1.0, kwargs...)

Create a new figure and plot uncertainty circles for localizations.

This function creates a new figure with proper axis configuration following
SMLMData coordinate conventions:
- Origin (0,0) at top-left corner  
- Y-axis flipped (increasing downward)
- Equal aspect ratio to prevent circle distortion
- Axis limits set from camera bounds or auto-calculated from data

# Arguments  
- `localizations`: Vector of localization objects with x, y, and uncertainty fields
- `camera`: Camera object to define axis bounds (default: nothing, auto-calculate from data)  
- `uncertainty_field::Symbol = :σx`: Field name for uncertainty (σx, σy, etc.)
- `scale_factor::Real = 1.0`: Scale factor for circle radii
- `figure_kwargs...`: additional arguments for Figure creation
- `axis_kwargs...`: additional arguments for Axis creation
- `plot_kwargs...`: styling arguments passed to circle plotting

# Returns
- `(fig, ax)`: Tuple of Figure and Axis objects

# Examples
```julia
# Basic uncertainty visualization (auto-calculated bounds)
fig, ax = sr_circles(localizations)

# Use camera bounds for consistent limits with sr_image
fig, ax = sr_circles(localizations, camera=camera)

# Customized plot with camera bounds
fig, ax = sr_circles(localizations, 
                    camera=camera,
                    scale_factor=2.0,
                    color=:red, 
                    linewidth=2,
                    figure=(size=(800, 600),),
                    axis=(title="Localization Uncertainties",))
```
"""
function sr_circles(localizations; 
                   camera=nothing,
                   uncertainty_field::Symbol=:σx, 
                   scale_factor::Real=1.0,
                   figure_kwargs=(;),
                   axis_kwargs=(;),
                   kwargs...)
    
    # Create figure
    fig = Figure(; figure_kwargs...)
    
    # Calculate axis bounds
    if camera !== nothing
        # Use camera bounds for consistent limits with sr_image
        x_edges = camera.pixel_edges_x
        y_edges = camera.pixel_edges_y
        x_min, x_max = extrema(x_edges)
        y_min, y_max = extrema(y_edges)
    else
        # Auto-calculate bounds from data
        if isempty(localizations)
            x_min, x_max = -1.0, 1.0
            y_min, y_max = -1.0, 1.0
        else
            x_coords = [loc.x for loc in localizations]
            y_coords = [loc.y for loc in localizations]
            
            # Get uncertainty values for margin calculation
            uncertainties = [getfield(loc, uncertainty_field) for loc in localizations]
            max_uncertainty = isempty(uncertainties) ? 0.1 : maximum(uncertainties)
            margin = max_uncertainty * scale_factor * 3  # 3x largest circle radius
            
            x_min, x_max = extrema(x_coords) .+ (-margin, margin)
            y_min, y_max = extrema(y_coords) .+ (-margin, margin)
        end
    end
    
    # Create axis with SMLMData coordinate conventions
    ax = Axis(fig[1, 1];
              limits=(x_min, x_max, y_min, y_max),
              yreversed=true,                    # SMLMData: (0,0) at top-left
              aspect=DataAspect(),               # Equal aspect ratio
              xlabel="x (μm)",
              ylabel="y (μm)",
              axis_kwargs...)
    
    # Plot circles
    sr_circles!(ax, localizations; uncertainty_field=uncertainty_field, 
               scale_factor=scale_factor, kwargs...)
    
    return fig, ax
end
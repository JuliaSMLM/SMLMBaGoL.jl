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
    CairoMakie.lines!(circle_x, circle_y; kwargs...)
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
    CairoMakie.lines!(ax, circle_x, circle_y; kwargs...)
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
        
        # Handle different types with different field names
        uncertainty = if isa(loc, Emitter2DFit)
            # Emitter2DFit uses σ_x, σ_y (with underscore)
            field_name = uncertainty_field == :σx ? :σ_x : :σ_y
            getfield(loc, field_name)
        else
            # Localization2D uses σx, σy (without underscore)
            getfield(loc, uncertainty_field)
        end
        
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
    fig = CairoMakie.Figure(; figure_kwargs...)
    
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
    ax = CairoMakie.Axis(fig[1, 1];
              limits=(x_min, x_max, y_min, y_max),
              yreversed=true,                    # SMLMData: (0,0) at top-left
              aspect=CairoMakie.DataAspect(),               # Equal aspect ratio
              xlabel="x (μm)",
              ylabel="y (μm)",
              axis_kwargs...)
    
    # Plot circles
    sr_circles!(ax, localizations; uncertainty_field=uncertainty_field, 
               scale_factor=scale_factor, kwargs...)
    
    return fig, ax
end

"""
    sr_circles_combined!(ax, localizations, mapn_results; 
                        loc_uncertainty_field=:σx, mapn_uncertainty_field=:σx,
                        loc_scale_factor=2.0, mapn_scale_factor=2.0,
                        loc_color=:black, loc_alpha=0.4, loc_linewidth=1,
                        mapn_color=:red, mapn_alpha=0.8, mapn_linewidth=2,
                        kwargs...)

Plot uncertainty circles for both localizations and MAPN results on the same axis.

This function creates a comparison visualization showing raw localizations 
(typically with more uncertainty) and refined MAPN emitter estimates on the
same plot with different styling to distinguish between them.

# Arguments
- `ax`: The axis to plot on
- `localizations`: Vector of localization objects (can be empty)
- `mapn_results`: Vector of MAPN emitter objects (can be empty)

# Localization styling (defaults)
- `loc_uncertainty_field::Symbol = :σx`: Uncertainty field for localizations
- `loc_scale_factor::Real = 2.0`: Scale factor for localization circles (2σ)
- `loc_color = :black`: Color for localization circles
- `loc_alpha::Real = 0.4`: Transparency for localization circles
- `loc_linewidth::Real = 1`: Line width for localization circles

# MAPN styling (defaults)  
- `mapn_uncertainty_field::Symbol = :σx`: Uncertainty field for MAPN results
- `mapn_scale_factor::Real = 2.0`: Scale factor for MAPN circles (2σ)
- `mapn_color = :red`: Color for MAPN circles
- `mapn_alpha::Real = 0.8`: Transparency for MAPN circles  
- `mapn_linewidth::Real = 2`: Line width for MAPN circles

# Examples
```julia
# Basic comparison plot
sr_circles_combined!(ax, localizations, mapn_results)

# Custom styling
sr_circles_combined!(ax, localizations, mapn_results,
                    loc_color=:gray, loc_alpha=0.3,
                    mapn_color=:blue, mapn_linewidth=3)

# Different scale factors
sr_circles_combined!(ax, localizations, mapn_results,
                    loc_scale_factor=1.0, mapn_scale_factor=3.0)
```
"""
function sr_circles_combined!(ax, localizations, mapn_results;
                             # Localization styling
                             loc_uncertainty_field::Symbol=:σx,
                             loc_scale_factor::Real=2.0,
                             loc_color=:black,
                             loc_alpha::Real=0.4,
                             loc_linewidth::Real=1,
                             # MAPN styling  
                             mapn_uncertainty_field::Symbol=:σx,
                             mapn_scale_factor::Real=2.0,
                             mapn_color=:red,
                             mapn_alpha::Real=0.8,
                             mapn_linewidth::Real=2,
                             # Additional kwargs passed to both
                             kwargs...)
    
    # Plot localizations first (background layer)
    if !isempty(localizations)
        sr_circles!(ax, localizations; 
                   uncertainty_field=loc_uncertainty_field,
                   scale_factor=loc_scale_factor,
                   color=loc_color,
                   alpha=loc_alpha,
                   linewidth=loc_linewidth,
                   kwargs...)
    end
    
    # Plot MAPN results on top (foreground layer)
    if !isempty(mapn_results)
        sr_circles!(ax, mapn_results;
                   uncertainty_field=mapn_uncertainty_field,
                   scale_factor=mapn_scale_factor,
                   color=mapn_color,
                   alpha=mapn_alpha,
                   linewidth=mapn_linewidth,
                   kwargs...)
    end
end

"""
    sr_circles_combined(localizations, mapn_results; camera=nothing,
                       figure_kwargs=(;), axis_kwargs=(;), kwargs...)

Create a new figure and plot uncertainty circles for both localizations and MAPN results.

This function creates a comparison visualization showing raw localizations and refined 
MAPN emitter estimates with appropriate styling defaults. The axis is configured 
following SMLMData coordinate conventions with optional camera bounds.

# Arguments
- `localizations`: Vector of localization objects (can be empty)
- `mapn_results`: Vector of MAPN emitter objects (can be empty)
- `camera`: Camera object to define axis bounds (default: auto-calculate from data)
- `figure_kwargs`: Keyword arguments for Figure creation
- `axis_kwargs`: Keyword arguments for Axis creation
- `kwargs...`: Additional arguments passed to sr_circles_combined!

# Returns
- `(fig, ax)`: Tuple of Figure and Axis objects

# Examples
```julia
# Basic comparison with auto-calculated bounds
fig, ax = sr_circles_combined(localizations, mapn_results)

# Use camera bounds for consistent field-of-view
fig, ax = sr_circles_combined(localizations, mapn_results, camera=camera)

# Customized figure and styling
fig, ax = sr_circles_combined(localizations, mapn_results,
                             figure_kwargs=(size=(1000, 800),),
                             axis_kwargs=(title="Localization vs MAPN Comparison",),
                             loc_alpha=0.2, mapn_color=:blue)
```
"""
function sr_circles_combined(localizations, mapn_results;
                            camera=nothing,
                            figure_kwargs=(;),
                            axis_kwargs=(;),
                            kwargs...)
    
    # Create figure
    fig = CairoMakie.Figure(; figure_kwargs...)
    
    # Calculate axis bounds - use combined data for bounds calculation
    if camera !== nothing
        # Use camera bounds for consistent limits
        x_edges = camera.pixel_edges_x
        y_edges = camera.pixel_edges_y
        x_min, x_max = extrema(x_edges)
        y_min, y_max = extrema(y_edges)
    else
        # Auto-calculate bounds from combined data
        # Extract coordinates from both datasets separately
        x_coords, y_coords = Float64[], Float64[]
        
        if !isempty(localizations)
            locs_x, locs_y = extract_coordinates(localizations)
            append!(x_coords, locs_x)
            append!(y_coords, locs_y)
        end
        
        if !isempty(mapn_results)
            mapn_x, mapn_y = extract_coordinates(mapn_results)
            append!(x_coords, mapn_x)
            append!(y_coords, mapn_y)
        end
        
        if isempty(x_coords) || isempty(y_coords)
            x_min, x_max = -1.0, 1.0
            y_min, y_max = -1.0, 1.0
        else
            # Calculate margins from both datasets
            loc_margin = isempty(localizations) ? 0.0 : get_data_margin(localizations)
            mapn_margin = isempty(mapn_results) ? 0.0 : get_data_margin(mapn_results)
            margin = max(loc_margin, mapn_margin, 0.1)  # At least 100nm margin
            
            # Calculate bounds with margin
            x_min, x_max = extrema(x_coords) .+ (-margin, margin)
            y_min, y_max = extrema(y_coords) .+ (-margin, margin)
        end
    end
    
    # Create axis with SMLMData coordinate conventions
    ax = CairoMakie.Axis(fig[1, 1];
              limits=(x_min, x_max, y_min, y_max),
              yreversed=true,                    # SMLMData: (0,0) at top-left
              aspect=CairoMakie.DataAspect(),               # Equal aspect ratio
              xlabel="x (μm)",
              ylabel="y (μm)",
              axis_kwargs...)
    
    # Plot combined circles
    sr_circles_combined!(ax, localizations, mapn_results; kwargs...)
    
    return fig, ax
end
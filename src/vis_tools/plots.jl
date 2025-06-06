# Main plot functions for VisTools
# Each function creates a complete figure with axis

"""
    plot_circles(observations; mapn_emitters=[], true_emitters=[], bounds=nothing, title="Circle Analysis")

Create a circle plot showing localizations, MAP-N estimates, and true emitters.

# Arguments
- `observations`: Vector of observations (must have .x, .y, .σ_x, .σ_y fields)
- `mapn_emitters`: Optional vector of MAP-N emitters (red circles)
- `true_emitters`: Optional vector of true emitters (green X markers)
- `bounds`: Optional (xmin, xmax, ymin, ymax), auto-calculated if nothing
- `title`: Plot title

# Returns
- `Figure`: Makie figure object
"""
function plot_circles(observations; mapn_emitters=[], true_emitters=[], bounds=nothing, title="Circle Analysis")
    # Auto-calculate bounds if not provided
    if isnothing(bounds)
        all_emitters = vcat(observations, mapn_emitters, true_emitters)
        if isempty(all_emitters)
            error("No emitters provided and no bounds specified")
        end
        
        xs = [em.x for em in all_emitters]
        ys = [em.y for em in all_emitters]
        
        # Add padding
        padding = 0.5  # Default padding in data units
        xmin, xmax = extrema(xs) .+ (-padding, padding)
        ymin, ymax = extrema(ys) .+ (-padding, padding)
        bounds = (xmin, xmax, ymin, ymax)
    end
    
    # Create figure and axis
    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1], 
        aspect=DataAspect(), 
        title=title,
        xlabel="x (μm)", 
        ylabel="y (μm)")
    
    # Plot localizations as black circles with uncertainty
    for obs in observations
        radius = 3 * mean([obs.σ_x, obs.σ_y])  # 3-sigma uncertainty
        draw_circle!(ax, obs.x, obs.y, radius; color=(:black, 0.3), strokewidth=1)
    end
    
    # Plot MAP-N emitters as red circles (twice as thick)
    if !isempty(mapn_emitters)
        for em in mapn_emitters
            # Use uncertainty if available, otherwise default size
            if hasfield(typeof(em), :σ_x) && hasfield(typeof(em), :σ_y)
                radius = 3 * mean([em.σ_x, em.σ_y])
            else
                radius = 0.05  # Default radius
            end
            draw_circle!(ax, em.x, em.y, radius; color=(:red, 0.7), strokewidth=2)
        end
    end
    
    # Plot true emitters as green X markers (2x line thickness, 1/2 median sigma size)
    if !isempty(true_emitters)
        # Calculate X size as 1/2 the median sigma of observations
        if !isempty(observations)
            sigmas = [mean([obs.σ_x, obs.σ_y]) for obs in observations]
            median_sigma = median(sigmas)
            x_size = 0.5 * median_sigma
        else
            x_size = 0.025  # Fallback if no observations
        end
        
        for em in true_emitters
            draw_x!(ax, em.x, em.y, x_size; color=:green, strokewidth=2)  # 2x circle strokewidth (1)
        end
    end
    
    # Set axis limits
    xmin, xmax, ymin, ymax = bounds
    xlims!(ax, xmin, xmax)
    ylims!(ax, ymin, ymax)
    
    return fig
end

"""
    plot_sr(observations; pixelsize=0.02, bounds=nothing, title="Super-Resolution Image")

Create a super-resolution image using Gaussian blobs.

# Arguments
- `observations`: Vector of observations (must have .x, .y, .σ_x, .σ_y fields)
- `pixelsize`: Pixel size for image generation
- `bounds`: Optional (xmin, xmax, ymin, ymax), auto-calculated if nothing
- `title`: Plot title

# Returns
- `Figure`: Makie figure object
"""
function plot_sr(observations; pixelsize=0.02, bounds=nothing, title="Super-Resolution Image")
    # Auto-calculate bounds if not provided
    if isnothing(bounds)
        if isempty(observations)
            error("No observations provided and no bounds specified")
        end
        
        xs = [obs.x for obs in observations]
        ys = [obs.y for obs in observations]
        
        # Add padding
        padding = 3 * pixelsize
        xmin, xmax = extrema(xs) .+ (-padding, padding)
        ymin, ymax = extrema(ys) .+ (-padding, padding)
        bounds = (xmin, xmax, ymin, ymax)
    end
    
    # Generate Gaussian blob image
    image = gauss_blob_image(observations, bounds, pixelsize)
    
    # Create figure
    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1], 
        aspect=DataAspect(), 
        title=title,
        xlabel="x (μm)", 
        ylabel="y (μm)")
    
    # Display image
    xmin, xmax, ymin, ymax = bounds
    heatmap!(ax, range(xmin, xmax, size(image, 2)), 
                 range(ymin, ymax, size(image, 1)), 
                 image, colormap=:inferno)
    
    return fig
end

"""
    plot_mapn(mapn_emitters; pixelsize=0.02, bounds=nothing, title="MAP-N Image")

Create a MAP-N image using Gaussian blobs.

# Arguments
- `mapn_emitters`: Vector of MAP-N emitters
- `pixelsize`: Pixel size for image generation
- `bounds`: Optional (xmin, xmax, ymin, ymax), auto-calculated if nothing
- `title`: Plot title

# Returns
- `Figure`: Makie figure object
"""
function plot_mapn(mapn_emitters; pixelsize=0.02, bounds=nothing, title="MAP-N Image")
    # Use the same function as plot_sr - just different title/data
    return plot_sr(mapn_emitters; pixelsize=pixelsize, bounds=bounds, title=title)
end

"""
    plot_posterior(observations; pixelsize=1.0, bounds=nothing, title="Posterior Image")

Create a posterior image using histogram binning.

# Arguments
- `observations`: Vector of observations
- `pixelsize`: Pixel size for image generation (larger for posterior)
- `bounds`: Optional (xmin, xmax, ymin, ymax), auto-calculated if nothing
- `title`: Plot title

# Returns
- `Figure`: Makie figure object
"""
function plot_posterior(observations; pixelsize=1.0, bounds=nothing, title="Posterior Image")
    # Auto-calculate bounds if not provided
    if isnothing(bounds)
        if isempty(observations)
            error("No observations provided and no bounds specified")
        end
        
        xs = [obs.x for obs in observations]
        ys = [obs.y for obs in observations]
        
        # Add padding
        padding = 3 * pixelsize
        xmin, xmax = extrema(xs) .+ (-padding, padding)
        ymin, ymax = extrema(ys) .+ (-padding, padding)
        bounds = (xmin, xmax, ymin, ymax)
    end
    
    # Generate histogram image
    image = histogram_image(observations, bounds, pixelsize)
    
    # Create figure
    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1], 
        aspect=DataAspect(), 
        title=title,
        xlabel="x (μm)", 
        ylabel="y (μm)")
    
    # Display image
    xmin, xmax, ymin, ymax = bounds
    heatmap!(ax, range(xmin, xmax, size(image, 2)), 
                 range(ymin, ymax, size(image, 1)), 
                 image, colormap=:inferno)
    
    return fig
end
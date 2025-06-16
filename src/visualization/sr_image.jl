# Super-resolution image generation for SMLMBaGoL
# Uses multiple dispatch to handle different data types

using Images: save

"""
    gen_sr_image(data; kwargs...) -> Matrix{Float64}

Generate super-resolution image from SMLMBaGoL data.

Supports multiple data types via dispatch:
- `Vector{<:AbstractLocalization}`: Renders localizations using their precision
- `Vector{<:AbstractEmitter}`: Renders emitters using their uncertainty estimates
- `RJMCMCChain`: Creates posterior histogram from emitter positions across samples

# Arguments
- `pixel_size::Real = 0.005`: Pixel size in microns (5 nm default)
- `image_size::Union{Nothing,Tuple{Int,Int}} = nothing`: Image size in pixels (auto-calculated if nothing)
- `bounds::Union{Nothing,NTuple{4,Real}} = nothing`: (x_min, x_max, y_min, y_max) in microns (auto-calculated if nothing)
- `filename::Union{Nothing,String} = nothing`: Optional PNG export filename
- `mode::Symbol = :posterior`: For RJMCMCChain, rendering mode

# Returns
- `Matrix{Float64}`: Super-resolution image array

# Examples
```julia
# From localizations (uses actual precision)
img1 = gen_sr_image(localizations, filename="localizations.png")

# From MAPN emitters (uses uncertainty estimates)
img2 = gen_sr_image(emitters, pixel_size=0.003)

# Posterior uncertainty from chain
img3 = gen_sr_image(chain, mode=:posterior, filename="posterior.png")
```
"""
function gen_sr_image(data; 
                     pixel_size::Real = 0.005,
                     image_size::Union{Nothing,Tuple{Int,Int}} = nothing,
                     bounds::Union{Nothing,NTuple{4,Real}} = nothing,
                     filename::Union{Nothing,String} = nothing,
                     kwargs...)
    
    # Auto-calculate bounds and image size if not provided
    if bounds === nothing || image_size === nothing
        calc_bounds, calc_size = calculate_image_bounds_and_size(data, pixel_size)
        bounds = bounds === nothing ? calc_bounds : bounds
        image_size = image_size === nothing ? calc_size : image_size
    end
    
    # Initialize image array
    image = zeros(Float64, image_size...)
    
    # Type-specific rendering (dispatches to methods below)
    render_sr_data!(image, data, pixel_size, bounds; kwargs...)
    
    # Optional PNG export
    if filename !== nothing
        save_sr_image(image, filename)
    end
    
    return image
end

"""
    gen_sr_image(smld::SMLMSim.BasicSMLD; pixel_size=nothing, filename=nothing, kwargs...)

Generate a super-resolution image from SMLMSim.BasicSMLD data using camera bounds.

This method automatically extracts the field of view from the camera stored in the SMLD
to define the image bounds, following SMLMSim conventions for spatial coordinate systems.

# Arguments
- `smld`: SMLMSim.BasicSMLD data structure (contains camera information)
- `pixel_size`: Reconstruction pixel size in microns (default: camera pixel size / 50 for super-resolution)
- `filename`: Optional PNG export filename
- `kwargs...`: Additional keyword arguments

# Returns
- Matrix{Float64}: Super-resolution image array

# Examples
```julia
# Generate image using SMLD camera bounds with auto pixel size
smld = simulate_static_smlm(npixelsx=128, npixelsy=64, pixelsize=0.1)
image = gen_sr_image(smld; filename="output.png")  # Uses 0.1/50 = 0.002μm = 2nm pixels

# Custom reconstruction pixel size
image = gen_sr_image(smld; pixel_size=0.001, filename="output.png")  # 1nm pixels
```
"""
function gen_sr_image(smld::SMLMSim.BasicSMLD; 
                     pixel_size::Union{Nothing,Real} = nothing,
                     filename::Union{Nothing,String} = nothing,
                     kwargs...)
    
    # Extract camera field of view bounds from SMLD
    camera = smld.camera
    x_edges = camera.pixel_edges_x
    y_edges = camera.pixel_edges_y
    
    # Get the bounds from the pixel edges
    x_min, x_max = extrema(x_edges)
    y_min, y_max = extrema(y_edges)
    bounds = (x_min, x_max, y_min, y_max)
    
    # Determine pixel size: use camera pixel size / 50 if not specified
    if pixel_size === nothing
        # Get camera pixel size (assuming uniform pixels)
        camera_pixel_size = (x_edges[2] - x_edges[1])  # Use first pixel width
        pixel_size = camera_pixel_size / 50  # 50x super-resolution
    end
    
    # Convert SMLD to localizations for image generation
    localizations = smld_to_localizations(smld)
    
    # Generate image with camera-defined bounds
    return gen_sr_image(localizations; 
                       pixel_size=pixel_size,
                       bounds=bounds,
                       filename=filename,
                       kwargs...)
end

"""
    render_sr_data!(image, localizations, pixel_size, bounds; kwargs...)

Render localizations as gaussian blobs using their localization precision.
"""
function render_sr_data!(image::Matrix{Float64}, locs::Vector{<:AbstractLocalization}, 
                        pixel_size::Real, bounds::NTuple{4,Real}; kwargs...)
    for loc in locs
        render_gaussian_blob!(image, loc.x, loc.y, loc.σx, loc.σy, pixel_size, bounds)
    end
end

"""
    render_sr_data!(image, emitters, pixel_size, bounds; kwargs...)

Render emitters as gaussian blobs using their uncertainty estimates from MAPN.
"""
function render_sr_data!(image::Matrix{Float64}, emitters::Vector{<:AbstractEmitter}, 
                        pixel_size::Real, bounds::NTuple{4,Real}; kwargs...)
    for emitter in emitters
        render_gaussian_blob!(image, emitter.x, emitter.y, emitter.σx, emitter.σy, pixel_size, bounds)
    end
end

"""
    render_sr_data!(image, chain, pixel_size, bounds; mode=:posterior, kwargs...)

Render RJMCMC chain as posterior histogram by counting emitter positions.
"""
function render_sr_data!(image::Matrix{Float64}, chain::RJMCMCChain, 
                        pixel_size::Real, bounds::NTuple{4,Real}; 
                        mode::Symbol = :posterior, kwargs...)
    if mode == :posterior
        render_posterior_counts!(image, chain, pixel_size, bounds)
    else
        error("Unsupported mode: $mode")
    end
end

"""
    render_gaussian_blob!(image, x, y, σx, σy, pixel_size, bounds)

Render a normalized 2D gaussian blob into the image array.
The gaussian is normalized so its integral equals 1.
"""
function render_gaussian_blob!(image::Matrix{Float64}, x::Real, y::Real, σx::Real, σy::Real, 
                              pixel_size::Real, bounds::NTuple{4,Real})
    # Standard 2D normal normalization: 1/(2π*σx*σy)
    normalization = 1.0 / (2π * σx * σy)
    
    x_min, x_max, y_min, y_max = bounds
    height, width = size(image)
    
    # Convert physical coordinates to pixel coordinates
    px = (x - x_min) / pixel_size
    py = (y - y_min) / pixel_size
    
    # Convert sigma to pixels
    σx_px = σx / pixel_size
    σy_px = σy / pixel_size
    
    # Calculate 3σ footprint in pixels
    x_range = max(1, floor(Int, px - 3σx_px)):min(width, ceil(Int, px + 3σx_px))
    y_range = max(1, floor(Int, py - 3σy_px)):min(height, ceil(Int, py + 3σy_px))
    
    # Render gaussian over footprint
    for i in y_range, j in x_range
        # Convert back to physical units for gaussian calculation
        dx = (j - px) * pixel_size
        dy = (i - py) * pixel_size
        
        # Normalized 2D gaussian
        gaussian_val = normalization * exp(-0.5 * ((dx/σx)^2 + (dy/σy)^2))
        
        # Scale by pixel area to preserve integral
        image[i, j] += gaussian_val * pixel_size^2
    end
end

"""
    render_posterior_counts!(image, chain, pixel_size, bounds)

Render RJMCMC chain as simple pixel counting histogram.
Each emitter position in each sample adds 1 to the corresponding pixel.
"""
function render_posterior_counts!(image::Matrix{Float64}, chain::RJMCMCChain, 
                                 pixel_size::Real, bounds::NTuple{4,Real})
    x_min, x_max, y_min, y_max = bounds
    height, width = size(image)
    
    for sample in chain.samples
        for emitter in sample.emitters
            # Convert to pixel coordinates (1-indexed)
            px = round(Int, (emitter.x - x_min) / pixel_size) + 1
            py = round(Int, (emitter.y - y_min) / pixel_size) + 1
            
            # Add count if within bounds
            if 1 ≤ px ≤ width && 1 ≤ py ≤ height
                image[py, px] += 1.0
            end
        end
    end
end

"""
    calculate_image_bounds_and_size(data, pixel_size) -> (bounds, size)

Auto-calculate image bounds and size from data extent.
Adds appropriate margin (3σ for localizations, reasonable default for others).
"""
function calculate_image_bounds_and_size(data, pixel_size::Real)
    # Extract coordinates and calculate margin
    x_coords, y_coords = extract_coordinates(data)
    margin = get_data_margin(data)
    
    # Handle empty data case
    if isempty(x_coords) || isempty(y_coords)
        # Default to small bounds if no data
        x_min, x_max = -margin, margin
        y_min, y_max = -margin, margin
    else
        # Calculate bounds with margin
        x_min, x_max = extrema(x_coords) .+ (-margin, margin)
        y_min, y_max = extrema(y_coords) .+ (-margin, margin)
    end
    
    # Calculate image size from bounds
    width = ceil(Int, (x_max - x_min) / pixel_size)
    height = ceil(Int, (y_max - y_min) / pixel_size)
    
    return (x_min, x_max, y_min, y_max), (height, width)
end

"""
    extract_coordinates(data) -> (x_coords, y_coords)

Extract x,y coordinates from different data types.
"""
function extract_coordinates(locs::Vector{<:AbstractLocalization})
    if !isempty(locs)
        return [loc.x for loc in locs], [loc.y for loc in locs]
    else
        return Float64[], Float64[]  # Return empty arrays for empty input
    end
end

function extract_coordinates(emitters::Vector{<:AbstractEmitter})
    return [emitter.x for emitter in emitters], [emitter.y for emitter in emitters]
end

function extract_coordinates(chain::RJMCMCChain)
    x_coords, y_coords = Float64[], Float64[]
    for sample in chain.samples
        for emitter in sample.emitters
            push!(x_coords, emitter.x)
            push!(y_coords, emitter.y)
        end
    end
    return x_coords, y_coords
end

"""
    get_data_margin(data) -> margin

Get appropriate margin for different data types.
"""
function get_data_margin(locs::Vector{<:AbstractLocalization})
    # Use 3σ of largest localization precision as margin
    if !isempty(locs)
        max_σ = maximum(max(loc.σx, loc.σy) for loc in locs)
        return 3 * max_σ
    else
        return 0.1  # Default 100 nm margin for empty data
    end
end

function get_data_margin(emitters::Vector{<:AbstractEmitter})
    # Use 3σ of largest emitter uncertainty as margin
    if !isempty(emitters)
        max_σ = maximum(max(emitter.σx, emitter.σy) for emitter in emitters)
        return 3 * max_σ
    else
        return 0.1  # Default 100 nm margin
    end
end

function get_data_margin(chain::RJMCMCChain)
    # Fixed margin for chain data
    return 0.1  # 100 nm margin
end

"""
    save_sr_image(image, filename)

Save image array as PNG file with proper normalization.
"""
function save_sr_image(image::Matrix{Float64}, filename::String)
    # Normalize to [0,1] for display
    if maximum(image) > 0
        normalized_image = image ./ maximum(image)
    else
        normalized_image = image
    end
    
    # Save using Images.jl
    save(filename, normalized_image)
end
# Image generation functions for VisTools
# Creates raster images from emitter data

using Distributions

"""
    gauss_blob_image(emitters, bounds, pixelsize)

Generate a Gaussian blob image from a vector of emitters.
Each emitter is rendered as a Gaussian blob with uncertainty.

# Arguments
- `emitters`: Vector of emitters (must have .x, .y, .σ_x, .σ_y fields)
- `bounds`: (xmin, xmax, ymin, ymax) in data coordinates
- `pixelsize`: Pixel size in data units

# Returns
- `Matrix{Float64}`: Image matrix
"""
function gauss_blob_image(emitters, bounds, pixelsize)
    xmin, xmax, ymin, ymax = bounds
    nx = ceil(Int, (xmax - xmin) / pixelsize)
    ny = ceil(Int, (ymax - ymin) / pixelsize)
    
    image = zeros(Float64, ny, nx)
    
    for em in emitters
        # Use uncertainty if available, otherwise default PSF
        σ_x = hasfield(typeof(em), :σ_x) ? em.σ_x : 0.1
        σ_y = hasfield(typeof(em), :σ_y) ? em.σ_y : 0.1
        
        # Create 2D Gaussian distribution
        μ = [em.x, em.y]
        Σ = [σ_x^2 0; 0 σ_y^2]
        dist = MvNormal(μ, Σ)
        
        # Add Gaussian blob to image
        for i in 1:nx, j in 1:ny
            x = xmin + (i - 0.5) * pixelsize
            y = ymin + (j - 0.5) * pixelsize
            image[j, i] += pdf(dist, [x, y])
        end
    end
    
    return image
end

"""
    histogram_image(emitters, bounds, pixelsize)

Generate a histogram image from a vector of emitters.
Each emitter contributes 1.0 to its pixel bin.

# Arguments
- `emitters`: Vector of emitters (must have .x, .y fields)
- `bounds`: (xmin, xmax, ymin, ymax) in data coordinates  
- `pixelsize`: Pixel size in data units

# Returns
- `Matrix{Float64}`: Image matrix
"""
function histogram_image(emitters, bounds, pixelsize)
    xmin, xmax, ymin, ymax = bounds
    nx = ceil(Int, (xmax - xmin) / pixelsize)
    ny = ceil(Int, (ymax - ymin) / pixelsize)
    
    image = zeros(Float64, ny, nx)
    
    for em in emitters
        # Convert to pixel coordinates
        i = clamp(round(Int, (em.x - xmin) / pixelsize) + 1, 1, nx)
        j = clamp(round(Int, (em.y - ymin) / pixelsize) + 1, 1, ny)
        image[j, i] += 1.0
    end
    
    return image
end

"""
    quantile_stretch!(image; max_quantile=0.99)

Apply quantile stretching to an image in-place.
Clips values above the specified quantile.
"""
function quantile_stretch!(image; max_quantile=0.99)
    if max_quantile < 1.0
        threshold = quantile(vec(image), max_quantile)
        image[image .> threshold] .= threshold
    end
    return image
end

"""
    gen_color_image(image; max_quantile=0.99, colormap=ColorSchemes.inferno)

Convert a grayscale image to a color image using a colormap.
"""
function gen_color_image(image; max_quantile=0.99, colormap=ColorSchemes.inferno)
    img_copy = copy(image)
    quantile_stretch!(img_copy; max_quantile=max_quantile)
    
    # Normalize to [0, 1]
    img_copy .-= minimum(img_copy)
    max_val = maximum(img_copy)
    if max_val > 0
        img_copy ./= max_val
    end
    
    # Apply colormap
    return get(colormap, img_copy)
end
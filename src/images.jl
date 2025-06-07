"""
    images.jl

Image handling utilities for SMLMBaGoL visualization and export.
"""

using Images

"""
    save_posterior_image(posterior::Matrix{T}, filepath::String; 
                        colormap=:viridis, normalize=true) where T

Save a posterior probability image as a PNG file with a nice colormap.

# Arguments
- `posterior::Matrix{T}`: 2D posterior probability matrix
- `filepath::String`: Output file path (should end in .png)
- `colormap::Symbol`: Colormap to use (default: :viridis)
- `normalize::Bool`: Whether to normalize values to [0,1] range (default: true)

# Example
```julia
result = bagol(emitters)
save_posterior_image(result.posterior, "posterior.png")
```
"""
function save_posterior_image(posterior::Matrix{T}, filepath::String; 
                             colormap::Symbol=:viridis, normalize::Bool=true) where T
    
    # Handle empty or invalid posterior
    if isempty(posterior) || all(iszero, posterior)
        @warn "Posterior image is empty or all zeros, creating blank image"
        blank_img = zeros(Gray{Float64}, size(posterior))
        save(filepath, blank_img)
        return
    end
    
    # Create a copy to avoid modifying the original
    img_data = Float64.(posterior)
    
    # Normalize to [0,1] range if requested
    if normalize
        min_val = minimum(img_data)
        max_val = maximum(img_data)
        
        if max_val > min_val
            img_data = (img_data .- min_val) ./ (max_val - min_val)
        else
            # All values are the same
            img_data .= 0.5
        end
    end
    
    # Apply colormap
    if colormap == :viridis
        # Viridis colormap approximation
        colored_img = apply_viridis_colormap(img_data)
    elseif colormap == :hot
        # Hot colormap (black -> red -> yellow -> white)
        colored_img = apply_hot_colormap(img_data)
    elseif colormap == :gray
        # Grayscale
        colored_img = Gray.(img_data)
    else
        @warn "Unknown colormap $colormap, using grayscale"
        colored_img = Gray.(img_data)
    end
    
    # Save the image
    save(filepath, colored_img)
    
    return nothing
end

"""
    apply_viridis_colormap(data::Matrix{Float64})

Apply a Viridis-like colormap to normalized data [0,1].
"""
function apply_viridis_colormap(data::Matrix{Float64})
    # Viridis colormap approximation
    # Purple (0) -> Blue -> Green -> Yellow (1)
    
    colored_img = Array{RGB{Float64}}(undef, size(data))
    
    for i in eachindex(data)
        val = clamp(data[i], 0.0, 1.0)
        
        if val < 0.25
            # Purple to blue
            t = val / 0.25
            r = 0.267004 * (1-t) + 0.127568 * t
            g = 0.004874 * (1-t) + 0.566949 * t  
            b = 0.329415 * (1-t) + 0.550556 * t
        elseif val < 0.5
            # Blue to green
            t = (val - 0.25) / 0.25
            r = 0.127568 * (1-t) + 0.365731 * t
            g = 0.566949 * (1-t) + 0.752475 * t
            b = 0.550556 * (1-t) + 0.194905 * t
        elseif val < 0.75
            # Green to yellow
            t = (val - 0.5) / 0.25
            r = 0.365731 * (1-t) + 0.906663 * t
            g = 0.752475 * (1-t) + 0.867586 * t
            b = 0.194905 * (1-t) + 0.107543 * t
        else
            # Yellow to light yellow
            t = (val - 0.75) / 0.25
            r = 0.906663 * (1-t) + 0.993248 * t
            g = 0.867586 * (1-t) + 0.906157 * t
            b = 0.107543 * (1-t) + 0.143936 * t
        end
        
        colored_img[i] = RGB{Float64}(r, g, b)
    end
    
    return colored_img
end

"""
    apply_hot_colormap(data::Matrix{Float64})

Apply a hot colormap (black -> red -> yellow -> white) to normalized data [0,1].
"""
function apply_hot_colormap(data::Matrix{Float64})
    colored_img = Array{RGB{Float64}}(undef, size(data))
    
    for i in eachindex(data)
        val = clamp(data[i], 0.0, 1.0)
        
        if val < 1/3
            # Black to red
            t = val * 3
            r = t
            g = 0.0
            b = 0.0
        elseif val < 2/3
            # Red to yellow
            t = (val - 1/3) * 3
            r = 1.0
            g = t
            b = 0.0
        else
            # Yellow to white
            t = (val - 2/3) * 3
            r = 1.0
            g = 1.0
            b = t
        end
        
        colored_img[i] = RGB{Float64}(r, g, b)
    end
    
    return colored_img
end
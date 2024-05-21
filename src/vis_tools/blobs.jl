function gauss_blob2D(x::T, y::T, σ_x::T, σ_y::T, i::Integer, j::Integer) where {T<:AbstractFloat}
    return one(T) / (2 * T(pi) * σ_x * σ_y) * exp(-((x - T(i))^2 / (2 * σ_x^2) + (y - T(j))^2 / (2 * σ_y^2)))
end

function add_blob!(im::BGLImage2D, center::Tuple{Real,Real}, σ::Tuple{Real,Real}; σ_range=3.0)
    # Convert coordinate and radius to pixel values (float)
    σ_y_pixels = σ[1] / im.pixelsize
    σ_x_pixels = σ[2] / im.pixelsize

    center_pixels = (
        (center[1] - im.x_start) / im.pixelsize,
        (center[2] - im.y_start) / im.pixelsize
    )

    # calculate pixel range for blob 
    x_min = max(1, round(Int, center_pixels[2] - σ_range * σ_x_pixels))
    x_max = min(im.width, round(Int, center_pixels[2] + σ_range * σ_x_pixels))
    y_min = max(1, round(Int, center_pixels[1] - σ_range * σ_y_pixels))
    y_max = min(im.height, round(Int, center_pixels[1] + σ_range * σ_y_pixels))

    # Add the blob to the image
    for i in x_min:x_max
        for j in y_min:y_max
            im.data[j, i] += gauss_blob2D(center_pixels[1], center_pixels[2], σ_y_pixels, σ_x_pixels, j, i)
        end
    end

end

function add_blobs!(im::BGLImage2D, obs::BGL.Observations)
    for loc in obs.ŷ
        add_blob!(im, (loc.y, loc.x), (loc.σ_y, loc.σ_x))
    end
end

function gen_sr_image(obs::BGL.Observations, pixelsize::Float64;
    imsize::Union{Tuple{Int,Int},Nothing}=nothing,
    imstart::Union{Tuple{Real,Real},Nothing}=nothing)

    # Create an empty image
    img = BGLImage2D(obs, pixelsize; imsize=imsize, imstart=imstart)

    add_blobs!(img, obs)
    normalize!(img)

    return img
end




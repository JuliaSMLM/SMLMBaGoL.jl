abstract type BGLImage end

struct BGLImage2D <: BGLImage
    data::Array{Gray{Float32},2}  # Using Gray{Float32} for grayscale image data
    height::Int
    width::Int
    y_start::Float64
    x_start::Float64
    pixelsize::Float64
end
function BGLImage2D(imsize::Tuple{Int,Int}, imstart::Tuple{Real,Real}, pixelsize::Real)
    return BGLImage2D(zeros(Gray{Float32}, imsize...), imsize[1], imsize[2], imstart[1], imstart[2], pixelsize)
end
function BGLImage2D(obs::Observations, pixelsize;
    imsize::Union{Tuple{Int,Int},Nothing}=nothing, 
    imstart::Union{Tuple{Real,Real},Nothing}=nothing)

    if isnothing(imstart)
        # find minimum of y- sigma_y and x- sigma_x
        x_start = Inf
        y_start = Inf
        for loc in obs.ŷ
            x_start = min(x_start, loc.x - loc.σ_x)
            y_start = min(y_start, loc.y - loc.σ_y)
        end

        # find maximum of y+ sigma_y and x+ sigma_x
        x_end = -Inf
        y_end = -Inf
        for loc in obs.ŷ
            x_end = max(x_end, loc.x + loc.σ_x)
            y_end = max(y_end, loc.y + loc.σ_y)
        end
    end

    # find the range of the image
    if isnothing(imsize)
        x_size = Int(round((x_end - x_start) / pixelsize))
        y_size = Int(round((y_end - y_start) / pixelsize))
    else
        x_size, y_size = imsize
    end
    imsize = (y_size, x_size)

    # Create an empty image
    img = BGLImage2D(imsize, (y_start, x_start), pixelsize)
    return img 
end

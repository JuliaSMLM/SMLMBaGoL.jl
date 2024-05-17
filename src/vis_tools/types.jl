abstract type BGLImage end

struct BGLImage2D <: BGLImage
    data::Array{Gray{Float32}, 2}  # Using Gray{Float32} for grayscale image data
    height::Int
    width::Int
    y_start::Float64
    x_start::Float64
    pixelsize::Float64
end
function BGLImage2D(imsize::Tuple{Int, Int}, imstart::Tuple{Real,Real}, pixelsize::Real)
    return BGLImage2D(zeros(Gray{Float32}, imsize...), imsize[1], imsize[2], imstart[1], imstart[2], pixelsize)    
end


function draw_circle!(im::BGLImage2D, center::Tuple{Real,Real}, radius::Real, color::Colorant)

    # convert coordinate and radius to pixel values (float)
    radius_pixels = radius / im.pixelsize
    center_pixels = (
        (center[1] - im.x_start) / im.pixelsize,
        (center[2] - im.y_start) / im.pixelsize
    )

    # generate line segments for the circle
    angles = range(0, 2π, length=100)
    circle_points = [
        (round(Int, center_pixels[1] + radius_pixels * cos(θ)), round(Int, center_pixels[2] + radius_pixels * sin(θ)))
        for θ in angles
    ]

    T = typeof(im.data[1, 1].val)
    # Draw line segments connecting the points
    for i in 1:length(circle_points)-1
        p1 = circle_points[i]
        p2 = circle_points[i+1]
        draw!(im.data, LineSegment(p1[1], p1[2], p2[1], p2[2]), Gray{T}(1),)
    end
    # Connect the last point to the first point
    p1 = circle_points[end]
    p2 = circle_points[1]
    draw!(im.data, LineSegment(p1[1], p1[2], p2[1], p2[2]), Gray{T}(1))
end

function draw_x!(im::BGLImage2D, center::Tuple{Real,Real}, size::Real, color::Colorant)
    T = typeof(im.data[1, 1])
    # Convert to pixel values
    center_pixels = (
        (center[1] - im.x_start) / im.pixelsize,
        (center[2] - im.y_start) / im.pixelsize
    )
    size_pixels = size / im.pixelsize

    # Draw the x
    draw!(im.data, LineSegment(center_pixels[1] - size_pixels, center_pixels[2] - size_pixels, center_pixels[1] + size_pixels, center_pixels[2] + size_pixels), color)
    draw!(im.data, LineSegment(center_pixels[1] - size_pixels, center_pixels[2] + size_pixels, center_pixels[1] + size_pixels, center_pixels[2] - size_pixels), color)
end

function gauss_blob2D(x::T, y::T, σ_x::T, σ_y::T, i::Integer, j::Integer) where {T<:AbstractFloat}
    return one(T) / (2 * T(pi) * σ_x * σ_y) * exp(-((x - T(i))^2 / (2 * σ_x^2) + (y - T(j))^2 / (2 * σ_y^2)))
end

function add_blob!(im::BGLImage2D, center::Tuple{Real,Real}, σ_x::Float64, σ_y::Float64, color::Colorant)

end


function draw_observations!(im::BGLImage2D, obs::BGL.Observations)


    T = typeof(im.data[1, 1].val)
    for loc in obs.ŷ
        r = sqrt(loc.σ_x^2 + loc.σ_y^2)
        draw_circle!(im, (loc.x, loc.y), r, Gray{T}(1))
    end

end

function gen_obs_image(obs::BGL.Observations, pixelsize::Float64;
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

    # Draw circles on the image
    draw_observations!(img, obs)
    return img 
end

function draw_emitters!(im::BGLImage2D, emitters::Vector{BGL.Emitters.Emitter2D})
    T = typeof(im.data[1, 1])
    for emitter in emitters
        r = 1.0
        draw_circle!(im, (emitter.x, emitter.y), r, Gray{T}(1))
    end
end

function draw_true!(im::BGLImage2D, emitters::Vector{BGL.Emitters.Emitter2D})
    T = typeof(im.data[1, 1])
    for emitter in emitters
        r = 5.0
        draw_x!(im, (emitter.x, emitter.y), r, Gray{T}(1))
    end
end








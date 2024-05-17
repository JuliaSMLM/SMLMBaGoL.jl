
function draw_circle!(im::BGLImage2D, center::Tuple{Real,Real}, radius::Real, color::Colorant)
    # Convert coordinate and radius to pixel values (float)
    radius_pixels = radius / im.pixelsize
    center_pixels = (
        (center[1] - im.y_start) / im.pixelsize,
        (center[2] - im.x_start) / im.pixelsize
    )

    # Generate line segments for the circle
    T = typeof(im.data[1, 1].val)
    num_points = 100
    step = 2π / num_points
    angle = 0.0

    # Note that draw and point uses (x,y) coordinates, not (row, col)
    p1 = (round(Int, center_pixels[2] + radius_pixels * cos(angle)), round(Int, center_pixels[1] + radius_pixels * sin(angle)))
    for i in 1:num_points
        angle += step
        p2 = (round(Int, center_pixels[2] + radius_pixels * cos(angle)), round(Int, center_pixels[1] + radius_pixels * sin(angle)))
        draw!(im.data, LineSegment(p1[1], p1[2], p2[1], p2[2]), Gray{T}(1))
        p1 = p2
    end

    # Connect the last point to the first point
    angle = 0.0
    p2 = (round(Int, center_pixels[2] + radius_pixels * cos(angle)), round(Int, center_pixels[1] + radius_pixels * sin(angle)))
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




function draw_observations!(im::BGLImage2D, obs::BGL.Observations)
    T = typeof(im.data[1, 1].val)
    for loc in obs.ŷ
        r = sqrt(loc.σ_x^2 + loc.σ_y^2)
        draw_circle!(im, (loc.y, loc.x), r, Gray{T}(1))
    end
end


function gen_obs_image(obs::BGL.Observations, pixelsize::Float64;
    imsize::Union{Tuple{Int,Int},Nothing}=nothing,
    imstart::Union{Tuple{Real,Real},Nothing}=nothing)

    # Create an empty image
    img = BGLImage2D(obs, pixelsize; imsize=imsize, imstart=imstart)

    # Draw circles on the image
    draw_observations!(img, obs)
    return img
end

function draw_emitters!(im::BGLImage2D, emitters::Vector{BGL.Emitters.Emitter2D})
    T = typeof(im.data[1, 1])
    for emitter in emitters
        r = 1.0
        draw_circle!(im, (emitter.y, emitter.x), r, Gray{T}(1))
    end
end

function draw_true!(im::BGLImage2D, emitters::Vector{BGL.Emitters.Emitter2D})
    T = typeof(im.data[1, 1])
    for emitter in emitters
        r = 5.0
        draw_x!(im, (emitter.y, emitter.x), r, Gray{T}(1))
    end
end






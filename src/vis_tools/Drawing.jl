module Drawing

using ..SMLMBaGoL
using ..SMLMBaGoL.Emitters
using ImageDraw

export draw_circle_on_image!, draw_x_on_image!
export draw_observations_on_image!, draw_emitters_on_image!
export draw_true_on_image!

# Image-based drawing functions (pixel arrays)
function draw_circle_on_image!(img::Matrix{T}, x::Real, y::Real, radius::Real; 
    color::T=one(T), fill::Bool=false) where T
    
    if fill
        draw!(img, FilledCircle(Point(round(Int, x), round(Int, y)), round(Int, radius)), color)
    else
        draw!(img, Circle(Point(round(Int, x), round(Int, y)), round(Int, radius)), color)
    end
    return img
end

function draw_x_on_image!(img::Matrix{T}, x::Real, y::Real; 
    size::Real=5, color::T=one(T)) where T
    
    ix, iy = round(Int, x), round(Int, y)
    s = round(Int, size)
    
    # Draw diagonal lines to form X
    for i in -s:s
        px1, py1 = ix + i, iy + i
        px2, py2 = ix + i, iy - i
        
        if checkbounds(Bool, img, px1, py1)
            img[px1, py1] = color
        end
        if checkbounds(Bool, img, px2, py2)
            img[px2, py2] = color
        end
    end
    
    return img
end

function draw_observations_on_image!(img::Matrix{T}, obs::Observations{<:Localization2D}; 
    scale::Real=3.0, color::T=one(T)) where T
    
    for ob in obs.ŷ
        radius = scale * mean([ob.σ_x, ob.σ_y])
        draw_circle_on_image!(img, ob.x, ob.y, radius; color=color, fill=false)
    end
    return img
end

function draw_emitters_on_image!(img::Matrix{T}, emitters::Vector{<:AbstractEmitter}; 
    markersize::Real=5, color::T=one(T)) where T
    
    for em in emitters
        draw_circle_on_image!(img, em.x, em.y, markersize; color=color, fill=true)
    end
    return img
end

function draw_true_on_image!(img::Matrix{T}, true_emitters::Vector{<:AbstractEmitter}; 
    markersize::Real=5, color::T=one(T)) where T
    
    for em in true_emitters
        draw_x_on_image!(img, em.x, em.y; size=markersize, color=color)
    end
    return img
end

end # module Drawing
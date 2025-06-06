# Core drawing primitives for VisTools
# All functions work in data coordinates on Makie axes

"""
    draw_circle!(ax, x, y, radius; color=:black, strokewidth=1, fillalpha=0.0)

Draw a circle on a Makie axis in data coordinates.

# Arguments
- `ax`: Makie axis object
- `x, y`: Center coordinates in data space
- `radius`: Radius in data space units
- `color`: Line color (default: :black)
- `strokewidth`: Line thickness (default: 1)
- `fillalpha`: Fill transparency, 0=no fill (default: 0.0)

# Returns
- `ax`: The modified axis
"""
function draw_circle!(ax, x, y, radius; color=:black, strokewidth=1, fillalpha=0.0)
    θ = range(0, 2π, length=100)
    xs = x .+ radius .* cos.(θ)
    ys = y .+ radius .* sin.(θ)
    
    if fillalpha > 0
        poly!(ax, Point2f.(xs, ys), color=(color, fillalpha))
    end
    lines!(ax, xs, ys, color=color, linewidth=strokewidth)
    
    return ax
end

"""
    draw_x!(ax, x, y, size; color=:black, strokewidth=2)

Draw an X marker on a Makie axis in data coordinates.

# Arguments
- `ax`: Makie axis object
- `x, y`: Center coordinates in data space
- `size`: Half-width of X marker in data space units
- `color`: Line color (default: :black)
- `strokewidth`: Line thickness (default: 2)

# Returns
- `ax`: The modified axis
"""
function draw_x!(ax, x, y, size; color=:black, strokewidth=2)
    # Draw X as two crossed lines in data coordinates
    lines!(ax, [x-size, x+size], [y-size, y+size], color=color, linewidth=strokewidth)
    lines!(ax, [x-size, x+size], [y+size, y-size], color=color, linewidth=strokewidth)
    return ax
end
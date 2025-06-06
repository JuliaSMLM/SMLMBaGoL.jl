module Combined

using ..SMLMBaGoL
using ..SMLMBaGoL.RJMCMC
using ..SMLMBaGoL.Emitters
using CairoMakie
using Statistics

# Import specific types from RJMCMC
import ..SMLMBaGoL.RJMCMC: RJMCMC_Chain, RJMCMC_ROI

export plot_combined_analysis, plot_mapn_with_uncertainty

"""
    plot_combined_analysis(obs::Observations, mapn_coords, true_emitters=nothing; kwargs...)

Create a comprehensive visualization combining:
- Localizations as uncertainty circles (blue)
- MAP-N estimates as crosses with uncertainty circles (green)
- Optional true emitter positions as filled circles (red)

# Arguments
- `obs::Observations`: Raw observations with uncertainty
- `mapn_coords`: Vector of MAP-N coordinate estimates with uncertainty
- `true_emitters=nothing`: Optional vector of true emitter positions
- `title="Combined Analysis"`: Plot title
- `figsize=(800, 600)`: Figure size
- `localization_alpha=0.4`: Transparency of localization circles
- `mapn_alpha=0.7`: Transparency of MAP-N uncertainty circles
- `show_legend=true`: Whether to show legend

# Returns
- `fig`: Makie Figure object
"""
function plot_combined_analysis(obs::Observations, mapn_coords, true_emitters=nothing;
    title="Combined Analysis: Localizations + MAP-N + Truth",
    figsize=(800, 600),
    localization_alpha=0.4,
    mapn_alpha=0.7,
    show_legend=true)
    
    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1], 
        aspect=DataAspect(), 
        title=title,
        xlabel="x (μm)", 
        ylabel="y (μm)")
    
    # Plot localizations as uncertainty circles (blue)
    for ob in obs.ŷ
        radius = 3 * mean([ob.σ_x, ob.σ_y])  # 3-sigma uncertainty
        draw_circle_on_axis!(ax, ob.x, ob.y, radius; 
            color=(:blue, localization_alpha), strokewidth=1)
    end
    
    # Add invisible point for legend
    scatter!(ax, [NaN], [NaN], color=:blue, alpha=localization_alpha, 
        marker=:circle, markersize=8, label="Localizations (3σ)")
    
    # Plot MAP-N estimates if provided
    if !isempty(mapn_coords)
        mapn_x = [coord.x for coord in mapn_coords]
        mapn_y = [coord.y for coord in mapn_coords]
        
        # Plot MAP-N estimates as crosses
        scatter!(ax, mapn_x, mapn_y, 
            color=:green, markersize=12, marker=:x, 
            label="MAP-N estimates (n=$(length(mapn_coords)))")
        
        # Plot MAP-N uncertainty circles
        for coord in mapn_coords
            radius = 3 * mean([coord.σ_x, coord.σ_y])  # 3-sigma uncertainty
            draw_circle_on_axis!(ax, coord.x, coord.y, radius; 
                color=(:green, mapn_alpha), strokewidth=2)
        end
    end
    
    # Plot true emitters if provided
    if !isnothing(true_emitters) && !isempty(true_emitters)
        true_x = [em.x for em in true_emitters]
        true_y = [em.y for em in true_emitters]
        scatter!(ax, true_x, true_y, 
            color=:red, markersize=8, marker=:circle,
            label="True emitters")
    end
    
    # Add legend if requested
    if show_legend
        axislegend(ax, position=:rt)
    end
    
    return fig
end

"""
    plot_mapn_with_uncertainty(mapn_coords; kwargs...)

Create a focused plot of MAP-N estimates with uncertainty circles.

# Arguments
- `mapn_coords`: Vector of MAP-N coordinate estimates with uncertainty
- `title="MAP-N Estimates with Uncertainty"`: Plot title
- `figsize=(600, 600)`: Figure size
- `uncertainty_alpha=0.6`: Transparency of uncertainty circles
- `sigma_level=3`: Number of standard deviations for uncertainty circles

# Returns
- `fig`: Makie Figure object
"""
function plot_mapn_with_uncertainty(mapn_coords;
    title="MAP-N Estimates with Uncertainty",
    figsize=(600, 600),
    uncertainty_alpha=0.6,
    sigma_level=3)
    
    if isempty(mapn_coords)
        error("No MAP-N coordinates provided")
    end
    
    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1], 
        aspect=DataAspect(), 
        title=title,
        xlabel="x (μm)", 
        ylabel="y (μm)")
    
    mapn_x = [coord.x for coord in mapn_coords]
    mapn_y = [coord.y for coord in mapn_coords]
    
    # Plot MAP-N estimates as crosses
    scatter!(ax, mapn_x, mapn_y, 
        color=:green, markersize=12, marker=:x, 
        label="MAP-N estimates")
    
    # Plot uncertainty circles
    for (i, coord) in enumerate(mapn_coords)
        radius_x = sigma_level * coord.σ_x
        radius_y = sigma_level * coord.σ_y
        radius = mean([radius_x, radius_y])  # Average for circular approximation
        
        draw_circle_on_axis!(ax, coord.x, coord.y, radius; 
            color=(:green, uncertainty_alpha), strokewidth=2)
    end
    
    # Add legend
    axislegend(ax, position=:rt)
    
    return fig
end

# Helper function for drawing circles on Makie axes
function draw_circle_on_axis!(ax, x, y, radius; color=:black, strokewidth=1, fillalpha=0.0)
    θ = range(0, 2π, length=100)
    xs = x .+ radius .* cos.(θ)
    ys = y .+ radius .* sin.(θ)
    
    if fillalpha > 0
        poly!(ax, Point2f.(xs, ys), color=(color, fillalpha))
    end
    lines!(ax, xs, ys, color=color, linewidth=strokewidth)
    
    return ax
end

end # module Combined
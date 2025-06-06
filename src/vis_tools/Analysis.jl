module Analysis

using ..SMLMBaGoL
using ..SMLMBaGoL.RJMCMC
using ..SMLMBaGoL.Emitters
using CairoMakie
using Distributions
using StatsBase

# Import specific types from RJMCMC
import ..SMLMBaGoL.RJMCMC: RJMCMC_Chain, RJMCMC_ROI

export plot_observations, plot_observations!
export plot_posterior, plot_sr
export plot_true_values!
export plot_prior_k, plot_prior_λ
export plot_state_length, plot_sld

# Observation plotting functions
function plot_observations(obs::Observations; kwargs...)
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=1)
    plot_observations!(ax, obs; kwargs...)
    return fig
end

function plot_observations!(ax, obs::Observations; 
    markersize=10, strokewidth=1, color=:blue, alpha=0.3)
    
    for ob in obs.ŷ
        draw_circle_on_axis!(ax, ob.x, ob.y, 3 * mean([ob.σ_x, ob.σ_y]); 
            color=(color, alpha), strokewidth=strokewidth)
    end
    return ax
end

# MCMC chain analysis plots
function plot_posterior(chain::RJMCMC_Chain, roi::RJMCMC_ROI;
    pixelsize=0.01, image_size=nothing)
    
    if isnothing(image_size)
        # Calculate image size from ROI bounds
        x_range = roi.limits.xmax - roi.limits.xmin
        y_range = roi.limits.ymax - roi.limits.ymin
        nx = ceil(Int, x_range / pixelsize)
        ny = ceil(Int, y_range / pixelsize)
        image_size = (nx, ny)
    end
    
    # Create posterior image
    posterior = zeros(Float64, image_size...)
    
    # Accumulate posterior from chain
    for state in chain.states
        for emitter in state.emitters
            # Convert to pixel coordinates
            px = round(Int, (emitter.x - roi.limits.xmin) / pixelsize) + 1
            py = round(Int, (emitter.y - roi.limits.ymin) / pixelsize) + 1
            
            if 1 <= px <= image_size[1] && 1 <= py <= image_size[2]
                posterior[px, py] += 1.0
            end
        end
    end
    
    # Normalize
    posterior ./= length(chain.states)
    
    # Create figure
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=1)
    heatmap!(ax, posterior', colormap=:hot)
    
    return fig, posterior
end

function plot_sr(obs::Observations, chain::RJMCMC_Chain; 
    pixelsize=0.01, psf_width=0.1)
    
    # Get bounds from observations
    xs = [ob.x for ob in obs.ŷ]
    ys = [ob.y for ob in obs.ŷ]
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    
    # Add padding
    padding = 3 * psf_width
    xmin -= padding
    xmax += padding
    ymin -= padding
    ymax += padding
    
    # Create SR image using prior distributions
    nx = ceil(Int, (xmax - xmin) / pixelsize)
    ny = ceil(Int, (ymax - ymin) / pixelsize)
    
    sr_image = zeros(Float64, nx, ny)
    
    # Add Gaussian blobs for each observation based on prior
    for ob in obs.ŷ
        # Create 2D Gaussian
        μ = [ob.x, ob.y]
        Σ = [ob.σ_x^2 0; 0 ob.σ_y^2]
        dist = MvNormal(μ, Σ)
        
        # Add to image
        for i in 1:nx, j in 1:ny
            x = xmin + (i - 0.5) * pixelsize
            y = ymin + (j - 0.5) * pixelsize
            sr_image[i, j] += pdf(dist, [x, y])
        end
    end
    
    # Normalize
    sr_image ./= maximum(sr_image)
    
    # Create figure
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=1)
    heatmap!(ax, sr_image', colormap=:viridis)
    
    return fig, sr_image
end

function plot_true_values!(ax, true_emitters::Vector{<:AbstractEmitter}; 
    markersize=10, color=:red)
    
    xs = [e.x for e in true_emitters]
    ys = [e.y for e in true_emitters]
    scatter!(ax, xs, ys, marker='x', markersize=markersize, color=color)
    return ax
end

# Prior distribution plots
function plot_prior_k(prior_k::Distribution; max_k=50)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Number of emitters (k)", ylabel="Prior probability")
    
    ks = 0:max_k
    probs = [pdf(prior_k, k) for k in ks]
    
    barplot!(ax, ks, probs)
    
    return fig
end

function plot_prior_λ(prior_λ::Distribution; max_λ=30)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Localizations per emitter (λ)", ylabel="Prior probability")
    
    λs = range(0, max_λ, length=1000)
    probs = [pdf(prior_λ, λ) for λ in λs]
    
    lines!(ax, λs, probs)
    
    return fig
end

# Chain diagnostics
function plot_state_length(chain::RJMCMC_Chain)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Iteration", ylabel="Number of emitters")
    
    lengths = [length(state.emitters) for state in chain.states]
    lines!(ax, lengths)
    
    return fig
end

function plot_sld(chain::RJMCMC_Chain; nbins=20)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Number of emitters", ylabel="Frequency")
    
    lengths = [length(state.emitters) for state in chain.states]
    hist!(ax, lengths, bins=nbins)
    
    return fig
end

# Animation functions are in the Animation module

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

end # module Analysis
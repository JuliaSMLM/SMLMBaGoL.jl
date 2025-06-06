# Basic 2D workflow 
using Pkg
Pkg.activate("dev")
using Revise
using SMLMBaGoL
BGL = SMLMBaGoL
using Distributions
using CairoMakie
using CairoMakie: Point2f
using StatsBase

# Setup Parameters
n_emitters = 10 
xy_range = 100.0 # for n_emitters == 2, this is separation between emitters
μ_λ = 10.0 # mean of λ
σ_λ = 3.0 # standard deviation of λ
# σ_λ = μ_λ # for Exponential distribution
n_burnin = 4000
n_jumps = 8000

## Build prior distributions
α = μ_λ^2 / σ_λ^2 # Shape
θ = σ_λ^2 / μ_λ # Scale
prior_λ = Gamma(α, θ)

# Create true emitters and observations
T = BGL.Emitter2D{Float64} 
if n_emitters == 2
    emitters = [BGL.Emitter2D([0.0, xy_range/2]), BGL.Emitter2D([0.0, -xy_range/2])]
else
    dist1 = Uniform(-xy_range/2, xy_range/2)
    dist2 = Uniform(-xy_range/2, xy_range/2)
    dist = Product([dist1, dist2])
    emitters = BGL.gen_emitters(T,n_emitters, dist)
end
obs = BGL.gen_observations(prior_λ, emitters; photons=1000.0)

chain, mapn_coords, chain_mapn, roi = BGL.rjmcmc(obs, prior_λ; n_burnin = n_burnin, n_jumps = n_jumps)

# Plots and Prints ----------------------------------------
# Create visualizations using new plot functions

# Plot observations with true emitters
@info "Creating circle plot with true emitters"
fig_obs = BGL.plot_circles(
    obs.ŷ;
    title="Observations and True Emitters"
)
display(fig_obs)

# Plot super-resolution image  
@info "Creating super-resolution image"
fig_sr = BGL.plot_sr(
    obs.ŷ;
    pixelsize=0.01,
    title="Super-Resolution Image"
)
display(fig_sr)

# Plot posterior distribution
@info "Creating posterior image"
fig_posterior = BGL.plot_posterior(
    obs.ŷ;
    pixelsize=0.05,
    title="Posterior Distribution"
)
display(fig_posterior)

# Plot MAP-N results if available
if !isempty(mapn_coords)
    @info "Creating MAP-N visualization"
    fig_mapn = BGL.plot_mapn(
        mapn_coords;
        pixelsize=0.01,
        title="MAP-N Estimates"
    )
    display(fig_mapn)
    
    # Combined plot
    fig_combined = BGL.plot_circles(
        obs.ŷ;
        mapn_emitters=mapn_coords,
        title="Combined: Observations + MAP-N"
    )
    display(fig_combined)
end

BGL.plot_state_length(chain)
BGL.plot_sld(chain)
n_map, n_vec = BGL.RJMCMC.find_mapn(chain)
n_true = length(emitters)
println("True N = $n_true, MAPN  = $n_map")

# slow
# BGL.animate_chain(chain, obs, emitters)







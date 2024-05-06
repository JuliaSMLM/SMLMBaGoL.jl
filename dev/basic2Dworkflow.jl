# Basic 2D workflow 
# using Pkg
#Pkg.activate("dev")
using Revise
using SMLMBaGoL
BGL = SMLMBaGoL
using SMLMBaGoL: RJMCMC
RJ = RJMCMC
using Distributions
using CairoMakie
using CairoMakie: Point2f0
using StatsBase

# Setup Parameters
n_emitters = 10 
xy_range = 50.0 # for n_emitters == 2, this is separation between emitters
μ_λ = 10.0 # mean of λ
σ_λ = 3.0 # standard deviation of λ
n_burnin = 4000
n_jumps = 8000

## Build prior distributions
α = μ_λ^2 / σ_λ^2 # Shape
θ = σ_λ^2 / μ_λ # Scale
prior_λ = Gamma(α, θ)

# Create true emitters and observations 
if n_emitters == 2
    emitters = RJ.Params([RJ.Emitter2D([0.0, xy_range/2]), RJ.Emitter2D([0.0, -xy_range/2])])
else
    dist1 = Uniform(-xy_range/2, xy_range/2)
    dist2 = Uniform(-xy_range/2, xy_range/2)
    dist = Product([dist1, dist2])
    emitters = RJ.gen_emitters2D(n_emitters, dist)
end
obs = RJ.gen_observations2D(prior_λ, emitters; photons=1000.0)

prior_k = RJ.build_prior_k(obs, prior_λ)
prior_y = RJ.build_prior_y(obs)


## RJMCMC
p_jump = Categorical([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7])
roi = RJ.RJMCMC_ROI(obs, prior_y, prior_k, p_jump, RJ.Emitter2D, prior_λ)
@time chain, z_chain = RJ.buildchain(roi, n_burnin, n_jumps);

# MAPN MCMC
best_state = RJ.find_mapn_ref_state(chain)
θ = RJ.Params(best_state.emitters)      
p_jump = Categorical([1, 0,0,0,0,0,0])
roi = RJ.RJMCMC_ROI(obs, prior_y, prior_k, p_jump, RJ.Emitter2D, prior_λ)
@time chain_mapn, = RJ.buildchain(roi, n_burnin, n_jumps; θ = θ);
RJ.sort_mapn_chain!(chain_mapn; n_iterate = 3)
mapn_coords = RJ.get_mapn_emitters(chain_mapn, obs)

# Plots and Prints ----------------------------------------
RJ.plot_prior_λ(prior_λ, obs)
RJ.plot_prior_k(prior_k, obs)

fig, ax = RJ.plot_observations(obs)
RJ.plot_true_values!(ax, emitters; markersize = 10)
display(fig)

fig, ax = RJ.plot_sr(obs, prior_y)
RJ.plot_true_values!(ax, emitters; markersize = 10)
display(fig)

fig, ax = RJ.plot_posterior(chain, obs)
RJ.plot_true_values!(ax, emitters; markersize = 10)
display(fig)

fig, ax = RJ.plot_posterior(chain_mapn, obs; title = "MAPN")
RJ.plot_true_values!(ax, emitters; markersize = 10)
display(fig)

fig, ax = RJ.plot_observations(RJ.Observations(mapn_coords))
RJ.plot_true_values!(ax, emitters; markersize = 10)
display(fig)

RJ.plot_state_length(chain)
RJ.plot_sld(chain)
n_map, n_vec = RJ.find_mapn(chain)
n_true = length(emitters.emitters)
println("True N = $n_true, MAPN  = $n_map")









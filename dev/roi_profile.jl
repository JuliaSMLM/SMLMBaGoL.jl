# Basic 2D workflow 
# using Pkg
#Pkg.activate("dev")
using Revise
using SMLMBaGoL
BGL = SMLMBaGoL
using Distributions
using CairoMakie
using CairoMakie: Point2f0
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

@time BGL.rjmcmc(obs, prior_λ; n_burnin = n_burnin, n_jumps = n_jumps);
@profview BGL.rjmcmc(obs, prior_λ; n_burnin = n_burnin, n_jumps = n_jumps)

z = Allocations(zeros(Int, length(obs.ŷ)))
θ = BGL.Params(emitters)

@code_warntype BGL.RJMCMC.allocate!(z, obs, θ )
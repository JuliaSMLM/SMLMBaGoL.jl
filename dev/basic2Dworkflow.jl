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


# Setup
emitter_gen_dist = MvNormal([0.0, 0.0], [200.0 0.0; 0.0 200.0])

# Prior distribution for λ: localizations per emitter
μ_λ = 10.0
σ_λ = 3.0
α = μ_λ^2 / σ_λ^2 # Shape
θ = σ_λ^2 / μ_λ

# A gamma prior with mean and variance given by μ_λ and σ_λ
prior_λ = Gamma(α, θ)

# An exponential like prior using Gamma and the mean
# prior_λ = Gamma(1.0 / μ_λ, 1.0 / μ_λ^2)

mean(prior_λ)
std(prior_λ)

n_emitters = 10

# Generate emitters and observations
emitters = RJ.gen_emitters2D(n_emitters, emitter_gen_dist)

# two close emitters 
d = 5.0
# emitters = RJ.Params([RJ.Emitter2D([0.0, d/2]), RJ.Emitter2D([0.0, -d/2])])

obs = RJ.gen_observations2D(prior_λ, emitters; photons=1000.0)
length(obs)

# Show plot of true emitters and observations with circles for standard deviation
function draw_circle!(axis, center::Point2f0, radius::Float64; points::Int=100, color=:black,
                      linewidth=1.0, linestyle=:solid, linealpha=1.0, linecolor=:black, fillalpha=0.0, fillcolor=:black)
    θ = LinRange(0, 2π, points)
    x = center[1] .+ radius * cos.(θ)
    y = center[2] .+ radius * sin.(θ)
    lines!(axis, x, y, color=color, linewidth=linewidth, linestyle=linestyle, linealpha=linealpha, linecolor=linecolor)
end

fig = Figure()
ax = Axis(fig[1, 1], aspect=DataAspect())
for emitter in emitters.emitters
    scatter!(ax, [emitter.x], [emitter.y], color=:red)
end
for obs in obs.ŷ
    draw_circle!(ax, Point2f0(obs.x, obs.y), obs.σ_x)
end
fig

## Build prior distributions
prior_y = RJ.build_prior_y(obs)
area = RJ.calc_area(obs)
# Plot prior distribution image for y
vals = rand(prior_y, 10000)
fig = Figure()
hist_data = fit(Histogram, (vals[2, :], vals[1, :]), nbins=(50, 50))
fig = Figure()
ax = Axis(fig[1, 1], aspect=DataAspect())
heatmap!(ax, hist_data.edges[1], hist_data.edges[2], hist_data.weights)
display(fig)

# Build the prior distribution for k
prior_k = RJ.build_prior_k(obs, prior_λ)

# Show prior distribution for k and compare to prior_λ
fig = Figure()
ax = Axis(fig[1, 1], xlabel="k", ylabel="pdf", title="Prior distribution for k")
k_vec = 0:length(obs.ŷ)
barplot!(ax, k_vec, pdf.(prior_k, k_vec), label="prior_k")
lines!(ax, k_vec, pdf.(prior_λ, k_vec), label="prior_λ")
axislegend()
display(fig)

length(obs.ŷ) / mean(prior_λ)
mode(prior_k)
mean(prior_k)
mean(prior_λ)


## Build the chain
p_jump = Categorical([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7])
roi = RJ.RJMCMC_ROI(obs, prior_y, area, prior_k, p_jump, RJ.Emitter2D, prior_λ)
n_burnin = 5000
n_jumps = 5000

@time chain, z_chain = RJ.buildchain(roi, n_burnin, n_jumps);

## Plot the chain
fig = Figure()
ax = Axis(fig[1, 1], aspect=DataAspect())
# Collate coordinates for chain
chain_x = Float64[]
chain_y = Float64[]
for state in chain.states
    for emitter in state.emitters
        push!(chain_x, emitter.x)
        push!(chain_y, emitter.y)
    end
end
hist_data = fit(Histogram, (chain_x, chain_y), nbins=(50, 50))
heatmap!(ax, hist_data.edges[1], hist_data.edges[2], hist_data.weights, colormap=:inferno)

scatter!(ax, chain_x, chain_y, color=:blue, transparency=0.1)
# Plot the circles for standard deviation
for obs in obs.ŷ
    draw_circle!(ax, Point2f0(obs.x, obs.y), obs.σ_x; color=:white)
end
# Collate coordinates for true values
true_x = [emitter.x for emitter in emitters.emitters]
true_y = [emitter.y for emitter in emitters.emitters]
for idx in 1:length(true_x)
    draw_circle!(ax, Point2f0(true_x[idx], true_y[idx]), obs.ŷ[idx].σ_x; color=:cyan)
end
display(fig)

# Try finding MAP in number of emitters
n_vec = length.(chain.states)
fig = Figure()
ax = Axis(fig[1, 1])
hist!(ax, n_vec, bins=0.5:1:maximum(n_vec)+0.5)
display(fig)

n_map = mode(n_vec)
println("MAP for number of emitters = $n_map")

# Plot the length of the chain over the Run
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Jump number", ylabel="Number of emitters", title="Chain length over run")
lines!(ax, 1:n_jumps, length.(chain.states))
display(fig)

# include("animate_chain.jl")
include("gen_posterior.jl")
gen_posterior(chain, emitters, obs)

# Distribution of allocations in last frame
fig = Figure()
ax = Axis(fig[1, 1])
hist!(ax, z_chain[end].idx, bins=0.5:1:maximum(z_chain[end].idx)+0.5)
display(fig)

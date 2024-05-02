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
emitter_gen_dist = MvNormal([0.0, 0.0], [100.0 0.0; 0.0 100.0])

# Prior distribution for λ
# Note for an Exponential distribution, the mean is μ = 1/λ and the variance is σ^2 = 1/λ^2
μ_λ = 10.0
σ_λ = 3.0
prior_λ = Gamma(μ_λ^2 / σ_λ^2, σ_λ^2 / μ_λ)
mean(prior_λ)
std(prior_λ)
n_emitters = 3

# Generate emitters and observations
# emitters = RJ.gen_emitters2D(n_emitters, emitter_gen_dist)
# two close emitters 
emitters = RJ.Params([RJ.Emitter2D([0.0, 10.0]), RJ.Emitter2D([0.0, -10.0])])

obs = RJ.gen_observations2D(prior_λ, emitters; photons=1000.0)
length(obs)

# Show plot of true emitters and observations with circles for standard deviation
function draw_circle!(axis, center::Point2f0, radius::Float64; points::Int=100, color=:black)
    θ = LinRange(0, 2π, points)
    x = center[1] .+ radius * cos.(θ)
    y = center[2] .+ radius * sin.(θ)
    lines!(axis, x, y, color=color)
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
p_jump = Categorical([0.2, 0.2, 0.2, 0.2, 0.2])
roi = RJ.RJMCMC_ROI(obs, prior_y, area, prior_k, p_jump, RJ.Emitter2D, prior_λ)
n_burnin = 100
n_jumps = 1000
chain, z_chain = RJ.buildchain(roi, n_burnin, n_jumps);

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

area * pdf(prior_y, [obs.ŷ[1].y, obs.ŷ[1].x])

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
ax = Axis(fig[1, 1])
lines!(ax, 1:n_jumps, length.(chain.states))
display(fig)


##  Animate the chain 

# Create an observable for the frame index
frame_index = Observable(1)

# Create the figure and axis
fig = Figure(resolution=(800, 800))
ax = Axis(fig[1, 1], aspect=DataAspect())

# Emitters
for emitter in emitters.emitters
    scatter!(ax, [emitter.x], [emitter.y], color=:red)
end

# Observations color coded by allocation
color_iterator = [:red, :green, :blue, :yellow, :cyan, :magenta, :black]

function calc_idx(loc_id::Int)
    idx = (1 + (loc_id - 1) % length(color_iterator))
    return idx
end

colors = @lift(
    [color_iterator[calc_idx(z_chain[$(frame_index)].idx[i])]
     for i in eachindex(obs.ŷ)]
)

circles = [draw_circle!(ax, Point2f0(obs.ŷ[i].x, obs.ŷ[i].y), obs.ŷ[i].σ_x, color=colors[][i]) for i in eachindex(obs.ŷ)]

# Function to extract coordinates for a given state
coords = @lift(
    [Point2f0.(chain.states[$(frame_index)].emitters[j].x, chain.states[$(frame_index)].emitters[j].y)
     for j in 1:length(chain.states[$(frame_index)].emitters)],
)

scatter!(ax, coords,
    color=:blue,
    marker=:circle,
    markersize=10)

# Record the animation
n_frames = length(chain.states)
record(fig, "scatter_animation.mp4", 1:n_frames; framerate=24) do i
    frame_index[] = i  # Update the frame index observable
    [circle.color = colors[][j] for (j, circle) in enumerate(circles)]
end

# Show final figure
display(fig)

# plot just the posterior distribution of the emitters with square pixels
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
pixelsize = 1.0
#calc bins from data range and pixelsize
xrange = maximum(chain_x) - minimum(chain_x)
yrange = maximum(chain_y) - minimum(chain_y)
nbins_x = Int(ceil(xrange / pixelsize))
nbins_y = Int(ceil(yrange / pixelsize))
hist_data = fit(Histogram, (chain_x, chain_y), nbins=(nbins_x, nbins_y))
heatmap!(ax, hist_data.edges[1], hist_data.edges[2], hist_data.weights, colormap=:inferno)
display(fig)
#save 
save("posterior_emitters.png", fig)

# Distribution of allocations in last frame
fig = Figure()
ax = Axis(fig[1, 1])
hist!(ax, z_chain[end].idx, bins=0.5:1:maximum(z_chain[end].idx)+0.5)
display(fig)

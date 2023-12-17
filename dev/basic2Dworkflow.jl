# Basic 2D workflow 
# using Pkg
# Pkg.activate("dev")
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
prior_λ = Gamma(2.0, 5.0)
n_emitters = 3

# Generate emitters and observations
emitters = RJ.gen_emitters2D(n_emitters, emitter_gen_dist)
obs = RJ.gen_observations2D(prior_λ, emitters; photons = 10000.0)
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
ax = Axis(fig[1, 1])
k_vec = 0:length(obs.ŷ)
lines!(ax, k_vec, pdf.(prior_k, k_vec))
lines!(ax, k_vec, pdf.(prior_λ, k_vec))
display(fig)
length(obs.ŷ) / mean(prior_λ)
mode(prior_k)
mean(prior_k)
mean(prior_λ)

## Build the chain
p_jump = Categorical([0.2, 0.2, 0.2, 0.2, 0.2])
roi = RJ.RJMCMC_ROI(obs, prior_y, prior_k, p_jump, RJ.Emitter2D)
n_burnin = 100
n_jumps = 1000
chain = RJ.buildchain(roi, n_burnin, n_jumps);

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
ax = Axis(fig[1, 1])
lines!(ax, 1:n_jumps, length.(chain.states))
display(fig)


##  Animate the chain 
# n_frames = length(chain.states)
n_frames = 100

# Create an observable for the frame index
frame_index = Observable(1)

# Function to extract coordinates for a given state
coords = @lift((
    [chain.states[$(frame_index)].emitters[j].x for j in 1:length(chain.states[$(frame_index)].emitters)],
    [chain.states[$(frame_index)].emitters[j].y for j in 1:length(chain.states[$(frame_index)].emitters)]
))

# Create the figure and initial scatter plot
# fig = Figure(resolution = (800, 800))
# ax = Axis(fig[1, 1], aspect=DataAspect())
fig = plot(coords[][1], coords[][2],
    color=:blue,
    marker=:circle,
    markersize=10)

# xlims!(ax, (-100, 100))
# ylims!(ax, (-100, 100))

# Record the animation
record(fig, "scatter_animation.mp4", 1:n_frames; framerate=24) do i
    frame_index[] = i  # Update the frame index observable
end



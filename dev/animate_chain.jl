##  Animate the chain 

# Create an observable for the frame index
frame_index = Observable(1)

# Create the figure and axis
fig = Figure(size=(800, 800))
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

module Animation

# Analysis module will be available through parent VisTools module
using ..SMLMBaGoL  
using ..SMLMBaGoL.RJMCMC
using ..SMLMBaGoL.Emitters
using CairoMakie

# Import specific types from RJMCMC
import ..SMLMBaGoL.RJMCMC: RJMCMC_Chain, RJMCMC_ROI

export create_chain_animation, create_posterior_animation, animate_chain

# Re-export the main animation function from Analysis
# (This will be handled by the parent VisTools module)

# Additional animation utilities
function create_posterior_animation(chain::RJMCMC_Chain, roi::RJMCMC_ROI;
    filename="posterior_animation.mp4", 
    pixelsize=0.01, 
    fps=10, 
    window_size=100,
    skip=10)
    
    # Calculate image dimensions
    x_range = roi.limits.xmax - roi.limits.xmin
    y_range = roi.limits.ymax - roi.limits.ymin
    nx = ceil(Int, x_range / pixelsize)
    ny = ceil(Int, y_range / pixelsize)
    
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], aspect=1, title="Posterior Evolution")
    
    # Create observable for heatmap
    posterior = Observable(zeros(Float64, nx, ny))
    hm = heatmap!(ax, posterior, colormap=:hot)
    
    # Animation
    indices = 1:skip:length(chain.states)
    record(fig, filename, indices; framerate=fps) do i
        # Calculate posterior for sliding window
        start_idx = max(1, i - window_size)
        end_idx = i
        
        # Reset posterior
        post = zeros(Float64, nx, ny)
        
        # Accumulate from window
        for j in start_idx:end_idx
            state = chain.states[j]
            for emitter in state.emitters
                px = round(Int, (emitter.x - roi.limits.xmin) / pixelsize) + 1
                py = round(Int, (emitter.y - roi.limits.ymin) / pixelsize) + 1
                
                if 1 <= px <= nx && 1 <= py <= ny
                    post[px, py] += 1.0
                end
            end
        end
        
        # Normalize and update
        if maximum(post) > 0
            post ./= maximum(post)
        end
        posterior[] = post
        
        # Update title
        ax.title = "Posterior Evolution (iterations $(start_idx)-$(end_idx))"
    end
    
    return filename
end

# Create animation showing convergence diagnostics
function create_convergence_animation(chain::RJMCMC_Chain;
    filename="convergence_animation.mp4",
    fps=10,
    window_size=100,
    skip=10)
    
    fig = Figure(resolution=(1200, 600))
    
    # State length plot
    ax1 = Axis(fig[1, 1], xlabel="Iteration", ylabel="Number of emitters", 
               title="Chain State Length")
    
    # Running mean plot
    ax2 = Axis(fig[1, 2], xlabel="Iteration", ylabel="Running mean k",
               title="Running Mean of State Length")
    
    # Prepare data
    lengths = [length(state.emitters) for state in chain.states]
    iterations = collect(1:length(lengths))
    
    # Create observables
    current_iter = Observable(1)
    current_lengths = Observable(lengths[1:1])
    current_iters = Observable(iterations[1:1])
    running_means = Observable([mean(lengths[1:1])])
    
    # Plot lines
    lines!(ax1, current_iters, current_lengths, color=:blue)
    scatter!(ax1, [current_iter], [lengths[1]], color=:red, markersize=10)
    
    lines!(ax2, current_iters, running_means, color=:green)
    
    # Animation
    indices = 1:skip:length(chain.states)
    record(fig, filename, indices; framerate=fps) do i
        current_iter[] = i
        current_lengths[] = lengths[1:i]
        current_iters[] = iterations[1:i]
        
        # Calculate running means
        means = Float64[]
        for j in 1:i
            push!(means, mean(lengths[1:j]))
        end
        running_means[] = means
        
        # Update scatter point
        empty!(ax1)
        lines!(ax1, current_iters[], current_lengths[], color=:blue)
        scatter!(ax1, [i], [lengths[i]], color=:red, markersize=10)
        
        # Adjust axes limits
        xlims!(ax1, 0, max(i + 10, 100))
        xlims!(ax2, 0, max(i + 10, 100))
    end
    
    return filename
end

# Main animation function for RJMCMC chains
function animate_chain(chain::RJMCMC_Chain, obs::Observations; 
    filename="chain_animation.mp4", fps=10, skip=10)
    
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], aspect=1)
    
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
    
    # Function to plot observations on axis
    function plot_observations_on_axis!(ax, obs::Observations; 
        markersize=10, strokewidth=1, color=:blue, alpha=0.3)
        
        for ob in obs.ŷ
            draw_circle_on_axis!(ax, ob.x, ob.y, 3 * mean([ob.σ_x, ob.σ_y]); 
                color=(color, alpha), strokewidth=strokewidth)
        end
        return ax
    end
    
    # Animation
    record(fig, filename, 1:skip:length(chain.states); framerate=fps) do i
        empty!(ax)
        plot_observations_on_axis!(ax, obs, alpha=0.2)
        
        # Plot current state
        state = chain.states[i]
        xs = [e.x for e in state.emitters]
        ys = [e.y for e in state.emitters]
        scatter!(ax, xs, ys, marker='o', markersize=10, color=:red)
        
        # Update title
        ax.title = "Iteration $i, k = $(length(state.emitters))"
    end
    
    return filename
end

end # module Animation
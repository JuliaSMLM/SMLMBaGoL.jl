function plot_hierarchical_evolution(chains::Vector{RJMCMCChain};
                                   filename::Union{Nothing,String} = nothing,
                                   figsize::Tuple{Int,Int} = (800, 600))
    
    # Collect hierarchical history from all chains
    all_histories = []
    for (i, chain) in enumerate(chains)
        if !isempty(chain.hierarchical_history)
            for (iter, α, β) in chain.hierarchical_history
                push!(all_histories, (iter, α, β, i))
            end
        end
    end
    
    if isempty(all_histories)
        @warn "No hierarchical history found. Ensure enable_hierarchical=true was used."
        return nothing
    end
    
    # Sort by iteration
    sort!(all_histories, by=x->x[1])
    
    # Extract data
    iterations = [h[1] for h in all_histories]
    alphas = [h[2] for h in all_histories]
    betas = [h[3] for h in all_histories]
    
    # Create figure
    fig = Figure(size=figsize)
    
    # Plot α evolution
    ax1 = CairoMakie.Axis(fig[1, 1], 
               xlabel="Iteration", 
               ylabel="α (shape parameter)",
               title="Hierarchical Prior Evolution: α")
    lines!(ax1, iterations, alphas, linewidth=2, color=:blue)
    scatter!(ax1, iterations, alphas, markersize=8, color=:blue)
    
    # Plot β evolution
    ax2 = CairoMakie.Axis(fig[2, 1], 
               xlabel="Iteration", 
               ylabel="β (scale parameter)",
               title="Hierarchical Prior Evolution: β")
    lines!(ax2, iterations, betas, linewidth=2, color=:red)
    scatter!(ax2, iterations, betas, markersize=8, color=:red)
    
    # Grid is already enabled by default in CairoMakie
    
    # Save if filename provided
    if !isnothing(filename)
        save(filename, fig, px_per_unit=2)
    end
    
    return fig
end

function plot_gamma_distributions(chains::Vector{RJMCMCChain};
                                filename::Union{Nothing,String} = nothing,
                                n_timepoints::Int = 5,
                                figsize::Tuple{Int,Int} = (800, 600))
    
    # Collect unique parameter sets from history
    all_params = Set{Tuple{Float64,Float64}}()
    iter_to_params = Dict{Int,Tuple{Float64,Float64}}()
    
    for chain in chains
        for (iter, α, β) in chain.hierarchical_history
            push!(all_params, (α, β))
            iter_to_params[iter] = (α, β)
        end
    end
    
    if isempty(all_params)
        @warn "No hierarchical history found. Ensure enable_hierarchical=true was used."
        return nothing
    end
    
    # Select timepoints to display
    sorted_iters = sort(collect(keys(iter_to_params)))
    if length(sorted_iters) <= n_timepoints
        selected_iters = sorted_iters
    else
        # Select evenly spaced timepoints
        indices = round.(Int, range(1, length(sorted_iters), length=n_timepoints))
        selected_iters = sorted_iters[indices]
    end
    
    # Create figure
    fig = Figure(size=figsize)
    ax = CairoMakie.Axis(fig[1, 1], 
              xlabel="Number of localizations per emitter (k)", 
              ylabel="P(k)",
              title="Evolution of Gamma Prior Distribution")
    
    # Color palette
    colors = [:blue, :green, :orange, :red, :purple]
    
    # Plot distributions
    k_max = 50
    k_values = 0:k_max
    
    for (i, iter) in enumerate(selected_iters)
        α, β = iter_to_params[iter]
        
        # Compute Gamma PDF values
        gamma_dist = Distributions.Gamma(α, β)
        pdf_values = [Distributions.pdf(gamma_dist, k) for k in k_values]
        
        color = colors[mod1(i, length(colors))]
        lines!(ax, k_values, pdf_values, 
               label="Iter $iter (α=$(round(α,digits=2)), β=$(round(β,digits=2)))",
               linewidth=2, color=color)
    end
    
    # Add legend
    axislegend(ax, position=:rt)
    
    # Save if filename provided
    if !isnothing(filename)
        save(filename, fig, px_per_unit=2)
    end
    
    return fig
end

function plot_emitter_count_histogram(chains::Vector{RJMCMCChain};
                                    filename::Union{Nothing,String} = nothing,
                                    figsize::Tuple{Int,Int} = (800, 600),
                                    true_mean::Union{Nothing,Real} = nothing)
    
    # Collect all emitter counts from samples
    all_counts = collect_emitter_counts(chains)
    
    if isempty(all_counts)
        @warn "No emitter count data found in chain samples."
        return nothing
    end
    
    # Get final hierarchical parameters if available
    final_α, final_β = nothing, nothing
    for chain in chains
        if !isempty(chain.hierarchical_history)
            last_entry = chain.hierarchical_history[end]
            final_α, final_β = last_entry[2], last_entry[3]
            break
        end
    end
    
    # Create figure
    fig = Figure(size=figsize)
    ax = CairoMakie.Axis(fig[1, 1], 
              xlabel="Localizations per emitter", 
              ylabel="Frequency",
              title="Empirical vs Fitted Distribution")
    
    # Plot histogram
    hist!(ax, all_counts, bins=0:maximum(all_counts)+1, 
          normalization=:pdf, color=(:blue, 0.6), 
          label="Empirical")
    
    # Overlay fitted Gamma if parameters available
    if !isnothing(final_α) && !isnothing(final_β)
        k_values = 0:0.1:maximum(all_counts)
        gamma_dist = Distributions.Gamma(final_α, final_β)
        pdf_values = [Distributions.pdf(gamma_dist, k) for k in k_values]
        lines!(ax, k_values, pdf_values, 
               color=:red, linewidth=3,
               label="Fitted Gamma(α=$(round(final_α,digits=2)), β=$(round(final_β,digits=2)))")
    end
    
    # Add true mean line if provided
    if !isnothing(true_mean)
        vlines!(ax, [true_mean], color=:green, linewidth=3, linestyle=:dash,
                label="True mean = $true_mean")
    end
    
    # Add legend
    axislegend(ax, position=:rt)
    
    # Save if filename provided
    if !isnothing(filename)
        save(filename, fig, px_per_unit=2)
    end
    
    return fig
end

function analyze_hierarchical_convergence(chains::Vector{RJMCMCChain};
                                        window_size::Int = 5)
    
    # Collect all parameter updates
    all_updates = []
    for chain in chains
        for (iter, α, β) in chain.hierarchical_history
            push!(all_updates, (iter, α, β))
        end
    end
    
    if length(all_updates) < window_size + 1
        return (converged=false, message="Insufficient updates for convergence analysis")
    end
    
    # Sort by iteration
    sort!(all_updates, by=x->x[1])
    
    # Check parameter stability in recent updates
    recent_alphas = [u[2] for u in all_updates[end-window_size+1:end]]
    recent_betas = [u[3] for u in all_updates[end-window_size+1:end]]
    
    α_cv = std(recent_alphas) / mean(recent_alphas)  # Coefficient of variation
    β_cv = std(recent_betas) / mean(recent_betas)
    
    converged = α_cv < 0.05 && β_cv < 0.05  # Less than 5% variation
    
    return (
        converged = converged,
        α_cv = α_cv,
        β_cv = β_cv,
        final_α = all_updates[end][2],
        final_β = all_updates[end][3],
        n_updates = length(all_updates),
        message = converged ? "Hierarchical parameters have converged" : "Hierarchical parameters still changing"
    )
end

function get_hierarchical_summary(chains::Vector{RJMCMCChain})
    
    # Count total updates
    total_updates = sum(length(chain.hierarchical_history) for chain in chains)
    
    if total_updates == 0
        return (
            enabled = false,
            n_updates = 0,
            message = "No hierarchical updates found. Was enable_hierarchical=true?"
        )
    end
    
    # Get initial and final parameters
    initial_α, initial_β = nothing, nothing
    final_α, final_β = nothing, nothing
    
    for chain in chains
        if !isempty(chain.hierarchical_history)
            if isnothing(initial_α)
                initial_α, initial_β = chain.hierarchical_history[1][2:3]
            end
            final_α, final_β = chain.hierarchical_history[end][2:3]
        end
    end
    
    # Calculate change
    α_change = (final_α - initial_α) / initial_α * 100
    β_change = (final_β - initial_β) / initial_β * 100
    
    return (
        enabled = true,
        n_updates = total_updates,
        initial_α = initial_α,
        initial_β = initial_β,
        final_α = final_α,
        final_β = final_β,
        α_change_percent = α_change,
        β_change_percent = β_change,
        convergence = analyze_hierarchical_convergence(chains),
        message = "Hierarchical prior adapted from α=$(round(initial_α,digits=2))→$(round(final_α,digits=2)), β=$(round(initial_β,digits=2))→$(round(final_β,digits=2))"
    )
end

function plot_true_vs_hierarchical_distribution(true_counts::Vector{Int}, chains::Vector{RJMCMCChain};
                                              filename::Union{Nothing,String} = nothing,
                                              figsize::Tuple{Int,Int} = (800, 600))
    
    # Get final hierarchical parameters if available
    final_α, final_β = nothing, nothing
    for chain in chains
        if !isempty(chain.hierarchical_history)
            last_entry = chain.hierarchical_history[end]
            final_α, final_β = last_entry[2], last_entry[3]
            break
        end
    end
    
    if isnothing(final_α)
        @warn "No hierarchical parameters found in chains."
        return nothing
    end
    
    # Create figure
    fig = Figure(size=figsize)
    ax = CairoMakie.Axis(fig[1, 1], 
              xlabel="Localizations per emitter", 
              ylabel="Probability density",
              title="True Distribution vs Hierarchical Prior")
    
    # Plot true distribution histogram
    hist!(ax, true_counts, bins=0:maximum(true_counts)+1, 
          normalization=:pdf, color=(:blue, 0.6), 
          label="True distribution")
    
    # Calculate true statistics
    true_mean = mean(true_counts)
    true_std = std(true_counts)
    
    # Overlay fitted Gamma from hierarchical prior
    k_values = 0:0.1:maximum(true_counts)+5
    gamma_dist = Distributions.Gamma(final_α, final_β)
    pdf_values = [Distributions.pdf(gamma_dist, k) for k in k_values]
    fitted_mean = final_α * final_β
    fitted_std = sqrt(final_α * final_β^2)
    
    lines!(ax, k_values, pdf_values, 
           color=:red, linewidth=3,
           label="Hierarchical prior Gamma($(round(final_α,digits=2)), $(round(final_β,digits=2)))")
    
    # Add vertical lines for means
    vlines!(ax, [true_mean], color=:blue, linewidth=2, linestyle=:dash,
            label="True mean = $(round(true_mean, digits=1))")
    vlines!(ax, [fitted_mean], color=:red, linewidth=2, linestyle=:dash,
            label="Prior mean = $(round(fitted_mean, digits=1))")
    
    # Add text box with statistics
    text_str = "True: μ=$(round(true_mean,digits=1)), σ=$(round(true_std,digits=1))\n" *
               "Prior: μ=$(round(fitted_mean,digits=1)), σ=$(round(fitted_std,digits=1))\n" *
               "Error: $(round(abs(fitted_mean-true_mean)/true_mean*100,digits=1))%"
    
    text!(ax, 0.95, 0.95, text=text_str, 
          align=(:right, :top), space=:relative,
          fontsize=14, font="mono")
    
    # Add legend
    axislegend(ax, position=:lt)
    
    # Save if filename provided
    if !isnothing(filename)
        save(filename, fig, px_per_unit=2)
    end
    
    return fig
end
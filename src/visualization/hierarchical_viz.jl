function plot_hierarchical_evolution(chains::Vector{<:RJMCMCChain};
                                   filename::Union{Nothing,String} = nothing,
                                   figsize::Tuple{Int,Int} = (800, 600))
    
    # Collect hierarchical history from all chains
    all_histories = []
    for (i, chain) in enumerate(chains)
        if !isempty(chain.hierarchical_history)
            for update in chain.hierarchical_history
                push!(all_histories, (update.iteration, update.μ, update.κ, i))
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
    mus = [h[2] for h in all_histories]
    kappas = [h[3] for h in all_histories]
    
    # Create figure
    fig = Figure(size=figsize)
    
    # Plot μ evolution
    ax1 = CairoMakie.Axis(fig[1, 1], 
               xlabel="Iteration", 
               ylabel="μ (mean localizations per emitter)",
               title="Hierarchical Prior Evolution: μ")
    lines!(ax1, iterations, mus, linewidth=2, color=:blue)
    scatter!(ax1, iterations, mus, markersize=8, color=:blue)
    
    # Plot κ evolution
    ax2 = CairoMakie.Axis(fig[2, 1], 
               xlabel="Iteration", 
               ylabel="κ (overdispersion parameter)",
               title="Hierarchical Prior Evolution: κ")
    lines!(ax2, iterations, kappas, linewidth=2, color=:red)
    scatter!(ax2, iterations, kappas, markersize=8, color=:red)
    
    # Grid is already enabled by default in CairoMakie
    
    # Save if filename provided
    if !isnothing(filename)
        save(filename, fig, px_per_unit=2)
    end
    
    return fig
end

function plot_negbinomial_distributions(chains::Vector{<:RJMCMCChain};
                                       filename::Union{Nothing,String} = nothing,
                                       n_timepoints::Int = 5,
                                       figsize::Tuple{Int,Int} = (800, 600))
    
    # Collect unique parameter sets from history
    all_params = Set{Tuple{Float64,Float64}}()
    iter_to_params = Dict{Int,Tuple{Float64,Float64}}()
    
    for chain in chains
        for update in chain.hierarchical_history
            push!(all_params, (update.μ, update.κ))
            iter_to_params[update.iteration] = (update.μ, update.κ)
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
              title="Evolution of Negative Binomial Prior Distribution")
    
    # Color palette
    colors = [:blue, :green, :orange, :red, :purple]
    
    # Plot distributions
    k_max = 50
    k_values = 0:k_max
    
    for (i, iter) in enumerate(selected_iters)
        μ, κ = iter_to_params[iter]
        
        # Compute Negative Binomial PMF values
        # Convert μ, κ parameterization to r, p parameterization for Distributions.jl
        # NegativeBinomial(r, p) where r = κ, p = κ/(κ + μ)
        r = κ
        p = κ / (κ + μ)
        nb_dist = Distributions.NegativeBinomial(r, p)
        pmf_values = [Distributions.pdf(nb_dist, k) for k in k_values]
        
        color = colors[mod1(i, length(colors))]
        lines!(ax, k_values, pmf_values, 
               label="Iter $iter (μ=$(round(μ,digits=2)), κ=$(round(κ,digits=2)))",
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

# Keep the old name for backward compatibility but call the new function
function plot_gamma_distributions(chains::Vector{<:RJMCMCChain}; kwargs...)
    return plot_negbinomial_distributions(chains; kwargs...)
end

function plot_emitter_count_histogram(chains::Vector{<:RJMCMCChain};
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
    final_μ, final_κ = nothing, nothing
    for chain in chains
        if !isempty(chain.hierarchical_history)
            last_update = chain.hierarchical_history[end]
            final_μ, final_κ = last_update.μ, last_update.κ
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
    
    # Overlay fitted Negative Binomial if parameters available
    if !isnothing(final_μ) && !isnothing(final_κ)
        k_values = 0:maximum(all_counts)
        # Convert μ, κ parameterization to r, p parameterization for Distributions.jl
        r = final_κ
        p = final_κ / (final_κ + final_μ)
        nb_dist = Distributions.NegativeBinomial(r, p)
        pmf_values = [Distributions.pdf(nb_dist, k) for k in k_values]
        scatter!(ax, k_values, pmf_values, 
                color=:red, markersize=6,
                label="Fitted NegBinom(μ=$(round(final_μ,digits=2)), κ=$(round(final_κ,digits=2)))")
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

function analyze_hierarchical_convergence(chains::Vector{<:RJMCMCChain};
                                        window_size::Int = 5)
    
    # Collect all parameter updates
    all_updates = []
    for chain in chains
        for update in chain.hierarchical_history
            push!(all_updates, (update.iteration, update.μ, update.κ))
        end
    end
    
    if length(all_updates) < window_size + 1
        return (converged=false, message="Insufficient updates for convergence analysis")
    end
    
    # Sort by iteration
    sort!(all_updates, by=x->x[1])
    
    # Check parameter stability in recent updates
    recent_mus = [u[2] for u in all_updates[end-window_size+1:end]]
    recent_kappas = [u[3] for u in all_updates[end-window_size+1:end]]
    
    μ_cv = std(recent_mus) / mean(recent_mus)  # Coefficient of variation
    κ_cv = std(recent_kappas) / mean(recent_kappas)
    
    converged = μ_cv < 0.05 && κ_cv < 0.05  # Less than 5% variation
    
    return (
        converged = converged,
        μ_cv = μ_cv,
        κ_cv = κ_cv,
        final_μ = all_updates[end][2],
        final_κ = all_updates[end][3],
        # For backward compatibility, also provide α, β names (μ=α*β, κ approximated)
        final_α = all_updates[end][3],  # κ as shape-like parameter
        final_β = all_updates[end][2] / all_updates[end][3],  # approximate scale
        n_updates = length(all_updates),
        message = converged ? "Hierarchical parameters have converged" : "Hierarchical parameters still changing"
    )
end

function get_hierarchical_summary(chains::Vector{<:RJMCMCChain})
    
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
    initial_μ, initial_κ = nothing, nothing
    final_μ, final_κ = nothing, nothing
    
    for chain in chains
        if !isempty(chain.hierarchical_history)
            if isnothing(initial_μ)
                first_update = chain.hierarchical_history[1]
                initial_μ, initial_κ = first_update.μ, first_update.κ
            end
            last_update = chain.hierarchical_history[end]
            final_μ, final_κ = last_update.μ, last_update.κ
        end
    end
    
    # Calculate change
    μ_change = (final_μ - initial_μ) / initial_μ * 100
    κ_change = (final_κ - initial_κ) / initial_κ * 100
    
    return (
        enabled = true,
        n_updates = total_updates,
        initial_μ = initial_μ,
        initial_κ = initial_κ,
        final_μ = final_μ,
        final_κ = final_κ,
        # For backward compatibility
        initial_α = initial_κ,
        initial_β = initial_μ / initial_κ,
        final_α = final_κ,
        final_β = final_μ / final_κ,
        μ_change_percent = μ_change,
        κ_change_percent = κ_change,
        # For backward compatibility
        α_change_percent = κ_change,
        β_change_percent = μ_change / initial_κ * 100,
        convergence = analyze_hierarchical_convergence(chains),
        message = "Hierarchical prior adapted from μ=$(round(initial_μ,digits=2))→$(round(final_μ,digits=2)), κ=$(round(initial_κ,digits=2))→$(round(final_κ,digits=2))"
    )
end
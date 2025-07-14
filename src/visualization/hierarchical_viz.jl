function plot_hierarchical_evolution(chains::Vector{<:RJMCMCChain};
                                   filename::Union{Nothing,String} = nothing,
                                   figsize::Tuple{Int,Int} = (800, 900))  # Increased height for 3 plots
    
    # Collect hierarchical history from all chains
    all_histories = []
    for (i, chain) in enumerate(chains)
        if !isempty(chain.hierarchical_history)
            for update in chain.hierarchical_history
                push!(all_histories, (update.iteration, update.μ, update.κ, update.τ², i))
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
    tau2s = [h[4] for h in all_histories]
    
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
    
    # Plot τ² evolution
    ax3 = CairoMakie.Axis(fig[3, 1], 
               xlabel="Iteration", 
               ylabel="τ² (nm²)",
               title="Hierarchical Prior Evolution: τ² (additional localization variance)")
    # Convert to nm² for better readability
    tau2s_nm2 = tau2s .* 1e6  # Convert from μm² to nm²
    lines!(ax3, iterations, tau2s_nm2, linewidth=2, color=:green)
    scatter!(ax3, iterations, tau2s_nm2, markersize=8, color=:green)
    
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
                                    true_mean::Union{Nothing,Real} = nothing,
                                    show_fit_stats::Bool = true)
    
    # Collect all emitter counts from samples
    all_counts = collect_emitter_counts(chains)
    
    if isempty(all_counts)
        @warn "No emitter count data found in chain samples."
        return nothing
    end
    
    # Get final hierarchical parameters if available
    final_μ, final_κ = nothing, nothing
    prior = nothing
    for chain in chains
        if !isempty(chain.hierarchical_history)
            last_update = chain.hierarchical_history[end]
            final_μ, final_κ = last_update.μ, last_update.κ
            # Get the actual prior object
            if isa(chain.current_state.count_prior, HierarchicalNegBinomialPrior)
                prior = chain.current_state.count_prior
            end
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
        
        # Add fit statistics if requested and prior is available
        if show_fit_stats && !isnothing(prior)
            stats = assess_negbinomial_fit(all_counts, prior)
            
            # Add text box with fit statistics
            textstr = """
            Fit Quality:
            χ² p-value: $(round(stats.chi_squared_pvalue, digits=3))
            KS p-value: $(round(stats.ks_pvalue, digits=3))
            
            Empirical: μ=$(round(stats.empirical_mean, digits=1)), σ²=$(round(stats.empirical_var, digits=1))
            Fitted: μ=$(round(stats.fitted_mean, digits=1)), σ²=$(round(stats.fitted_var, digits=1))
            """
            
            text!(ax, 0.95, 0.5, text=textstr, 
                  align=(:right, :center),
                  fontsize=12,
                  space=:relative)
        end
    end
    
    # Add true mean line if provided
    if !isnothing(true_mean)
        vlines!(ax, [true_mean], color=:green, linewidth=3, linestyle="-",
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
            push!(all_updates, (update.iteration, update.μ, update.κ, update.τ²))
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
    recent_tau2s = [u[4] for u in all_updates[end-window_size+1:end]]
    
    μ_cv = std(recent_mus) / mean(recent_mus)  # Coefficient of variation
    κ_cv = std(recent_kappas) / mean(recent_kappas)
    τ²_cv = std(recent_tau2s) / mean(recent_tau2s)
    
    converged = μ_cv < 0.05 && κ_cv < 0.05 && τ²_cv < 0.05  # Less than 5% variation
    
    return (
        converged = converged,
        μ_cv = μ_cv,
        κ_cv = κ_cv,
        τ²_cv = τ²_cv,
        final_μ = all_updates[end][2],
        final_κ = all_updates[end][3],
        final_τ² = all_updates[end][4],
        n_updates = length(all_updates),
        message = converged ? "Hierarchical parameters have converged" : "Hierarchical parameters still changing"
    )
end

function plot_negbinomial_diagnostic(chains::Vector{<:RJMCMCChain};
                                   filename::Union{Nothing,String} = nothing,
                                   figsize::Tuple{Int,Int} = (1200, 800),
                                   show_qq::Bool = true,
                                   show_residuals::Bool = true)
    
    # Collect data
    all_counts = collect_emitter_counts(chains)
    if isempty(all_counts)
        @warn "No emitter count data found."
        return nothing
    end
    
    # Get final parameters and prior
    prior = nothing
    for chain in chains
        if isa(chain.current_state.count_prior, HierarchicalNegBinomialPrior)
            prior = chain.current_state.count_prior
            break
        end
    end
    
    if isnothing(prior)
        @warn "No hierarchical prior found."
        return nothing
    end
    
    # Create multi-panel figure
    fig = Figure(size=figsize)
    
    # Panel 1: Histogram comparison (top left)
    ax1 = CairoMakie.Axis(fig[1, 1], 
                title="Empirical vs Fitted Distribution",
                xlabel="Localizations per emitter", 
                ylabel="Frequency")
    
    hist!(ax1, all_counts, bins=0:maximum(all_counts)+1, 
          normalization=:pdf, color=(:blue, 0.6), label="Empirical")
    
    # Overlay fitted
    k_values = 0:maximum(all_counts)
    r = prior.κ
    p = prior.κ / (prior.κ + prior.μ)
    nb_dist = NegativeBinomial(r, p)
    pmf_values = [pdf(nb_dist, k) for k in k_values]
    scatter!(ax1, k_values, pmf_values, color=:red, markersize=6,
            label="Fitted NegBinom(μ=$(round(prior.μ,digits=2)), κ=$(round(prior.κ,digits=2)))")
    axislegend(ax1, position=:rt)
    
    # Panel 2: Q-Q plot (top right)
    if show_qq
        ax2 = CairoMakie.Axis(fig[1, 2], 
                    title="Q-Q Plot",
                    xlabel="Theoretical Quantiles", 
                    ylabel="Empirical Quantiles")
        
        # Compute quantiles
        sorted_counts = sort(all_counts)
        n = length(sorted_counts)
        theoretical_quantiles = Float64[]
        empirical_quantiles = Float64[]
        
        for i in 1:n
            p = (i - 0.5) / n
            # Find theoretical quantile
            q_theory = quantile(nb_dist, p)
            push!(theoretical_quantiles, q_theory)
            push!(empirical_quantiles, sorted_counts[i])
        end
        
        scatter!(ax2, theoretical_quantiles, empirical_quantiles, 
                markersize=4, color=:blue)
        
        # Add diagonal reference line
        lims = [minimum([theoretical_quantiles; empirical_quantiles]),
                maximum([theoretical_quantiles; empirical_quantiles])]
        lines!(ax2, lims, lims, color=:red, linestyle="-", linewidth=2)
    end
    
    # Panel 3: Residual plot (bottom left)
    if show_residuals
        ax3 = CairoMakie.Axis(fig[2, 1], 
                    title="Standardized Residuals",
                    xlabel="Count value", 
                    ylabel="Standardized Residual")
        
        # Compute residuals
        max_count = maximum(all_counts)
        count_vals = Int[]
        residuals = Float64[]
        
        # Get observed frequencies
        obs_freq = zeros(max_count + 1)
        for c in all_counts
            obs_freq[c + 1] += 1
        end
        obs_freq ./= length(all_counts)
        
        for k in 0:max_count
            expected = pdf(nb_dist, k)
            observed = obs_freq[k + 1]
            
            if expected > 0
                # Standardized residual
                std_resid = (observed - expected) / sqrt(expected * (1 - expected) / length(all_counts))
                push!(count_vals, k)
                push!(residuals, std_resid)
            end
        end
        
        scatter!(ax3, count_vals, residuals, markersize=6, color=:blue)
        hlines!(ax3, [-2, 0, 2], color=[:red, :black, :red], 
                linewidth=[1, 2, 1])
    end
    
    # Panel 4: Fit statistics (bottom right)
    ax4 = CairoMakie.Axis(fig[2, 2], title="Fit Statistics")
    hidedecorations!(ax4)
    hidespines!(ax4)
    
    # Compute fit statistics
    stats = assess_negbinomial_fit(all_counts, prior)
    
    # Create text summary
    text_lines = [
        "Goodness-of-Fit Tests:",
        "  χ² test: p = $(round(stats.chi_squared_pvalue, digits=3))",
        "  KS test: p = $(round(stats.ks_pvalue, digits=3))",
        "",
        "Moment Comparison:",
        "  Empirical: μ = $(round(stats.empirical_mean, digits=2)), σ² = $(round(stats.empirical_var, digits=2))",
        "  Fitted:    μ = $(round(stats.fitted_mean, digits=2)), σ² = $(round(stats.fitted_var, digits=2))",
        "",
        "PMF Errors:",
        "  Mean absolute: $(round(stats.mean_abs_error, digits=4))",
        "  Maximum: $(round(stats.max_abs_error, digits=4))",
        "",
        "Information Criteria:",
        "  AIC = $(round(stats.aic, digits=2))",
        "  BIC = $(round(stats.bic, digits=2))"
    ]
    
    # Add diagnosis if poor fit
    if stats.chi_squared_pvalue < 0.05 || stats.ks_pvalue < 0.05
        push!(text_lines, "")
        push!(text_lines, "⚠️ Fit Issues Detected")
    else
        push!(text_lines, "")
        push!(text_lines, "✓ Fit Appears Adequate")
    end
    
    text!(ax4, 0.1, 0.9, text=join(text_lines, "\n"),
          align=(:left, :top), fontsize=12, font="monospace")
    
    # Save if requested
    if !isnothing(filename)
        save(filename, fig, px_per_unit=2)
    end
    
    return fig
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
    initial_μ, initial_κ, initial_τ² = nothing, nothing, nothing
    final_μ, final_κ, final_τ² = nothing, nothing, nothing
    
    for chain in chains
        if !isempty(chain.hierarchical_history)
            if isnothing(initial_μ)
                first_update = chain.hierarchical_history[1]
                initial_μ, initial_κ, initial_τ² = first_update.μ, first_update.κ, first_update.τ²
            end
            last_update = chain.hierarchical_history[end]
            final_μ, final_κ, final_τ² = last_update.μ, last_update.κ, last_update.τ²
        end
    end
    
    # Calculate change
    μ_change = (final_μ - initial_μ) / initial_μ * 100
    κ_change = (final_κ - initial_κ) / initial_κ * 100
    τ²_change = (final_τ² - initial_τ²) / initial_τ² * 100
    
    return (
        enabled = true,
        n_updates = total_updates,
        initial_μ = initial_μ,
        initial_κ = initial_κ,
        initial_τ² = initial_τ²,
        final_μ = final_μ,
        final_κ = final_κ,
        final_τ² = final_τ²,
        μ_change_percent = μ_change,
        κ_change_percent = κ_change,
        τ²_change_percent = τ²_change,
        convergence = analyze_hierarchical_convergence(chains),
        message = "Hierarchical prior adapted from μ=$(round(initial_μ,digits=2))→$(round(final_μ,digits=2)), κ=$(round(initial_κ,digits=2))→$(round(final_κ,digits=2)), τ²=$(round(initial_τ²*1e6,digits=2))→$(round(final_τ²*1e6,digits=2)) nm²"
    )
end
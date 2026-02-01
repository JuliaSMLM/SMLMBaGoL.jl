# Extended Diagnostics for BaGoL Evaluation
# ==========================================
# Additional diagnostic plots and exports for AI-assisted evaluation.
#
# Usage:
#   include("viz_extended_diagnostics.jl")
#   plot_convergence_diagnostics(chain; save_path="convergence.png")
#   plot_uncertainty_calibration(emitters, true_positions; save_path="calibration.png")
#   plot_n_recovery_histogram(per_cluster_metrics; save_path="n_recovery.png")
#   write_diagnostic_json(filename, global_metrics, per_cluster_metrics, diagnostics, chain, ...)

using CairoMakie
using Statistics
using JSON
using SMLMData
using SMLMBaGoL: RJMCMCChain, BaGoLDiagnostics

# Include viz_metrics for match_positions
include(joinpath(@__DIR__, "viz_metrics.jl"))

"""
Plot convergence diagnostics: N_emitters trace, autocorrelation, ESS.

# Arguments
- `chain`: RJMCMCChain with samples
- `burn_in`: Number of iterations for burn-in (default 2000)
- `save_path`: Optional path to save figure
"""
function plot_convergence_diagnostics(
    chain::RJMCMCChain;
    burn_in::Int = 2000,
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(1200, 400))

    # Extract K (emitter count) from samples
    ks = [length(s.emitters) for s in chain.samples]
    n_samples = length(ks)

    # 1. K trace with burn-in marked
    ax1 = Axis(fig[1, 1], title="Emitter Count Trace",
               xlabel="Sample", ylabel="K")
    lines!(ax1, 1:n_samples, ks, color=:steelblue, linewidth=0.5)

    # Mark burn-in (convert iterations to samples)
    burn_in_sample = burn_in  # samples stored every iteration
    if burn_in_sample < n_samples
        vlines!(ax1, [burn_in_sample], color=:red, linestyle=:dash,
                linewidth=2, label="Burn-in")
    end

    # Add mean line (post burn-in)
    post_burn_ks = ks[burn_in_sample+1:end]
    if !isempty(post_burn_ks)
        hlines!(ax1, [mean(post_burn_ks)], color=:green, linestyle=:solid,
                linewidth=2, label="Mean = $(round(mean(post_burn_ks), digits=1))")
    end
    axislegend(ax1, position=:rt)

    # 2. Autocorrelation of K
    ax2 = Axis(fig[1, 2], title="Autocorrelation of K",
               xlabel="Lag", ylabel="Autocorrelation")

    if length(post_burn_ks) > 50
        max_lag = min(100, length(post_burn_ks) ÷ 4)
        acf = Float64[]
        k_centered = post_burn_ks .- mean(post_burn_ks)
        var_k = var(post_burn_ks)

        for lag in 0:max_lag
            if var_k > 0
                corr = sum(k_centered[1:end-lag] .* k_centered[lag+1:end]) / ((length(k_centered) - lag) * var_k)
                push!(acf, corr)
            else
                push!(acf, 0.0)
            end
        end

        barplot!(ax2, 0:max_lag, acf, color=:steelblue)
        hlines!(ax2, [0], color=:black, linewidth=1)
        hlines!(ax2, [1.96/sqrt(length(post_burn_ks)), -1.96/sqrt(length(post_burn_ks))],
                color=:red, linestyle=:dash, linewidth=1, label="95% CI")
    end

    # 3. ESS calculation and display
    ax3 = Axis(fig[1, 3], title="Effective Sample Size")
    hidedecorations!(ax3)
    hidespines!(ax3)

    ess = compute_ess(post_burn_ks)

    if !isnan(ess)
        text!(ax3, 0.5, 0.7, text="ESS(K) = $(round(Int, ess))",
              align=(:center, :center), fontsize=18)
        text!(ax3, 0.5, 0.5, text="N samples (post burn-in) = $(length(post_burn_ks))",
              align=(:center, :center), fontsize=14)
        text!(ax3, 0.5, 0.3, text="ESS ratio = $(round(ess/length(post_burn_ks), digits=2))",
              align=(:center, :center), fontsize=14)
    else
        text!(ax3, 0.5, 0.5, text="Insufficient samples\nfor ESS calculation",
              align=(:center, :center), fontsize=14, color=:gray)
    end

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

"""
Compute effective sample size using autocorrelation.
"""
function compute_ess(samples::Vector)
    if length(samples) < 50 || var(samples) <= 0
        return NaN
    end

    k_centered = samples .- mean(samples)
    var_k = var(samples)
    acf_vals = Float64[]

    for lag in 0:min(100, length(samples) ÷ 4)
        corr = sum(k_centered[1:end-lag] .* k_centered[lag+1:end]) / ((length(k_centered) - lag) * var_k)
        push!(acf_vals, corr)
        corr < 0 && break
    end

    sum_acf = sum(acf_vals[2:end])
    ess = length(samples) / (1 + 2 * max(0, sum_acf))
    return min(ess, length(samples))  # Cap at n
end

"""
Plot uncertainty calibration: σ_estimated vs actual_error, coverage probability.

# Arguments
- `emitters`: Vector of estimated emitters
- `true_positions`: Vector of (x, y) ground truth positions
- `threshold`: Maximum distance (μm) for matching (default 0.100)
- `save_path`: Optional path to save figure
"""
function plot_uncertainty_calibration(
    emitters::Vector{<:SMLMData.AbstractEmitter},
    true_positions::Vector{Tuple{Float64, Float64}};
    threshold::Float64 = 0.100,
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(1000, 400))

    # Match emitters to true positions
    assignments, cost, _ = match_positions(emitters, true_positions, threshold)

    # Collect (σ_est, actual_error) pairs for matched emitters
    σ_values = Float64[]
    errors = Float64[]

    for (i, j) in enumerate(assignments)
        if j > 0
            σ = sqrt(emitters[i].σ_x^2 + emitters[i].σ_y^2)
            err = cost[i, j]
            push!(σ_values, σ)
            push!(errors, err)
        end
    end

    if isempty(σ_values)
        @warn "No matched emitters for calibration"
        return fig
    end

    # 1. σ_estimated vs actual_error scatter
    ax1 = Axis(fig[1, 1], title="Uncertainty Calibration",
               xlabel="σ_estimated (nm)", ylabel="Actual error (nm)")

    scatter!(ax1, σ_values .* 1000, errors .* 1000, color=(:steelblue, 0.5), markersize=6)

    # Add y=x line
    max_val = max(maximum(σ_values), maximum(errors)) * 1000
    lines!(ax1, [0, max_val], [0, max_val], color=:red, linewidth=2, label="y = x")
    axislegend(ax1, position=:rb)

    # 2. Coverage plot
    ax2 = Axis(fig[1, 2], title="Coverage Probability",
               xlabel="Confidence level (σ)", ylabel="Empirical coverage")

    # Compute coverage at different σ levels
    σ_levels = [1.0, 2.0, 3.0]
    theoretical = [0.683, 0.954, 0.997]  # 2D Gaussian coverage
    empirical = Float64[]

    for level in σ_levels
        covered = sum(errors .< level .* σ_values)
        push!(empirical, covered / length(errors))
    end

    # Plot
    barplot!(ax2, 1:3, empirical, color=:steelblue, label="Empirical")
    scatter!(ax2, 1:3, theoretical, color=:red, markersize=12, label="Theoretical")

    ax2.xticks = (1:3, ["1σ", "2σ", "3σ"])
    axislegend(ax2, position=:rb)

    # Add calibration factor text
    if !isempty(errors) && !isempty(σ_values)
        cal_factor = median(errors ./ σ_values)
        text!(ax2, 0.05, 0.95, text="Cal. factor: $(round(cal_factor, digits=2))",
              align=(:left, :top), fontsize=12, space=:relative)
    end

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

"""
Plot N-recovery histogram showing (estimated - true) distribution.

# Arguments
- `per_cluster_metrics`: Vector of NamedTuples with `n_error` field
- `save_path`: Optional path to save figure
"""
function plot_n_recovery_histogram(
    per_cluster_metrics::Vector{<:NamedTuple};
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(600, 400))

    ax = Axis(fig[1, 1], title="N-Recovery Distribution",
              xlabel="N_estimated - N_true", ylabel="Count")

    n_errors = [m.n_error for m in per_cluster_metrics]

    # Compute bins centered on integers
    min_err = minimum(n_errors)
    max_err = maximum(n_errors)
    bins = (min_err - 0.5):(max_err + 0.5)

    hist!(ax, n_errors, bins=bins, color=:steelblue)

    # Mark zero line
    vlines!(ax, [0], color=:red, linestyle=:dash, linewidth=2, label="Correct")

    # Add statistics text
    n_correct = count(==(0), n_errors)
    accuracy = n_correct / length(n_errors)
    text!(ax, 0.95, 0.95, text="Accuracy: $(round(accuracy*100, digits=1))%\n($(n_correct)/$(length(n_errors)) correct)",
          align=(:right, :top), fontsize=12, space=:relative)

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

"""
Plot per-cluster metrics as boxplots with individual points.

# Arguments
- `per_cluster_metrics`: Vector of NamedTuples with jaccard, rmse, n_error fields
- `save_path`: Optional path to save figure
"""
function plot_per_cluster_metrics(
    per_cluster_metrics::Vector{<:NamedTuple};
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(1200, 400))

    # Extract metrics
    jaccards = [m.jaccard for m in per_cluster_metrics]
    rmses = [m.rmse * 1000 for m in per_cluster_metrics]  # Convert to nm
    n_errors = [m.n_error for m in per_cluster_metrics]
    valid_rmses = filter(!isnan, rmses)

    # 1. Jaccard boxplot
    ax1 = Axis(fig[1, 1], title="Jaccard Index per Cluster", ylabel="Value")
    boxplot!(ax1, ones(length(jaccards)), jaccards, color=:steelblue)
    scatter!(ax1, ones(length(jaccards)) .+ 0.15 .* (rand(length(jaccards)) .- 0.5),
             jaccards, color=(:black, 0.4), markersize=5)
    ax1.xticks = ([1], ["JI"])

    # 2. RMSE boxplot (in nm)
    ax2 = Axis(fig[1, 2], title="RMSE per Cluster", ylabel="nm")
    if !isempty(valid_rmses)
        boxplot!(ax2, ones(length(valid_rmses)), valid_rmses, color=:seagreen)
        scatter!(ax2, ones(length(valid_rmses)) .+ 0.15 .* (rand(length(valid_rmses)) .- 0.5),
                 valid_rmses, color=(:black, 0.4), markersize=5)
    end
    ax2.xticks = ([1], ["RMSE"])

    # 3. N-error histogram
    ax3 = Axis(fig[1, 3], title="N-Error per Cluster", xlabel="N_est - N_true", ylabel="Count")
    min_err = minimum(n_errors)
    max_err = maximum(n_errors)
    bins = (min_err - 0.5):(max_err + 0.5)
    hist!(ax3, n_errors, bins=bins, color=:darkorange)
    vlines!(ax3, [0], color=:red, linestyle=:dash, linewidth=2)

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

"""
Plot prior vs posterior comparison for μ and shape.

# Arguments
- `chain`: RJMCMCChain with samples
- `prior_μ`: Initial/expected μ value (e.g., from data)
- `prior_shape`: Initial shape value
- `save_path`: Optional path to save figure
"""
function plot_prior_posterior(
    chain::RJMCMCChain;
    prior_μ::Union{Float64, Nothing} = nothing,
    prior_shape::Float64 = 2.0,
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(1000, 400))

    # Extract μ and shape samples
    μ_samples = [s.μ for s in chain.samples]
    shape_samples = [s.shape for s in chain.samples]

    # Compute prior μ from chain if not provided
    if prior_μ === nothing
        prior_μ = μ_samples[1]
    end

    # 1. μ comparison
    ax1 = Axis(fig[1, 1], title="μ: Prior vs Posterior",
               xlabel="μ (locs/emitter)", ylabel="Density")

    hist!(ax1, μ_samples, bins=30, normalization=:pdf, color=(:steelblue, 0.7), label="Posterior")
    vlines!(ax1, [prior_μ], color=:red, linestyle=:dash, linewidth=2, label="Prior μ = $(round(prior_μ, digits=1))")
    vlines!(ax1, [mean(μ_samples)], color=:green, linewidth=2, label="Post. mean = $(round(mean(μ_samples), digits=1))")
    axislegend(ax1, position=:rt)

    # 2. Shape comparison (only if shape was learned)
    shape_learned = length(unique(shape_samples)) > 1

    ax2 = Axis(fig[1, 2], title="Shape: Prior vs Posterior",
               xlabel="Shape", ylabel="Density")

    if shape_learned
        hist!(ax2, shape_samples, bins=30, normalization=:pdf, color=(:darkorange, 0.7), label="Posterior")
        vlines!(ax2, [prior_shape], color=:red, linestyle=:dash, linewidth=2, label="Prior = $(prior_shape)")
        vlines!(ax2, [mean(shape_samples)], color=:green, linewidth=2, label="Post. mean = $(round(mean(shape_samples), digits=2))")
        axislegend(ax2, position=:rt)
    else
        text!(ax2, 0.5, 0.5, text="Shape fixed at $(shape_samples[1])",
              align=(:center, :center), fontsize=14, color=:gray)
    end

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

"""
Write diagnostic data to JSON for programmatic analysis.

# Arguments
- `filename`: Output JSON file path
- `global_metrics`: NamedTuple from compute_all_metrics
- `per_cluster_metrics`: Vector of per-cluster metric NamedTuples
- `diagnostics`: BaGoLDiagnostics struct
- `chain`: RJMCMCChain (for ESS calculation)
- `n_true_emitters`: Total number of true emitters
- `n_locs`: Total number of localizations
- `burn_in`: Burn-in iterations (default 2000)
"""
function write_diagnostic_json(
    filename::String,
    global_metrics::NamedTuple,
    per_cluster_metrics::Vector{<:NamedTuple},
    diagnostics::BaGoLDiagnostics,
    chain::RJMCMCChain,
    n_true_emitters::Int,
    n_locs::Int;
    burn_in::Int = 2000
)
    # Compute ESS
    ks = [length(s.emitters) for s in chain.samples]
    burn_in_sample = burn_in  # samples stored every iteration
    post_burn_ks = ks[burn_in_sample+1:end]
    ess = compute_ess(post_burn_ks)

    data = Dict(
        "global_metrics" => Dict(
            "n_estimated" => global_metrics.n_estimated,
            "n_true" => global_metrics.n_true,
            "n_matched" => global_metrics.n_matched,
            "jaccard" => global_metrics.jaccard,
            "precision" => global_metrics.precision,
            "recall" => global_metrics.recall,
            "f1" => global_metrics.f1,
            "rmse_nm" => isnan(global_metrics.rmse) ? nothing : global_metrics.rmse * 1000
        ),
        "per_cluster" => [
            Dict(
                "id" => i,
                "jaccard" => m.jaccard,
                "n_true" => m.n_true,
                "n_estimated" => m.n_estimated,
                "n_error" => m.n_error,
                "rmse_nm" => isnan(m.rmse) ? nothing : m.rmse * 1000
            )
            for (i, m) in enumerate(per_cluster_metrics)
        ],
        "hierarchical" => Dict(
            "final_mu" => diagnostics.final_μ,
            "final_shape" => diagnostics.final_shape
        ),
        "convergence" => Dict(
            "ess_n_emitters" => isnan(ess) ? nothing : round(Int, ess),
            "n_samples" => length(chain.samples),
            "burn_in" => burn_in
        ),
        "simulation" => Dict(
            "n_true_emitters" => n_true_emitters,
            "n_localizations" => n_locs,
            "mean_locs_per_emitter" => n_locs / n_true_emitters
        )
    )

    open(filename, "w") do io
        JSON.print(io, data, 2)
    end
end

"""
Write text diagnostic report.

# Arguments
- `filename`: Output file path
- `global_metrics`: NamedTuple from compute_all_metrics
- `per_cluster_metrics`: Vector of per-cluster metric NamedTuples
- `diagnostics`: BaGoLDiagnostics struct
- `n_true_emitters`: Total number of true emitters
- `n_locs`: Total number of localizations
- `title`: Report title (default "BaGoL Diagnostic Report")
"""
function write_diagnostic_report(
    filename::String,
    global_metrics::NamedTuple,
    per_cluster_metrics::Vector{<:NamedTuple},
    diagnostics::BaGoLDiagnostics,
    n_true_emitters::Int,
    n_locs::Int;
    title::String = "BaGoL Diagnostic Report"
)
    open(filename, "w") do io
        println(io, "="^60)
        println(io, title)
        println(io, "="^60)
        println(io)

        println(io, "SIMULATION")
        println(io, "-"^40)
        println(io, "True emitters: $n_true_emitters")
        println(io, "Localizations: $n_locs")
        println(io, "Mean locs/emitter: $(round(n_locs / n_true_emitters, digits=1))")
        println(io)

        println(io, "BAGOL RESULTS")
        println(io, "-"^40)
        println(io, "Partitions: $(diagnostics.n_partitions)")
        println(io, "Final μ: $(round(diagnostics.final_μ, digits=2))")
        println(io, "Final shape: $(round(diagnostics.final_shape, digits=2))")
        println(io)

        println(io, "GLOBAL METRICS")
        println(io, "-"^40)
        println(io, "Estimated emitters: $(global_metrics.n_estimated)")
        println(io, "True emitters: $(global_metrics.n_true)")
        println(io, "Matched: $(global_metrics.n_matched)")
        println(io)
        println(io, "Jaccard Index: $(round(global_metrics.jaccard, digits=3))")
        println(io, "Precision: $(round(global_metrics.precision, digits=3))")
        println(io, "Recall: $(round(global_metrics.recall, digits=3))")
        println(io, "F1 Score: $(round(global_metrics.f1, digits=3))")
        if !isnan(global_metrics.rmse)
            println(io, "RMSE: $(round(global_metrics.rmse * 1000, digits=1)) nm")
        end
        println(io)

        println(io, "ACCEPTANCE RATES")
        println(io, "-"^40)
        for (move, rate) in diagnostics.acceptance_rates
            println(io, "  $move: $(round(rate * 100, digits=1))%")
        end
        println(io)

        println(io, "PER-CLUSTER STATISTICS")
        println(io, "-"^40)
        jaccards = [m.jaccard for m in per_cluster_metrics]
        rmses = [m.rmse * 1000 for m in per_cluster_metrics if !isnan(m.rmse)]
        n_errors = [m.n_error for m in per_cluster_metrics]

        println(io, "Jaccard Index:")
        println(io, "  Mean: $(round(mean(jaccards), digits=3))")
        println(io, "  Std:  $(round(std(jaccards), digits=3))")
        println(io, "  Min:  $(round(minimum(jaccards), digits=3))")
        println(io, "  Max:  $(round(maximum(jaccards), digits=3))")
        println(io)

        if !isempty(rmses)
            println(io, "RMSE (nm):")
            println(io, "  Mean: $(round(mean(rmses), digits=1))")
            println(io, "  Std:  $(round(std(rmses), digits=1))")
            println(io, "  Min:  $(round(minimum(rmses), digits=1))")
            println(io, "  Max:  $(round(maximum(rmses), digits=1))")
            println(io)
        end

        println(io, "N-Recovery:")
        n_correct = count(==(0), n_errors)
        println(io, "  Correct (N_error=0): $(n_correct)/$(length(n_errors)) ($(round(n_correct/length(n_errors)*100, digits=1))%)")
        println(io, "  Mean N-error: $(round(mean(n_errors), digits=2))")
        println(io, "  N-error range: [$(minimum(n_errors)), $(maximum(n_errors))]")
        println(io)

        println(io, "="^60)
    end
end

"""
Write failure mode catalog listing problematic clusters.

# Arguments
- `filename`: Output file path
- `per_cluster_metrics`: Vector of per-cluster metric NamedTuples
- `cluster_info`: Optional Vector of (center_x, center_y) for each cluster
- `threshold`: N-error threshold for failure (default 1)
"""
function write_failure_modes(
    filename::String,
    per_cluster_metrics::Vector{<:NamedTuple};
    cluster_info::Vector{Tuple{Float64, Float64}} = Tuple{Float64, Float64}[],
    threshold::Int = 1
)
    failures = [(i, m) for (i, m) in enumerate(per_cluster_metrics) if abs(m.n_error) > threshold]

    open(filename, "w") do io
        println(io, "Failure Mode Catalog")
        println(io, "="^50)
        println(io, "Threshold: |N_error| > $threshold")
        println(io, "Total failures: $(length(failures)) / $(length(per_cluster_metrics))")
        println(io)

        if isempty(failures)
            println(io, "No failures detected.")
            return
        end

        println(io, "FAILURE DETAILS")
        println(io, "-"^50)

        for (idx, m) in failures
            println(io)
            println(io, "Cluster $idx:")
            println(io, "  N_true: $(m.n_true)")
            println(io, "  N_estimated: $(m.n_estimated)")
            println(io, "  N_error: $(m.n_error)")
            println(io, "  Jaccard: $(round(m.jaccard, digits=3))")
            if !isnan(m.rmse)
                println(io, "  RMSE: $(round(m.rmse * 1000, digits=1)) nm")
            end
            if !isempty(cluster_info) && idx <= length(cluster_info)
                cx, cy = cluster_info[idx]
                println(io, "  Center: ($(round(cx, digits=3)), $(round(cy, digits=3))) μm")
            end
        end

        println(io)
        println(io, "="^50)

        # Summary statistics
        under = count(m -> m.n_error < -threshold, per_cluster_metrics)
        over = count(m -> m.n_error > threshold, per_cluster_metrics)
        println(io, "Under-estimates (N_error < -$threshold): $under")
        println(io, "Over-estimates (N_error > $threshold): $over")
    end
end

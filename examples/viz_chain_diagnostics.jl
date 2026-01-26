# BaGoL Chain Diagnostics Visualization
# ======================================
# Reference visualization functions for BaGoL chain diagnostics.
# These are example implementations - copy/adapt for your needs.
#
# Usage: include("viz_chain_diagnostics.jl") in your script
#
# Functions:
#   plot_bagol(chain, emitters, posterior_k, locs; true_positions, save_path)
#   plot_mapn(emitters, posterior_k, locs; true_positions, save_path)
#   plot_hierarchical_diagnostics(chain; true_locs_per_emitter, save_path)

using CairoMakie
using Statistics
using Distributions: NegativeBinomial, pdf
using SMLMData
using SMLMBaGoL: RJMCMCChain

"""
Draw a circle at (x, y) with radius r.
"""
function draw_circle!(ax, x, y, r; color=:black, linewidth=1.0, alpha=1.0)
    θ = range(0, 2π, length=50)
    cx = x .+ r .* cos.(θ)
    cy = y .+ r .* sin.(θ)
    lines!(ax, cx, cy, color=(color, alpha), linewidth=linewidth)
end

"""
Draw an X marker at (x, y) with half-size r.
"""
function draw_x!(ax, x, y, r; color=:blue, linewidth=2.0, alpha=1.0)
    lines!(ax, [x - r, x + r], [y + r, y - r], color=(color, alpha), linewidth=linewidth)
    lines!(ax, [x - r, x + r], [y - r, y + r], color=(color, alpha), linewidth=linewidth)
end

"""
Plot BaGoL results with proper visualization:
- Localizations as 1σ circles (gray)
- Chain emitter positions as scatter (light red)
- MAP-N emitters as 1σ circles (red)
- True positions as X markers (blue) if provided

# Arguments
- `chain`: RJMCMCChain for showing sample positions
- `emitters`: Vector of Emitter2DFit from estimate_mapn
- `posterior_k`: Histogram of K from estimate_mapn
- `locs`: Input localizations
- `true_positions`: Optional ground truth positions
- `save_path`: Optional path to save figure
"""
function plot_bagol(
    chain::RJMCMCChain,
    emitters::Vector{SMLMData.Emitter2DFit},
    posterior_k::Vector{Int},
    locs::Vector{<:SMLMData.AbstractEmitter};
    true_positions::Vector{Tuple{Float64, Float64}} = Tuple{Float64, Float64}[],
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(1200, 500))

    # Left: Main visualization
    ax1 = Axis(fig[1, 1], title="BaGoL Results",
               xlabel="x (μm)", ylabel="y (μm)",
               aspect=DataAspect())

    # 1. Localizations as 1σ circles (gray)
    for loc in locs
        σ = mean([loc.σ_x, loc.σ_y])
        draw_circle!(ax1, loc.x, loc.y, σ; color=:gray, linewidth=0.5, alpha=0.4)
    end

    # 2. Chain emitter positions as scatter (all samples)
    chain_xs = Float64[]
    chain_ys = Float64[]
    for sample in chain.samples
        for emitter in sample.emitters
            push!(chain_xs, emitter.x)
            push!(chain_ys, emitter.y)
        end
    end
    if !isempty(chain_xs)
        scatter!(ax1, chain_xs, chain_ys,
            color=(:red, 0.05), markersize=3, label="Chain samples")
    end

    # 3. MAP-N emitters as 1σ circles (red)
    for e in emitters
        σ = mean([e.σ_x, e.σ_y])
        if σ > 0
            draw_circle!(ax1, e.x, e.y, σ; color=:red, linewidth=2.0)
        end
        scatter!(ax1, [e.x], [e.y], color=:red, markersize=8)
    end

    # 4. True positions as X (blue) - size based on median loc sigma
    if !isempty(true_positions)
        median_σ = median([mean([loc.σ_x, loc.σ_y]) for loc in locs])
        for (tx, ty) in true_positions
            draw_x!(ax1, tx, ty, median_σ / 2; color=:blue, linewidth=2.0)
        end
    end

    # Right: Posterior on K
    ax2 = Axis(fig[1, 2], title="Posterior P(K)",
               xlabel="Number of emitters", ylabel="Probability")

    k_vals = 0:(length(posterior_k) - 1)
    probs = posterior_k ./ sum(posterior_k)
    barplot!(ax2, k_vals, probs, color=:steelblue)

    # Mark true K and MAP-N
    n_emitters = length(emitters)
    if !isempty(true_positions)
        vlines!(ax2, [length(true_positions)], color=:blue,
            linestyle=:dash, linewidth=2, label="True K")
    end
    vlines!(ax2, [n_emitters], color=:red,
        linestyle=:solid, linewidth=2, label="MAP-N = $n_emitters")
    axislegend(ax2, position=:rt)

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

"""
Simple plot without chain samples (for quick visualization).

# Arguments
- `emitters`: Vector of Emitter2DFit from estimate_mapn
- `posterior_k`: Histogram of K from estimate_mapn
- `locs`: Input localizations
- `true_positions`: Optional ground truth positions
- `save_path`: Optional path to save figure
"""
function plot_mapn(
    emitters::Vector{SMLMData.Emitter2DFit},
    posterior_k::Vector{Int},
    locs::Vector{<:SMLMData.AbstractEmitter};
    true_positions::Vector{Tuple{Float64, Float64}} = Tuple{Float64, Float64}[],
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(1200, 500))

    ax1 = Axis(fig[1, 1], title="Localizations + MAP-N Emitters",
               xlabel="x (μm)", ylabel="y (μm)",
               aspect=DataAspect())

    # Localizations as 1σ circles
    for loc in locs
        σ = mean([loc.σ_x, loc.σ_y])
        draw_circle!(ax1, loc.x, loc.y, σ; color=:gray, linewidth=0.5, alpha=0.4)
    end

    # MAP-N emitters as 1σ circles
    for e in emitters
        σ = mean([e.σ_x, e.σ_y])
        if σ > 0
            draw_circle!(ax1, e.x, e.y, σ; color=:red, linewidth=2.0)
        end
        scatter!(ax1, [e.x], [e.y], color=:red, markersize=8)
    end

    # True positions as X - size based on median loc sigma
    if !isempty(true_positions)
        median_σ = median([mean([loc.σ_x, loc.σ_y]) for loc in locs])
        for (tx, ty) in true_positions
            draw_x!(ax1, tx, ty, median_σ / 2; color=:blue, linewidth=2.0)
        end
    end

    # Posterior on K
    ax2 = Axis(fig[1, 2], title="Posterior P(K)",
               xlabel="Number of emitters", ylabel="Probability")

    k_vals = 0:(length(posterior_k) - 1)
    probs = posterior_k ./ sum(posterior_k)
    barplot!(ax2, k_vals, probs, color=:steelblue)

    n_emitters = length(emitters)
    if !isempty(true_positions)
        vlines!(ax2, [length(true_positions)], color=:blue,
            linestyle=:dash, linewidth=2, label="True K")
    end
    vlines!(ax2, [n_emitters], color=:red,
        linestyle=:solid, linewidth=2, label="MAP-N")
    axislegend(ax2, position=:rt)

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

"""
Plot hierarchical Bayes diagnostics:
- μ trace over iterations (and α if learned)
- μ posterior histogram (and α if learned)
- Locs/emitter distribution: true vs estimated vs NegBin model

# Arguments
- `chain`: RJMCMCChain with samples
- `true_locs_per_emitter`: Vector of true counts (from simulation), or nothing
- `save_path`: Optional path to save figure
"""
function plot_hierarchical_diagnostics(
    chain::RJMCMCChain;
    true_locs_per_emitter::Union{Vector{Int}, Nothing} = nothing,
    save_path::Union{String, Nothing} = nothing
)
    # Extract μ and α values from samples
    μ_samples = [s.μ for s in chain.samples]
    α_samples = [s.α for s in chain.samples]

    if isempty(μ_samples)
        @warn "No samples in chain"
        return Figure()
    end

    # Check if α was learned (varies across samples)
    α_learned = length(unique(α_samples)) > 1

    # Determine layout based on whether α was learned
    if α_learned
        fig = Figure(size=(1400, 700))
    else
        fig = Figure(size=(1400, 400))
    end

    # Row 1: μ diagnostics
    # 1. μ trace plot
    ax1 = Axis(fig[1, 1], title="μ Trace",
               xlabel="Sample", ylabel="μ (locs/emitter)")
    lines!(ax1, 1:length(μ_samples), μ_samples, color=:steelblue)

    # Add true μ if we have true counts
    if true_locs_per_emitter !== nothing && !isempty(true_locs_per_emitter)
        true_μ = mean(true_locs_per_emitter)
        hlines!(ax1, [true_μ], color=:red, linestyle=:dash,
            linewidth=2, label="True μ = $(round(true_μ, digits=1))")
        axislegend(ax1, position=:rt)
    end

    # 2. μ posterior histogram
    ax2 = Axis(fig[1, 2], title="μ Posterior",
               xlabel="μ (locs/emitter)", ylabel="Density")
    hist!(ax2, μ_samples, bins=30, normalization=:pdf, color=:steelblue)

    μ_mean = mean(μ_samples)
    vlines!(ax2, [μ_mean], color=:black, linewidth=2,
        label="Mean = $(round(μ_mean, digits=1))")

    if true_locs_per_emitter !== nothing && !isempty(true_locs_per_emitter)
        true_μ = mean(true_locs_per_emitter)
        vlines!(ax2, [true_μ], color=:red, linestyle=:dash,
            linewidth=2, label="True = $(round(true_μ, digits=1))")
    end
    axislegend(ax2, position=:rt)

    # 3. Locs/emitter count distribution
    ax3 = Axis(fig[1, 3], title="Locs/Emitter Distribution",
               xlabel="Count", ylabel="Probability")

    # Get allocation counts from chain samples
    all_counts = Int[]
    for sample in chain.samples
        for emitter in sample.emitters
            push!(all_counts, length(emitter.allocated))
        end
    end

    if !isempty(all_counts)
        max_count = max(maximum(all_counts),
                        true_locs_per_emitter !== nothing ? maximum(true_locs_per_emitter) : 0)
        bins = 0:(max_count + 1)

        # Estimated histogram (from chain)
        hist!(ax3, all_counts, bins=bins, normalization=:probability,
            color=(:steelblue, 0.6), label="Estimated")

        # True histogram (if provided)
        if true_locs_per_emitter !== nothing && !isempty(true_locs_per_emitter)
            hist!(ax3, true_locs_per_emitter, bins=bins, normalization=:probability,
                color=(:red, 0.4), label="True")
        end

        # NegBin model curve with posterior mean μ and α
        α_post = mean(α_samples)
        μ_post = mean(μ_samples)
        p = α_post / (α_post + μ_post)
        negbin = NegativeBinomial(α_post, p)

        x_model = 0:max_count
        y_model = [pdf(negbin, k) for k in x_model]
        lines!(ax3, x_model, y_model, color=:black, linewidth=2,
            label="NegBin(α=$(round(α_post, digits=1)), μ=$(round(μ_post, digits=1)))")

        axislegend(ax3, position=:rt)
    end

    # Row 2: α diagnostics (only if α was learned)
    if α_learned
        # 4. α trace plot
        ax4 = Axis(fig[2, 1], title="α Trace",
                   xlabel="Sample", ylabel="α (shape)")
        lines!(ax4, 1:length(α_samples), α_samples, color=:darkorange)

        # 5. α posterior histogram
        ax5 = Axis(fig[2, 2], title="α Posterior",
                   xlabel="α (shape)", ylabel="Density")
        hist!(ax5, α_samples, bins=30, normalization=:pdf, color=:darkorange)

        α_mean = mean(α_samples)
        vlines!(ax5, [α_mean], color=:black, linewidth=2,
            label="Mean = $(round(α_mean, digits=2))")
        axislegend(ax5, position=:rt)

        # 6. Interpretation panel
        ax6 = Axis(fig[2, 3], title="α Interpretation",
                   xlabel="α value", ylabel="")
        hidedecorations!(ax6, label=false, ticklabels=false, ticks=false)

        # Show where α falls on the scale
        α_mean = mean(α_samples)
        text!(ax6, 0.5, 0.8, text="Posterior mean: α = $(round(α_mean, digits=2))",
            align=(:center, :center), fontsize=14)

        if α_mean < 1.5
            interp = "Exponential-like (dSTORM/photobleaching)"
        elseif α_mean > 5.0
            interp = "Poisson-like (DNA-PAINT/constant rate)"
        else
            interp = "Intermediate heterogeneity"
        end
        text!(ax6, 0.5, 0.5, text=interp,
            align=(:center, :center), fontsize=12)

        # Add reference lines
        text!(ax6, 0.5, 0.2, text="α ≈ 1: Exponential | α → ∞: Poisson",
            align=(:center, :center), fontsize=10, color=:gray)
    end

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

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
using Distributions: Gamma, pdf, cdf
using Hungarian
using SMLMData
using SMLMBaGoL: RJMCMCChain, BaGoLSample

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
Draw an ellipse at (x, y) with semi-axes rx, ry.
"""
function draw_ellipse!(ax, x, y, rx, ry; color=:black, linewidth=1.0, alpha=1.0)
    θ = range(0, 2π, length=50)
    ex = x .+ rx .* cos.(θ)
    ey = y .+ ry .* sin.(θ)
    lines!(ax, ex, ey, color=(color, alpha), linewidth=linewidth)
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
               aspect=DataAspect(), yreversed=true)

    # 1. Localizations as 1σ circles (gray)
    for loc in locs
        σ = mean([loc.σ_x, loc.σ_y])
        draw_circle!(ax1, loc.x, loc.y, σ; color=:gray, linewidth=1.0, alpha=0.6)
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

    # 3. MAP-N emitters as 1σ ellipses (red)
    for e in emitters
        if e.σ_x > 0 && e.σ_y > 0
            draw_ellipse!(ax1, e.x, e.y, e.σ_x, e.σ_y; color=:red, linewidth=2.0)
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
               aspect=DataAspect(), yreversed=true)

    # Localizations as 1σ circles
    for loc in locs
        σ = mean([loc.σ_x, loc.σ_y])
        draw_circle!(ax1, loc.x, loc.y, σ; color=:gray, linewidth=1.0, alpha=0.6)
    end

    # MAP-N emitters as 1σ ellipses
    for e in emitters
        if e.σ_x > 0 && e.σ_y > 0
            draw_ellipse!(ax1, e.x, e.y, e.σ_x, e.σ_y; color=:red, linewidth=2.0)
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
- μ trace over iterations (and shape if learned)
- μ posterior histogram (and shape if learned)
- Locs/emitter distribution: true vs estimated vs Gamma model

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
    # Extract μ and shape values from samples
    μ_samples = [s.μ for s in chain.samples]
    shape_samples = [s.shape for s in chain.samples]

    if isempty(μ_samples)
        @warn "No samples in chain"
        return Figure()
    end

    # Check if shape was learned (varies across samples)
    shape_learned = length(unique(shape_samples)) > 1

    # Determine layout based on whether shape was learned
    if shape_learned
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

    # 3. Locs/emitter count distribution (last hierarchical chunk only)
    # Use only samples from the last hierarchical update period, where μ/shape were fixed
    ax3 = Axis(fig[1, 3], title="Locs/Emitter (last chunk)",
               xlabel="Count", ylabel="Probability")

    # Find samples from the last hierarchical chunk
    # These all share the same μ/shape (the final values)
    n_samples = length(chain.samples)
    hierarchical_interval = chain.config.hierarchical_interval
    chunk_size = min(hierarchical_interval, n_samples)
    last_chunk_samples = chain.samples[end-chunk_size+1:end]

    # Get counts from last chunk only
    chunk_counts = Int[]
    for sample in last_chunk_samples
        for emitter in sample.emitters
            push!(chunk_counts, length(emitter.allocated))
        end
    end

    if !isempty(chunk_counts)
        max_count = max(maximum(chunk_counts),
                        true_locs_per_emitter !== nothing ? maximum(true_locs_per_emitter) : 0)
        bins = 0:(max_count + 1)

        # Estimated histogram (from last chunk)
        hist!(ax3, chunk_counts, bins=bins, normalization=:probability,
            color=(:steelblue, 0.6), label="Last chunk")

        # True histogram (if provided)
        if true_locs_per_emitter !== nothing && !isempty(true_locs_per_emitter)
            hist!(ax3, true_locs_per_emitter, bins=bins, normalization=:probability,
                color=(:red, 0.4), label="True")
        end

        # 1. Prior curve: Gamma using μ/shape that were active during this chunk
        prior_μ = last_chunk_samples[end].μ
        prior_shape = last_chunk_samples[end].shape
        prior_scale = prior_μ / prior_shape
        prior_dist = Gamma(prior_shape, prior_scale)

        x_model = 1:max_count
        y_prior = [cdf(prior_dist, k + 0.5) - cdf(prior_dist, k - 0.5) for k in x_model]
        lines!(ax3, x_model, y_prior, color=:black, linewidth=2,
            label="Prior(shape=$(round(prior_shape, digits=1)), μ=$(round(prior_μ, digits=1)))")

        # 2. Empirical fit: Gamma with parameters from observed counts (method of moments)
        emp_μ = mean(chunk_counts)
        emp_var = var(chunk_counts)
        if emp_var > 0 && emp_μ > 0
            emp_shape = emp_μ^2 / emp_var
            emp_scale = emp_var / emp_μ
            emp_dist = Gamma(emp_shape, emp_scale)

            y_emp = [cdf(emp_dist, k + 0.5) - cdf(emp_dist, k - 0.5) for k in x_model]
            lines!(ax3, x_model, y_emp, color=:green, linewidth=2, linestyle=:dash,
                label="Fit(shape=$(round(emp_shape, digits=1)), μ=$(round(emp_μ, digits=1)))")
        end

        axislegend(ax3, position=:rt)
    end

    # Row 2: shape diagnostics (only if shape was learned)
    if shape_learned
        # 4. shape trace plot
        ax4 = Axis(fig[2, 1], title="Shape Trace",
                   xlabel="Sample", ylabel="shape")
        lines!(ax4, 1:length(shape_samples), shape_samples, color=:darkorange)

        # 5. shape posterior histogram
        ax5 = Axis(fig[2, 2], title="Shape Posterior",
                   xlabel="shape", ylabel="Density")
        hist!(ax5, shape_samples, bins=30, normalization=:pdf, color=:darkorange)

        shape_mean = mean(shape_samples)
        vlines!(ax5, [shape_mean], color=:black, linewidth=2,
            label="Mean = $(round(shape_mean, digits=2))")
        axislegend(ax5, position=:rt)

        # 6. Interpretation panel
        ax6 = Axis(fig[2, 3], title="Shape Interpretation",
                   xlabel="shape value", ylabel="")
        hidedecorations!(ax6, label=false, ticklabels=false, ticks=false)

        # Show where shape falls on the scale
        shape_mean = mean(shape_samples)
        cv = 1.0 / sqrt(shape_mean)
        text!(ax6, 0.5, 0.8, text="Posterior mean: shape = $(round(shape_mean, digits=2)) (CV = $(round(cv, digits=2)))",
            align=(:center, :center), fontsize=14)

        if shape_mean < 1.5
            interp = "Exponential-like (dSTORM/photobleaching)"
        elseif shape_mean > 5.0
            interp = "Peaked (DNA-PAINT/constant rate)"
        else
            interp = "Intermediate spread"
        end
        text!(ax6, 0.5, 0.5, text=interp,
            align=(:center, :center), fontsize=12)

        # Add reference lines
        text!(ax6, 0.5, 0.2, text="shape ≈ 1: Exponential | shape → ∞: Delta",
            align=(:center, :center), fontsize=10, color=:gray)
    end

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

"""
Plot move type histogram showing proposed vs accepted counts.

# Arguments
- `chain`: RJMCMCChain with acceptance statistics
- `save_path`: Optional path to save figure
"""
function plot_move_histogram(
    chain::RJMCMCChain;
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(700, 400))

    ax = Axis(fig[1, 1], title="RJMCMC Move Statistics",
              xlabel="Move Type", ylabel="Count")

    move_types = [:birth, :death, :move, :allocate]
    proposed = [chain.acceptance[m][2] for m in move_types]
    accepted = [chain.acceptance[m][1] for m in move_types]

    # X positions for grouped bars
    x = 1:length(move_types)
    bar_width = 0.35

    # Proposed (blue) and Accepted (green)
    barplot!(ax, x .- bar_width/2, proposed,
        color=:steelblue, label="Proposed", width=bar_width)
    barplot!(ax, x .+ bar_width/2, accepted,
        color=:seagreen, label="Accepted", width=bar_width)

    # Add acceptance rate labels
    for (i, (p, a)) in enumerate(zip(proposed, accepted))
        rate = p > 0 ? round(100 * a / p, digits=1) : 0.0
        text!(ax, i, max(p, a) * 1.05, text="$(rate)%",
            align=(:center, :bottom), fontsize=10)
    end

    ax.xticks = (x, string.(move_types))
    axislegend(ax, position=:rt)

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

"""
Plot 2D posterior density of emitter positions from chain samples.

# Arguments
- `chain`: RJMCMCChain with samples
- `bins`: Tuple of (n_x_bins, n_y_bins) for 2D histogram (default: (100, 100))
- `save_path`: Optional path to save figure
"""
function plot_posterior_density(
    chain::RJMCMCChain;
    bins::Tuple{Int, Int} = (100, 100),
    save_path::Union{String, Nothing} = nothing
)
    # Collect all emitter positions from samples
    xs = Float64[]
    ys = Float64[]
    for sample in chain.samples
        for emitter in sample.emitters
            push!(xs, emitter.x)
            push!(ys, emitter.y)
        end
    end

    if isempty(xs)
        @warn "No emitter positions in chain samples"
        return Figure()
    end

    fig = Figure(size=(600, 550))

    ax = Axis(fig[1, 1], title="Posterior Position Density",
              xlabel="x (μm)", ylabel="y (μm)",
              aspect=DataAspect(), yreversed=true)

    # 2D histogram with colorbar
    hm = hexbin!(ax, xs, ys, bins=bins[1],
        colormap=:viridis)
    Colorbar(fig[1, 2], hm, label="Count")

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

"""
Plot side-by-side comparison of standard vs topology-preserving MAP-N.

Shows both estimates with their uncertainty circles, highlighting the
difference in uncertainty estimates between the two methods.

# Arguments
- `emitters_std`: Emitters from standard `estimate_mapn`
- `emitters_topo`: Emitters from `estimate_mapn_topology`
- `posterior_k`: Histogram of K from either method
- `locs`: Input localizations
- `true_positions`: Optional ground truth positions
- `save_path`: Optional path to save figure
"""
function plot_mapn_comparison(
    emitters_std::Vector{<:SMLMData.AbstractEmitter},
    emitters_topo::Vector{<:SMLMData.AbstractEmitter},
    posterior_k::Vector{Int},
    locs::Vector{<:SMLMData.AbstractEmitter};
    true_positions::Vector{Tuple{Float64, Float64}} = Tuple{Float64, Float64}[],
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(1200, 500))

    # Compute bounds from localizations
    xs_loc = [loc.x for loc in locs]
    ys_loc = [loc.y for loc in locs]
    margin = 0.05
    x_range = (minimum(xs_loc) - margin, maximum(xs_loc) + margin)
    y_range = (minimum(ys_loc) - margin, maximum(ys_loc) + margin)

    titles = ["Standard Hungarian", "Iterative (median-based)"]
    emitter_sets = [emitters_std, emitters_topo]

    for (col, (title, emitters)) in enumerate(zip(titles, emitter_sets))
        ax = Axis(fig[1, col], title=title,
                  xlabel="x (μm)", ylabel="y (μm)",
                  aspect=DataAspect(), yreversed=true)
        xlims!(ax, x_range)
        ylims!(ax, y_range)

        # Draw localizations (gray circles at 1σ)
        for loc in locs
            σ = mean([loc.σ_x, loc.σ_y])
            draw_circle!(ax, loc.x, loc.y, σ; color=:gray, linewidth=1.0, alpha=0.6)
        end

        # Draw true positions
        if !isempty(true_positions)
            for (tx, ty) in true_positions
                draw_x!(ax, tx, ty, 0.005; color=:blue, linewidth=2.0)
            end
        end

        # Draw MAP-N emitters with uncertainty ellipses
        for e in emitters
            scatter!(ax, [e.x], [e.y], color=:red, markersize=8)
            draw_ellipse!(ax, e.x, e.y, e.σ_x, e.σ_y; color=:red, linewidth=2.0, alpha=0.8)
        end

        # Add mean uncertainty label
        σ_mean = mean([sqrt(e.σ_x^2 + e.σ_y^2) for e in emitters])
        text!(ax, 0.02, 0.98, text="mean σ = $(round(σ_mean*1000, digits=1)) nm",
            align=(:left, :top), fontsize=12, space=:relative,
            color=:red)
    end

    # Right panel: uncertainty comparison bar chart
    ax_bar = Axis(fig[1, 3], title="Uncertainty Comparison",
                  xlabel="Emitter", ylabel="σ (nm)")

    σ_std = [sqrt(e.σ_x^2 + e.σ_y^2) * 1000 for e in emitters_std]
    σ_topo = [sqrt(e.σ_x^2 + e.σ_y^2) * 1000 for e in emitters_topo]

    n_emitters = min(length(σ_std), length(σ_topo))
    x = 1:n_emitters
    bar_width = 0.35

    barplot!(ax_bar, x .- bar_width/2, σ_std[1:n_emitters],
        color=:steelblue, label="Standard", width=bar_width)
    barplot!(ax_bar, x .+ bar_width/2, σ_topo[1:n_emitters],
        color=:seagreen, label="Topology", width=bar_width)

    axislegend(ax_bar, position=:rt)

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

# Color palette for matched position clusters
const CLUSTER_COLORS = [
    colorant"#e41a1c",  # red
    colorant"#377eb8",  # blue
    colorant"#4daf4a",  # green
    colorant"#984ea3",  # purple
    colorant"#ff7f00",  # orange
    colorant"#ffff33",  # yellow
    colorant"#a65628",  # brown
    colorant"#f781bf",  # pink
    colorant"#999999",  # gray
    colorant"#66c2a5",  # teal
    colorant"#fc8d62",  # salmon
    colorant"#8da0cb",  # periwinkle
]

"""
Visualize the matched positions used to compute MAP-N μ and σ.

Shows a scatter plot where each point is a position from a chain sample,
colored by the emitter identity assigned via Hungarian matching. This reveals
label switching: if an emitter's cluster has points from multiple true positions,
Hungarian assigned them incorrectly.

# Arguments
- `chain`: RJMCMCChain with samples
- `true_positions`: Optional ground truth for comparison
- `save_path`: Optional path to save figure

# Returns
- `fig`: The figure
- `matched_positions`: Vector of position vectors for each emitter (for further analysis)
"""
function plot_mapn_matched_positions(
    chain::RJMCMCChain;
    true_positions::Vector{Tuple{Float64, Float64}} = Tuple{Float64, Float64}[],
    save_path::Union{String, Nothing} = nothing
)
    samples = chain.samples

    if isempty(samples)
        @warn "No samples in chain"
        return Figure(), Vector{Tuple{Float64,Float64}}[]
    end

    # Find MAP-N
    ks = [length(s.emitters) for s in samples]
    k_max = maximum(ks)
    posterior_k = zeros(Int, k_max + 1)
    for k in ks
        posterior_k[k + 1] += 1
    end
    map_n = argmax(posterior_k) - 1

    if map_n == 0
        @warn "MAP-N is 0"
        return Figure(), Vector{Tuple{Float64,Float64}}[]
    end

    # Get MAP-N samples
    map_samples = filter(s -> length(s.emitters) == map_n, samples)

    # Use first sample as reference (same as estimate_mapn)
    reference = map_samples[1]

    # Collect matched positions
    matched_positions = [Vector{Tuple{Float64, Float64}}() for _ in 1:map_n]

    for sample in map_samples
        # Build cost matrix
        cost = zeros(map_n, map_n)
        for i in 1:map_n
            for j in 1:map_n
                ref_e = reference.emitters[i]
                samp_e = sample.emitters[j]
                cost[i, j] = (ref_e.x - samp_e.x)^2 + (ref_e.y - samp_e.y)^2
            end
        end

        # Hungarian matching
        assignment, _ = Hungarian.hungarian(cost)

        # Record matched positions
        for (i, j) in enumerate(assignment)
            if j <= length(sample.emitters)
                e = sample.emitters[j]
                push!(matched_positions[i], (Float64(e.x), Float64(e.y)))
            end
        end
    end

    # Compute stats for each cluster
    cluster_stats = NamedTuple[]
    for (i, positions) in enumerate(matched_positions)
        if !isempty(positions)
            xs = [p[1] for p in positions]
            ys = [p[2] for p in positions]
            push!(cluster_stats, (
                id = i,
                n = length(positions),
                μx = mean(xs),
                μy = mean(ys),
                σx = std(xs),
                σy = std(ys),
                σ = sqrt(std(xs)^2 + std(ys)^2)
            ))
        end
    end

    # Create figure
    fig = Figure(size=(900, 800))

    ax = Axis(fig[1, 1],
        title="MAP-N Matched Positions (K=$map_n, $(length(map_samples)) samples)",
        xlabel="x (μm)", ylabel="y (μm)",
        aspect=DataAspect(),
        yreversed=true)

    # Plot each cluster with different color
    for (i, positions) in enumerate(matched_positions)
        if isempty(positions)
            continue
        end
        xs = [p[1] for p in positions]
        ys = [p[2] for p in positions]
        color = CLUSTER_COLORS[mod1(i, length(CLUSTER_COLORS))]

        # Scatter with transparency to show density
        scatter!(ax, xs, ys, color=(color, 0.3), markersize=3)

        # Mark the mean (MAP-N position)
        μx, μy = mean(xs), mean(ys)
        scatter!(ax, [μx], [μy], color=color, markersize=12,
            strokecolor=:black, strokewidth=2, marker=:diamond)
    end

    # Plot ground truth if provided
    if !isempty(true_positions)
        for (tx, ty) in true_positions
            scatter!(ax, [tx], [ty], marker=:xcross, color=:black,
                markersize=15, strokewidth=3)
        end
    end

    # Add legend/stats panel
    ax_stats = Axis(fig[1, 2], title="Cluster Statistics")
    hidedecorations!(ax_stats)
    hidespines!(ax_stats)

    stats_text = "Emitter   N      σ (nm)\n" * "─"^25 * "\n"
    for s in cluster_stats
        stats_text *= "   $(s.id)    $(lpad(s.n, 5))   $(lpad(round(s.σ * 1000, digits=1), 5))\n"
    end
    stats_text *= "─"^25 * "\n"
    mean_σ = mean(s.σ for s in cluster_stats) * 1000
    stats_text *= "  mean          $(lpad(round(mean_σ, digits=1), 5))"

    text!(ax_stats, 0.1, 0.9, text=stats_text,
        align=(:left, :top), fontsize=11)

    # Add note about what we're seeing
    text!(ax_stats, 0.1, 0.3,
        text="Each color = one emitter identity\nScatter = positions from K=$map_n samples\nDiamond = computed mean (MAP-N)\nX = ground truth",
        align=(:left, :top), fontsize=10, color=:gray)

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig, matched_positions
end

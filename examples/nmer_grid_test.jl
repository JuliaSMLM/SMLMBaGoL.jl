# Grid of Identical N-mers Test
# =============================
# Statistics over replicas with controlled, identical geometry.
# All clusters are the same N-mer type for controlled evaluation.
# Uses partitioning - each partition treated as one n-mer.
#
# Run with: julia --threads=auto --project=examples examples/nmer_grid_test.jl

using Pkg
Pkg.activate(@__DIR__)

using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions
using Hungarian
using CairoMakie
using JSON

# Include visualization functions
include(joinpath(@__DIR__, "viz_chain_diagnostics.jl"))
include(joinpath(@__DIR__, "viz_metrics.jl"))
include(joinpath(@__DIR__, "viz_smlmrender.jl"))

# =============================================================================
# ADJUSTABLE PARAMETERS
# =============================================================================

# Random seed (set to nothing for random results)
const SEED = 42

# Grid configuration
const GRID_SIZE = 4                # 4×4 = 16 replicas (minimum for stats)
const GRID_SPACING = 1.0           # μm between cluster centers
const N_EMITTERS = 8               # All clusters are 8-mers
const CLUSTER_DIAMETER = 0.050     # 50 nm

# Field of view (calculated from grid)
const FOV_SIZE = (GRID_SIZE + 1) * GRID_SPACING  # Extra margin
const PIXEL_SIZE = 0.100           # μm per pixel (100 nm)
const CAMERA_PIXELS = round(Int, FOV_SIZE / PIXEL_SIZE)

# Photophysics - Fixed labeling with Poisson blink count
const PSF_SIGMA = 0.130            # PSF sigma in μm (130 nm)
const PHOTON_MEAN = 500.0          # Mean photons (exponential distribution)
const PHOTON_MIN = 100.0           # Minimum photons
const BLINK_MEAN = 10.0            # Mean blinks per emitter (Poisson)

# Precision filter
const PRECISION_MAX = 0.010        # Max σ in μm (10 nm) - reject imprecise locs

# BaGoL parameters
const N_ITERATIONS = 15000
const BURN_IN = 3000

# Partitioning parameters
const NSIGMA = 3.0
const MAX_PARTITION_SIZE = 1000

# Output directory
const OUTPUT_DIR = joinpath(@__DIR__, "output", "nmer_grid")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
Generate a grid of identical nmers.

Returns:
- all_positions: Vector of (x, y) tuples for all true emitter positions
- cluster_positions: Vector of vectors, each containing positions for one cluster
- cluster_centers: Vector of (x, y) tuples for cluster centers
"""
function generate_identical_nmer_grid(;
    grid_size::Int = GRID_SIZE,
    grid_spacing::Float64 = GRID_SPACING,
    cluster_diameter::Float64 = CLUSTER_DIAMETER,
    n_emitters::Int = N_EMITTERS,
    fov_size::Float64 = FOV_SIZE
)
    all_positions = Tuple{Float64, Float64}[]
    cluster_positions = Vector{Tuple{Float64, Float64}}[]
    cluster_centers = Tuple{Float64, Float64}[]

    # Calculate grid offset to center in FOV
    grid_extent = (grid_size - 1) * grid_spacing
    offset = (fov_size - grid_extent) / 2

    cluster_radius = cluster_diameter / 2

    for i in 1:grid_size
        for j in 1:grid_size
            # Cluster center position
            cx = offset + (i - 1) * grid_spacing
            cy = offset + (j - 1) * grid_spacing

            push!(cluster_centers, (cx, cy))

            # Place emitters in circle around center
            cluster_emitters = Tuple{Float64, Float64}[]
            for k in 1:n_emitters
                θ = 2π * (k - 1) / n_emitters
                ex = cx + cluster_radius * cos(θ)
                ey = cy + cluster_radius * sin(θ)
                push!(all_positions, (ex, ey))
                push!(cluster_emitters, (ex, ey))
            end
            push!(cluster_positions, cluster_emitters)
        end
    end

    return all_positions, cluster_positions, cluster_centers
end

"""
Generate localizations from true emitter positions using fixed labeling.
Each emitter gets exactly n_blinks localizations (Poisson-sampled).
"""
function generate_localizations(
    positions::Vector{Tuple{Float64, Float64}};
    psf_sigma::Float64 = PSF_SIGMA,
    photon_mean::Float64 = PHOTON_MEAN,
    photon_min::Float64 = PHOTON_MIN,
    blink_mean::Float64 = BLINK_MEAN
)
    locs = SMLMData.Emitter2DFit[]
    blink_counts = Int[]
    loc_id = 1

    blink_dist = Poisson(blink_mean)
    photon_dist = Exponential(photon_mean)

    for (ex, ey) in positions
        # Fixed labeling: sample number of blinks from Poisson
        n_blinks = max(1, rand(blink_dist))
        actual_blinks = 0

        for _ in 1:n_blinks
            N = rand(photon_dist)
            N < photon_min && continue

            σ = psf_sigma / sqrt(N)
            x = ex + σ * randn()
            y = ey + σ * randn()

            push!(locs, SMLMData.Emitter2DFit(
                x, y,
                N, 10.0,
                σ, σ, 0.0,
                sqrt(N), 1.0,
                loc_id, 1, 0, loc_id
            ))
            loc_id += 1
            actual_blinks += 1
        end
        push!(blink_counts, actual_blinks)
    end

    return locs, blink_counts
end

"""
Match estimated emitters to true positions for a partition.
Uses partition bounding box to identify which true emitters belong to it.
"""
function match_partition_emitters(
    partition_emitters::Vector{<:SMLMData.AbstractEmitter},
    true_positions::Vector{Tuple{Float64, Float64}},
    partition::SMLMBaGoL.Partition;
    margin::Float64 = 0.1,
    threshold::Float64 = 0.020
)
    # Get bounding box from partition locs
    p_locs = partition.locs
    x_min = minimum(loc.x for loc in p_locs) - margin
    x_max = maximum(loc.x for loc in p_locs) + margin
    y_min = minimum(loc.y for loc in p_locs) - margin
    y_max = maximum(loc.y for loc in p_locs) + margin

    # Filter true positions within partition bounding box
    partition_true = filter(p ->
        x_min <= p[1] <= x_max && y_min <= p[2] <= y_max, true_positions)

    if isempty(partition_emitters) || isempty(partition_true)
        return (
            n_estimated = length(partition_emitters),
            n_true = length(partition_true),
            n_matched = 0,
            jaccard = 0.0,
            rmse = NaN,
            n_locs = length(p_locs)
        )
    end

    metrics = compute_all_metrics(partition_emitters, partition_true; threshold)

    return (
        n_estimated = metrics.n_estimated,
        n_true = metrics.n_true,
        n_matched = metrics.n_matched,
        jaccard = metrics.jaccard,
        rmse = metrics.rmse,
        n_locs = length(p_locs)
    )
end

"""
Assign partition IDs to localizations for coloring.
Returns new vector with track_id set to partition ID.
"""
function assign_partition_ids(
    locs::Vector{<:SMLMData.AbstractEmitter},
    partitions::Vector{<:SMLMBaGoL.Partition}
)
    idx_to_partition = zeros(Int, length(locs))
    for partition in partitions
        for idx in partition.original_indices
            idx_to_partition[idx] = partition.id
        end
    end

    new_locs = SMLMData.Emitter2DFit[]
    for (i, loc) in enumerate(locs)
        push!(new_locs, SMLMData.Emitter2DFit(
            loc.x, loc.y,
            loc.photons, loc.bg,
            loc.σ_x, loc.σ_y, loc.σ_xy,
            loc.σ_photons, loc.σ_bg,
            loc.frame, loc.dataset, idx_to_partition[i], loc.id
        ))
    end
    return new_locs
end

"""
Plot N-recovery histogram showing N_estimated per partition with line at N_true.
"""
function plot_n_recovery_histogram(
    per_partition_metrics::Vector{<:NamedTuple},
    n_true::Int;
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(600, 400))

    ax = Axis(fig[1, 1], title="N-Recovery: Estimated Emitters per Partition",
              xlabel="N_estimated", ylabel="Count")

    n_estimated = [m.n_estimated for m in per_partition_metrics]

    # Bins for histogram
    min_n = min(minimum(n_estimated), n_true) - 1
    max_n = max(maximum(n_estimated), n_true) + 1
    bins = (min_n - 0.5):(max_n + 0.5)

    hist!(ax, n_estimated, bins=bins, color=:steelblue)

    # Mark true N with vertical line
    vlines!(ax, [n_true], color=:red, linestyle=:solid, linewidth=3,
            label="N_true = $n_true")

    # Add statistics
    n_correct = count(==(n_true), n_estimated)
    mean_est = mean(n_estimated)
    text!(ax, 0.95, 0.95,
          text="Mean: $(round(mean_est, digits=1))\nCorrect: $(n_correct)/$(length(n_estimated))",
          align=(:right, :top), fontsize=12, space=:relative)

    axislegend(ax, position=:lt)

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

"""
Plot per-partition metrics as histograms (not boxplots).
"""
function plot_per_partition_histograms(
    per_partition_metrics::Vector{<:NamedTuple};
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(1200, 400))

    # Extract metrics
    jaccards = [m.jaccard for m in per_partition_metrics]
    rmses = [m.rmse * 1000 for m in per_partition_metrics]  # nm
    valid_rmses = filter(!isnan, rmses)

    # 1. Jaccard histogram
    ax1 = Axis(fig[1, 1], title="Jaccard Index Distribution",
               xlabel="Jaccard Index", ylabel="Count")
    hist!(ax1, jaccards, bins=10, color=:steelblue)
    vlines!(ax1, [mean(jaccards)], color=:red, linewidth=2,
            label="Mean = $(round(mean(jaccards), digits=2))")
    axislegend(ax1, position=:lt)

    # 2. RMSE histogram (in nm)
    ax2 = Axis(fig[1, 2], title="RMSE Distribution",
               xlabel="RMSE (nm)", ylabel="Count")
    if !isempty(valid_rmses)
        hist!(ax2, valid_rmses, bins=10, color=:seagreen)
        vlines!(ax2, [mean(valid_rmses)], color=:red, linewidth=2,
                label="Mean = $(round(mean(valid_rmses), digits=1)) nm")
        axislegend(ax2, position=:rt)
    end

    # 3. N_locs per partition histogram
    n_locs = [m.n_locs for m in per_partition_metrics]
    ax3 = Axis(fig[1, 3], title="Localizations per Partition",
               xlabel="N_locs", ylabel="Count")
    hist!(ax3, n_locs, bins=10, color=:darkorange)
    vlines!(ax3, [mean(n_locs)], color=:red, linewidth=2,
            label="Mean = $(round(mean(n_locs), digits=0))")
    axislegend(ax3, position=:rt)

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

"""
Plot convergence diagnostics: N_emitters trace, autocorrelation, ESS.
"""
function plot_convergence_diagnostics(
    chain::SMLMBaGoL.RJMCMCChain;
    burn_in::Int = BURN_IN,
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(1200, 400))

    ks = [length(s.emitters) for s in chain.samples]
    n_samples = length(ks)

    ax1 = Axis(fig[1, 1], title="Emitter Count Trace",
               xlabel="Sample", ylabel="K")
    lines!(ax1, 1:n_samples, ks, color=:steelblue, linewidth=0.5)

    burn_in_sample = burn_in
    if burn_in_sample < n_samples
        vlines!(ax1, [burn_in_sample], color=:red, linestyle=:dash,
                linewidth=2, label="Burn-in")
    end

    post_burn_ks = ks[burn_in_sample+1:end]
    if !isempty(post_burn_ks)
        hlines!(ax1, [mean(post_burn_ks)], color=:green, linestyle=:solid,
                linewidth=2, label="Mean = $(round(mean(post_burn_ks), digits=1))")
    end
    axislegend(ax1, position=:rt)

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
    end

    ax3 = Axis(fig[1, 3], title="Effective Sample Size")
    hidedecorations!(ax3)
    hidespines!(ax3)

    if length(post_burn_ks) > 50 && var(post_burn_ks) > 0
        acf_vals = Float64[]
        k_centered = post_burn_ks .- mean(post_burn_ks)
        var_k = var(post_burn_ks)

        for lag in 0:min(100, length(post_burn_ks) ÷ 4)
            corr = sum(k_centered[1:end-lag] .* k_centered[lag+1:end]) / ((length(k_centered) - lag) * var_k)
            push!(acf_vals, corr)
            corr < 0 && break
        end

        sum_acf = sum(acf_vals[2:end])
        ess = length(post_burn_ks) / (1 + 2 * max(0, sum_acf))
        ess = min(ess, length(post_burn_ks))

        text!(ax3, 0.5, 0.7, text="ESS(K) = $(round(Int, ess))",
              align=(:center, :center), fontsize=18)
        text!(ax3, 0.5, 0.5, text="N samples = $(length(post_burn_ks))",
              align=(:center, :center), fontsize=14)
        text!(ax3, 0.5, 0.3, text="ESS ratio = $(round(ess/length(post_burn_ks), digits=2))",
              align=(:center, :center), fontsize=14)
    end

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

"""
Plot uncertainty calibration.
"""
function plot_uncertainty_calibration(
    emitters::Vector{<:SMLMData.AbstractEmitter},
    true_positions::Vector{Tuple{Float64, Float64}};
    threshold::Float64 = 0.100,
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(1000, 400))

    assignments, cost, _ = match_positions(emitters, true_positions, threshold)

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

    ax1 = Axis(fig[1, 1], title="Uncertainty Calibration",
               xlabel="σ_estimated (nm)", ylabel="Actual error (nm)")

    scatter!(ax1, σ_values .* 1000, errors .* 1000, color=(:steelblue, 0.5), markersize=6)
    max_val = max(maximum(σ_values), maximum(errors)) * 1000
    lines!(ax1, [0, max_val], [0, max_val], color=:red, linewidth=2, label="y = x")
    axislegend(ax1, position=:rb)

    ax2 = Axis(fig[1, 2], title="Coverage Probability",
               xlabel="Confidence level", ylabel="Empirical coverage")

    σ_levels = [1.0, 2.0, 3.0]
    theoretical = [0.683, 0.954, 0.997]
    empirical = [sum(errors .< level .* σ_values) / length(errors) for level in σ_levels]

    barplot!(ax2, 1:3, empirical, color=:steelblue, label="Empirical")
    scatter!(ax2, 1:3, theoretical, color=:red, markersize=12, label="Theoretical")
    ax2.xticks = (1:3, ["1σ", "2σ", "3σ"])
    axislegend(ax2, position=:rb)

    cal_factor = median(errors ./ σ_values)
    text!(ax2, 0.05, 0.95, text="Cal. factor: $(round(cal_factor, digits=2))",
          align=(:left, :top), fontsize=12, space=:relative)

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

"""
Write diagnostic data to JSON.
"""
function write_diagnostic_json(
    filename::String,
    global_metrics::NamedTuple,
    per_partition_metrics::Vector{<:NamedTuple},
    diagnostics::SMLMBaGoL.BaGoLDiagnostics,
    n_true_emitters::Int,
    n_locs::Int
)
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
        "per_partition" => [
            Dict(
                "id" => i,
                "jaccard" => m.jaccard,
                "n_true" => m.n_true,
                "n_estimated" => m.n_estimated,
                "n_locs" => m.n_locs,
                "rmse_nm" => isnan(m.rmse) ? nothing : m.rmse * 1000
            )
            for (i, m) in enumerate(per_partition_metrics)
        ],
        "hierarchical" => Dict(
            "final_mu" => diagnostics.final_μ,
            "final_shape" => diagnostics.final_shape
        ),
        "simulation" => Dict(
            "n_partitions" => length(per_partition_metrics),
            "n_emitters_per_cluster" => N_EMITTERS,
            "n_true_emitters" => n_true_emitters,
            "n_localizations" => n_locs,
            "mean_locs_per_emitter" => n_locs / n_true_emitters,
            "blink_mean" => BLINK_MEAN
        )
    )

    open(filename, "w") do io
        JSON.print(io, data, 2)
    end
end

"""
Write markdown diagnostic report with nice formatting.
"""
function write_diagnostic_report(
    filename::String,
    global_metrics::NamedTuple,
    per_partition_metrics::Vector{<:NamedTuple},
    diagnostics::SMLMBaGoL.BaGoLDiagnostics,
    n_true_emitters::Int,
    n_locs::Int,
    true_blink_counts::Vector{Int}
)
    open(filename, "w") do io
        println(io, "# N-mer Grid Test Diagnostic Report")
        println(io)

        println(io, "## Simulation Parameters")
        println(io)
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        println(io, "| Grid | $(GRID_SIZE) × $(GRID_SIZE) = $(GRID_SIZE^2) clusters |")
        println(io, "| N-mer size | $(N_EMITTERS) emitters/cluster |")
        println(io, "| Cluster diameter | $(CLUSTER_DIAMETER * 1000) nm |")
        println(io, "| Grid spacing | $(GRID_SPACING * 1000) nm |")
        println(io, "| Blink distribution | Poisson(λ=$(BLINK_MEAN)) |")
        println(io, "| True emitters | $n_true_emitters |")
        println(io, "| Localizations | $n_locs |")
        println(io, "| Mean blinks/emitter | $(round(mean(true_blink_counts), digits=1)) |")
        println(io)

        println(io, "## BaGoL Parameters")
        println(io)
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        println(io, "| Iterations | $N_ITERATIONS |")
        println(io, "| Burn-in | $BURN_IN |")
        println(io, "| DBSCAN nsigma | $NSIGMA |")
        println(io, "| Partitions | $(diagnostics.n_partitions) |")
        println(io, "| Final μ | $(round(diagnostics.final_μ, digits=2)) |")
        println(io, "| Final shape | $(round(diagnostics.final_shape, digits=2)) |")
        println(io)

        println(io, "## Global Metrics")
        println(io)
        println(io, "| Metric | Value |")
        println(io, "|--------|-------|")
        println(io, "| Estimated emitters | $(global_metrics.n_estimated) |")
        println(io, "| True emitters | $(global_metrics.n_true) |")
        println(io, "| Matched | $(global_metrics.n_matched) |")
        println(io, "| **Jaccard Index** | **$(round(global_metrics.jaccard, digits=3))** |")
        println(io, "| Precision | $(round(global_metrics.precision, digits=3)) |")
        println(io, "| Recall | $(round(global_metrics.recall, digits=3)) |")
        println(io, "| F1 Score | $(round(global_metrics.f1, digits=3)) |")
        if !isnan(global_metrics.rmse)
            println(io, "| **RMSE** | **$(round(global_metrics.rmse * 1000, digits=1)) nm** |")
        end
        println(io)

        println(io, "## Acceptance Rates")
        println(io)
        println(io, "| Move | Rate |")
        println(io, "|------|------|")
        for (move, rate) in diagnostics.acceptance_rates
            println(io, "| $move | $(round(rate * 100, digits=1))% |")
        end
        println(io)

        println(io, "## Per-Partition Statistics")
        println(io)

        jaccards = [m.jaccard for m in per_partition_metrics]
        rmses = [m.rmse * 1000 for m in per_partition_metrics if !isnan(m.rmse)]
        n_estimated = [m.n_estimated for m in per_partition_metrics]

        println(io, "### Jaccard Index")
        println(io, "- Mean: $(round(mean(jaccards), digits=3))")
        println(io, "- Std: $(round(std(jaccards), digits=3))")
        println(io, "- Range: [$(round(minimum(jaccards), digits=3)), $(round(maximum(jaccards), digits=3))]")
        println(io)

        if !isempty(rmses)
            println(io, "### RMSE (nm)")
            println(io, "- Mean: $(round(mean(rmses), digits=1))")
            println(io, "- Std: $(round(std(rmses), digits=1))")
            println(io, "- Range: [$(round(minimum(rmses), digits=1)), $(round(maximum(rmses), digits=1))]")
            println(io)
        end

        println(io, "### N-Recovery")
        n_correct = count(==(N_EMITTERS), n_estimated)
        println(io, "- Correct (N=$(N_EMITTERS)): $(n_correct)/$(length(n_estimated)) ($(round(n_correct/length(n_estimated)*100, digits=1))%)")
        println(io, "- Mean N_estimated: $(round(mean(n_estimated), digits=1))")
        println(io, "- Range: [$(minimum(n_estimated)), $(maximum(n_estimated))]")
    end
end

"""
Plot zoomed-in view of each cluster in a grid layout.
Each subplot shows localizations, BaGoL emitters, and true positions for one cluster.
"""
function plot_cluster_grid(
    locs::Vector{<:SMLMData.AbstractEmitter},
    emitters::Vector{SMLMData.Emitter2DFit},
    true_positions::Vector{Tuple{Float64, Float64}},
    cluster_centers::Vector{Tuple{Float64, Float64}},
    cluster_positions::Vector{Vector{Tuple{Float64, Float64}}};
    grid_size::Int = GRID_SIZE,
    zoom_radius::Float64 = 0.060,  # μm around each center (60 nm)
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(280 * grid_size, 280 * grid_size + 40))

    # Layout matches spatial grid with camera convention (y increases downward)
    # Generation loop: outer i → cx, inner j → cy
    # idx encodes: slow index = gen_i (cx), fast index = gen_j (cy)
    for idx in 1:length(cluster_centers)
        cx, cy = cluster_centers[idx]
        gen_i = (idx - 1) ÷ grid_size + 1  # cx direction → figure column
        gen_j = (idx - 1) % grid_size + 1  # cy direction → figure row
        fig_row = gen_j  # camera convention: small cy at top
        fig_col = gen_i  # small cx at left

        # Convert zoom window to nm relative to center for display
        zoom_nm = zoom_radius * 1000  # nm
        ax = Axis(fig[fig_row, fig_col], aspect=DataAspect(), yreversed=true,
                  xticklabelsize=7, yticklabelsize=7,
                  xticks=WilkinsonTicks(3), yticks=WilkinsonTicks(3))
        xlims!(ax, -zoom_nm, zoom_nm)
        ylims!(ax, -zoom_nm, zoom_nm)

        # Only show axis labels on edges
        if fig_row < grid_size
            hidexdecorations!(ax, ticks=false, grid=false)
        end
        if fig_col > 1
            hideydecorations!(ax, ticks=false, grid=false)
        end

        # Filter data within zoom window (with margin)
        r = zoom_radius * 1.2
        nearby_locs = filter(l -> abs(l.x - cx) < r && abs(l.y - cy) < r, locs)
        nearby_emitters = filter(e -> abs(e.x - cx) < r && abs(e.y - cy) < r, emitters)
        nearby_true = filter(p -> abs(p[1] - cx) < r && abs(p[2] - cy) < r, true_positions)

        # Localizations as 1σ circles (centered at 0,0 in nm)
        for loc in nearby_locs
            σ = mean([loc.σ_x, loc.σ_y]) * 1000
            draw_circle!(ax, (loc.x - cx) * 1000, (loc.y - cy) * 1000, σ;
                        color=:gray30, linewidth=0.8, alpha=0.5)
        end

        # True positions as blue X
        for (tx, ty) in nearby_true
            draw_x!(ax, (tx - cx) * 1000, (ty - cy) * 1000, 4.0;
                    color=:blue, linewidth=2.0)
        end

        # BaGoL emitters as red dots + uncertainty ellipses
        for e in nearby_emitters
            scatter!(ax, [(e.x - cx) * 1000], [(e.y - cy) * 1000],
                    color=:red, markersize=5)
            if e.σ_x > 0 && e.σ_y > 0
                draw_ellipse!(ax, (e.x - cx) * 1000, (e.y - cy) * 1000,
                             e.σ_x * 1000, e.σ_y * 1000;
                             color=:red, linewidth=1.0, alpha=0.7)
            end
        end

        # Label: n_est / n_true
        n_est = length(nearby_emitters)
        n_true = length(nearby_true)
        label_color = n_est == n_true ? :forestgreen : :orangered
        text!(ax, 0.03, 0.97, text="$(n_est)/$(n_true)",
              align=(:left, :top), fontsize=12, space=:relative,
              color=label_color, font=:bold)
    end

    # Shared axis labels
    Label(fig[grid_size + 1, :], "nm from center", fontsize=12)
    Label(fig[1:grid_size, 0], "nm from center", fontsize=12, rotation=π/2)
    Label(fig[0, :], "BaGoL Cluster Results  (blue=true, red=estimated, gray=localizations)",
          fontsize=14)

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

"""
Plot partition-colored localizations.
"""
function plot_partition_circles(
    locs_with_partition::Vector{SMLMData.Emitter2DFit},
    partitions::Vector{<:SMLMBaGoL.Partition};
    fov_size::Float64 = FOV_SIZE,
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(800, 800))

    ax = Axis(fig[1, 1], title="Localizations by Partition ($(length(partitions)) partitions)",
              xlabel="x (μm)", ylabel="y (μm)",
              aspect=DataAspect(), yreversed=true)
    xlims!(ax, 0, fov_size)
    ylims!(ax, 0, fov_size)

    # Generate distinct colors
    n_partitions = length(partitions)
    colors = [Makie.wong_colors()[mod1(i, 7)] for i in 1:n_partitions]

    for partition in partitions
        p_locs = partition.locs
        xs = [loc.x for loc in p_locs]
        ys = [loc.y for loc in p_locs]
        scatter!(ax, xs, ys, color=(colors[partition.id], 0.7), markersize=4)
    end

    if save_path !== nothing
        save(save_path, fig)
    end

    return fig
end

# =============================================================================
# MAIN SCRIPT
# =============================================================================

mkpath(OUTPUT_DIR)
if SEED !== nothing
    Random.seed!(SEED)
end

println("="^60)
println("N-mer Grid Test ($(GRID_SIZE)×$(GRID_SIZE) identical $(N_EMITTERS)-mers)")
println("="^60)

# -----------------------------------------------------------------------------
# Generate simulation
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Generating $(GRID_SIZE)×$(GRID_SIZE) grid of identical $(N_EMITTERS)-mers...")

true_positions, cluster_positions, cluster_centers = generate_identical_nmer_grid()

println("  Total clusters: $(length(cluster_centers))")
println("  Emitters per cluster: $N_EMITTERS")
println("  Total true emitters: $(length(true_positions))")

println("\nGenerating localizations (Poisson blinks, mean=$(BLINK_MEAN))...")
locs, true_blink_counts = generate_localizations(true_positions)
println("  Generated $(length(locs)) localizations")
println("  Mean blinks/emitter: $(round(mean(true_blink_counts), digits=1))")

# Precision filter
n_before = length(locs)
locs = filter(loc -> max(loc.σ_x, loc.σ_y) <= PRECISION_MAX, locs)
println("  Precision filter (σ ≤ $(PRECISION_MAX*1000) nm): $(n_before) → $(length(locs)) locs ($(n_before - length(locs)) removed)")

camera = SMLMData.IdealCamera(CAMERA_PIXELS, CAMERA_PIXELS, PIXEL_SIZE)
locs_smld = SMLMData.BasicSMLD(locs, camera, 1, 1)

# -----------------------------------------------------------------------------
# Partition the data
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Partitioning localizations...")

partitions, skipped = SMLMBaGoL.partition_locs(locs;
    nsigma=NSIGMA, min_size=0, max_size=MAX_PARTITION_SIZE)

println("  Created $(length(partitions)) partitions")
if !isempty(skipped)
    println("  Skipped $(length(skipped)) oversized clusters")
end

# Assign partition IDs for visualization
locs_with_partition = assign_partition_ids(locs, partitions)

# -----------------------------------------------------------------------------
# Run BaGoL
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Running BaGoL...")

fov_extent = Float64(CAMERA_PIXELS * PIXEL_SIZE)
result_smld, diagnostics, partition_chains = run_bagol(locs_smld;
    nsigma = NSIGMA,
    max_partition_size = MAX_PARTITION_SIZE,
    n_iterations = N_ITERATIONS,
    burn_in = BURN_IN,
    posterior_pixel_size = 0.002,
    posterior_xlim = (0.0, fov_extent),
    posterior_ylim = (0.0, fov_extent),
    return_chains = true,
    verbose = true)

emitters = result_smld.emitters
println("\nResult: $(length(emitters)) emitters from $(length(true_positions)) true")

# Get single-chain for convergence diagnostics
println("\nRunning single-partition chain for diagnostics...")
chain = run_bagol_chain(locs;
    λ_K = Float64(length(true_positions) ÷ 2),
    n_iterations = N_ITERATIONS,
    burn_in = BURN_IN,
    verbose = false)

# -----------------------------------------------------------------------------
# Compute per-partition metrics
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Computing per-partition metrics...")

# Match emitters to partitions
per_partition_metrics = NamedTuple[]
for partition in partitions
    # Get bounding box of partition
    p_locs = partition.locs
    x_min = minimum(loc.x for loc in p_locs) - 0.1
    x_max = maximum(loc.x for loc in p_locs) + 0.1
    y_min = minimum(loc.y for loc in p_locs) - 0.1
    y_max = maximum(loc.y for loc in p_locs) + 0.1

    # Filter emitters within partition bounds
    partition_emitters = filter(e ->
        x_min <= e.x <= x_max && y_min <= e.y <= y_max, emitters)

    metrics = match_partition_emitters(partition_emitters, true_positions, partition)
    push!(per_partition_metrics, metrics)
end

# Global metrics
global_metrics = compute_all_metrics(emitters, true_positions; threshold=0.020)

println("\nGlobal metrics:")
println("  Jaccard: $(round(global_metrics.jaccard, digits=3))")
println("  Precision: $(round(global_metrics.precision, digits=3))")
println("  Recall: $(round(global_metrics.recall, digits=3))")
println("  RMSE: $(round(global_metrics.rmse * 1000, digits=1)) nm")

n_estimated = [m.n_estimated for m in per_partition_metrics]
n_correct = count(==(N_EMITTERS), n_estimated)
println("  N-recovery: $(n_correct)/$(length(partitions)) correct ($(round(n_correct/length(partitions)*100, digits=1))%)")

# -----------------------------------------------------------------------------
# Generate visualizations
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Generating visualizations...")

# 0. Posterior image (raw PNG from partitioned chains)
if diagnostics.posterior_image !== nothing
    println("  [0/14] Posterior image PNG...")
    save_posterior_png(joinpath(OUTPUT_DIR, "posterior_image.png"), diagnostics.posterior_image; percentile=0.99)
    post = diagnostics.posterior_image
    println("    Image size: $(size(post.image, 1))×$(size(post.image, 2)), $(sum(post.image)) counts")
end

# 1. SMLMRender suite (uses partitioned result, not single chain)
println("  [1/11] SMLMRender suite...")
bagol_smld = SMLMData.BasicSMLD(emitters, camera, 1, 1)
camera_fov = (0.0, Float64(CAMERA_PIXELS * PIXEL_SIZE),
              0.0, Float64(CAMERA_PIXELS * PIXEL_SIZE))
target = render_bagol_suite(locs_smld, bagol_smld;
    true_positions = true_positions,
    prefix = "render",
    output_dir = OUTPUT_DIR,
    pixel_size = 1.0,
    fov = camera_fov)

# 2. Posterior histogram from partition chains
println("  [2/13] Posterior histogram (all partitions)...")
render_posterior_histogram(partition_chains, locs_smld;
    target = target,
    prefix = "render",
    output_dir = OUTPUT_DIR)

# 3. MAP-N histogram from partition chains
println("  [3/13] MAP-N histogram (all partitions)...")
render_mapn_histogram(partition_chains, locs_smld;
    target = target,
    prefix = "render",
    output_dir = OUTPUT_DIR)

# 4. Zoomed cluster grid
println("  [4/13] Zoomed cluster grid...")
plot_cluster_grid(locs, emitters, true_positions, cluster_centers, cluster_positions;
    save_path = joinpath(OUTPUT_DIR, "bagol_result.png"))

# 5. Partition circles
println("  [5/13] Partition circles...")
plot_partition_circles(locs_with_partition, partitions;
    save_path = joinpath(OUTPUT_DIR, "partition_circles.png"))

# 6. MAP-N result (uses partitioned result, not single chain)
println("  [6/13] MAP-N result...")
plot_mapn(emitters, diagnostics.posterior_k, locs;
    true_positions = true_positions,
    save_path = joinpath(OUTPUT_DIR, "mapn_result.png"))

# 7. Hierarchical diagnostics (single chain - for convergence diagnostics only)
println("  [7/13] Hierarchical diagnostics...")
plot_hierarchical_diagnostics(chain;
    true_locs_per_emitter = true_blink_counts,
    save_path = joinpath(OUTPUT_DIR, "hierarchical_diagnostics.png"))

# 8. Move histogram
println("  [8/13] Move histogram...")
plot_move_histogram(chain;
    save_path = joinpath(OUTPUT_DIR, "move_histogram.png"))

# 9. Per-partition histograms
println("  [9/13] Per-partition histograms...")
plot_per_partition_histograms(per_partition_metrics;
    save_path = joinpath(OUTPUT_DIR, "per_partition_histograms.png"))

# 10. N-recovery histogram
println("  [10/13] N-recovery histogram...")
plot_n_recovery_histogram(per_partition_metrics, N_EMITTERS;
    save_path = joinpath(OUTPUT_DIR, "n_recovery_histogram.png"))

# 11. Convergence diagnostics (single chain)
println("  [11/13] Convergence diagnostics...")
plot_convergence_diagnostics(chain;
    save_path = joinpath(OUTPUT_DIR, "convergence_diagnostics.png"))

# 12. Uncertainty calibration
println("  [12/13] Uncertainty calibration...")
plot_uncertainty_calibration(emitters, true_positions;
    save_path = joinpath(OUTPUT_DIR, "calibration.png"))

# 13. Write reports
println("  [13/13] Writing reports...")
write_diagnostic_report(
    joinpath(OUTPUT_DIR, "diagnostic_report.md"),
    global_metrics,
    per_partition_metrics,
    diagnostics,
    length(true_positions),
    length(locs),
    true_blink_counts
)

write_diagnostic_json(
    joinpath(OUTPUT_DIR, "diagnostic_data.json"),
    global_metrics,
    per_partition_metrics,
    diagnostics,
    length(true_positions),
    length(locs)
)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
println("\n" * "="^60)
println("Output files saved to: $OUTPUT_DIR")
println("="^60)
println("\nRaw images:")
println("  - posterior_image.png")
println("\nSMLMRender outputs:")
println("  - render_mapn_gaussian.png    (Gaussian render of MAP-N)")
println("  - render_sr_gaussian.png      (Gaussian SR of input locs)")
println("  - render_circles.png          (locs gray + MAP-N red)")
println("  - render_comparison.png       (locs gray + MAP-N red + GT blue)")
println("  - render_posterior.png        (histogram of ALL chain samples)")
println("  - render_mapn_histogram.png   (histogram of K=MAP-N samples only)")
println("\nPlots:")
println("  - bagol_result.png")
println("  - partition_circles.png")
println("  - mapn_result.png")
println("  - hierarchical_diagnostics.png")
println("  - move_histogram.png")
println("  - per_partition_histograms.png")
println("  - n_recovery_histogram.png")
println("  - convergence_diagnostics.png")
println("  - calibration.png")
println("\nReports:")
println("  - diagnostic_report.md")
println("  - diagnostic_data.json")
println("\nKey results:")
println("  True: $(length(true_positions)) ($(GRID_SIZE^2) × $(N_EMITTERS))")
println("  Estimated: $(length(emitters))")
println("  Jaccard: $(round(global_metrics.jaccard, digits=3))")
println("  N-recovery: $(n_correct)/$(length(partitions)) correct")
println("  RMSE: $(round(global_metrics.rmse * 1000, digits=1)) nm")

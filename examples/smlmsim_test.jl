# SMLMSim Realistic Photophysics Test
# ====================================
# End-to-end validation with realistic photophysics from SMLMSim package.
# Uses realistic blinking kinetics and localization uncertainty.
#
# Run with: julia --threads=auto --project=examples examples/smlmsim_test.jl

using Pkg
Pkg.activate(@__DIR__)

using SMLMBaGoL
using SMLMData
using SMLMSim
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

# Camera/FOV
const CAMERA_PIXELS = 64
const PIXEL_SIZE = 0.100           # μm → 6.4×6.4 μm FOV

# SMLMSim simulation
const DENSITY = 2.0                # patterns per μm² → ~80 n-mers (faster)
const N_EMITTERS = 8               # 8-mer
const CLUSTER_DIAMETER = 0.050     # 50 nm

# StaticSMLMConfig
const PSF_SIGMA = 0.130            # 130 nm PSF
const NFRAMES = 1000               # 1000 frames (faster)
const FRAMERATE = 50.0             # 50 fps
const MIN_PHOTONS = 100            # minimum photons for detection

# Fluorophore model
# With k_off=50Hz (τ_on=20ms) and k_on=0.5Hz, expect ~10 blinks in 20s acquisition
const PHOTON_RATE = 50000.0        # 50k photons/s emission rate
const K_OFF = 50.0                 # 50 Hz → 20 ms ON time
const K_ON = 0.5                   # 0.5 Hz → ~10 blinks in 20s
const EXPECTED_BLINKS = 10         # For reporting

# Precision filter
const PRECISION_MAX = 0.010        # Max σ in μm (10 nm) - reject imprecise locs

# BaGoL parameters
const N_ITERATIONS = 15000
const BURN_IN = 3000

# Partitioning parameters
const NSIGMA = 3.0
const MAX_PARTITION_SIZE = 1000

# Output directory
const OUTPUT_DIR = joinpath(@__DIR__, "output", "smlmsim")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
Extract ground truth positions from smld_true.

SMLMSim assigns:
- track_id: identifies each fluorophore within a pattern
- id: identifies the pattern instance

Returns (positions, pattern_ids, cluster_sizes) where:
- positions: Vector of (x, y) tuples
- pattern_ids: Vector mapping each position to its pattern id
- cluster_sizes: Dict mapping pattern_id to number of emitters
"""
function extract_ground_truth(smld_true::SMLMData.SMLD)
    positions = Tuple{Float64, Float64}[]
    pattern_ids = Int[]
    cluster_sizes = Dict{Int, Int}()

    # Group emitters by pattern id
    for e in smld_true.emitters
        push!(positions, (e.x, e.y))
        push!(pattern_ids, e.id)

        if haskey(cluster_sizes, e.id)
            cluster_sizes[e.id] += 1
        else
            cluster_sizes[e.id] = 1
        end
    end

    return positions, pattern_ids, cluster_sizes
end

"""
Get unique emitter positions per pattern from smld_true.

Returns Vector of Tuple{Float64, Float64} with unique true positions.
"""
function get_unique_true_positions(smld_true::SMLMData.SMLD)
    # Use (pattern_id, track_id) as unique identifier
    seen = Set{Tuple{Int, Int}}()
    positions = Tuple{Float64, Float64}[]

    for e in smld_true.emitters
        key = (e.id, e.track_id)
        if key ∉ seen
            push!(seen, key)
            push!(positions, (e.x, e.y))
        end
    end

    return positions
end

"""
Match estimated emitters to true positions within a cluster's bounding box.
"""
function match_cluster_emitters(
    emitters::Vector{<:SMLMData.AbstractEmitter},
    true_positions::Vector{Tuple{Float64, Float64}},
    pattern_positions::Vector{Tuple{Float64, Float64}};
    margin::Float64 = 0.1,
    threshold::Float64 = 0.020
)
    if isempty(pattern_positions)
        return (
            n_estimated = 0,
            n_true = 0,
            n_matched = 0,
            jaccard = 0.0,
            rmse = NaN,
            n_error = 0
        )
    end

    # Get bounding box of this pattern
    xs = [p[1] for p in pattern_positions]
    ys = [p[2] for p in pattern_positions]
    x_min, x_max = minimum(xs) - margin, maximum(xs) + margin
    y_min, y_max = minimum(ys) - margin, maximum(ys) + margin

    # Filter emitters within bounding box
    cluster_emitters = filter(e ->
        x_min <= e.x <= x_max && y_min <= e.y <= y_max, emitters)

    # Filter true positions within bounding box
    cluster_true = filter(p ->
        x_min <= p[1] <= x_max && y_min <= p[2] <= y_max, true_positions)

    if isempty(cluster_emitters) || isempty(cluster_true)
        return (
            n_estimated = length(cluster_emitters),
            n_true = length(cluster_true),
            n_matched = 0,
            jaccard = 0.0,
            rmse = NaN,
            n_error = length(cluster_emitters) - length(cluster_true)
        )
    end

    metrics = compute_all_metrics(cluster_emitters, cluster_true; threshold)

    return (
        n_estimated = metrics.n_estimated,
        n_true = metrics.n_true,
        n_matched = metrics.n_matched,
        jaccard = metrics.jaccard,
        rmse = metrics.rmse,
        n_error = metrics.n_estimated - metrics.n_true
    )
end

"""
Plot N-recovery histogram showing N_estimated distribution with N_true marked.
"""
function plot_n_recovery_histogram(
    per_pattern_metrics::Vector{<:NamedTuple},
    n_true::Int;
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(700, 500))

    ax = Axis(fig[1, 1], title="N-Recovery: Estimated Emitter Counts",
              xlabel="N_estimated", ylabel="Count")

    n_estimated = [m.n_estimated for m in per_pattern_metrics]

    # Compute bins centered on integers
    min_n = minimum(n_estimated)
    max_n = maximum(n_estimated)
    bins = (min_n - 0.5):(max_n + 0.5)

    hist!(ax, n_estimated, bins=bins, color=:steelblue)

    # Mark true N with vertical line
    vlines!(ax, [n_true], color=:red, linestyle=:solid, linewidth=3,
            label="N_true = $n_true")

    # Add statistics
    n_correct = count(==(n_true), n_estimated)
    accuracy = n_correct / length(n_estimated)
    mean_n = mean(n_estimated)

    text!(ax, 0.95, 0.95, text="N_true = $n_true\nMean N_est = $(round(mean_n, digits=1))\nExact match: $(n_correct)/$(length(n_estimated)) ($(round(accuracy*100, digits=1))%)",
          align=(:right, :top), fontsize=12, space=:relative)

    axislegend(ax, position=:lt)

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

    # Extract K from samples
    ks = [length(s.emitters) for s in chain.samples]
    n_samples = length(ks)

    # 1. K trace with burn-in marked
    ax1 = Axis(fig[1, 1], title="Emitter Count Trace",
               xlabel="Sample", ylabel="K")
    lines!(ax1, 1:n_samples, ks, color=:steelblue, linewidth=0.5)

    burn_in_sample = burn_in  # samples stored every iteration
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
                color=:red, linestyle=:dash, linewidth=1)
    end

    # 3. ESS calculation
    ax3 = Axis(fig[1, 3], title="Effective Sample Size")
    hidedecorations!(ax3)
    hidespines!(ax3)

    if length(post_burn_ks) > 50 && var(post_burn_ks) > 0
        k_centered = post_burn_ks .- mean(post_burn_ks)
        var_k = var(post_burn_ks)
        acf_vals = Float64[]

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
Plot uncertainty calibration: σ_estimated vs actual_error, coverage.
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

    # Collect (σ_est, actual_error) pairs
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

    max_val = max(maximum(σ_values), maximum(errors)) * 1000
    lines!(ax1, [0, max_val], [0, max_val], color=:red, linewidth=2, label="y = x")
    axislegend(ax1, position=:rb)

    # 2. Coverage plot
    ax2 = Axis(fig[1, 2], title="Coverage Probability",
               xlabel="Confidence level (σ)", ylabel="Empirical coverage")

    σ_levels = [1.0, 2.0, 3.0]
    theoretical = [0.683, 0.954, 0.997]
    empirical = Float64[]

    for level in σ_levels
        covered = sum(errors .< level .* σ_values)
        push!(empirical, covered / length(errors))
    end

    barplot!(ax2, 1:3, empirical, color=:steelblue, label="Empirical")
    scatter!(ax2, 1:3, theoretical, color=:red, markersize=12, label="Theoretical")

    ax2.xticks = (1:3, ["1σ", "2σ", "3σ"])
    axislegend(ax2, position=:rb)

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
Write simulation parameters to markdown file.
"""
function write_simulation_params(filename::String)
    open(filename, "w") do io
        println(io, "# SMLMSim Simulation Parameters\n")

        println(io, "## Camera/FOV\n")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        println(io, "| Pixels | $(CAMERA_PIXELS) × $(CAMERA_PIXELS) |")
        println(io, "| Pixel size | $(PIXEL_SIZE * 1000) nm |")
        println(io, "| FOV | $(CAMERA_PIXELS * PIXEL_SIZE) × $(CAMERA_PIXELS * PIXEL_SIZE) μm |")
        println(io)

        println(io, "## Pattern\n")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        println(io, "| Density | $(DENSITY) patterns/μm² |")
        println(io, "| Pattern | $(N_EMITTERS)-mer |")
        println(io, "| Cluster diameter | $(CLUSTER_DIAMETER * 1000) nm |")
        println(io)

        println(io, "## Acquisition\n")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        println(io, "| PSF sigma | $(PSF_SIGMA * 1000) nm |")
        println(io, "| Frames | $(NFRAMES) |")
        println(io, "| Frame rate | $(FRAMERATE) fps |")
        println(io, "| Min photons | $(MIN_PHOTONS) |")
        println(io)

        println(io, "## Fluorophore (GenericFluor)\n")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        println(io, "| Photon rate | $(PHOTON_RATE) photons/s |")
        println(io, "| k_off | $(K_OFF) Hz (τ_on = $(round(1000/K_OFF, digits=1)) ms) |")
        println(io, "| k_on | $(K_ON) Hz |")
        println(io, "| Expected blinks | ~$(EXPECTED_BLINKS) |")
        println(io)

        println(io, "## BaGoL\n")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        println(io, "| Iterations | $(N_ITERATIONS) |")
        println(io, "| Burn-in | $(BURN_IN) |")
        println(io, "| DBSCAN nsigma | $(NSIGMA) |")
        println(io, "| Max partition size | $(MAX_PARTITION_SIZE) |")
    end
end

"""
Write diagnostic data to JSON.
"""
function write_diagnostic_json(
    filename::String,
    global_metrics::NamedTuple,
    per_pattern_metrics::Vector{<:NamedTuple},
    diagnostics::SMLMBaGoL.BaGoLDiagnostics,
    chain::SMLMBaGoL.RJMCMCChain,
    n_true_emitters::Int,
    n_patterns::Int,
    n_locs::Int
)
    # Compute ESS
    ks = [length(s.emitters) for s in chain.samples]
    burn_in_sample = BURN_IN
    post_burn_ks = ks[burn_in_sample+1:end]

    ess = NaN
    if length(post_burn_ks) > 50 && var(post_burn_ks) > 0
        k_centered = post_burn_ks .- mean(post_burn_ks)
        var_k = var(post_burn_ks)
        acf_vals = Float64[]

        for lag in 0:min(100, length(post_burn_ks) ÷ 4)
            corr = sum(k_centered[1:end-lag] .* k_centered[lag+1:end]) / ((length(k_centered) - lag) * var_k)
            push!(acf_vals, corr)
            corr < 0 && break
        end

        sum_acf = sum(acf_vals[2:end])
        ess = length(post_burn_ks) / (1 + 2 * max(0, sum_acf))
        ess = min(ess, length(post_burn_ks))
    end

    data = Dict(
        "global_metrics" => Dict(
            "n_estimated" => global_metrics.n_estimated,
            "n_true" => global_metrics.n_true,
            "n_matched" => global_metrics.n_matched,
            "jaccard" => global_metrics.jaccard,
            "precision" => global_metrics.precision,
            "recall" => global_metrics.recall,
            "f1" => global_metrics.f1,
            "rmse_nm" => global_metrics.rmse * 1000
        ),
        "per_pattern" => [
            Dict(
                "id" => i,
                "jaccard" => m.jaccard,
                "n_true" => m.n_true,
                "n_estimated" => m.n_estimated,
                "n_error" => m.n_error,
                "rmse_nm" => isnan(m.rmse) ? nothing : m.rmse * 1000
            )
            for (i, m) in enumerate(per_pattern_metrics)
        ],
        "hierarchical" => Dict(
            "final_mu" => diagnostics.final_μ,
            "final_shape" => diagnostics.final_shape
        ),
        "convergence" => Dict(
            "ess_n_emitters" => isnan(ess) ? nothing : round(Int, ess),
            "n_samples" => length(chain.samples),
            "burn_in" => BURN_IN
        ),
        "simulation" => Dict(
            "density" => DENSITY,
            "n_patterns" => n_patterns,
            "n_emitters_per_pattern" => N_EMITTERS,
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
Write diagnostic report in markdown format.
"""
function write_diagnostic_report(
    filename::String,
    global_metrics::NamedTuple,
    per_pattern_metrics::Vector{<:NamedTuple},
    diagnostics::SMLMBaGoL.BaGoLDiagnostics,
    n_true_emitters::Int,
    n_patterns::Int,
    n_locs::Int
)
    open(filename, "w") do io
        println(io, "# SMLMSim Test Diagnostic Report\n")

        println(io, "## Simulation\n")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        println(io, "| Density | $(DENSITY) patterns/μm² |")
        println(io, "| Patterns | $n_patterns ($(N_EMITTERS)-mers) |")
        println(io, "| True emitters | $n_true_emitters |")
        println(io, "| Localizations | $n_locs |")
        println(io, "| Mean locs/emitter | $(round(n_locs / n_true_emitters, digits=1)) |")
        println(io)

        println(io, "## BaGoL Parameters\n")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        println(io, "| Iterations | $N_ITERATIONS |")
        println(io, "| Burn-in | $BURN_IN |")
        println(io, "| DBSCAN nsigma | $NSIGMA |")
        println(io, "| Partitions | $(diagnostics.n_partitions) |")
        println(io)

        println(io, "## Global Metrics\n")
        println(io, "| Metric | Value |")
        println(io, "|--------|-------|")
        println(io, "| Estimated emitters | $(global_metrics.n_estimated) |")
        println(io, "| True emitters | $(global_metrics.n_true) |")
        println(io, "| Matched | $(global_metrics.n_matched) |")
        println(io, "| **Jaccard Index** | **$(round(global_metrics.jaccard, digits=3))** |")
        println(io, "| Precision | $(round(global_metrics.precision, digits=3)) |")
        println(io, "| Recall | $(round(global_metrics.recall, digits=3)) |")
        println(io, "| F1 Score | $(round(global_metrics.f1, digits=3)) |")
        println(io, "| **RMSE** | **$(round(global_metrics.rmse * 1000, digits=1)) nm** |")
        println(io)

        println(io, "## Hierarchical Parameters\n")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        println(io, "| Final μ | $(round(diagnostics.final_μ, digits=2)) |")
        println(io, "| Final shape | $(round(diagnostics.final_shape, digits=2)) |")
        println(io)

        println(io, "## Acceptance Rates\n")
        println(io, "| Move Type | Rate |")
        println(io, "|-----------|------|")
        for (move, rate) in diagnostics.acceptance_rates
            println(io, "| $move | $(round(rate * 100, digits=1))% |")
        end
        println(io)

        println(io, "## Per-Pattern Statistics\n")
        jaccards = [m.jaccard for m in per_pattern_metrics]
        rmses = [m.rmse * 1000 for m in per_pattern_metrics if !isnan(m.rmse)]
        n_errors = [m.n_error for m in per_pattern_metrics]
        n_estimated = [m.n_estimated for m in per_pattern_metrics]

        println(io, "### Jaccard Index\n")
        println(io, "| Statistic | Value |")
        println(io, "|-----------|-------|")
        println(io, "| Mean | $(round(mean(jaccards), digits=3)) |")
        println(io, "| Std | $(round(std(jaccards), digits=3)) |")
        println(io, "| Min | $(round(minimum(jaccards), digits=3)) |")
        println(io, "| Max | $(round(maximum(jaccards), digits=3)) |")
        println(io)

        if !isempty(rmses)
            println(io, "### RMSE (nm)\n")
            println(io, "| Statistic | Value |")
            println(io, "|-----------|-------|")
            println(io, "| Mean | $(round(mean(rmses), digits=1)) |")
            println(io, "| Std | $(round(std(rmses), digits=1)) |")
            println(io, "| Min | $(round(minimum(rmses), digits=1)) |")
            println(io, "| Max | $(round(maximum(rmses), digits=1)) |")
            println(io)
        end

        println(io, "### N-Recovery\n")
        n_correct = count(==(N_EMITTERS), n_estimated)
        println(io, "| Statistic | Value |")
        println(io, "|-----------|-------|")
        println(io, "| Exact match (N=$(N_EMITTERS)) | $(n_correct)/$(length(n_estimated)) ($(round(n_correct/length(n_estimated)*100, digits=1))%) |")
        println(io, "| Mean N_estimated | $(round(mean(n_estimated), digits=2)) |")
        println(io, "| N_estimated range | [$(minimum(n_estimated)), $(maximum(n_estimated))] |")
        println(io, "| Mean N-error | $(round(mean(n_errors), digits=2)) |")
    end
end

"""
Plot full FOV result: locs (gray) + MAP-N (red) + GT (blue).
"""
function plot_full_fov(
    locs::Vector{<:SMLMData.AbstractEmitter},
    emitters::Vector{SMLMData.Emitter2DFit},
    true_positions::Vector{Tuple{Float64, Float64}};
    fov_size::Float64 = CAMERA_PIXELS * PIXEL_SIZE,
    save_path::Union{String, Nothing} = nothing
)
    fig = Figure(size=(800, 800))

    ax = Axis(fig[1, 1], title="BaGoL Results (SMLMSim $(N_EMITTERS)-mers)",
              xlabel="x (μm)", ylabel="y (μm)",
              aspect=DataAspect(), yreversed=true)
    xlims!(ax, 0, fov_size)
    ylims!(ax, 0, fov_size)

    # Localizations as small gray dots
    locs_x = [loc.x for loc in locs]
    locs_y = [loc.y for loc in locs]
    scatter!(ax, locs_x, locs_y, color=(:gray, 0.3), markersize=2)

    # MAP-N emitters as red ellipses
    for e in emitters
        scatter!(ax, [e.x], [e.y], color=:red, markersize=6)
        if e.σ_x > 0 && e.σ_y > 0
            draw_ellipse!(ax, e.x, e.y, e.σ_x, e.σ_y; color=:red, linewidth=1.0, alpha=0.7)
        end
    end

    # True positions as blue X markers
    for (tx, ty) in true_positions
        draw_x!(ax, tx, ty, 0.015; color=:blue, linewidth=1.5)
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

fov_size = CAMERA_PIXELS * PIXEL_SIZE

println("="^60)
println("SMLMSim Realistic Photophysics Test")
println("="^60)
println("\nSimulation parameters:")
println("  FOV: $(fov_size) × $(fov_size) μm")
println("  Density: $(DENSITY) patterns/μm² → ~$(round(Int, DENSITY * fov_size^2)) $(N_EMITTERS)-mers")
println("  Frames: $NFRAMES @ $(FRAMERATE) fps")
println("  Fluorophore: $(PHOTON_RATE) photons/s, τ_on=$(round(1000/K_OFF, digits=1))ms, k_on=$(K_ON)Hz (~$(EXPECTED_BLINKS) blinks)")

# -----------------------------------------------------------------------------
# Run SMLMSim simulation
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Running SMLMSim simulation...")

# Create StaticSMLMConfig
params = SMLMSim.StaticSMLMConfig(
    density = DENSITY,
    σ_psf = PSF_SIGMA,
    minphotons = MIN_PHOTONS,
    ndatasets = 1,
    nframes = NFRAMES,
    framerate = FRAMERATE,
    ndims = 2
)

# Create pattern and fluorophore
pattern = SMLMSim.Nmer2D(n=N_EMITTERS, d=CLUSTER_DIAMETER)
fluor = SMLMSim.GenericFluor(photons=PHOTON_RATE, k_off=K_OFF, k_on=K_ON)

# Create camera
camera = SMLMData.IdealCamera(CAMERA_PIXELS, CAMERA_PIXELS, PIXEL_SIZE)

# Run simulation - returns (smld_noisy, SimInfo)
smld_noisy, sim_info = SMLMSim.simulate(params;
    pattern = pattern,
    molecule = fluor,
    camera = camera
)
smld_true = sim_info.smld_true

# Extract ground truth
true_positions = get_unique_true_positions(smld_true)
n_patterns = sim_info.n_patterns
n_locs = length(smld_noisy.emitters)

println("  Generated $n_patterns patterns with $(length(true_positions)) unique emitters")
println("  Total localizations: $n_locs")
println("  Mean locs/emitter: $(round(n_locs / length(true_positions), digits=1))")

# Compute localization precision stats
sigmas = [loc.σ_x for loc in smld_noisy.emitters]
println("  Localization precision: $(round(mean(sigmas)*1000, digits=1)) ± $(round(std(sigmas)*1000, digits=1)) nm")

# Precision filter
n_before = length(smld_noisy.emitters)
filtered_locs = filter(loc -> max(loc.σ_x, loc.σ_y) <= PRECISION_MAX, smld_noisy.emitters)
println("  Precision filter (σ ≤ $(PRECISION_MAX*1000) nm): $(n_before) → $(length(filtered_locs)) locs ($(n_before - length(filtered_locs)) removed)")
smld_noisy = SMLMData.BasicSMLD(filtered_locs, camera, 1, 1)
n_locs = length(filtered_locs)

# -----------------------------------------------------------------------------
# Run BaGoL
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Running BaGoL...")

result_smld, diagnostics, partition_chains = run_bagol(smld_noisy;
    nsigma = NSIGMA,
    max_partition_size = MAX_PARTITION_SIZE,
    n_iterations = N_ITERATIONS,
    burn_in = BURN_IN,
    posterior_pixel_size = 0.002,
    posterior_xlim = (0.0, Float64(fov_size)),
    posterior_ylim = (0.0, Float64(fov_size)),
    return_chains = true,
    verbose = true)

emitters = result_smld.emitters
println("\nResult: $(length(emitters)) emitters from $(length(true_positions)) true")

# Also run single chain for diagnostics
println("\nRunning single-partition BaGoL for chain diagnostics...")
locs = smld_noisy.emitters
chain = run_bagol_chain(locs;
    λ_K = Float64(length(true_positions) ÷ 2),
    n_iterations = N_ITERATIONS,
    burn_in = BURN_IN,
    verbose = false)

# -----------------------------------------------------------------------------
# Compute metrics
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Computing metrics...")

# Global metrics
global_metrics = compute_all_metrics(emitters, true_positions; threshold=0.020)

println("\nGlobal metrics:")
println("  Jaccard: $(round(global_metrics.jaccard, digits=3))")
println("  Precision: $(round(global_metrics.precision, digits=3))")
println("  Recall: $(round(global_metrics.recall, digits=3))")
println("  F1: $(round(global_metrics.f1, digits=3))")
println("  RMSE: $(round(global_metrics.rmse * 1000, digits=1)) nm")

# Per-pattern metrics
println("\nComputing per-pattern metrics...")

# Group true positions by pattern
pattern_positions = Dict{Int, Vector{Tuple{Float64, Float64}}}()
for e in smld_true.emitters
    if !haskey(pattern_positions, e.id)
        pattern_positions[e.id] = Tuple{Float64, Float64}[]
    end
    # Only add if not already present (unique positions)
    pos = (e.x, e.y)
    if pos ∉ pattern_positions[e.id]
        push!(pattern_positions[e.id], pos)
    end
end

per_pattern_metrics = [match_cluster_emitters(emitters, true_positions, positions)
                       for (_, positions) in sort(collect(pattern_positions))]

# N-recovery statistics
n_errors = [m.n_error for m in per_pattern_metrics]
n_correct = count(==(0), n_errors)
println("  N-recovery accuracy: $(n_correct)/$(length(n_errors)) ($(round(n_correct/length(n_errors)*100, digits=1))%)")

# -----------------------------------------------------------------------------
# Generate all visualizations
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Generating visualizations...")

# 0. Posterior image (raw PNG from partitioned chains)
if diagnostics.posterior_image !== nothing
    println("  [0/10] Posterior image PNG...")
    save_posterior_png(joinpath(OUTPUT_DIR, "posterior_image.png"), diagnostics.posterior_image; percentile=0.99)
    post = diagnostics.posterior_image
    println("    Image size: $(size(post.image, 1))×$(size(post.image, 2)), $(sum(post.image)) counts")
end

# 1. Full FOV result
println("  [1/10] Full FOV result...")
plot_full_fov(locs, emitters, true_positions;
    save_path = joinpath(OUTPUT_DIR, "bagol_result.png"))

# 2. MAP-N result (uses partitioned result, not single chain)
println("  [2/10] MAP-N result...")
plot_mapn(emitters, diagnostics.posterior_k, locs;
    true_positions = true_positions,
    save_path = joinpath(OUTPUT_DIR, "mapn_result.png"))

# 3. Hierarchical diagnostics (single chain - diagnostics only)
println("  [3/10] Hierarchical diagnostics...")
plot_hierarchical_diagnostics(chain;
    save_path = joinpath(OUTPUT_DIR, "hierarchical_diagnostics.png"))

# 4. Move histogram
println("  [4/10] Move histogram...")
plot_move_histogram(chain;
    save_path = joinpath(OUTPUT_DIR, "move_histogram.png"))

# 5. Posterior density (single chain)
println("  [5/10] Posterior density...")
plot_posterior_density(chain;
    save_path = joinpath(OUTPUT_DIR, "posterior_density.png"))

# 6. N-recovery histogram
println("  [6/10] N-recovery histogram...")
plot_n_recovery_histogram(per_pattern_metrics, N_EMITTERS;
    save_path = joinpath(OUTPUT_DIR, "n_recovery_histogram.png"))

# 7. Convergence diagnostics (single chain)
println("  [7/10] Convergence diagnostics...")
plot_convergence_diagnostics(chain;
    save_path = joinpath(OUTPUT_DIR, "convergence_diagnostics.png"))

# 8. Uncertainty calibration
println("  [8/10] Uncertainty calibration...")
plot_uncertainty_calibration(emitters, true_positions;
    save_path = joinpath(OUTPUT_DIR, "calibration.png"))

# 9. SMLMRender suite (uses partitioned result, not single chain)
println("  [9/10] SMLMRender suite...")
bagol_smld = SMLMData.BasicSMLD(emitters, camera, 1, 1)
locs_smld = SMLMData.BasicSMLD(locs, camera, 1, 1)
camera_fov = (0.0, Float64(fov_size), 0.0, Float64(fov_size))
target = render_bagol_suite(locs_smld, bagol_smld;
    true_positions = true_positions,
    prefix = "render",
    output_dir = OUTPUT_DIR,
    pixel_size = 1.0,
    fov = camera_fov)

# Posterior histogram from partition chains
render_posterior_histogram(partition_chains, locs_smld;
    target = target,
    prefix = "render",
    output_dir = OUTPUT_DIR)

# MAP-N histogram from partition chains
render_mapn_histogram(partition_chains, locs_smld;
    target = target,
    prefix = "render",
    output_dir = OUTPUT_DIR)

# 10. Write reports
println("  [10/10] Writing diagnostic reports...")
write_simulation_params(joinpath(OUTPUT_DIR, "simulation_params.md"))

write_diagnostic_report(
    joinpath(OUTPUT_DIR, "diagnostic_report.md"),
    global_metrics,
    per_pattern_metrics,
    diagnostics,
    length(true_positions),
    n_patterns,
    n_locs
)

# JSON export
println("  Writing JSON data...")
write_diagnostic_json(
    joinpath(OUTPUT_DIR, "diagnostic_data.json"),
    global_metrics,
    per_pattern_metrics,
    diagnostics,
    chain,
    length(true_positions),
    n_patterns,
    n_locs
)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
println("\n" * "="^60)
println("Output files saved to: $OUTPUT_DIR")
println("="^60)
println("\nCairoMakie plots:")
println("  - bagol_result.png            (full FOV: locs + MAP-N + GT)")
println("  - mapn_result.png             (MAP-N + GT, no chain samples)")
println("  - hierarchical_diagnostics.png (μ/shape traces and posteriors)")
println("  - move_histogram.png          (proposed vs accepted moves)")
println("  - posterior_density.png       (2D position density)")
println("  - n_recovery_histogram.png    (N-error distribution)")
println("  - convergence_diagnostics.png (trace, autocorr, ESS)")
println("  - calibration.png             (σ vs error, coverage)")
println("\nRaw images:")
println("  - posterior_image.png         (grayscale posterior, 0.99 percentile)")
println("\nSMLMRender outputs:")
println("  - render_mapn_gaussian.png    (Gaussian render of MAP-N)")
println("  - render_sr_gaussian.png      (Gaussian SR of input locs)")
println("  - render_circles.png          (locs gray + MAP-N red)")
println("  - render_comparison.png       (locs gray + MAP-N red + GT blue)")
println("  - render_posterior.png        (histogram of ALL chain samples)")
println("  - render_mapn_histogram.png   (histogram of K=MAP-N samples only)")
println("\nDiagnostic reports:")
println("  - simulation_params.md        (SMLMSim configuration)")
println("  - diagnostic_report.md        (full text metrics)")
println("  - diagnostic_data.json        (structured JSON for analysis)")
println("\nKey results:")
println("  Patterns: $n_patterns $(N_EMITTERS)-mers")
println("  True emitters: $(length(true_positions))")
println("  Estimated: $(length(emitters))")
println("  Jaccard: $(round(global_metrics.jaccard, digits=3))")
println("  N-recovery: $(n_correct)/$(length(n_errors)) correct ($(round(n_correct/length(n_errors)*100, digits=1))%)")
println("  RMSE: $(round(global_metrics.rmse * 1000, digits=1)) nm")

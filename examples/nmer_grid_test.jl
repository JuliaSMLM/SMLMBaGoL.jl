# Grid of Identical N-mers
# =======================
# Generates a grid of identical N-mers, runs BaGoL, and evaluates results.
# Uses Rao-Blackwellized posterior image; emitters from ClusterStats posterior.
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

# Include shared visualization functions
include(joinpath(@__DIR__, "viz_metrics.jl"))
include(joinpath(@__DIR__, "viz_smlmrender.jl"))

# =============================================================================
# ADJUSTABLE PARAMETERS
# =============================================================================

const SEED = 42

# Grid configuration
const GRID_SIZE = 4                # 4×4 = 16 replicas
const GRID_SPACING = 1.0           # μm between cluster centers
const N_EMITTERS = 8               # All clusters are 8-mers
const CLUSTER_DIAMETER = 0.050     # 50 nm

# Field of view
const FOV_SIZE = (GRID_SIZE + 1) * GRID_SPACING
const PIXEL_SIZE = 0.100           # μm per pixel
const CAMERA_PIXELS = round(Int, FOV_SIZE / PIXEL_SIZE)

# Photophysics
const PSF_SIGMA = 0.130            # μm (130 nm)
const PHOTON_MEAN = 500.0
const PHOTON_MIN = 100.0
const BLINK_MEAN = 10.0

# Precision filter
const PRECISION_MAX = 0.010        # μm (10 nm)

# BaGoL parameters
const N_ITERATIONS = 15000
const BURN_IN = 3000

# Partitioning
const NSIGMA = 3.0
const MAX_PARTITION_SIZE = 1000

# Output
const OUTPUT_DIR = joinpath(@__DIR__, "output", "nmer_grid")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

function generate_identical_nmer_grid(;
    grid_size=GRID_SIZE, grid_spacing=GRID_SPACING,
    cluster_diameter=CLUSTER_DIAMETER, n_emitters=N_EMITTERS, fov_size=FOV_SIZE
)
    all_positions = Tuple{Float64, Float64}[]
    cluster_positions = Vector{Tuple{Float64, Float64}}[]
    cluster_centers = Tuple{Float64, Float64}[]

    grid_extent = (grid_size - 1) * grid_spacing
    offset = (fov_size - grid_extent) / 2
    cluster_radius = cluster_diameter / 2

    for i in 1:grid_size
        for j in 1:grid_size
            cx = offset + (i - 1) * grid_spacing
            cy = offset + (j - 1) * grid_spacing
            push!(cluster_centers, (cx, cy))

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

function generate_localizations(positions; psf_sigma=PSF_SIGMA,
    photon_mean=PHOTON_MEAN, photon_min=PHOTON_MIN, blink_mean=BLINK_MEAN)
    locs = SMLMData.Emitter2DFit[]
    blink_counts = Int[]
    loc_id = 1
    blink_dist = Poisson(blink_mean)
    photon_dist = Exponential(photon_mean)

    for (ex, ey) in positions
        n_blinks = max(1, rand(blink_dist))
        actual_blinks = 0
        for _ in 1:n_blinks
            N = rand(photon_dist)
            N < photon_min && continue
            σ = psf_sigma / sqrt(N)
            x = ex + σ * randn()
            y = ey + σ * randn()
            push!(locs, SMLMData.Emitter2DFit(
                x, y, N, 10.0, σ, σ, 0.0, sqrt(N), 1.0, loc_id, 1, 0, loc_id
            ))
            loc_id += 1
            actual_blinks += 1
        end
        push!(blink_counts, actual_blinks)
    end
    return locs, blink_counts
end

function draw_circle!(ax, x, y, r; color=:black, linewidth=1.0, alpha=1.0)
    poly!(ax, Circle(Point2f(x, y), Float32(r)),
          color=(:white, 0.0), strokecolor=(color, alpha), strokewidth=linewidth)
end

function draw_ellipse!(ax, x, y, rx, ry; color=:black, linewidth=1.0, alpha=1.0)
    θ = range(0, 2π, length=50)
    lines!(ax, x .+ rx .* cos.(θ), y .+ ry .* sin.(θ),
           color=(color, alpha), linewidth=linewidth)
end

function draw_x!(ax, x, y, r; color=:blue, linewidth=2.0, alpha=1.0)
    lines!(ax, [x-r, x+r], [y+r, y-r], color=(color, alpha), linewidth=linewidth)
    lines!(ax, [x-r, x+r], [y-r, y+r], color=(color, alpha), linewidth=linewidth)
end

# =============================================================================
# MAIN SCRIPT
# =============================================================================

mkpath(OUTPUT_DIR)
if SEED !== nothing
    Random.seed!(SEED)
end

println("="^60)
println("N-mer Grid — Collapsed Gibbs ($(GRID_SIZE)×$(GRID_SIZE) identical $(N_EMITTERS)-mers)")
println("="^60)

# Generate grid
println("\n" * "-"^60)
println("Generating $(GRID_SIZE)×$(GRID_SIZE) grid of identical $(N_EMITTERS)-mers...")

true_positions, cluster_positions, cluster_centers = generate_identical_nmer_grid()
println("  Total clusters: $(length(cluster_centers))")
println("  Total true emitters: $(length(true_positions))")

println("\nGenerating localizations...")
locs, true_blink_counts = generate_localizations(true_positions)
println("  Generated $(length(locs)) localizations")
println("  Mean blinks/emitter: $(round(mean(true_blink_counts), digits=1))")

# Precision filter
n_before = length(locs)
locs = filter(loc -> max(loc.σ_x, loc.σ_y) <= PRECISION_MAX, locs)
println("  Precision filter: $(n_before) → $(length(locs)) locs")

camera = SMLMData.IdealCamera(CAMERA_PIXELS, CAMERA_PIXELS, PIXEL_SIZE)
locs_smld = SMLMData.BasicSMLD(locs, camera, 1, 1)

# Partition for per-partition analysis later
partitions, skipped = SMLMBaGoL.partition_locs(locs;
    nsigma=NSIGMA, min_size=0, max_size=MAX_PARTITION_SIZE)
println("  Partitions: $(length(partitions))")

# -----------------------------------------------------------------------------
# Run BaGoL (collapsed Gibbs is default)
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Running BaGoL (collapsed Gibbs)...")

fov_extent = Float64(CAMERA_PIXELS * PIXEL_SIZE)
result_smld, diagnostics = run_bagol(locs_smld;
    nsigma = NSIGMA,
    max_partition_size = MAX_PARTITION_SIZE,
    n_iterations = N_ITERATIONS,
    burn_in = BURN_IN,
    posterior_pixel_size = 0.002,
    posterior_xlim = (0.0, fov_extent),
    posterior_ylim = (0.0, fov_extent),
    verbose = true)

emitters = result_smld.emitters
println("\nResult: $(length(emitters)) emitters from $(length(true_positions)) true")

# -----------------------------------------------------------------------------
# Per-partition metrics
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Computing per-partition metrics...")

per_partition_metrics = NamedTuple[]
for partition in partitions
    p_locs = partition.locs
    margin = 0.1
    x_min = minimum(loc.x for loc in p_locs) - margin
    x_max = maximum(loc.x for loc in p_locs) + margin
    y_min = minimum(loc.y for loc in p_locs) - margin
    y_max = maximum(loc.y for loc in p_locs) + margin

    partition_emitters = filter(e ->
        x_min <= e.x <= x_max && y_min <= e.y <= y_max, emitters)

    partition_true = filter(p ->
        x_min <= p[1] <= x_max && y_min <= p[2] <= y_max, true_positions)

    if isempty(partition_emitters) || isempty(partition_true)
        push!(per_partition_metrics, (
            n_estimated=length(partition_emitters),
            n_true=length(partition_true),
            n_matched=0, jaccard=0.0, rmse=NaN,
            n_locs=length(p_locs)))
    else
        m = compute_all_metrics(partition_emitters, partition_true; threshold=0.020)
        push!(per_partition_metrics, (
            n_estimated=m.n_estimated, n_true=m.n_true,
            n_matched=m.n_matched, jaccard=m.jaccard, rmse=m.rmse,
            n_locs=length(p_locs)))
    end
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
println("  N-recovery: $(n_correct)/$(length(partitions)) correct")

# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Generating visualizations...")

# 0. Posterior image
if diagnostics.posterior_image !== nothing
    println("  [0/8] Posterior image PNG...")
    SMLMBaGoL.save_posterior_png(joinpath(OUTPUT_DIR, "posterior_image.png"),
                                 diagnostics.posterior_image; percentile=0.99)
    post = diagnostics.posterior_image
    println("    Image size: $(size(post.image, 1))×$(size(post.image, 2))")
end

# 1. SMLMRender suite
println("  [1/8] SMLMRender suite...")
bagol_smld = SMLMData.BasicSMLD(emitters, camera, 1, 1)
camera_fov = (0.0, Float64(CAMERA_PIXELS * PIXEL_SIZE),
              0.0, Float64(CAMERA_PIXELS * PIXEL_SIZE))
render_bagol_suite(locs_smld, bagol_smld;
    true_positions = true_positions,
    prefix = "render",
    output_dir = OUTPUT_DIR,
    pixel_size = 1.0,
    fov = camera_fov)

# 2. Zoomed cluster grid
println("  [2/8] Zoomed cluster grid...")
zoom_radius = 0.060  # μm

fig_grid = Figure(size=(280 * GRID_SIZE, 280 * GRID_SIZE + 40))
for idx in 1:length(cluster_centers)
    cx, cy = cluster_centers[idx]
    gen_i = (idx - 1) ÷ GRID_SIZE + 1
    gen_j = (idx - 1) % GRID_SIZE + 1
    fig_row, fig_col = gen_j, gen_i

    zoom_nm = zoom_radius * 1000
    ax = Axis(fig_grid[fig_row, fig_col], aspect=DataAspect(), yreversed=true,
              xticklabelsize=7, yticklabelsize=7,
              xticks=WilkinsonTicks(3), yticks=WilkinsonTicks(3))
    xlims!(ax, -zoom_nm, zoom_nm)
    ylims!(ax, -zoom_nm, zoom_nm)

    if fig_row < GRID_SIZE
        hidexdecorations!(ax, ticks=false, grid=false)
    end
    if fig_col > 1
        hideydecorations!(ax, ticks=false, grid=false)
    end

    r = zoom_radius * 1.2
    nearby_locs = filter(l -> abs(l.x - cx) < r && abs(l.y - cy) < r, locs)
    nearby_emitters = filter(e -> abs(e.x - cx) < r && abs(e.y - cy) < r, emitters)
    nearby_true = filter(p -> abs(p[1] - cx) < r && abs(p[2] - cy) < r, true_positions)

    for loc in nearby_locs
        σ = mean([loc.σ_x, loc.σ_y]) * 1000
        draw_circle!(ax, (loc.x - cx) * 1000, (loc.y - cy) * 1000, σ;
                    color=:gray30, linewidth=0.8, alpha=0.5)
    end
    for (tx, ty) in nearby_true
        draw_x!(ax, (tx - cx) * 1000, (ty - cy) * 1000, 4.0;
                color=:blue, linewidth=2.0)
    end
    for e in nearby_emitters
        scatter!(ax, [(e.x - cx) * 1000], [(e.y - cy) * 1000],
                color=:red, markersize=5)
        if e.σ_x > 0 && e.σ_y > 0
            draw_ellipse!(ax, (e.x - cx) * 1000, (e.y - cy) * 1000,
                         e.σ_x * 1000, e.σ_y * 1000;
                         color=:red, linewidth=1.0, alpha=0.7)
        end
    end

    n_est = length(nearby_emitters)
    n_true = length(nearby_true)
    label_color = n_est == n_true ? :forestgreen : :orangered
    text!(ax, 0.03, 0.97, text="$(n_est)/$(n_true)",
          align=(:left, :top), fontsize=12, space=:relative,
          color=label_color, font=:bold)
end

Label(fig_grid[GRID_SIZE + 1, :], "nm from center", fontsize=12)
Label(fig_grid[1:GRID_SIZE, 0], "nm from center", fontsize=12, rotation=π/2)
Label(fig_grid[0, :],
    "Collapsed Gibbs Cluster Results  (blue=true, red=estimated, gray=localizations)",
    fontsize=14)
save(joinpath(OUTPUT_DIR, "bagol_result.png"), fig_grid)

# 3. MAP-N result plot (posterior K histogram)
println("  [3/8] MAP-N result...")
fig_mapn = Figure(size=(800, 400))
ax_mapn = Axis(fig_mapn[1, 1], title="Posterior K (Number of Emitters)",
               xlabel="K", ylabel="Count")
pk = diagnostics.posterior_k
if !isempty(pk)
    k_min, k_max = minimum(pk), maximum(pk)
    hist!(ax_mapn, pk, bins=(k_min - 0.5):(k_max + 0.5), color=:steelblue)
    map_n = length(emitters)
    vlines!(ax_mapn, [map_n], color=:red, linewidth=3, label="MAP-N = $map_n")
    vlines!(ax_mapn, [length(true_positions)], color=:blue, linewidth=2,
            linestyle=:dash, label="True = $(length(true_positions))")
    axislegend(ax_mapn, position=:rt)
end
save(joinpath(OUTPUT_DIR, "mapn_result.png"), fig_mapn)

# 4. Per-partition histograms
println("  [4/8] Per-partition histograms...")
fig_hist = Figure(size=(1200, 400))

jaccards = [m.jaccard for m in per_partition_metrics]
rmses = [m.rmse * 1000 for m in per_partition_metrics]
valid_rmses = filter(!isnan, rmses)

ax1 = Axis(fig_hist[1, 1], title="Jaccard Index Distribution",
           xlabel="Jaccard Index", ylabel="Count")
hist!(ax1, jaccards, bins=10, color=:steelblue)
vlines!(ax1, [mean(jaccards)], color=:red, linewidth=2,
        label="Mean = $(round(mean(jaccards), digits=2))")
axislegend(ax1, position=:lt)

ax2 = Axis(fig_hist[1, 2], title="RMSE Distribution",
           xlabel="RMSE (nm)", ylabel="Count")
if !isempty(valid_rmses)
    hist!(ax2, valid_rmses, bins=10, color=:seagreen)
    vlines!(ax2, [mean(valid_rmses)], color=:red, linewidth=2,
            label="Mean = $(round(mean(valid_rmses), digits=1)) nm")
    axislegend(ax2, position=:rt)
end

n_locs_list = [m.n_locs for m in per_partition_metrics]
ax3 = Axis(fig_hist[1, 3], title="Localizations per Partition",
           xlabel="N_locs", ylabel="Count")
hist!(ax3, n_locs_list, bins=10, color=:darkorange)
vlines!(ax3, [mean(n_locs_list)], color=:red, linewidth=2,
        label="Mean = $(round(mean(n_locs_list), digits=0))")
axislegend(ax3, position=:rt)
save(joinpath(OUTPUT_DIR, "per_partition_histograms.png"), fig_hist)

# 5. N-recovery histogram
println("  [5/8] N-recovery histogram...")
fig_nrec = Figure(size=(600, 400))
ax_nrec = Axis(fig_nrec[1, 1], title="N-Recovery: Estimated Emitters per Partition",
               xlabel="N_estimated", ylabel="Count")
min_n = min(minimum(n_estimated), N_EMITTERS) - 1
max_n = max(maximum(n_estimated), N_EMITTERS) + 1
hist!(ax_nrec, n_estimated, bins=(min_n - 0.5):(max_n + 0.5), color=:steelblue)
vlines!(ax_nrec, [N_EMITTERS], color=:red, linewidth=3, label="N_true = $N_EMITTERS")
text!(ax_nrec, 0.95, 0.95,
      text="Mean: $(round(mean(n_estimated), digits=1))\nCorrect: $(n_correct)/$(length(n_estimated))",
      align=(:right, :top), fontsize=12, space=:relative)
axislegend(ax_nrec, position=:lt)
save(joinpath(OUTPUT_DIR, "n_recovery_histogram.png"), fig_nrec)

# 6. Uncertainty calibration
println("  [6/8] Uncertainty calibration...")
fig_cal = Figure(size=(1000, 400))

assignments, cost, _ = match_positions(emitters, true_positions, 0.100)
σ_values = Float64[]
errors = Float64[]
for (i, j) in enumerate(assignments)
    if j > 0
        push!(σ_values, sqrt(emitters[i].σ_x^2 + emitters[i].σ_y^2))
        push!(errors, cost[i, j])
    end
end

ax_cal1 = Axis(fig_cal[1, 1], title="Uncertainty Calibration",
               xlabel="σ_estimated (nm)", ylabel="Actual error (nm)")
if !isempty(σ_values)
    scatter!(ax_cal1, σ_values .* 1000, errors .* 1000, color=(:steelblue, 0.5), markersize=6)
    max_val = max(maximum(σ_values), maximum(errors)) * 1000
    lines!(ax_cal1, [0, max_val], [0, max_val], color=:red, linewidth=2, label="y = x")
    axislegend(ax_cal1, position=:rb)
end

ax_cal2 = Axis(fig_cal[1, 2], title="Coverage Probability",
               xlabel="Confidence level", ylabel="Empirical coverage")
if !isempty(σ_values)
    σ_levels = [1.0, 2.0, 3.0]
    theoretical = [0.683, 0.954, 0.997]
    empirical = [sum(errors .< l .* σ_values) / length(errors) for l in σ_levels]
    barplot!(ax_cal2, 1:3, empirical, color=:steelblue, label="Empirical")
    scatter!(ax_cal2, 1:3, theoretical, color=:red, markersize=12, label="Theoretical")
    ax_cal2.xticks = (1:3, ["1σ", "2σ", "3σ"])
    axislegend(ax_cal2, position=:rb)
    cal_factor = median(errors ./ σ_values)
    text!(ax_cal2, 0.05, 0.95, text="Cal. factor: $(round(cal_factor, digits=2))",
          align=(:left, :top), fontsize=12, space=:relative)
end
save(joinpath(OUTPUT_DIR, "calibration.png"), fig_cal)

# 7. Posterior image heatmap
println("  [7/8] Posterior heatmap...")
if diagnostics.posterior_image !== nothing
    post = diagnostics.posterior_image
    fig_post = Figure(size=(800, 700))
    ax_post = Axis(fig_post[1, 1], title="Rao-Blackwellized Posterior Image",
                   xlabel="x (μm)", ylabel="y (μm)", aspect=DataAspect(), yreversed=true)
    img = Float64.(post.image)
    sorted_vals = sort(filter(>(0), vec(img)))
    vmax = isempty(sorted_vals) ? 1.0 : sorted_vals[min(end, round(Int, length(sorted_vals) * 0.99))]
    clipped = min.(img, vmax)
    heatmap!(ax_post, post.edges_x[1:end-1], post.edges_y[1:end-1], clipped',
             colormap=:inferno)
    for (tx, ty) in true_positions
        scatter!(ax_post, [tx], [ty], color=:cyan, markersize=4, marker=:xcross)
    end
    Colorbar(fig_post[1, 2], limits=(0, vmax), colormap=:inferno, label="Density")
    save(joinpath(OUTPUT_DIR, "posterior_heatmap.png"), fig_post)
end

# 8. Write reports
println("  [8/8] Writing reports...")
report_data = Dict(
    "sampler" => "collapsed_gibbs",
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
    "hierarchical" => Dict(
        "final_mu" => diagnostics.final_μ,
        "final_shape" => diagnostics.final_shape
    ),
    "simulation" => Dict(
        "n_partitions" => length(partitions),
        "n_emitters_per_cluster" => N_EMITTERS,
        "n_true_emitters" => length(true_positions),
        "n_localizations" => length(locs),
        "n_recovery_correct" => n_correct,
        "n_recovery_total" => length(partitions)
    )
)
open(joinpath(OUTPUT_DIR, "diagnostic_data.json"), "w") do io
    JSON.print(io, report_data, 2)
end

# Summary
println("\n" * "="^60)
println("Output files saved to: $OUTPUT_DIR")
println("="^60)
println("\nKey results (collapsed Gibbs):")
println("  True: $(length(true_positions)) ($(GRID_SIZE^2) × $(N_EMITTERS))")
println("  Estimated: $(length(emitters))")
println("  Jaccard: $(round(global_metrics.jaccard, digits=3))")
println("  N-recovery: $(n_correct)/$(length(partitions)) correct")
println("  RMSE: $(round(global_metrics.rmse * 1000, digits=1)) nm")
println("  Final μ: $(round(diagnostics.final_μ, digits=2))")
println("  Final shape: $(round(diagnostics.final_shape, digits=2))")

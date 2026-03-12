# SMLMSim Realistic Photophysics (No Hierarchical)
# ===============================================
# BaGoL with realistic SMLMSim photophysics.
# μ and shape are held fixed at their true simulation values.
#
# Run with: julia --threads=auto --project=examples examples/smlmsim_test_nohier.jl

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

# Include shared visualization functions
include(joinpath(@__DIR__, "viz_metrics.jl"))
include(joinpath(@__DIR__, "viz_smlmrender.jl"))

# =============================================================================
# ADJUSTABLE PARAMETERS
# =============================================================================

const SEED = 42

# Camera/FOV
const CAMERA_PIXELS = 64
const PIXEL_SIZE = 0.100           # μm → 6.4×6.4 μm FOV

# SMLMSim simulation
const DENSITY = 2.0                # patterns per μm²
const N_EMITTERS = 8               # 8-mer
const CLUSTER_DIAMETER = 0.050     # 50 nm

# StaticSMLMConfig
const PSF_SIGMA = 0.130            # 130 nm PSF
const NFRAMES = 1000
const FRAMERATE = 50.0
const MIN_PHOTONS = 100

# Fluorophore model
const PHOTON_RATE = 50000.0
const K_OFF = 50.0
const K_ON = 0.5
const EXPECTED_BLINKS = 10

# Precision filter
const PRECISION_MAX = 0.010        # μm (10 nm)

# BaGoL parameters
const N_ITERATIONS = 15000
const BURN_IN = 3000

# Partitioning
const NSIGMA = 3.0
const MAX_PARTITION_SIZE = 1000

# Output
const OUTPUT_DIR = joinpath(@__DIR__, "output", "smlmsim_nohier")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

function get_unique_true_positions(smld_true::SMLMData.SMLD)
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

function match_cluster_emitters(emitters, true_positions, pattern_positions;
                                 margin=0.1, threshold=0.020)
    isempty(pattern_positions) && return (
        n_estimated=0, n_true=0, n_matched=0, jaccard=0.0, rmse=NaN, n_error=0)

    xs = [p[1] for p in pattern_positions]
    ys = [p[2] for p in pattern_positions]
    x_min, x_max = minimum(xs) - margin, maximum(xs) + margin
    y_min, y_max = minimum(ys) - margin, maximum(ys) + margin

    cluster_emitters = filter(e ->
        x_min <= e.x <= x_max && y_min <= e.y <= y_max, emitters)
    cluster_true = filter(p ->
        x_min <= p[1] <= x_max && y_min <= p[2] <= y_max, true_positions)

    if isempty(cluster_emitters) || isempty(cluster_true)
        return (n_estimated=length(cluster_emitters), n_true=length(cluster_true),
                n_matched=0, jaccard=0.0, rmse=NaN,
                n_error=length(cluster_emitters) - length(cluster_true))
    end

    m = compute_all_metrics(cluster_emitters, cluster_true; threshold)
    return (n_estimated=m.n_estimated, n_true=m.n_true,
            n_matched=m.n_matched, jaccard=m.jaccard, rmse=m.rmse,
            n_error=m.n_estimated - m.n_true)
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

fov_size = CAMERA_PIXELS * PIXEL_SIZE

println("="^60)
println("SMLMSim — Collapsed Gibbs Sampler (No Hierarchical)")
println("="^60)
println("\nSimulation parameters:")
println("  FOV: $(fov_size) × $(fov_size) μm")
println("  Density: $(DENSITY) patterns/μm² → ~$(round(Int, DENSITY * fov_size^2)) $(N_EMITTERS)-mers")
println("  Frames: $NFRAMES @ $(FRAMERATE) fps")
println("  Fluorophore: $(PHOTON_RATE) photons/s, τ_on=$(round(1000/K_OFF, digits=1))ms")

# -----------------------------------------------------------------------------
# Run SMLMSim simulation
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Running SMLMSim simulation...")

params = SMLMSim.StaticSMLMConfig(
    density = DENSITY,
    σ_psf = PSF_SIGMA,
    minphotons = MIN_PHOTONS,
    ndatasets = 1,
    nframes = NFRAMES,
    framerate = FRAMERATE,
    ndims = 2
)

pattern = SMLMSim.Nmer2D(n=N_EMITTERS, d=CLUSTER_DIAMETER)
fluor = SMLMSim.GenericFluor(photons=PHOTON_RATE, k_off=K_OFF, k_on=K_ON)
camera = SMLMData.IdealCamera(CAMERA_PIXELS, CAMERA_PIXELS, PIXEL_SIZE)

smld_noisy, sim_info = SMLMSim.simulate(params;
    pattern = pattern, molecule = fluor, camera = camera)
smld_true = sim_info.smld_true

true_positions = get_unique_true_positions(smld_true)
n_patterns = sim_info.n_patterns
n_locs = length(smld_noisy.emitters)

println("  Generated $n_patterns patterns with $(length(true_positions)) unique emitters")
println("  Total localizations: $n_locs")
println("  Mean locs/emitter: $(round(n_locs / length(true_positions), digits=1))")

# Precision filter
n_before = length(smld_noisy.emitters)
filtered_locs = filter(loc -> max(loc.σ_x, loc.σ_y) <= PRECISION_MAX, smld_noisy.emitters)
println("  Precision filter: $(n_before) → $(length(filtered_locs)) locs")
smld_noisy = SMLMData.BasicSMLD(filtered_locs, camera, 1, 1)
n_locs = length(filtered_locs)

TRUE_MU = n_locs / length(true_positions)
TRUE_SHAPE = 1000.0  # Poisson blinking ≈ NegBin with large α
println("  No hierarchical — fixed: μ = $(round(TRUE_MU, digits=1)), shape = $(TRUE_SHAPE)")

# -----------------------------------------------------------------------------
# Run BaGoL (collapsed Gibbs)
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Running BaGoL (collapsed Gibbs)...")

result_smld, diagnostics = run_bagol(smld_noisy;
    nsigma = NSIGMA,
    max_partition_size = MAX_PARTITION_SIZE,
    n_iterations = N_ITERATIONS,
    burn_in = BURN_IN,
    μ_prior_shape = TRUE_MU,
    μ_prior_scale = 1.0,
    shape = TRUE_SHAPE,
    learn_shape = false,
    sync_interval = N_ITERATIONS + 1,
    posterior_pixel_size = 0.002,
    posterior_xlim = (0.0, Float64(fov_size)),
    posterior_ylim = (0.0, Float64(fov_size)),
    verbose = true)

emitters = result_smld.emitters
locs = smld_noisy.emitters
println("\nResult: $(length(emitters)) emitters from $(length(true_positions)) true")

# -----------------------------------------------------------------------------
# Compute metrics
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Computing metrics...")

global_metrics = compute_all_metrics(emitters, true_positions; threshold=0.020)

println("\nGlobal metrics:")
println("  Jaccard: $(round(global_metrics.jaccard, digits=3))")
println("  Precision: $(round(global_metrics.precision, digits=3))")
println("  Recall: $(round(global_metrics.recall, digits=3))")
println("  F1: $(round(global_metrics.f1, digits=3))")
println("  RMSE: $(round(global_metrics.rmse * 1000, digits=1)) nm")

# Per-pattern metrics
pattern_positions = Dict{Int, Vector{Tuple{Float64, Float64}}}()
for e in smld_true.emitters
    if !haskey(pattern_positions, e.id)
        pattern_positions[e.id] = Tuple{Float64, Float64}[]
    end
    pos = (e.x, e.y)
    if pos ∉ pattern_positions[e.id]
        push!(pattern_positions[e.id], pos)
    end
end

per_pattern_metrics = [match_cluster_emitters(emitters, true_positions, positions)
                       for (_, positions) in sort(collect(pattern_positions))]

n_errors = [m.n_error for m in per_pattern_metrics]
n_correct = count(==(0), n_errors)
println("  N-recovery: $(n_correct)/$(length(n_errors)) correct ($(round(n_correct/length(n_errors)*100, digits=1))%)")

# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Generating visualizations...")

# 0. Posterior image
if diagnostics.posterior_image !== nothing
    println("  [0/9] Posterior image PNG...")
    SMLMBaGoL.save_posterior_png(joinpath(OUTPUT_DIR, "posterior_image.png"),
                                 diagnostics.posterior_image; percentile=0.99)
end

# 1. Full FOV result
println("  [1/9] Full FOV result...")
fig_fov = Figure(size=(800, 800))
ax_fov = Axis(fig_fov[1, 1],
    title="Collapsed Gibbs Results (SMLMSim $(N_EMITTERS)-mers)",
    xlabel="x (μm)", ylabel="y (μm)",
    aspect=DataAspect(), yreversed=true)
xlims!(ax_fov, 0, fov_size)
ylims!(ax_fov, 0, fov_size)

scatter!(ax_fov, [l.x for l in locs], [l.y for l in locs],
         color=(:gray, 0.3), markersize=2)
for e in emitters
    scatter!(ax_fov, [e.x], [e.y], color=:red, markersize=6)
    if e.σ_x > 0 && e.σ_y > 0
        draw_ellipse!(ax_fov, e.x, e.y, e.σ_x, e.σ_y;
                     color=:red, linewidth=1.0, alpha=0.7)
    end
end
for (tx, ty) in true_positions
    draw_x!(ax_fov, tx, ty, 0.015; color=:blue, linewidth=1.5)
end
save(joinpath(OUTPUT_DIR, "bagol_result.png"), fig_fov)

# 2. MAP-N result plot (posterior K histogram)
println("  [2/9] MAP-N result...")
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

# 3. N-recovery histogram
println("  [3/9] N-recovery histogram...")
fig_nrec = Figure(size=(700, 500))
ax_nrec = Axis(fig_nrec[1, 1], title="N-Recovery: Estimated Emitter Counts",
               xlabel="N_estimated", ylabel="Count")
n_estimated = [m.n_estimated for m in per_pattern_metrics]
min_n, max_n = minimum(n_estimated), maximum(n_estimated)
hist!(ax_nrec, n_estimated, bins=(min_n - 0.5):(max_n + 0.5), color=:steelblue)
vlines!(ax_nrec, [N_EMITTERS], color=:red, linewidth=3, label="N_true = $N_EMITTERS")
accuracy = n_correct / length(n_estimated)
text!(ax_nrec, 0.95, 0.95,
    text="N_true = $N_EMITTERS\nMean = $(round(mean(n_estimated), digits=1))\nCorrect: $(n_correct)/$(length(n_estimated)) ($(round(accuracy*100, digits=1))%)",
    align=(:right, :top), fontsize=12, space=:relative)
axislegend(ax_nrec, position=:lt)
save(joinpath(OUTPUT_DIR, "n_recovery_histogram.png"), fig_nrec)

# 4. Uncertainty calibration
println("  [4/9] Uncertainty calibration...")
fig_cal = Figure(size=(1000, 400))

assignments, cost, _ = match_positions(emitters, true_positions, 0.100)
σ_values = Float64[]
errors_cal = Float64[]
for (i, j) in enumerate(assignments)
    if j > 0
        push!(σ_values, sqrt(emitters[i].σ_x^2 + emitters[i].σ_y^2))
        push!(errors_cal, cost[i, j])
    end
end

ax_cal1 = Axis(fig_cal[1, 1], title="Uncertainty Calibration",
               xlabel="σ_estimated (nm)", ylabel="Actual error (nm)")
if !isempty(σ_values)
    scatter!(ax_cal1, σ_values .* 1000, errors_cal .* 1000,
             color=(:steelblue, 0.5), markersize=6)
    max_val = max(maximum(σ_values), maximum(errors_cal)) * 1000
    lines!(ax_cal1, [0, max_val], [0, max_val], color=:red, linewidth=2, label="y = x")
    axislegend(ax_cal1, position=:rb)
end

ax_cal2 = Axis(fig_cal[1, 2], title="Coverage Probability",
               xlabel="Confidence level", ylabel="Empirical coverage")
if !isempty(σ_values)
    σ_levels = [1.0, 2.0, 3.0]
    theoretical = [0.683, 0.954, 0.997]
    empirical = [sum(errors_cal .< l .* σ_values) / length(errors_cal) for l in σ_levels]
    barplot!(ax_cal2, 1:3, empirical, color=:steelblue, label="Empirical")
    scatter!(ax_cal2, 1:3, theoretical, color=:red, markersize=12, label="Theoretical")
    ax_cal2.xticks = (1:3, ["1σ", "2σ", "3σ"])
    axislegend(ax_cal2, position=:rb)
end
save(joinpath(OUTPUT_DIR, "calibration.png"), fig_cal)

# 5. NN distance histogram from estimated emitters
println("  [5/9] NN distance histogram...")
if length(emitters) >= 2
    nn_dists = Float64[]
    for i in eachindex(emitters)
        d_min = Inf
        for j in eachindex(emitters)
            i == j && continue
            d = sqrt((emitters[i].x - emitters[j].x)^2 + (emitters[i].y - emitters[j].y)^2)
            d < d_min && (d_min = d)
        end
        push!(nn_dists, d_min * 1000)  # μm → nm
    end
    fig_nn = Figure(size=(600, 400))
    ax_nn = Axis(fig_nn[1, 1], title="Nearest-Neighbor Distances (Estimated Emitters)",
                 xlabel="Distance (nm)", ylabel="Count")
    hist!(ax_nn, nn_dists, bins=50, color=:steelblue)
    nn_true = CLUSTER_DIAMETER * sin(π / N_EMITTERS) * 1000
    vlines!(ax_nn, [nn_true], color=:red, linewidth=2,
            label="True NN = $(round(nn_true, digits=1)) nm")
    axislegend(ax_nn, position=:rt)
    save(joinpath(OUTPUT_DIR, "nn_distances.png"), fig_nn)
end

# 6. Posterior heatmap
println("  [6/9] Posterior heatmap...")
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
    Colorbar(fig_post[1, 2], limits=(0, vmax), colormap=:inferno, label="Density")
    save(joinpath(OUTPUT_DIR, "posterior_heatmap.png"), fig_post)
end

# 7. SMLMRender suite
println("  [7/9] SMLMRender suite...")
bagol_smld = SMLMData.BasicSMLD(emitters, camera, 1, 1)
locs_smld = SMLMData.BasicSMLD(locs, camera, 1, 1)
camera_fov = (0.0, Float64(fov_size), 0.0, Float64(fov_size))
render_bagol_suite(locs_smld, bagol_smld;
    true_positions = true_positions,
    prefix = "render",
    output_dir = OUTPUT_DIR,
    pixel_size = 1.0,
    fov = camera_fov)

# 8. Write reports
println("  [8/9] Writing reports...")

open(joinpath(OUTPUT_DIR, "simulation_params.md"), "w") do io
    println(io, "# SMLMSim Simulation Parameters (Collapsed Gibbs)\n")
    println(io, "| Parameter | Value |")
    println(io, "|-----------|-------|")
    println(io, "| Sampler | Collapsed Gibbs |")
    println(io, "| FOV | $(fov_size) × $(fov_size) μm |")
    println(io, "| Density | $(DENSITY) patterns/μm² |")
    println(io, "| Patterns | $n_patterns ($(N_EMITTERS)-mers) |")
    println(io, "| Iterations | $N_ITERATIONS |")
    println(io, "| Burn-in | $BURN_IN |")
    println(io, "| Final μ | $(round(diagnostics.final_μ, digits=2)) |")
    println(io, "| Final shape | $(round(diagnostics.final_shape, digits=2)) |")
end

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
        "rmse_nm" => global_metrics.rmse * 1000
    ),
    "hierarchical" => Dict(
        "final_mu" => diagnostics.final_μ,
        "final_shape" => diagnostics.final_shape
    ),
    "simulation" => Dict(
        "density" => DENSITY,
        "n_patterns" => n_patterns,
        "n_emitters_per_pattern" => N_EMITTERS,
        "n_true_emitters" => length(true_positions),
        "n_localizations" => n_locs
    )
)
open(joinpath(OUTPUT_DIR, "diagnostic_data.json"), "w") do io
    JSON.print(io, report_data, 2)
end

# 9. Summary
println("  [9/9] Done.")

println("\n" * "="^60)
println("Output files saved to: $OUTPUT_DIR")
println("="^60)
println("\nKey results (collapsed Gibbs):")
println("  Patterns: $n_patterns $(N_EMITTERS)-mers")
println("  True emitters: $(length(true_positions))")
println("  Estimated: $(length(emitters))")
println("  Jaccard: $(round(global_metrics.jaccard, digits=3))")
println("  N-recovery: $(n_correct)/$(length(n_errors)) correct")
println("  RMSE: $(round(global_metrics.rmse * 1000, digits=1)) nm")
println("  Final μ: $(round(diagnostics.final_μ, digits=2))")
println("  Final shape: $(round(diagnostics.final_shape, digits=2))")

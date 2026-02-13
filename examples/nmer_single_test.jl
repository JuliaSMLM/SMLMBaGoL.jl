# Single N-mer Visualization Test
# ================================
# Generates a single n-mer cluster and produces all visualization outputs.
# Adjust parameters at top of script for testing.
#
# Run with: julia --project=examples examples/09_nmer_single_test.jl

using Pkg
Pkg.activate(@__DIR__)

using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions

# Include visualization functions
include(joinpath(@__DIR__, "viz_chain_diagnostics.jl"))
include(joinpath(@__DIR__, "viz_animation.jl"))
include(joinpath(@__DIR__, "viz_metrics.jl"))
include(joinpath(@__DIR__, "viz_smlmrender.jl"))

# =============================================================================
# ADJUSTABLE PARAMETERS
# =============================================================================

# Random seed for reproducibility (set to nothing for random results)
const SEED = nothing
# const SEED = 42

# N-mer geometry
const N_EMITTERS = 8              # Number of emitters in cluster
const CLUSTER_DIAMETER = 0.050    # Diameter in μm (50 nm)

# Photophysics
const PSF_SIGMA = 0.130           # PSF sigma in μm (130 nm)
const PHOTON_MEAN = 500.0         # Mean photons (exponential distribution)
const PHOTON_MIN = 100.0          # Minimum photons (reject below this)

# Blink count distribution - Poisson with mean 10
const BLINK_MEAN = 10.0           # Mean blinks per emitter

# Precision filter
const PRECISION_MAX = 0.010       # Max σ in μm (10 nm) - reject imprecise locs

# BaGoL parameters
const N_ITERATIONS = 20000
const BURN_IN = 4000
const CALLBACK_INTERVAL = 50      # Record every N iterations for animation

# Camera (for SMLD creation)
const CAMERA_PIXELS = 256
const PIXEL_SIZE = 0.100          # μm per pixel (FOV = 25.6 μm)

# Output
const OUTPUT_DIR = joinpath(@__DIR__, "output", "nmer_single")

# =============================================================================
# SETUP
# =============================================================================

mkpath(OUTPUT_DIR)
if SEED !== nothing
    Random.seed!(SEED)
end

println("="^60)
println("Single $(N_EMITTERS)-mer Visualization Test")
println("="^60)
println("\nParameters:")
println("  Cluster diameter: $(CLUSTER_DIAMETER * 1000) nm")
println("  PSF sigma: $(PSF_SIGMA * 1000) nm")
println("  Photon mean: $PHOTON_MEAN (min: $PHOTON_MIN)")
println("  Blink distribution: Poisson($(BLINK_MEAN))")
println("  Expected blinks/emitter: $(BLINK_MEAN)")

# =============================================================================
# GENERATE N-MER
# =============================================================================

println("\n" * "-"^60)
println("Generating $(N_EMITTERS)-mer...")

# Place emitters in circle at center of camera FOV
cluster_radius = CLUSTER_DIAMETER / 2
fov_size = CAMERA_PIXELS * PIXEL_SIZE  # 25.6 μm
center_x, center_y = fov_size / 2, fov_size / 2  # Center of camera FOV

true_positions = Tuple{Float64, Float64}[]
for i in 1:N_EMITTERS
    θ = 2π * (i - 1) / N_EMITTERS
    ex = center_x + cluster_radius * cos(θ)
    ey = center_y + cluster_radius * sin(θ)
    push!(true_positions, (ex, ey))
end

# Generate localizations
blink_dist = Poisson(BLINK_MEAN)
photon_dist = Exponential(PHOTON_MEAN)

function generate_localizations(positions, blink_dist, photon_dist)
    locs = SMLMData.Emitter2DFit[]
    blink_counts = Int[]
    loc_id = 1

    for (ex, ey) in positions
        # Draw number of blinks from Poisson
        n_blinks = max(1, rand(blink_dist))
        actual_blinks = 0

        for _ in 1:n_blinks
            N = rand(photon_dist)
            N < PHOTON_MIN && continue

            σ = PSF_SIGMA / sqrt(N)
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

locs, true_blink_counts = generate_localizations(true_positions, blink_dist, photon_dist)

println("  Generated $(length(locs)) localizations from $(N_EMITTERS) emitters")
println("  Blinks per emitter: $(true_blink_counts)")
println("  Mean blinks: $(round(mean(true_blink_counts), digits=1))")

# Compute localization precision statistics
sigmas = [loc.σ_x for loc in locs]
println("  Localization precision: $(round(mean(sigmas)*1000, digits=1)) ± $(round(std(sigmas)*1000, digits=1)) nm")

# Precision filter
n_before = length(locs)
locs = filter(loc -> max(loc.σ_x, loc.σ_y) <= PRECISION_MAX, locs)
println("  Precision filter (σ ≤ $(PRECISION_MAX*1000) nm): $(n_before) → $(length(locs)) locs ($(n_before - length(locs)) removed)")

# Create camera and SMLD
camera = SMLMData.IdealCamera(CAMERA_PIXELS, CAMERA_PIXELS, PIXEL_SIZE)
locs_smld = SMLMData.BasicSMLD(locs, camera, 1, 1)

# =============================================================================
# RUN BAGOL WITH CALLBACK
# =============================================================================

println("\n" * "-"^60)
println("Running BaGoL...")

# Create animation collector
collector = AnimationCollector()
callback = make_animation_callback(collector, CALLBACK_INTERVAL)

chain = run_bagol_chain(
    locs;
    λ_K = Float64(N_EMITTERS),
    n_iterations = N_ITERATIONS,
    burn_in = BURN_IN,
    callback = callback,
    callback_interval = CALLBACK_INTERVAL,
    verbose = true
)

# MAP-N estimation (iterative Hungarian + MAD-based σ)
emitters, posterior_k = estimate_mapn(chain)

println("\n" * "-"^60)
println("Results:")
println("  True emitters: $(N_EMITTERS)")
println("  MAP-N estimate: $(length(emitters))")

# Show uncertainties
σ_values = [sqrt(e.σ_x^2 + e.σ_y^2) for e in emitters]
println("  Mean σ: $(round(mean(σ_values)*1000, digits=2)) nm")

# =============================================================================
# METRICS
# =============================================================================

println("\n" * "-"^60)
println("Evaluation Metrics:")
metrics = print_metrics(emitters, true_positions; threshold=0.020)

# =============================================================================
# GENERATE ALL PLOTS
# =============================================================================

println("\n" * "-"^60)
println("Generating visualizations...")

# 1. Main BaGoL result plot (chain samples + MAP-N + GT)
println("  [1/8] BaGoL result plot...")
fig1 = plot_bagol(chain, emitters, posterior_k, locs;
    true_positions = true_positions,
    save_path = joinpath(OUTPUT_DIR, "bagol_result.png"))

# 2. Simple MAP-N plot (no chain samples)
println("  [2/8] MAP-N plot...")
fig2 = plot_mapn(emitters, posterior_k, locs;
    true_positions = true_positions,
    save_path = joinpath(OUTPUT_DIR, "mapn_result.png"))

# 3. Hierarchical diagnostics (μ trace, posterior, count distribution)
println("  [3/8] Hierarchical diagnostics...")
fig3 = plot_hierarchical_diagnostics(chain;
    true_locs_per_emitter = true_blink_counts,
    save_path = joinpath(OUTPUT_DIR, "hierarchical_diagnostics.png"))

# 4. Move type histogram
println("  [4/8] Move histogram...")
fig4 = plot_move_histogram(chain;
    save_path = joinpath(OUTPUT_DIR, "move_histogram.png"))

# 5. Posterior density
println("  [5/8] Posterior density...")
fig5 = plot_posterior_density(chain;
    save_path = joinpath(OUTPUT_DIR, "posterior_density.png"))

# 6. Chain snapshots
println("  [6/8] Chain snapshots...")
fig6 = plot_chain_snapshots(collector, locs;
    burn_in = BURN_IN,
    save_path = joinpath(OUTPUT_DIR, "chain_snapshots.png"))

# 7. Animation (MP4)
println("  [7/8] Chain animation (MP4)...")
animate_chain(collector, locs;
    filename = joinpath(OUTPUT_DIR, "chain_animation.mp4"),
    fps = 30,
    true_positions = true_positions)

# 8. Posterior image (raw PNG)
println("  [8/11] Posterior image PNG...")
post = posterior_image(chain; pixel_size=0.002)
save_posterior_png(joinpath(OUTPUT_DIR, "posterior_image.png"), post; percentile=0.99)
println("    Image size: $(size(post.image, 1))×$(size(post.image, 2)), $(sum(post.image)) counts")

# 9. SMLMRender outputs
println("  [9/11] SMLMRender suite...")
bagol_smld = SMLMData.BasicSMLD(emitters, camera, 1, 1)
target = render_bagol_suite(locs_smld, bagol_smld;
    true_positions = true_positions,
    prefix = "render",
    output_dir = OUTPUT_DIR,
    pixel_size = 1.0)

# 10. Posterior histogram from chain (all samples)
println("  [10/11] Posterior histogram (all K)...")
render_posterior_histogram(chain, locs_smld;
    target = target,
    prefix = "render",
    output_dir = OUTPUT_DIR)

# 11. MAP-N histogram (only K=MAP-N samples) - for comparison with Gaussian render
println("  [11/11] MAP-N histogram (K=MAP-N only)...")
render_mapn_histogram(chain, locs_smld;
    target = target,
    prefix = "render",
    output_dir = OUTPUT_DIR)

# =============================================================================
# SUMMARY
# =============================================================================

println("\n" * "="^60)
println("Output files saved to: $OUTPUT_DIR")
println("="^60)
println("\nCairoMakie plots:")
println("  - bagol_result.png        (chain samples + MAP-N + GT)")
println("  - mapn_result.png         (MAP-N + GT, no chain samples)")
println("  - hierarchical_diagnostics.png (μ/shape traces and posteriors)")
println("  - move_histogram.png      (proposed vs accepted moves)")
println("  - posterior_density.png   (2D position density)")
println("  - chain_snapshots.png     (burn-in, middle, final states)")
println("  - chain_animation.mp4     (full chain evolution)")
println("\nRaw images:")
println("  - posterior_image.png     (grayscale posterior, 0.99 percentile)")
println("\nSMLMRender outputs:")
println("  - render_mapn_gaussian.png  (Gaussian render of MAP-N)")
println("  - render_sr_gaussian.png    (Gaussian SR of input locs)")
println("  - render_circles.png        (locs gray + MAP-N red)")
println("  - render_comparison.png     (locs gray + MAP-N red + GT blue)")
println("  - render_posterior.png      (histogram of ALL chain samples)")
println("  - render_mapn_histogram.png (histogram of K=MAP-N samples only)")
println("\nMetrics:")
println("  Jaccard Index: $(round(metrics.jaccard, digits=3))")
println("  F1 Score: $(round(metrics.f1, digits=3))")
println("  RMSE: $(round(metrics.rmse * 1000, digits=1)) nm")

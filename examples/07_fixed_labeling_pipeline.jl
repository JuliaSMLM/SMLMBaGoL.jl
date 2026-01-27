# Full Pipeline: Simulation → Frame Connection → Partitioned BaGoL
# =================================================================
# Demonstrates the complete workflow with FIXED labeling:
# 1. Simulate 6-mers with Fixed labeling (1 fluorophore per site) using SMLMSim
# 2. Apply SMLMFrameConnection to combine repeated localizations
# 3. Run partitioned BaGoL for parallel processing
#
# Run with: julia --threads=auto --project=examples examples/07_fixed_labeling_pipeline.jl

using Pkg
Pkg.activate(@__DIR__)

using SMLMSim
using SMLMFrameConnection
using SMLMBaGoL
using SMLMData
using SMLMRender
using CairoMakie
using Statistics
using Random

const OUTPUT_DIR = joinpath(@__DIR__, "output", "fixed")
mkpath(OUTPUT_DIR)

println("="^70)
println("Full Pipeline: Fixed Labeling (1 per site)")
println("="^70)
println("Threads available: $(Threads.nthreads())")

Random.seed!(42)

# =============================================================================
# 1. SETUP SIMULATION PARAMETERS
# =============================================================================
println("\n--- Setting up simulation ---")

# Camera: 64×64 pixels @ 0.1 μm/pixel = 6.4×6.4 μm FOV
pixelsize = 0.1  # μm
camera = IdealCamera(1:64, 1:64, pixelsize)
fov_size = 64 * pixelsize  # 6.4 μm
fov_area = fov_size^2      # 40.96 μm²

# Simulation parameters
params = StaticSMLMParams(
    density = 10.0,       # 10 nmers per μm² → ~410 nmers in FOV
    σ_psf = 0.13,         # 130 nm PSF
    minphotons = 50,
    ndatasets = 20,       # 20 datasets
    nframes = 5000,       # 5000 frames each
    framerate = 50.0,     # 50 Hz
    ndims = 2
)

# Pattern: 6-mer with 25 nm diameter
pattern = Nmer2D(n=6, d=0.025)

# Labeling: Fixed 1 fluorophore per binding site (perfect labeling)
labeling = FixedLabeling(1)

# Fluorophore kinetics for ~5 blinks per molecule
total_time = params.ndatasets * params.nframes / params.framerate  # 2000 s
k_on = 5.0 / total_time  # 0.0025 Hz → average 5 blinks over acquisition

fluor = GenericFluor(
    photons = 1000.0 * params.framerate,  # 1000 photons/frame at 50 fps
    k_off = params.framerate,             # ~1 frame on-time (20 ms)
    k_on = k_on
)

println("FOV: $(fov_size)×$(fov_size) μm ($(64)×$(64) pixels)")
println("Expected nmers: ~$(round(Int, params.density * fov_area))")
println("Expected fluorophores: ~$(round(Int, 6 * params.density * fov_area)) (6 sites × density × 1 fixed)")
println("Total acquisition time: $(total_time) s")
println("Expected blinks per molecule: ~$(k_on * total_time)")

# =============================================================================
# 2. SIMULATE SMLM DATA
# =============================================================================
println("\n--- Running simulation ---")

smld_true, smld_model, smld_noisy = simulate(
    params;
    pattern = pattern,
    labeling = labeling,
    molecule = fluor,
    camera = camera
)

println("True emitters (fluorophore positions): $(length(smld_true.emitters))")
println("Kinetic model localizations: $(length(smld_model.emitters))")
println("Noisy localizations: $(length(smld_noisy.emitters))")

# Extract ground truth emitter positions for later comparison
true_positions = [(e.x, e.y) for e in smld_true.emitters]

# =============================================================================
# 3. FRAME CONNECTION
# =============================================================================
println("\n--- Running frame connection ---")

# Frame connect combines repeated localizations of the same fluorophore
fc_result = frameconnect(
    smld_noisy;
    maxframegap = 10,    # Allow gaps for blinking
    nsigmadev = 5.0      # Connection radius threshold
)
smld_combined = fc_result.combined
smld_connected = fc_result.connected
fc_params = fc_result.params

println("After frame connection: $(length(smld_combined.emitters)) combined localizations")
println("Compression ratio: $(round(length(smld_noisy.emitters) / length(smld_combined.emitters), digits=1))×")

# Report photophysics estimates from frame connection
println("\nEstimated photophysics:")
println("  k_on:     $(round(fc_params.k_on, digits=4)) per frame")
println("  k_off:    $(round(fc_params.k_off, digits=4)) per frame")
println("  k_bleach: $(round(fc_params.k_bleach, digits=4)) per frame")
println("  p_miss:   $(round(fc_params.p_miss, digits=4))")

# =============================================================================
# 4. RUN BAGOL
# =============================================================================
println("\n--- Running BaGoL ---")

# Estimate expected number of emitters (fluorophores, not nmers)
expected_fluors = length(true_positions)

result_smld, diagnostics = run_bagol(
    smld_combined;
    # Partitioning parameters
    nsigma = 3.0,
    max_partition_size = 200,
    # BaGoL parameters
    α = :auto,
    learn_α = true,
    λ_K = Float64(expected_fluors) / 100,  # Per-partition prior
    n_iterations = 10000,
    burn_in = 2000,
    verbose = true
)

n_bagol = diagnostics.n_emitters

println("\n" * "="^70)
println("RESULTS")
println("="^70)
println("True fluorophores: $(length(true_positions))")
println("Frame-connected localizations: $(length(smld_combined.emitters))")
println("Partitions: $(diagnostics.n_partitions)")
println("BaGoL MAP-N estimate: $n_bagol")
println("Final μ: $(round(diagnostics.final_μ, digits=2)) locs/emitter")

# =============================================================================
# 5. RESULT VISUALIZATION
# =============================================================================
println("\n--- Generating result visualization ---")

fig_results = Figure(size=(1200, 600))

# Left: Input localizations
ax1 = Axis(fig_results[1, 1], title="Frame-Connected Localizations",
           xlabel="x (μm)", ylabel="y (μm)", aspect=DataAspect())

locs = smld_combined.emitters
scatter!(ax1, [loc.x for loc in locs], [loc.y for loc in locs],
         color=(:gray, 0.3), markersize=2)

# Right: MAP-N emitters vs ground truth
ax2 = Axis(fig_results[1, 2], title="MAP-N Emitters ($n_bagol total)",
           xlabel="x (μm)", ylabel="y (μm)", aspect=DataAspect())

# Plot true positions
scatter!(ax2, [p[1] for p in true_positions], [p[2] for p in true_positions],
         marker='x', color=:black, markersize=6, alpha=0.3, label="True ($(length(true_positions)))")

# Plot MAP-N emitters
scatter!(ax2, [e.x for e in result_smld.emitters], [e.y for e in result_smld.emitters],
         color=:red, markersize=8, label="MAP-N ($n_bagol)")

axislegend(ax2, position=:rt, framevisible=false)

save(joinpath(OUTPUT_DIR, "pipeline_results.png"), fig_results)
println("Saved: $(joinpath(OUTPUT_DIR, "pipeline_results.png"))")

# =============================================================================
# 6. SUPER-RESOLUTION RENDERING
# =============================================================================
println("\n--- Generating SR renders ---")

sr_pixel_size = 5.0  # 5 nm pixels for SR reconstruction

# Create ground truth SMLD from true positions (for rendering)
true_emitters = [SMLMData.Emitter2DFit{Float64}(
    pos[1], pos[2],           # x, y
    1000.0,                   # photons
    0.0,                      # bg
    0.005, 0.005,             # σ_x, σ_y (5 nm - tight for ground truth)
    10.0, 0.0;                # σ_photons, σ_bg
    frame=1, dataset=1, track_id=0, id=i
) for (i, pos) in enumerate(true_positions)]
smld_truth = BasicSMLD(true_emitters, camera, 1, 1)

# BaGoL result SMLD (already have uncertainties)
smld_bagol = result_smld

# Primary output: Two-color circle plot (1 nm pixels)
# Frame-connected (cyan) vs BaGoL MAP-N (red)
println("  Rendering comparison circles (1 nm pixels)...")
render([smld_combined, smld_bagol]; colors=[:cyan, :red], pixel_size=1.0,
    strategy=CircleRender(), filename=joinpath(OUTPUT_DIR, "pipeline_comparison_circles.png"))

println("Saved: pipeline_comparison_circles.png (frame-connected cyan vs BaGoL red)")

# =============================================================================
# 7. SUMMARY STATISTICS
# =============================================================================
println("\n--- Summary Statistics ---")
println("Partitions processed: $(diagnostics.n_partitions)")
println("Acceptance rates:")
for (move, rate) in diagnostics.acceptance_rates
    println("  $move: $(round(100*rate, digits=1))%")
end

println("\n" * "="^70)
println("Pipeline complete!")
println("="^70)

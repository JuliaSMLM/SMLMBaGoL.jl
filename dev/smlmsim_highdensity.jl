# High-Density 6-mer Benchmark — GenMAb-Like Photophysics
# ========================================================
# Dense hexamers (25 nm diameter) with blinking stats learned from
# GenMAb HexaBody RGY data (Cell_01): μ ≈ 8.7 locs/emitter, shape ≈ 1.5
# High density so hexamers frequently overlap — stress test for BaGoL.
#
# Run with: julia --threads=auto --project=dev dev/smlmsim_highdensity.jl
# Set GEN_PLOTS=false to skip rendering (timing-only mode)

GEN_PLOTS = get(ENV, "GEN_PLOTS", "true") != "false"
PROGRESS_FILE = get(ENV, "PROGRESS_FILE", "/tmp/smlmsim_progress.txt")
_log(msg) = (println(msg); open(io -> println(io, msg), PROGRESS_FILE, "a"); flush(stdout))

_log("Loading packages...")
using SMLMBaGoL, SMLMData, SMLMSim
if GEN_PLOTS
    using SMLMRender, CairoMakie
end
using Statistics, Random
_log("Packages loaded.")

OUTPUT_DIR = joinpath(@__DIR__, "output", "smlmsim_highdensity")
rm(OUTPUT_DIR; force=true, recursive=true)
mkpath(OUTPUT_DIR)

Random.seed!(42)

# =============================================================================
# Parameters — calibrated to match GenMAb blinking stats
# =============================================================================

# GenMAb learned values (Cell_01 HexaBody RGY, 2×2 μm ROI):
#   μ = 8.72 locs/emitter, shape = 1.47
#   Photons: median 2700 (IQR 1385–5548)
#   Precision: median σ ≈ 7.3 nm (IQR 5.2–10.2 nm)
#   Density: ~4875 locs/μm² → ~560 emitters/μm²

# Camera/FOV — small FOV keeps runtime reasonable
CAMERA_PIXELS = 64
PIXEL_SIZE = 0.100                  # μm → 6.4 × 6.4 μm FOV
fov_size = CAMERA_PIXELS * PIXEL_SIZE

# Pattern — hexamers at high density so clusters frequently overlap
# d=0.025 μm (25 nm) hexamer diameter
# At density=8 patterns/μm² → ~327 hexamers in FOV → ~1964 emitters
# Mean inter-hexamer distance ≈ 1/√density ≈ 0.35 μm, but with Poisson
# clustering many will be within 50-100 nm → overlapping blink clouds
DENSITY = 8.0                       # hexamers/μm²
PATTERN_N = 6                       # hexamers
PATTERN_D = 0.025                   # 25 nm diameter

# Photophysics — tuned for ~8.7 detected blinks per emitter
# Expected blinks ≈ T × k_on where T = nframes/framerate
# Want ~8.7, with T = 500/50 = 10s → k_on ≈ 0.87
#
# Photons per blink ≈ PHOTON_RATE × τ_on = PHOTON_RATE / K_OFF
# SMLMSim CRLB: σ ≈ PSF/√photons. For σ≈7nm: photons ≈ (130/7)² ≈ 345
# GenMAb has σ≈7nm at ~2700 photons (real noise > CRLB), so we target
# σ≈7nm directly via lower photon count in the ideal SMLMSim model.
NFRAMES = 500
FRAMERATE = 50.0                    # T_acquisition = 10 s
PSF_SIGMA = 0.130                   # 130 nm PSF
MIN_PHOTONS = 50
PHOTON_RATE = 35000.0               # photons/s → ~350 photons/blink → σ≈7nm CRLB
K_OFF = 100.0                       # τ_on = 10 ms (fast blinking)
K_ON = 0.87                         # tuned for ~8.7 blinks in 10 s

# Precision filter — GenMAb used all data (median σ≈7nm), keep ≤ 15 nm
PRECISION_MAX = 0.015               # μm

# BaGoL parameters
N_ITERATIONS = 15_000
BURN_IN = 3_000
PARTITION_SIGMA = 2.0                   # tighter partitioning for dense data

n_patterns_est = round(Int, DENSITY * fov_size^2)
n_emitters_est = n_patterns_est * PATTERN_N

println("="^60)
println("High-Density 6-mer Benchmark — GenMAb-Like Photophysics")
println("="^60)
println("  FOV: $(fov_size) × $(fov_size) μm")
println("  Pattern: $(PATTERN_N)-mer, d=$(PATTERN_D*1000) nm, density=$(DENSITY)/μm²")
println("  Expected: ~$(n_patterns_est) hexamers → ~$(n_emitters_est) emitters")
println("  Mean hexamer spacing: ~$(round(1/sqrt(DENSITY)*1000, digits=0)) nm")
println("  Acquisition: $(NFRAMES) frames @ $(FRAMERATE) fps ($(NFRAMES/FRAMERATE)s)")
println("  Expected blinks/emitter: ~$(round(NFRAMES/FRAMERATE * K_ON, digits=1))")
println("  τ_on = $(round(1000/K_OFF, digits=1)) ms, τ_off = $(round(1000/K_ON, digits=0)) ms")

# =============================================================================
# Simulate
# =============================================================================

_log("\n" * "-"^60)
_log("Running SMLMSim...")

params = SMLMSim.StaticSMLMConfig(
    density = DENSITY,
    σ_psf = PSF_SIGMA,
    minphotons = MIN_PHOTONS,
    ndatasets = 1,
    nframes = NFRAMES,
    framerate = FRAMERATE,
    ndims = 2
)

pattern = SMLMSim.Nmer2D(n=PATTERN_N, d=PATTERN_D)
fluor = SMLMSim.GenericFluor(photons=PHOTON_RATE, k_off=K_OFF, k_on=K_ON)
camera = SMLMData.IdealCamera(CAMERA_PIXELS, CAMERA_PIXELS, PIXEL_SIZE)

smld_noisy, sim_info = SMLMSim.simulate(params; pattern=pattern, molecule=fluor, camera=camera)
smld_true = sim_info.smld_true

# Extract unique true emitter positions
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

true_positions = get_unique_true_positions(smld_true)
n_true = length(true_positions)
n_locs_raw = length(smld_noisy.emitters)

println("  True emitters: $n_true")
println("  Raw localizations: $n_locs_raw")
println("  Mean locs/emitter: $(round(n_locs_raw / n_true, digits=1))")

# Precision filter
filtered = filter(loc -> max(loc.σ_x, loc.σ_y) <= PRECISION_MAX, smld_noisy.emitters)
smld_noisy = SMLMData.BasicSMLD(filtered, camera, 1, 1)
n_locs = length(filtered)

median_sigma = median([max(l.σ_x, l.σ_y) for l in filtered]) * 1000
println("  After filter (σ ≤ $(PRECISION_MAX*1000) nm): $n_locs locs")
println("  Median σ: $(round(median_sigma, digits=1)) nm")
println("  Locs/emitter after filter: $(round(n_locs / n_true, digits=1))")
println("  Effective density: $(round(n_locs / fov_size^2, digits=0)) locs/μm²")

# =============================================================================
# Run BaGoL
# =============================================================================

_log("\n" * "-"^60)
_log("Running BaGoL (collapsed Gibbs)...")

# Compute true μ and shape from simulation's per-emitter blink counts
emitter_counts = Dict{Int, Int}()
for loc in smld_noisy.emitters
    tid = loc.track_id
    emitter_counts[tid] = get(emitter_counts, tid, 0) + 1
end
counts = collect(values(emitter_counts))
TRUE_MU = mean(counts)
TRUE_SHAPE = TRUE_MU^2 / var(counts)  # MoM Gamma fit

println("  True μ = $(round(TRUE_MU, digits=2)), shape = $(round(TRUE_SHAPE, digits=2)) (from $(length(counts)) emitters)")
println("  Hierarchical: learn_distribution=true, initial shape=2.0, μ prior = Gamma(2, $(round(TRUE_MU/2, digits=1)))")

fov = compute_fov(smld_noisy)

t_bagol = @elapsed begin
    result_smld, diag = run_bagol(smld_noisy;
        partition_sigma = PARTITION_SIGMA,
        n_iterations = N_ITERATIONS,
        burn_in = BURN_IN,
        μ = TRUE_MU,
        shape = 2.0,
        learn_distribution = true,
        sync_interval = 500,
        posterior_pixel_size = GEN_PLOTS ? 0.002 : 0.0,
        posterior_xlim = (fov[1], fov[2]),
        posterior_ylim = (fov[3], fov[4]),
        verbose = true)
end
_log("\n  run_bagol: $(round(t_bagol, digits=1))s")
GC.gc()
_log("  Live bytes: $(round(Base.gc_live_bytes() / 1024^2, digits=1)) MB")

# =============================================================================
# Report
# =============================================================================

_log("\n" * "-"^60)
_log("Computing report...")

report = compute_report(result_smld, diag;
    true_positions = true_positions,
    locs_smld = smld_noisy,
    count_params = (μ = TRUE_MU, shape = TRUE_SHAPE))

write_report(report; output_dir=OUTPUT_DIR)

println("\n-- Results --")
println("  True emitters: $(report.k_true)")
println("  Estimated emitters: $(report.n_emitters)")
println("  Jaccard: $(round(report.jaccard, digits=3))")
println("  Precision: $(round(report.precision, digits=3))")
println("  Recall: $(round(report.recall, digits=3))")
println("  RMSE: $(round(report.rmse * 1000, digits=1)) nm")
println("  Learned μ: $(round(report.final_mu, digits=2))")
println("  Learned shape: $(round(report.final_shape, digits=2))")

if !GEN_PLOTS
    println("\n" * "="^60)
    println("Plots skipped (GEN_PLOTS=false)")
    println("="^60)
    exit(0)
end

_log("Generating plots and renders...")

plot_report(report; output_dir=OUTPUT_DIR)
render_report(smld_noisy, result_smld;
    output_dir = OUTPUT_DIR,
    true_positions = true_positions,
    fov = fov)

println("\n" * "="^60)
println("Output: $OUTPUT_DIR")
println("="^60)

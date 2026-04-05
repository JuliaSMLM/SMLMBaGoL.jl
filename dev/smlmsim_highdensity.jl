# High-Density 6-mer + Monomer Benchmark — GenMAb-Like Photophysics
# ==================================================================
# Dense hexamers (25 nm diameter) + equal density of monomers, with
# blinking stats learned from GenMAb HexaBody RGY data (Cell_01):
# μ ≈ 8.7 locs/emitter, shape ≈ 1.5
# 128/μm² hexamers + 128/μm² monomers — extreme density stress test
# (~36K emitters, ~350K locs in 6.4×6.4 μm FOV)
#
# Run with: julia --threads=auto --project=dev dev/smlmsim_highdensity.jl
# Set GEN_PLOTS=false to skip rendering (timing-only mode)

GEN_PLOTS = get(ENV, "GEN_PLOTS", "true") != "false"

using SMLMBaGoL, SMLMData, SMLMSim
if GEN_PLOTS
    using SMLMRender, CairoMakie
end
using Statistics, Random

OUTPUT_DIR = joinpath(@__DIR__, "output", "smlmsim_highdensity")
rm(OUTPUT_DIR; force=true, recursive=true)
mkpath(OUTPUT_DIR)

PROGRESS_FILE = joinpath(OUTPUT_DIR, "progress.log")
_log(msg) = (println(msg); open(io -> println(io, msg), PROGRESS_FILE, "a"); flush(stdout))

_log("Packages loaded.")

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

# Pattern — hexamers at extreme density so clusters frequently overlap
# d=0.025 μm (25 nm) hexamer diameter
# At density=128 patterns/μm² → ~5243 hexamers in FOV → ~31457 emitters
# Plus ~5243 monomers scattered randomly (equal pattern count)
# Mean inter-pattern distance ≈ 1/√(density_total) ≈ 62 nm
DENSITY_NMER = 128.0                # hexamers/μm²
DENSITY_MONO = DENSITY_NMER         # monomers/μm² (equal count)
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

n_nmer_est = round(Int, DENSITY_NMER * fov_size^2)
n_mono_est = round(Int, DENSITY_MONO * fov_size^2)
n_emitters_est = n_nmer_est * PATTERN_N + n_mono_est
density_total = DENSITY_NMER + DENSITY_MONO

println("="^60)
println("High-Density 6-mer + Monomer Benchmark — GenMAb-Like Photophysics")
println("="^60)
println("  FOV: $(fov_size) × $(fov_size) μm")
println("  Hexamers: $(PATTERN_N)-mer, d=$(PATTERN_D*1000) nm, density=$(DENSITY_NMER)/μm²")
println("  Monomers: density=$(DENSITY_MONO)/μm²")
println("  Expected: ~$(n_nmer_est) hexamers + ~$(n_mono_est) monomers → ~$(n_emitters_est) emitters")
println("  Mean inter-pattern spacing: ~$(round(1/sqrt(density_total)*1000, digits=0)) nm")
println("  Acquisition: $(NFRAMES) frames @ $(FRAMERATE) fps ($(NFRAMES/FRAMERATE)s)")
println("  Expected blinks/emitter: ~$(round(NFRAMES/FRAMERATE * K_ON, digits=1))")
println("  τ_on = $(round(1000/K_OFF, digits=1)) ms, τ_off = $(round(1000/K_ON, digits=0)) ms")

# =============================================================================
# Simulate
# =============================================================================

_log("\n" * "-"^60)
_log("Running SMLMSim (hexamers + monomers)...")

fluor = SMLMSim.GenericFluor(photons=PHOTON_RATE, k_off=K_OFF, k_on=K_ON)
camera = SMLMData.IdealCamera(CAMERA_PIXELS, CAMERA_PIXELS, PIXEL_SIZE)

# --- Hexamer simulation ---
params_nmer = SMLMSim.StaticSMLMConfig(
    density = DENSITY_NMER,
    σ_psf = PSF_SIGMA,
    minphotons = MIN_PHOTONS,
    ndatasets = 1,
    nframes = NFRAMES,
    framerate = FRAMERATE,
    ndims = 2
)
pattern_nmer = SMLMSim.Nmer2D(n=PATTERN_N, d=PATTERN_D)
smld_nmer, info_nmer = SMLMSim.simulate(params_nmer; pattern=pattern_nmer, molecule=fluor, camera=camera)

# --- Monomer simulation ---
params_mono = SMLMSim.StaticSMLMConfig(
    density = DENSITY_MONO,
    σ_psf = PSF_SIGMA,
    minphotons = MIN_PHOTONS,
    ndatasets = 1,
    nframes = NFRAMES,
    framerate = FRAMERATE,
    ndims = 2
)
pattern_mono = SMLMSim.Nmer2D(n=1, d=0.0)
smld_mono, info_mono = SMLMSim.simulate(params_mono; pattern=pattern_mono, molecule=fluor, camera=camera)

# --- Merge simulations ---
# Offset monomer track_ids and pattern ids to avoid collisions with hexamer IDs
function offset_smld(smld::SMLMData.SMLD, track_offset, id_offset)
    new_emitters = [SMLMData.Emitter2DFit{Float64}(
        e.x, e.y, e.photons, e.bg, e.σ_x, e.σ_y, e.σ_photons, e.σ_bg;
        σ_xy=e.σ_xy, frame=e.frame, dataset=e.dataset,
        track_id=e.track_id + track_offset, id=e.id + id_offset
    ) for e in smld.emitters]
    SMLMData.BasicSMLD(new_emitters, smld.camera, smld.n_frames, smld.n_datasets)
end

smld_mono_offset = offset_smld(smld_mono,
    maximum(e.track_id for e in smld_nmer.emitters),
    maximum(e.id for e in smld_nmer.emitters))
smld_noisy = SMLMData.cat_smld(smld_nmer, smld_mono_offset)

true_mono_offset = offset_smld(info_mono.smld_true,
    maximum(e.track_id for e in info_nmer.smld_true.emitters),
    maximum(e.id for e in info_nmer.smld_true.emitters))
smld_true = SMLMData.cat_smld(info_nmer.smld_true, true_mono_offset)

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

println("  Hexamers: $(info_nmer.n_patterns) patterns, $(info_nmer.n_emitters) emitters, $(info_nmer.n_localizations) locs")
println("  Monomers: $(info_mono.n_patterns) patterns, $(info_mono.n_emitters) emitters, $(info_mono.n_localizations) locs")

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
        progress_file = PROGRESS_FILE,
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

# Skip true_positions — Hungarian matching is O(n³), intractable at ~36K emitters.
# True emitter count printed separately for reference.
report = compute_report(result_smld, diag;
    locs_smld = smld_noisy,
    count_params = (μ = TRUE_MU, shape = TRUE_SHAPE))

write_report(report; output_dir=OUTPUT_DIR)

println("\n-- Results --")
println("  True emitters: $n_true")
println("  Estimated emitters: $(report.n_emitters)")
println("  Compression: $(round(n_locs / report.n_emitters, digits=1))×")
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
    partition_ids = report.partition_ids,
    fov = fov)

println("\n" * "="^60)
println("Output: $OUTPUT_DIR")
println("="^60)

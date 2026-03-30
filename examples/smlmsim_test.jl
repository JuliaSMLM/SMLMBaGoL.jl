# SMLMSim Realistic Photophysics — Standard Report
# ==================================================
# BaGoL with realistic SMLMSim photophysics (Nmer2D pattern + GenericFluor).
#
# Run with: julia --threads=auto --project=examples examples/smlmsim_test.jl

using Pkg; Pkg.activate(@__DIR__)
using SMLMBaGoL
using SMLMData
using SMLMSim
using CairoMakie
using SMLMRender
using Random
using Statistics

# =============================================================================
# Parameters
# =============================================================================

const SEED = 42
const CAMERA_PIXELS = 64
const PIXEL_SIZE = 0.100          # μm → 6.4×6.4 μm FOV
const DENSITY = 2.0               # patterns per μm²
const N_EMITTERS = 8              # 8-mer
const CLUSTER_DIAMETER = 0.050    # 50 nm
const PSF_SIGMA = 0.130           # 130 nm
const NFRAMES = 1000
const FRAMERATE = 50.0
const MIN_PHOTONS = 100

# Fluorophore: k_off=50Hz (τ_on=20ms), k_on=0.5Hz → ~10 blinks in 20s
const PHOTON_RATE = 50000.0
const K_OFF = 50.0
const K_ON = 0.5

const N_ITERATIONS = 15000
const BURN_IN = 3000

# =============================================================================
# Simulate with SMLMSim
# =============================================================================

Random.seed!(SEED)

camera = SMLMData.IdealCamera(CAMERA_PIXELS, CAMERA_PIXELS, PIXEL_SIZE)

params = SMLMSim.StaticSMLMConfig(;
    density=DENSITY, nframes=NFRAMES, framerate=FRAMERATE,
    ndatasets=1, σ_psf=PSF_SIGMA, minphotons=MIN_PHOTONS
)

pattern = SMLMSim.Nmer2D(n=N_EMITTERS, d=CLUSTER_DIAMETER)
fluor = SMLMSim.GenericFluor(photons=PHOTON_RATE, k_off=K_OFF, k_on=K_ON)

smld_sim, info = SMLMSim.simulate(params;
    pattern=pattern, molecule=fluor, camera=camera
)

# Extract true positions
true_emitters = info.smld_true.emitters
true_positions = unique([(e.x, e.y) for e in true_emitters])
n_locs = length(smld_sim.emitters)
println("SMLMSim: $n_locs locs from $(info.n_emitters) emitters in $(info.n_patterns) patterns")
println("  $(length(true_positions)) unique true positions")
println("  Mean locs/emitter: $(round(n_locs / length(true_positions), digits=1))")
println("  Mean σ: $(round(mean(SMLMBaGoL.mean_sigma(e) for e in smld_sim.emitters) * 1000, digits=1)) nm")

fov = compute_fov(smld_sim)

# =============================================================================
# Run BaGoL
# =============================================================================

result_smld, diagnostics = run_bagol(smld_sim;
    n_iterations=N_ITERATIONS, burn_in=BURN_IN,
    posterior_pixel_size=0.002,
    posterior_xlim=(fov[1], fov[2]),
    posterior_ylim=(fov[3], fov[4])
)

# =============================================================================
# Standard outputs
# =============================================================================

output_dir = joinpath(@__DIR__, "output", "smlmsim")

# Empirical count params from simulation (shape unknown, use Poisson approx)
empirical_μ = n_locs / length(true_positions)
report = compute_report(result_smld, diagnostics;
    true_positions=true_positions, locs_smld=smld_sim,
    count_params=(μ=empirical_μ, shape=1000.0))

write_report(report; output_dir)
plot_report(report; output_dir)
render_report(smld_sim, result_smld;
    output_dir, true_positions=true_positions, partition_ids=report.partition_ids, fov=fov)

println("\nResults in $output_dir")

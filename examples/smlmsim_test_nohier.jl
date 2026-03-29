# SMLMSim Realistic Photophysics — No Hierarchical Learning
# ===========================================================
# Run with: julia --threads=auto --project=examples examples/smlmsim_test_nohier.jl

using Pkg; Pkg.activate(@__DIR__)
using SMLMBaGoL
using SMLMData
using SMLMSim
using CairoMakie
using SMLMRender
using Random
using Statistics

# =============================================================================
# Parameters (same as smlmsim_test.jl)
# =============================================================================

const SEED = 42
const CAMERA_PIXELS = 64
const PIXEL_SIZE = 0.100
const DENSITY = 2.0
const N_EMITTERS = 8
const CLUSTER_DIAMETER = 0.050
const PSF_SIGMA = 0.130
const NFRAMES = 1000
const FRAMERATE = 50.0
const MIN_PHOTONS = 100
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

true_emitters = info.smld_true.emitters
true_positions = unique([(e.x, e.y) for e in true_emitters])
n_locs = length(smld_sim.emitters)

# Empirical μ from SMLMSim (not from our count model)
TRUE_MU = n_locs / length(true_positions)
println("SMLMSim: $n_locs locs, $(length(true_positions)) true emitters, μ=$(round(TRUE_MU, digits=2))")

fov = compute_fov(smld_sim)

# =============================================================================
# Run BaGoL (fixed parameters)
# =============================================================================

result_smld, diagnostics = run_bagol(smld_sim;
    n_iterations=N_ITERATIONS, burn_in=BURN_IN,
    μ=TRUE_MU, shape=1000.0, learn_distribution=false,
    sync_interval=N_ITERATIONS + 1,
    posterior_pixel_size=0.002,
    posterior_xlim=(fov[1], fov[2]),
    posterior_ylim=(fov[3], fov[4])
)

# =============================================================================
# Standard outputs
# =============================================================================

output_dir = joinpath(@__DIR__, "output", "smlmsim_nohier")

report = compute_report(result_smld, diagnostics;
    true_positions=true_positions, locs_smld=smld_sim)

write_report(report; output_dir)
plot_report(report; output_dir)
render_report(smld_sim, result_smld;
    output_dir, true_positions=true_positions, partition_ids=report.partition_ids, fov=fov)

println("\nResults in $output_dir")

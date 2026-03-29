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
# Parameters
# =============================================================================

const SEED = 42
const CAMERA_PIXELS = 64
const PIXEL_SIZE = 0.100
const DENSITY = 2.0
const NFRAMES = 1000
const MIN_PHOTONS = 100
const N_ITERATIONS = 15000
const BURN_IN = 3000

# =============================================================================
# Simulate with SMLMSim
# =============================================================================

Random.seed!(SEED)

camera = SMLMData.IdealCamera(CAMERA_PIXELS, CAMERA_PIXELS, PIXEL_SIZE)

config = SMLMSim.StaticSMLMConfig(;
    density=DENSITY, nframes=NFRAMES, framerate=50.0,
    ndatasets=1, minphotons=MIN_PHOTONS
)

smld_sim, info = SMLMSim.simulate(config; camera=camera)
println("SMLMSim: $(info.n_localizations) locs from $(info.n_emitters) emitters")

true_emitters = info.smld_true.emitters
true_positions = [(e.x, e.y) for e in true_emitters]

# SMLMSim doesn't use our count model, so compute empirical μ
TRUE_MU = length(smld_sim.emitters) / length(true_positions)
println("Empirical μ = $(round(TRUE_MU, digits=2))")

fov = compute_fov(smld_sim)

# =============================================================================
# Run BaGoL (fixed parameters)
# =============================================================================

result_smld, diagnostics = run_bagol(smld_sim;
    n_iterations=N_ITERATIONS, burn_in=BURN_IN,
    μ=TRUE_MU, shape=1000.0, learn_shape=false,
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
    output_dir, true_positions=true_positions, fov=fov)

println("\nResults in $output_dir")

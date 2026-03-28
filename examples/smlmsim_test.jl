# SMLMSim Realistic Photophysics — Standard Report
# ==================================================
# BaGoL with realistic SMLMSim photophysics.
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
println("SMLMSim: $(info.n_localizations) locs from $(info.n_emitters) emitters in $(info.n_patterns) patterns")

# Extract true positions from model SMLD
true_emitters = info.smld_true.emitters
true_positions = [(e.x, e.y) for e in true_emitters]
println("  $(length(true_positions)) true emitter positions")
println("  Mean σ: $(round(mean(SMLMBaGoL.mean_sigma(e) for e in smld_sim.emitters) * 1000, digits=1)) nm")

# =============================================================================
# Run BaGoL
# =============================================================================

result_smld, diagnostics = run_bagol(smld_sim;
    n_iterations=N_ITERATIONS, burn_in=BURN_IN,
    posterior_pixel_size=0.002
)

# =============================================================================
# Standard outputs
# =============================================================================

output_dir = joinpath(@__DIR__, "output", "smlmsim")

report = compute_report(result_smld, diagnostics;
    true_positions=true_positions, locs_smld=smld_sim)

write_report(report; output_dir)
plot_report(report; output_dir)
render_report(smld_sim, result_smld;
    output_dir, true_positions=true_positions)

println("\nResults in $output_dir")

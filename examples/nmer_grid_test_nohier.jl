# Grid of Identical N-mers — No Hierarchical Learning
# ====================================================
# Run with: julia --threads=auto --project=examples examples/nmer_grid_test_nohier.jl

using Pkg; Pkg.activate(@__DIR__)
using SMLMBaGoL
using SMLMData
using CairoMakie
using SMLMRender
using Random

# =============================================================================
# Parameters
# =============================================================================

const SEED = 42
const GRID_SIZE = 4
const GRID_SPACING = 1.0
const N_EMITTERS = 8
const CLUSTER_DIAMETER = 0.050
const PSF_SIGMA = 0.130
const PHOTON_MEAN = 500.0
const PHOTON_MIN = 100.0
const BLINK_MEAN = 10.0
const N_ITERATIONS = 15000
const BURN_IN = 3000
const TRUE_SHAPE = 1000.0

# =============================================================================
# Simulate
# =============================================================================

Random.seed!(SEED)

sim = simulate_nmer_grid(;
    n_per_cluster=N_EMITTERS, cluster_diameter=CLUSTER_DIAMETER,
    grid_nx=GRID_SIZE, grid_ny=GRID_SIZE, grid_spacing=GRID_SPACING,
    mean_count=BLINK_MEAN, psf_sigma=PSF_SIGMA,
    mean_photons=PHOTON_MEAN, min_photons=PHOTON_MIN, pixel_size=0.100
)
print_simulation_summary(sim)

TRUE_MU = length(sim.smld.emitters) / (GRID_SIZE^2 * N_EMITTERS)
println("True μ = $(round(TRUE_MU, digits=2))")

fov = compute_fov(sim.smld)

# =============================================================================
# Run BaGoL (fixed parameters)
# =============================================================================

result_smld, diagnostics = run_bagol(sim.smld;
    n_iterations=N_ITERATIONS, burn_in=BURN_IN,
    shape=TRUE_SHAPE, learn_shape=false,
    sync_interval=N_ITERATIONS + 1,
    μ=TRUE_MU,
    posterior_pixel_size=0.002,
    posterior_xlim=(fov[1], fov[2]),
    posterior_ylim=(fov[3], fov[4])
)

# =============================================================================
# Standard outputs
# =============================================================================

output_dir = joinpath(@__DIR__, "output", "nmer_grid_nohier")

report = compute_report(result_smld, diagnostics;
    true_positions=sim.true_positions, locs_smld=sim.smld)

write_report(report; output_dir)
plot_report(report; output_dir)
render_report(sim.smld, result_smld;
    output_dir, true_positions=sim.true_positions, fov=fov)

println("\nResults in $output_dir")

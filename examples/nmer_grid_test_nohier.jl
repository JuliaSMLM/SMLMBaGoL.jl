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
const GRID_SPACING = 0.250           # μm (250 nm)
const N_EMITTERS = 8
const CLUSTER_DIAMETER = 0.050
const PSF_SIGMA = 0.130
const PHOTON_MEAN = 500.0
const PHOTON_MIN = 100.0
const BLINK_MEAN = 10.0
const N_ITERATIONS = 15000
const BURN_IN = 3000

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
println("Count params: μ=$(round(sim.count_params.μ, digits=2)), shape=$(round(sim.count_params.shape, digits=1))")

fov = compute_fov(sim.smld; margin=GRID_SPACING / 2)

# =============================================================================
# Run BaGoL (fixed parameters from simulation)
# =============================================================================

result_smld, diagnostics = run_bagol(sim.smld;
    n_iterations=N_ITERATIONS, burn_in=BURN_IN,
    μ=sim.count_params.μ, shape=sim.count_params.shape, learn_shape=false,
    sync_interval=N_ITERATIONS + 1,
    posterior_pixel_size=0.002,
    posterior_xlim=(fov[1], fov[2]),
    posterior_ylim=(fov[3], fov[4])
)

# =============================================================================
# Standard outputs
# =============================================================================

output_dir = joinpath(@__DIR__, "output", "nmer_grid_nohier")

report = compute_report(result_smld, diagnostics;
    true_positions=sim.true_positions, locs_smld=sim.smld, count_params=sim.count_params)

write_report(report; output_dir)
plot_report(report; output_dir)
render_report(sim.smld, result_smld;
    output_dir, true_positions=sim.true_positions, partition_ids=report.partition_ids, fov=fov)

println("\nResults in $output_dir")

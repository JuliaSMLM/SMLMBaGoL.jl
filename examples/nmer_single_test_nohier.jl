# Single N-mer — No Hierarchical Learning
# =========================================
# Same as nmer_single_test but μ and shape fixed at true values.
#
# Run with: julia --project=examples examples/nmer_single_test_nohier.jl

using Pkg; Pkg.activate(@__DIR__)
using SMLMBaGoL
using SMLMData
using CairoMakie
using SMLMRender
using Random

# =============================================================================
# Parameters
# =============================================================================

const SEED = nothing
const N_EMITTERS = 6
const CLUSTER_DIAMETER = 0.025    # μm (25 nm)
const PSF_SIGMA = 0.130
const PHOTON_MEAN = 500.0
const PHOTON_MIN = 100.0
const BLINK_MEAN = 10.0
const N_ITERATIONS = 20000
const BURN_IN = 4000

# =============================================================================
# Simulate
# =============================================================================

SEED !== nothing && Random.seed!(SEED)

sim = simulate_nmer(;
    n=N_EMITTERS, diameter=CLUSTER_DIAMETER,
    mean_count=BLINK_MEAN, psf_sigma=PSF_SIGMA,
    mean_photons=PHOTON_MEAN, min_photons=PHOTON_MIN,
    pixel_size=0.100, field_size=25.6
)
print_simulation_summary(sim)
println("Count params: μ=$(round(sim.count_params.μ, digits=2)), shape=$(round(sim.count_params.shape, digits=1))")

fov = compute_fov(sim.smld)

# =============================================================================
# Run BaGoL (fixed parameters from simulation)
# =============================================================================

result_smld, diagnostics = run_bagol(sim.smld;
    n_iterations=N_ITERATIONS, burn_in=BURN_IN,
    partition_sigma=Inf,
    μ=sim.count_params.μ, shape=sim.count_params.shape, learn_distribution=false,
    sync_interval=N_ITERATIONS + 1,
    posterior_pixel_size=0.001,
    posterior_xlim=(fov[1], fov[2]),
    posterior_ylim=(fov[3], fov[4])
)

# =============================================================================
# Standard outputs
# =============================================================================

output_dir = joinpath(@__DIR__, "output", "nmer_single_nohier")

report = compute_report(result_smld, diagnostics;
    true_positions=sim.true_positions, locs_smld=sim.smld, count_params=sim.count_params)

write_report(report; output_dir)
plot_report(report; output_dir)
render_report(sim.smld, result_smld;
    output_dir, true_positions=sim.true_positions, partition_ids=report.partition_ids, fov=fov)

println("\nResults in $output_dir")

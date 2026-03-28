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
const TRUE_SHAPE = 1000.0         # large shape ≈ Poisson

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

TRUE_MU = length(sim.smld.emitters) / N_EMITTERS
println("True μ = $(round(TRUE_MU, digits=2))")

fov = compute_fov(sim.smld)

# =============================================================================
# Run BaGoL (fixed parameters)
# =============================================================================

result_smld, diagnostics = run_bagol(sim.smld;
    n_iterations=N_ITERATIONS, burn_in=BURN_IN,
    nsigma=Inf,
    shape=TRUE_SHAPE, learn_shape=false,
    sync_interval=N_ITERATIONS + 1,
    μ_prior_shape=TRUE_MU,
    posterior_pixel_size=0.001,
    posterior_xlim=(fov[1], fov[2]),
    posterior_ylim=(fov[3], fov[4])
)

# =============================================================================
# Standard outputs
# =============================================================================

output_dir = joinpath(@__DIR__, "output", "nmer_single_nohier")

report = compute_report(result_smld, diagnostics;
    true_positions=sim.true_positions, locs_smld=sim.smld)

write_report(report; output_dir)
plot_report(report; output_dir)
render_report(sim.smld, result_smld;
    output_dir, true_positions=sim.true_positions, fov=fov)

println("\nResults in $output_dir")

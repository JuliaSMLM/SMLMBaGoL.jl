# Single N-mer — Standard Report
# ===============================
# Run with: julia --project=examples examples/nmer_single_test.jl

using Pkg; Pkg.activate(@__DIR__)
using SMLMBaGoL
using SMLMData
using CairoMakie   # activates BaGoLMakieExt
using SMLMRender   # activates BaGoLRenderExt
using Random

# =============================================================================
# Parameters
# =============================================================================

const SEED = nothing
const N_EMITTERS = 6
const CLUSTER_DIAMETER = 0.025    # μm (25 nm)
const PSF_SIGMA = 0.130           # μm (130 nm)
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

# =============================================================================
# Run BaGoL
# =============================================================================

result_smld, diagnostics = run_bagol(sim.smld;
    n_iterations=N_ITERATIONS, burn_in=BURN_IN,
    nsigma=Inf,  # single cluster, no partitioning
    posterior_pixel_size=0.001
)

# =============================================================================
# Standard outputs
# =============================================================================

output_dir = joinpath(@__DIR__, "output", "nmer_single")

report = compute_report(result_smld, diagnostics;
    true_positions=sim.true_positions, locs_smld=sim.smld)

write_report(report; output_dir)
plot_report(report; output_dir)
render_report(sim.smld, result_smld;
    output_dir, true_positions=sim.true_positions)

println("\nResults in $output_dir")

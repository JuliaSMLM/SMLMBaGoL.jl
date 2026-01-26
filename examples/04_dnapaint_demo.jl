# DNA-PAINT Simulation Demo
# =========================
# Simulates DNA-PAINT-like data with Poisson count distribution (α large).
# Uses α=:auto to estimate shape parameter from frame statistics.
#
# DNA-PAINT characteristics:
# - Imager strands bind transiently at constant rate
# - Number of binding events ~ Poisson(λT) for acquisition time T
# - Variance ≈ mean (Fano factor ≈ 1)
# - Count distribution is approximately Poisson (α → ∞)
#
# Run with: julia --project=examples --threads=4 examples/04_dnapaint_demo.jl

using Pkg
Pkg.activate(@__DIR__)

using SMLMBaGoL
using SMLMData
using Random
using Statistics

# Include visualization functions from examples/
include(joinpath(@__DIR__, "viz_chain_diagnostics.jl"))

const OUTPUT_DIR = joinpath(@__DIR__, "output")
mkpath(OUTPUT_DIR)

println("="^60)
println("DNA-PAINT Simulation Demo")
println("="^60)

Random.seed!(123)

# -----------------------------------------------------------------------------
# Simulate DNA-PAINT-like data
# -----------------------------------------------------------------------------
# High α gives Poisson-like count distribution (variance ≈ mean)
# This models constant-rate binding where each emitter gets similar counts.

println("\nSimulating DNA-PAINT-like data...")
println("  α = 20.0 (Poisson-like, low variance)")
println("  μ = 10.0 (mean localizations per emitter)")

sim = simulate_nmers(
    n_dimers = 4,
    n_trimers = 2,
    emitter_spacing = 0.040,  # 40 nm
    cluster_spacing = 0.200,  # 200 nm
    μ = 10.0,
    α = 20.0,  # High α → Poisson-like → low variance
    σ_psf = 0.130,        # 130 nm PSF
    mean_photons = 800.0, # DNA-PAINT often has more photons
    background = 5.0,     # Lower background
    n_frames = 1000
)

print_simulation_summary(sim)

n_emitters = length(sim.true_positions)
println("\nTrue emitter count: $n_emitters")

# -----------------------------------------------------------------------------
# Run BaGoL with α=:auto
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Running BaGoL with α=:auto")
println("-"^60)
println("Will estimate α from frame statistics (Fano factor)...")

chain = run_bagol_chain(
    sim.localizations;
    α = :auto,  # Estimate from frame statistics
    λ_K = Float64(n_emitters),
    n_iterations = 15000,
    burn_in = 3000,
    verbose = true
)

emitters, posterior_k = estimate_mapn(chain)

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
println("\n" * "="^60)
println("Results")
println("="^60)
println("True emitters: $n_emitters")
println("MAP-N: $(length(emitters))")
println("True α: 20.0 (Poisson-like)")
println("Estimated α: $(round(chain.α, digits=1))")

# Compute Fano factor from frame statistics for comparison
frames = [loc.frame for loc in sim.localizations]
frame_counts = zeros(Int, maximum(frames))
for f in frames
    frame_counts[f] += 1
end
nonzero = filter(x -> x > 0, frame_counts)
fano = var(nonzero) / mean(nonzero)
println("\nFrame statistics:")
println("  Fano factor: $(round(fano, digits=2)) (1.0 = Poisson)")

# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
fig = plot_bagol(chain, emitters, posterior_k, sim.localizations;
    true_positions=sim.true_positions,
    save_path=joinpath(OUTPUT_DIR, "dnapaint_result.png"))
println("\nSaved: $(joinpath(OUTPUT_DIR, "dnapaint_result.png"))")

fig2 = plot_hierarchical_diagnostics(chain;
    true_locs_per_emitter=sim.true_counts,
    save_path=joinpath(OUTPUT_DIR, "dnapaint_hierarchical.png"))
println("Saved: $(joinpath(OUTPUT_DIR, "dnapaint_hierarchical.png"))")

# Show count distribution comparison
println("\n" * "-"^60)
println("Count Distribution Comparison")
println("-"^60)
println("True counts:     $(sim.true_counts)")
println("True mean:       $(round(mean(sim.true_counts), digits=1))")
println("True std:        $(round(std(sim.true_counts), digits=1))")
println("Expected std (Poisson): $(round(sqrt(sim.μ), digits=1))")
println("Expected std (α=20):    $(round(sqrt(sim.μ * (1 + sim.μ/20)), digits=1))")

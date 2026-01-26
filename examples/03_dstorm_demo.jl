# dSTORM Simulation Demo
# ======================
# Simulates dSTORM-like data with exponential count distribution (α ≈ 1).
# Uses learn_α=true to infer the shape parameter from data.
#
# dSTORM characteristics:
# - Fluorophores blink multiple times before photobleaching
# - High variance in number of localizations per emitter
# - Count distribution is approximately exponential (α ≈ 1)
#
# Run with: julia --project=examples --threads=4 examples/03_dstorm_demo.jl

using Pkg
Pkg.activate(@__DIR__)

using SMLMBaGoL
using SMLMData
using Random
using CairoMakie
using Statistics

const OUTPUT_DIR = joinpath(@__DIR__, "output")
mkpath(OUTPUT_DIR)

println("="^60)
println("dSTORM Simulation Demo")
println("="^60)

Random.seed!(42)

# -----------------------------------------------------------------------------
# Simulate dSTORM-like data
# -----------------------------------------------------------------------------
# α ≈ 1 gives exponential-like count distribution (high variance)
# This models photobleaching where some fluorophores blink many times,
# others blink only a few times before bleaching.

println("\nSimulating dSTORM-like data...")
println("  α = 1.2 (exponential-like, high variance)")
println("  μ = 8.0 (mean localizations per emitter)")

sim = simulate_nmers(
    n_dimers = 4,
    n_trimers = 2,
    emitter_spacing = 0.040,  # 40 nm
    cluster_spacing = 0.200,  # 200 nm
    μ = 8.0,
    α = 1.2,  # Low α → exponential-like → high variance
    σ_psf = 0.130,        # 130 nm PSF
    mean_photons = 500.0, # Typical dSTORM photon count
    background = 10.0,
    n_frames = 1000
)

print_simulation_summary(sim)

n_emitters = length(sim.true_positions)
println("\nTrue emitter count: $n_emitters")

# -----------------------------------------------------------------------------
# Run BaGoL with α learning
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Running BaGoL with learn_α=true")
println("-"^60)
println("Starting from α=2.0, will learn true value from data...")

chain = run_bagol_chain(
    sim.localizations;
    α = 2.0,        # Initial guess (will be updated)
    learn_α = true, # Learn α from data
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
println("True α: 1.2")
println("Learned α: $(round(chain.α, digits=2))")

α_samples = [s.α for s in chain.samples]
println("α posterior: $(round(mean(α_samples), digits=2)) ± $(round(std(α_samples), digits=2))")

# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
fig = plot_bagol(chain, emitters, posterior_k, sim.localizations;
    true_positions=sim.true_positions,
    save_path=joinpath(OUTPUT_DIR, "dstorm_result.png"))
println("\nSaved: $(joinpath(OUTPUT_DIR, "dstorm_result.png"))")

fig2 = plot_hierarchical_diagnostics(chain;
    true_locs_per_emitter=sim.true_counts,
    save_path=joinpath(OUTPUT_DIR, "dstorm_hierarchical.png"))
println("Saved: $(joinpath(OUTPUT_DIR, "dstorm_hierarchical.png"))")

# Show count distribution comparison
println("\n" * "-"^60)
println("Count Distribution Comparison")
println("-"^60)
println("True counts:     $(sim.true_counts)")
println("True mean:       $(round(mean(sim.true_counts), digits=1))")
println("True std:        $(round(std(sim.true_counts), digits=1))")
println("Expected std (α=1.2): $(round(sqrt(sim.μ * (1 + sim.μ/1.2)), digits=1))")

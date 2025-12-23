# Basic BaGoL Demo
# =================
# Demonstrates the core API workflow.
#
# Run with: julia --project=examples examples/01_basic_demo.jl

using Pkg
Pkg.activate(@__DIR__)

using SMLMBaGoL
using SMLMData
using Random
using CairoMakie

const OUTPUT_DIR = joinpath(@__DIR__, "output")

println("="^60)
println("Basic BaGoL Demo")
println("="^60)

# -----------------------------------------------------------------------------
# Create synthetic data: 2 well-separated emitters
# -----------------------------------------------------------------------------
Random.seed!(42)

# Two emitters 150nm apart (well-separated)
true_positions = [(0.100, 0.100), (0.250, 0.100)]
σ_loc = 0.008  # 8 nm precision
n_locs_per_emitter = 12

# Track true locs per emitter for diagnostics
true_locs_per_emitter = fill(n_locs_per_emitter, length(true_positions))

localizations = SMLMData.Emitter2DFit[]
for (i, (ex, ey)) in enumerate(true_positions)
    for j in 1:n_locs_per_emitter
        x = ex + randn() * σ_loc
        y = ey + randn() * σ_loc
        σ = σ_loc * (0.9 + 0.2 * rand())
        push!(localizations, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 50.0, 1.0, j, 1, 0, (i-1)*n_locs_per_emitter + j))
    end
end

println("\nGenerated $(length(localizations)) localizations from $(length(true_positions)) emitters")

# -----------------------------------------------------------------------------
# Run BaGoL
# -----------------------------------------------------------------------------
println("\nRunning BaGoL analysis...")

# τ: systematic uncertainty
chain = run_bagol(
    localizations;
    τ = 0.010,  # 10 nm
    λ_K = 2.0,  # Prior: expect ~2 emitters
    n_iterations = 10000,
    burn_in = 2000,
    verbose = true
)

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
result = estimate_mapn(chain)

println("\n" * "="^60)
println("Results")
println("="^60)
println("True: $(length(true_positions)) emitters")
println("MAP-N: $(result.n_emitters) emitters")

# Visualization with full chain samples
fig = plot_bagol(chain, result, localizations;
    true_positions=true_positions,
    save_path=joinpath(OUTPUT_DIR, "basic_result.png"))
println("\nSaved: $(joinpath(OUTPUT_DIR, "basic_result.png"))")

# Hierarchical Bayes diagnostics
fig2 = plot_hierarchical_diagnostics(chain;
    true_locs_per_emitter=true_locs_per_emitter,
    save_path=joinpath(OUTPUT_DIR, "basic_hierarchical.png"))
println("Saved: $(joinpath(OUTPUT_DIR, "basic_hierarchical.png"))")

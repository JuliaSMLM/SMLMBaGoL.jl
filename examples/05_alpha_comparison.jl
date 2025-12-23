# Alpha Parameter Comparison Demo
# ================================
# Compares three approaches to handling the α parameter:
# 1. Fixed α (user-specified)
# 2. Auto-estimated α (from frame statistics)
# 3. Learned α (MCMC inference)
#
# Run with: julia --project=examples --threads=4 examples/05_alpha_comparison.jl

using Pkg
Pkg.activate(@__DIR__)

using SMLMBaGoL
using SMLMData
using Random
using CairoMakie
using Statistics
using Printf

const OUTPUT_DIR = joinpath(@__DIR__, "output")
mkpath(OUTPUT_DIR)

println("="^60)
println("Alpha Parameter Comparison Demo")
println("="^60)

Random.seed!(456)

# -----------------------------------------------------------------------------
# Simulate data with intermediate α
# -----------------------------------------------------------------------------
# α = 3 is between exponential (α=1) and Poisson (α→∞)

TRUE_ALPHA = 3.0
TRUE_MU = 8.0

println("\nSimulating data with intermediate statistics...")
println("  True α = $TRUE_ALPHA")
println("  True μ = $TRUE_MU")

sim = simulate_nmers(
    n_dimers = 5,
    n_trimers = 3,
    emitter_spacing = 0.040,
    cluster_spacing = 0.200,
    μ = TRUE_MU,
    α = TRUE_ALPHA,
    σ_psf = 0.130,
    mean_photons = 600.0,
    background = 8.0,
    n_frames = 1000
)

print_simulation_summary(sim)
n_emitters = length(sim.true_positions)

# Common MCMC settings
mcmc_kwargs = (
    τ = 0.010,
    λ_K = Float64(n_emitters),
    n_iterations = 12000,
    burn_in = 2000,
    verbose = false
)

# -----------------------------------------------------------------------------
# Approach 1: Fixed α (wrong value)
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Approach 1: Fixed α = 2.0 (default, potentially wrong)")
println("-"^60)

chain1 = run_bagol(sim.localizations; α=2.0, mcmc_kwargs...)
result1 = estimate_mapn(chain1)
println("MAP-N: $(result1.n_emitters) (true: $n_emitters)")

# -----------------------------------------------------------------------------
# Approach 2: Auto-estimated α
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Approach 2: Auto-estimated α from frame statistics")
println("-"^60)

chain2 = run_bagol(sim.localizations; α=:auto, mcmc_kwargs...)
result2 = estimate_mapn(chain2)
println("Estimated α: $(round(chain2.α, digits=2))")
println("MAP-N: $(result2.n_emitters) (true: $n_emitters)")

# -----------------------------------------------------------------------------
# Approach 3: Learned α via MCMC
# -----------------------------------------------------------------------------
println("\n" * "-"^60)
println("Approach 3: Learned α via MCMC (starting from α=2.0)")
println("-"^60)

chain3 = run_bagol(sim.localizations; α=2.0, learn_α=true, mcmc_kwargs...)
result3 = estimate_mapn(chain3)

α_samples = [s.α for s in chain3.samples]
α_mean = mean(α_samples)
α_std = std(α_samples)
println("Learned α: $(round(α_mean, digits=2)) ± $(round(α_std, digits=2))")
println("MAP-N: $(result3.n_emitters) (true: $n_emitters)")

# -----------------------------------------------------------------------------
# Summary comparison
# -----------------------------------------------------------------------------
println("\n" * "="^60)
println("Summary")
println("="^60)
println("True: $n_emitters emitters, α=$TRUE_ALPHA, μ=$TRUE_MU")
println()
println("┌─────────────────┬────────┬─────────┬──────────┐")
println("│ Approach        │ α used │ MAP-N   │ Error    │")
println("├─────────────────┼────────┼─────────┼──────────┤")
@printf("│ Fixed (α=2)     │ %5.2f  │ %7d │ %+8d │\n", 2.0, result1.n_emitters, result1.n_emitters - n_emitters)
@printf("│ Auto-estimated  │ %5.2f  │ %7d │ %+8d │\n", chain2.α, result2.n_emitters, result2.n_emitters - n_emitters)
@printf("│ MCMC learned    │ %5.2f  │ %7d │ %+8d │\n", α_mean, result3.n_emitters, result3.n_emitters - n_emitters)
println("└─────────────────┴────────┴─────────┴──────────┘")
println("\nTrue α = $TRUE_ALPHA")

# -----------------------------------------------------------------------------
# Create comparison figure
# -----------------------------------------------------------------------------
fig = CairoMakie.Figure(size=(1400, 900))

# Row 1: Posterior on K for each approach
ax1 = CairoMakie.Axis(fig[1, 1], title="Fixed α=2.0",
                       xlabel="K", ylabel="P(K)")
k_vals = 0:(length(result1.posterior_k) - 1)
probs1 = result1.posterior_k ./ sum(result1.posterior_k)
CairoMakie.barplot!(ax1, k_vals, probs1, color=:steelblue)
CairoMakie.vlines!(ax1, [n_emitters], color=:red, linestyle=:dash, linewidth=2)
CairoMakie.vlines!(ax1, [result1.n_emitters], color=:black, linewidth=2)

ax2 = CairoMakie.Axis(fig[1, 2], title="Auto α=$(round(chain2.α, digits=1))",
                       xlabel="K", ylabel="P(K)")
probs2 = result2.posterior_k ./ sum(result2.posterior_k)
CairoMakie.barplot!(ax2, 0:(length(probs2)-1), probs2, color=:steelblue)
CairoMakie.vlines!(ax2, [n_emitters], color=:red, linestyle=:dash, linewidth=2)
CairoMakie.vlines!(ax2, [result2.n_emitters], color=:black, linewidth=2)

ax3 = CairoMakie.Axis(fig[1, 3], title="Learned α=$(round(α_mean, digits=1))",
                       xlabel="K", ylabel="P(K)")
probs3 = result3.posterior_k ./ sum(result3.posterior_k)
CairoMakie.barplot!(ax3, 0:(length(probs3)-1), probs3, color=:steelblue)
CairoMakie.vlines!(ax3, [n_emitters], color=:red, linestyle=:dash, linewidth=2)
CairoMakie.vlines!(ax3, [result3.n_emitters], color=:black, linewidth=2)

# Row 2: α trace (only for learned), count distributions
ax4 = CairoMakie.Axis(fig[2, 1], title="Count Distribution (True)",
                       xlabel="Locs/emitter", ylabel="Count")
CairoMakie.hist!(ax4, sim.true_counts, bins=0:maximum(sim.true_counts)+2, color=:steelblue)

ax5 = CairoMakie.Axis(fig[2, 2], title="α Learning Trace",
                       xlabel="Sample", ylabel="α")
CairoMakie.lines!(ax5, 1:length(α_samples), α_samples, color=:darkorange)
CairoMakie.hlines!(ax5, [TRUE_ALPHA], color=:red, linestyle=:dash, linewidth=2, label="True α")
CairoMakie.axislegend(ax5)

ax6 = CairoMakie.Axis(fig[2, 3], title="α Posterior",
                       xlabel="α", ylabel="Density")
CairoMakie.hist!(ax6, α_samples, bins=30, normalization=:pdf, color=:darkorange)
CairoMakie.vlines!(ax6, [TRUE_ALPHA], color=:red, linestyle=:dash, linewidth=2, label="True")
CairoMakie.vlines!(ax6, [α_mean], color=:black, linewidth=2, label="Mean")
CairoMakie.axislegend(ax6)

CairoMakie.save(joinpath(OUTPUT_DIR, "alpha_comparison.png"), fig)
println("\nSaved: $(joinpath(OUTPUT_DIR, "alpha_comparison.png"))")

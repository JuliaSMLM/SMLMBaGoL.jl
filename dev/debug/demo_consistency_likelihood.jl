#!/usr/bin/env julia

"""
Demonstrate how consistency likelihood addresses over-segmentation.
"""

using Pkg
cd(@__DIR__)
Pkg.activate("../..")
using SMLMBaGoL
using Statistics
using Random
using Printf

Random.seed!(42)

println("🎯 DEMONSTRATING CONSISTENCY LIKELIHOOD CONCEPT")
println("="^70)

# Create a simple scenario: 100 localizations from 1 emitter
n_locs = 100
true_x = 0.0
true_y = 0.0
σ_loc = 0.01  # 10 nm
τ = 0.001     # 1 nm systematic noise

# Generate localizations
localizations = SMLMBaGoL.Localization2D{Float64}[]
for i in 1:n_locs
    x = true_x + σ_loc * randn()
    y = true_y + σ_loc * randn()
    push!(localizations, SMLMBaGoL.Localization2D(x, y, σ_loc, σ_loc, i))
end

println("Generated $(n_locs) localizations from 1 emitter at origin")
println("Localization uncertainty: σ = $(σ_loc*1000) nm")
println("Systematic noise: τ = $(τ*1000) nm")

# Create spatial prior
spatial_prior = SMLMBaGoL.UniformSpatialPrior(-0.1, 0.1, -0.1, 0.1)

# Test Case 1: Correct model (1 emitter at truth)
println("\n📊 CASE 1: Correct model (1 emitter at truth)")
println("-"^70)

emitters_1 = [SMLMBaGoL.Emitter2D(true_x, true_y, 1000.0)]  # Add photons parameter
allocations_1 = fill(1, n_locs)
latent_positions_1 = [(true_x, true_y) for _ in 1:n_locs]

# Create dummy count prior (not used for likelihood calculation)
count_prior = SMLMBaGoL.HierarchicalNegBinomialPrior(
    40.0, 10.0, τ^2,
    (2.0, 5.0), (5.0, 2.0), (3.0, 3.0e-6)
)

state_1 = SMLMBaGoL.BaGoLState(
    emitters_1,
    localizations,
    allocations_1,
    latent_positions_1,
    spatial_prior,
    count_prior,
    τ^2,
    0.0
)

ll_standard_1 = SMLMBaGoL.standard_log_likelihood(state_1)
ll_consistency_1 = SMLMBaGoL.consistency_log_likelihood(state_1; α=1.0)

# Analyze residuals
residuals_x_1 = [(loc.x - true_x) / sqrt(loc.σx^2 + τ^2) for loc in localizations]
residuals_y_1 = [(loc.y - true_y) / sqrt(loc.σy^2 + τ^2) for loc in localizations]
var_x_1 = var(residuals_x_1)
var_y_1 = var(residuals_y_1)

println("Standard likelihood:    $(round(ll_standard_1, digits=2))")
println("Consistency likelihood: $(round(ll_consistency_1, digits=2))")
println("Normalized residual variance: var_x = $(round(var_x_1, digits=3)), var_y = $(round(var_y_1, digits=3))")
println("(Should be ≈ 1.0 for correct model)")

# Test Case 2: Over-segmented model (2 emitters)
println("\n📊 CASE 2: Over-segmented model (2 emitters, split data)")
println("-"^70)

# Split data in half and fit each half separately
half1_idx = 1:50
half2_idx = 51:100

mean_x_1 = mean(loc.x for loc in localizations[half1_idx])
mean_y_1 = mean(loc.y for loc in localizations[half1_idx])
mean_x_2 = mean(loc.x for loc in localizations[half2_idx])
mean_y_2 = mean(loc.y for loc in localizations[half2_idx])

emitters_2 = [
    SMLMBaGoL.Emitter2D(mean_x_1, mean_y_1, 1000.0),
    SMLMBaGoL.Emitter2D(mean_x_2, mean_y_2, 1000.0)
]

allocations_2 = vcat(fill(1, 50), fill(2, 50))
latent_positions_2 = vcat(
    [(mean_x_1, mean_y_1) for _ in 1:50],
    [(mean_x_2, mean_y_2) for _ in 1:50]
)

state_2 = SMLMBaGoL.BaGoLState(
    emitters_2,
    localizations,
    allocations_2,
    latent_positions_2,
    spatial_prior,
    count_prior,
    τ^2,
    0.0
)

ll_standard_2 = SMLMBaGoL.standard_log_likelihood(state_2)
ll_consistency_2 = SMLMBaGoL.consistency_log_likelihood(state_2; α=1.0)

# Analyze residuals for over-segmented model
residuals_x_2 = Float64[]
residuals_y_2 = Float64[]
for i in 1:n_locs
    emitter = emitters_2[allocations_2[i]]
    loc = localizations[i]
    push!(residuals_x_2, (loc.x - emitter.x) / sqrt(loc.σx^2 + τ^2))
    push!(residuals_y_2, (loc.y - emitter.y) / sqrt(loc.σy^2 + τ^2))
end
var_x_2 = var(residuals_x_2)
var_y_2 = var(residuals_y_2)

println("Standard likelihood:    $(round(ll_standard_2, digits=2))")
println("Consistency likelihood: $(round(ll_consistency_2, digits=2))")
println("Normalized residual variance: var_x = $(round(var_x_2, digits=3)), var_y = $(round(var_y_2, digits=3))")
println("(< 1.0 indicates overfitting!)")

# Compare improvements
println("\n⚖️  LIKELIHOOD COMPARISON:")
println("-"^70)
improvement_standard = ll_standard_2 - ll_standard_1
improvement_consistency = ll_consistency_2 - ll_consistency_1

println("Standard likelihood improvement (2 vs 1 emitter):")
println("  ΔLL = $(round(improvement_standard, digits=2)) (positive = prefers 2 emitters)")

println("\nConsistency likelihood improvement (2 vs 1 emitter):")
println("  ΔLL = $(round(improvement_consistency, digits=2)) (should be negative = prefers 1 emitter)")

# Explain the penalty
println("\n💡 HOW IT WORKS:")
println("-"^70)
println("The consistency likelihood adds a penalty based on KL divergence:")
println("  Penalty = α * n/2 * (var - log(var) - 1)")
println("")
println("This penalty:")
println("• = 0 when var = 1.0 (perfect consistency)")
println("• > 0 when var ≠ 1.0 (both over- and under-dispersion)")
println("• Increases with sample size n")

# Calculate penalties explicitly
n = n_locs
penalty_1 = n/2 * ((var_x_1 - log(var_x_1) - 1) + (var_y_1 - log(var_y_1) - 1))
penalty_2 = n/2 * ((var_x_2 - log(var_x_2) - 1) + (var_y_2 - log(var_y_2) - 1))

println("\nActual penalties:")
println("1 emitter:  $(round(penalty_1, digits=2))")
println("2 emitters: $(round(penalty_2, digits=2)) (larger penalty for overfitting)")

# Test adaptive version
println("\n🔧 ADAPTIVE CONSISTENCY LIKELIHOOD:")
println("-"^70)
ll_adaptive_1 = SMLMBaGoL.adaptive_consistency_likelihood(state_1)
ll_adaptive_2 = SMLMBaGoL.adaptive_consistency_likelihood(state_2)
improvement_adaptive = ll_adaptive_2 - ll_adaptive_1

println("Adaptive penalty weight:")
println("  1 emitter: α = $(round(1.0 + 2.0 * 1 / n_locs, digits=3))")
println("  2 emitters: α = $(round(1.0 + 2.0 * 2 / n_locs, digits=3))")
println("\nAdaptive likelihood improvement (2 vs 1):")
println("  ΔLL = $(round(improvement_adaptive, digits=2)) (more negative = stronger preference for 1)")

println("\n✅ Demonstration complete!")
println("\nKey takeaway: The consistency likelihood penalizes models that")
println("make residuals smaller than statistically expected, thus")
println("preventing over-segmentation!")
#!/usr/bin/env julia

"""
Basic SMLMBaGoL Demo: Simulation and Analysis

This example demonstrates the complete workflow:
1. Simulate synthetic n-mer data
2. Analyze with BaGoL algorithm  
3. Compare results to ground truth

The simulation generates a 6-mer complex with realistic photon noise,
then BaGoL recovers the emitter positions and count.
"""

using Pkg; Pkg.activate("examples")
using SMLMBaGoL

println("=== SMLMBaGoL Basic Demo ===\n")

# Step 1: Generate synthetic data
println("1. Generating synthetic 6-mer data...")
localizations, prior = simulate_n_mer(
    n = 6,                                        # 6-mer complex
    diameter = 0.3,                               # 0.3 μm diameter  
    localizations_per_emitter_mean = 6.0,        # ~6 localizations per emitter
    localizations_per_emitter_variance = 6.0,    # Poisson-like variation
    photons = 800,                                # 800 photons average
    prior_K_mean = 6.0,                          # Expect ~6 emitters  
    prior_K_variance = 2.0                       # Low uncertainty
)

println("   Generated $(length(localizations)) total localizations")
prior_mean = round(prior.K_prior.α / prior.K_prior.β, digits=1)
prior_var = round(prior.K_prior.α / prior.K_prior.β^2, digits=1)
println("   Prior on emitter count: mean=$prior_mean, variance=$prior_var")
println("   Internal Gamma parameters: α=$(round(prior.K_prior.α, digits=1)), β=$(round(prior.K_prior.β, digits=1))")

println("\n   Localization summary:")
show(stdout, MIME("text/plain"), localizations)
println()

# Step 2: Analyze with BaGoL
println("\n2. Running BaGoL analysis...")
result = run_bagol(
    localizations,
    prior = prior,
    n_iterations = 2000,
    burn_in = 500,
    partition_data = false  # Single partition for simple demo
)

# Step 3: Extract results
println("\n3. Analysis Results:")
final_state = result.current_state
n_estimated = length(final_state.emitters)
println("   Estimated emitter count: $n_estimated (true: 6)")

println("\n   Final state summary:")
show(stdout, MIME("text/plain"), final_state)
println()

if n_estimated > 0
    println("\n   Estimated emitters:")
    show(stdout, MIME("text/plain"), final_state.emitters)
    println()
end

println("\n4. Ground Truth Comparison:")
println("   True positions (6-mer on circle with radius 0.15 μm):")
true_radius = 0.15
for i in 1:6
    angle = 2π * (i-1) / 6
    true_x = true_radius * cos(angle)
    true_y = true_radius * sin(angle)
    println("     True $i: ($(round(true_x, digits=3)), $(round(true_y, digits=3)))")
end

# Step 5: Summary statistics
println("\n5. Summary:")
println("   Total RJMCMC samples: $(length(result.samples))")
println("   Final log-likelihood: $(round(final_state.log_likelihood, digits=1))")

recovery_success = abs(n_estimated - 6) <= 1
println("   Recovery success: $(recovery_success ? "✓" : "✗") (within ±1 emitter)")

println("\n=== Demo Complete ===")
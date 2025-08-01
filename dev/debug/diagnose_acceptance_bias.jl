#!/usr/bin/env julia

"""
Diagnose how the K prior biases birth/death acceptance ratios.
"""

using Pkg; Pkg.activate("../..")
using SMLMBaGoL
using Distributions

# Scenario from our nmer_demo results
N = 227  # Total localizations  
true_k = 6
true_μ = N / true_k  # ≈ 37.8

# Algorithm's converged values (from our test)
converged_μ = 12.6
converged_κ = 1.0

println("=== Birth/Death Acceptance Ratio Bias Analysis ===\n")

# Function to compute log prior ratio for K transitions
function k_prior_log_ratio(k_from, k_to, μ, κ)
    return SMLMBaGoL.log_prior_k_given_N(k_to, N, μ, κ) - 
           SMLMBaGoL.log_prior_k_given_N(k_from, N, μ, κ)
end

# Test transitions from the true answer (K=6)
println("Starting from TRUE K=6, what does the prior favor?")
println(repeat("=", 55))

for target_k in 4:10
    # Prior ratio using converged μ (what algorithm sees)
    biased_ratio = k_prior_log_ratio(6, target_k, converged_μ, converged_κ)
    
    # Prior ratio using true μ (what it should be)
    correct_ratio = k_prior_log_ratio(6, target_k, true_μ, converged_κ)
    
    biased_preference = biased_ratio > 0 ? "FAVORS" : "opposes"
    correct_preference = correct_ratio > 0 ? "FAVORS" : "opposes"
    
    println("K=6 → K=$target_k:")
    println("  With μ=$(converged_μ): log ratio = $(round(biased_ratio, digits=2)) ($biased_preference)")
    println("  With μ=$(round(true_μ, digits=1)): log ratio = $(round(correct_ratio, digits=2)) ($correct_preference)")
    
    if biased_ratio > 0 && correct_ratio < 0
        println("  ⚠️  BIAS: Algorithm incorrectly favors this transition!")
    elseif biased_ratio < 0 && correct_ratio > 0
        println("  ⚠️  BIAS: Algorithm incorrectly opposes this transition!")
    end
    println()
end

println("\n" * repeat("=", 60))
println("Testing birth moves from K=6 to K=7 (most common error):")
println(repeat("=", 60))

# Calculate the actual bias magnitude
birth_bias = k_prior_log_ratio(6, 7, converged_μ, converged_κ)
correct_bias = k_prior_log_ratio(6, 7, true_μ, converged_κ)

println("Birth move K=6 → K=7:")
println("  Biased prior contribution: $(round(birth_bias, digits=3))")
println("  Correct prior contribution: $(round(correct_bias, digits=3))")
println("  Net bias toward birth: $(round(birth_bias - correct_bias, digits=3))")

odds_multiplier = exp(birth_bias - correct_bias)
println("  This makes birth moves $(round(odds_multiplier, digits=2))x more likely to be accepted!")

println("\nDeath move K=7 → K=6:")
death_bias = -birth_bias  # Death is inverse of birth
correct_death_bias = -correct_bias
println("  Biased prior contribution: $(round(death_bias, digits=3))")
println("  Correct prior contribution: $(round(correct_death_bias, digits=3))")
println("  Net bias against death: $(round(death_bias - correct_death_bias, digits=3))")

death_odds_multiplier = exp(death_bias - correct_death_bias)
println("  This makes death moves $(round(death_odds_multiplier, digits=2))x less likely to be accepted!")

println("\n" * repeat("=", 60))
println("KEY INSIGHT: The Feedback Loop")
println(repeat("=", 60))
println("1. Algorithm converges to μ=$(converged_μ) instead of true μ=$(round(true_μ, digits=1))")
println("2. Prior P(K|N,μ,κ) with small μ expects more emitters")
println("3. Birth moves become $(round(odds_multiplier, digits=2))x more likely to be accepted")
println("4. Death moves become $(round(1/death_odds_multiplier, digits=2))x less likely to be accepted") 
println("5. This drives K upward, reinforcing the small μ estimate")
println("6. The cycle continues until equilibrium at wrong K")

println("\n📈 SOLUTION: Break the circular dependency!")
println("   Option 1: Use a K prior that doesn't depend on μ")
println("   Option 2: Use a different parameterization")
println("   Option 3: Marginalize over μ uncertainty in the K prior")
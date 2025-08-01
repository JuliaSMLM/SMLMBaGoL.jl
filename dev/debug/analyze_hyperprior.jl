#!/usr/bin/env julia

"""
Analyze the current κ hyperprior and propose a better one.
"""

using Pkg; Pkg.activate("examples")
using Distributions
using Statistics

println("=== Analyzing Current κ Hyperprior ===\n")

# Current hyperprior
current_hyperprior = Gamma(0.5, 20.0)
println("Current hyperprior: Gamma(0.5, 20.0)")
println("  Mean: $(mean(current_hyperprior))")
println("  Variance: $(var(current_hyperprior))")
println("  Mode: 0 (degenerate at boundary since shape < 1)")
println("  Standard deviation: $(std(current_hyperprior))")

# Sample from it to see what values it prefers
samples = rand(current_hyperprior, 10000)
println("\n  Sample statistics:")
println("    5th percentile: $(quantile(samples, 0.05))")
println("    25th percentile: $(quantile(samples, 0.25))")
println("    50th percentile (median): $(quantile(samples, 0.50))")
println("    75th percentile: $(quantile(samples, 0.75))")
println("    95th percentile: $(quantile(samples, 0.95))")

fraction_below_1 = mean(samples .< 1.0)
fraction_below_5 = mean(samples .< 5.0)
println("    Fraction < 1.0: $(round(fraction_below_1, digits=3))")
println("    Fraction < 5.0: $(round(fraction_below_5, digits=3))")

println("\n  ⚠ Problem: Shape parameter 0.5 < 1 creates strong bias toward κ < 1!")

# Proposed better hyperpriors
println("\n=== Proposed Better Hyperpriors ===")

proposals = [
    (2.0, 5.0, "Gamma(2, 5) - moderate concentration around mean=10"),
    (5.0, 2.0, "Gamma(5, 2) - tighter around mean=10"),
    (10.0, 1.0, "Gamma(10, 1) - concentrated at mean=10"),
    (1.0, 10.0, "Gamma(1, 10) - exponential-like, mean=10"),
    (3.0, 3.0, "Gamma(3, 3) - balanced, mean=9")
]

for (shape, scale, desc) in proposals
    dist = Gamma(shape, scale)
    samples_prop = rand(dist, 10000)
    
    println("\n$desc:")
    println("  Mean: $(round(mean(dist), digits=1)), Var: $(round(var(dist), digits=1)), Mode: $(round(mode(dist), digits=1))")
    println("  Percentiles: 5%=$(round(quantile(samples_prop, 0.05), digits=1)), 50%=$(round(quantile(samples_prop, 0.50), digits=1)), 95%=$(round(quantile(samples_prop, 0.95), digits=1))")
    println("  Fraction < 1.0: $(round(mean(samples_prop .< 1.0), digits=3))")
    println("  Fraction < 5.0: $(round(mean(samples_prop .< 5.0), digits=3))")
end

println("\n=== Recommendation ===")
println("Replace (0.5, 20.0) with (2.0, 5.0) or (5.0, 2.0)")
println("Both have mean=10 but avoid the pathological small-κ bias.")
println("Gamma(5, 2) is more concentrated and will prevent drift to extreme values.")
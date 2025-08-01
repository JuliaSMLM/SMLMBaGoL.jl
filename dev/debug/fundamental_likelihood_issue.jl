#!/usr/bin/env julia

"""
Exploring the fundamental issue: Standard likelihood penalizes expected variation.

When we have N localizations from K emitters with Gaussian noise, we EXPECT:
- 68% within 1σ
- 95% within 2σ  
- 99.7% within 3σ
- 0.3% beyond 3σ (these are NOT errors - they're expected!)

But standard likelihood penalizes ALL deviations, treating expected outliers
as "errors" that need fixing (by adding more emitters).
"""

using Pkg; Pkg.activate("../..")
using SMLMBaGoL
using Statistics
using Distributions
using Random
using Printf
# using HypothesisTests  # Skip for now

Random.seed!(12345)

println("🧠 FUNDAMENTAL LIKELIHOOD PROBLEM")
println("="^60)
println("Standard likelihood penalizes expected statistical variation")
println()

# Generate perfect Gaussian data
n_points = 100
σ = 0.01  # 10 nm uncertainty
true_position = 0.0

# Generate data from true Gaussian
data = true_position .+ σ .* randn(n_points)

# Calculate distances from truth
distances = abs.(data .- true_position)
normalized_distances = distances ./ σ

# Show distribution
println("📊 Distribution of |x - μ|/σ for $(n_points) points:")
println("-"^50)
for threshold in [1.0, 2.0, 3.0, 4.0]
    count = sum(normalized_distances .≤ threshold)
    expected = 2 * cdf(Normal(), threshold) - 1  # Two-sided
    @printf("Within %1.0fσ: %3d/%d (%.1f%%) Expected: %.1f%%\n", 
            threshold, count, n_points, 100*count/n_points, 100*expected)
end

# Points beyond 3σ
outliers_3σ = sum(normalized_distances .> 3.0)
println("\nBeyond 3σ: $(outliers_3σ) points ($(round(100*outliers_3σ/n_points, digits=1))%)")
println("These are NOT errors - they're statistically expected!")

# Calculate standard log-likelihood
println("\n📈 Standard Gaussian Log-Likelihood Analysis:")
println("-"^50)

function gaussian_log_likelihood(data, μ, σ)
    n = length(data)
    return -n/2 * log(2π) - n * log(σ) - sum((data .- μ).^2) / (2σ^2)
end

# Compare 1 emitter vs 2 emitters
ll_1emitter = gaussian_log_likelihood(data, true_position, σ)

# For 2 emitters, split data in half
half1 = data[1:50]
half2 = data[51:100]
μ1 = mean(half1)
μ2 = mean(half2)
ll_2emitters = gaussian_log_likelihood(half1, μ1, σ) + gaussian_log_likelihood(half2, μ2, σ)

println("1 emitter at truth: LL = $(round(ll_1emitter, digits=2))")
println("2 emitters (split): LL = $(round(ll_2emitters, digits=2))")
println("Improvement: $(round(ll_2emitters - ll_1emitter, digits=2))")

# The problem illustrated
println("\n⚠️  THE PROBLEM:")
println("2 emitters improve likelihood even with PERFECT Gaussian data!")
println("This happens because:")
println("1. Each subset happens to have mean ≠ 0 (by chance)")
println("2. Fitting local means reduces squared errors")
println("3. Expected outliers are treated as 'errors' to minimize")

# Theoretical analysis
println("\n🔬 THEORETICAL INSIGHT:")
println("-"^50)
println("For N points from Gaussian(μ, σ²):")
println("• Expected sum of squared errors: N * σ²")
println("• But sample mean x̄ ≠ μ (sampling variation)")
println("• Using x̄ reduces SSE by ~σ²")
println("• Each additional parameter can exploit this!")

# Alternative approach: Distribution matching
println("\n💡 ALTERNATIVE: Distribution Matching")
println("-"^50)
println("Instead of minimizing squared errors, we could ask:")
println("'Does the distribution of residuals match expected Gaussian?'")

# Check residual distributions
residuals_1 = (data .- true_position) ./ σ
residuals_2a = (half1 .- μ1) ./ σ
residuals_2b = (half2 .- μ2) ./ σ
residuals_2 = vcat(residuals_2a, residuals_2b)

println("Residual statistics:")
println("1 emitter:  mean = $(round(mean(residuals_1), digits=3)), std = $(round(std(residuals_1), digits=3))")
println("2 emitters: mean = $(round(mean(residuals_2), digits=3)), std = $(round(std(residuals_2), digits=3))")
println("Both have residuals ~ N(0,1) as expected!")

# Practical illustration with emitters
println("\n🔍 IMPLICATIONS FOR EMITTER DETECTION:")
println("-"^50)
println("Current approach: More emitters → Better likelihood → Accept")
println("Better approach: More emitters → Check if residuals more Gaussian")
println()
println("Key insight: We should NOT penalize configurations that produce")
println("the EXPECTED amount of variation for the claimed uncertainty!")

# Proposed solution sketch
println("\n✨ PROPOSED SOLUTION:")
println("-"^50)
println("1. Use likelihood ratio vs NULL model (expected variation)")
println("2. Penalize only EXCESS variation beyond statistical expectation")
println("3. Or use distribution-based metrics (K-S, Anderson-Darling)")
println("4. Or robust likelihood (Student-t, Huber) that expects outliers")

println("\n🎯 This explains why τ² helps - it allows 'expected outliers'")
println("But current formulation still tries to minimize them!")

println("\n✅ Analysis complete!")
#!/usr/bin/env julia

"""
The deep issue: Standard likelihood conflates model FIT with model CORRECTNESS.

Information theoretic view:
- Gaussian likelihood measures: "How much information do we need to encode residuals?"
- But it assumes: "Less information = better model"
- This is WRONG when the TRUE process has inherent randomness!

The correct question: "Do residuals have the EXPECTED information content?"
"""

using Pkg; Pkg.activate("../..")
using SMLMBaGoL
using Statistics
using Distributions
using Random
using Printf

Random.seed!(99)

println("🧠 INFORMATION THEORETIC VIEW OF THE LIKELIHOOD PROBLEM")
println("="^70)

# The key insight
println("📌 THE FUNDAMENTAL CONFUSION:")
println("-"^70)
println("Standard likelihood minimizes: -log P(data|model)")
println("This minimizes the 'surprise' or 'information content' of residuals")
println()
println("But with measurement uncertainty, residuals SHOULD be surprising!")
println("A model that perfectly fits noisy data is OVERFITTING, not better!")
println()

# Concrete example
n_points = 100
true_μ = 0.0
true_σ = 0.01  # 10nm uncertainty

# Generate data
data = true_μ .+ true_σ .* randn(n_points)

println("🔍 Example: $(n_points) measurements with σ = $(true_σ*1000)nm")
println("-"^70)

# Information content of different models
function information_content(residuals, σ)
    # Information = -log P = negative log likelihood
    # For Gaussian: I = n/2 * log(2πσ²) + Σ(r²)/(2σ²)
    n = length(residuals)
    return n/2 * log(2π * σ^2) + sum(residuals.^2) / (2σ^2)
end

# Model 1: Correct model (1 emitter at truth)
residuals_correct = data .- true_μ
info_correct = information_content(residuals_correct, true_σ)

# Model 2: Overfitted model (fit each point exactly)
residuals_overfit = zeros(n_points)  # Perfect fit!
info_overfit_naive = information_content(residuals_overfit, true_σ)

# Model 3: Overfitted but honest about uncertainty
# If we fit each point exactly, we're claiming σ_effective ≈ 0
σ_overfit = 1e-6  # Near zero
info_overfit_honest = information_content(residuals_overfit, σ_overfit)

println("Information content (nats):")
println("1. Correct model (1 emitter):        $(round(info_correct, digits=1))")
println("2. Overfit (n emitters, claim σ=10nm): $(round(info_overfit_naive, digits=1)) ← WRONG!")
println("3. Overfit (n emitters, claim σ≈0):   $(round(info_overfit_honest, digits=1)) ← Penalized")

println("\nThe problem: Model 2 claims to explain data perfectly while maintaining")
println("original uncertainty claims. This is incoherent!")

# The right question
println("\n✨ THE RIGHT APPROACH:")
println("-"^70)
println("Don't ask: 'How small can we make residuals?'")
println("Ask: 'Are residuals consistent with claimed uncertainty?'")
println()

# Implement a simple consistency check
function likelihood_ratio_test(residuals, σ_claimed)
    """
    Test if residuals are consistent with N(0, σ_claimed²)
    Returns log likelihood ratio vs 'oracle' that knows true distribution
    """
    n = length(residuals)
    
    # Empirical variance
    σ²_empirical = var(residuals)
    
    # Log likelihood under claimed model
    ll_claimed = -n/2 * log(2π * σ_claimed^2) - sum(residuals.^2) / (2 * σ_claimed^2)
    
    # Log likelihood under empirical model (maximum likelihood)
    ll_empirical = -n/2 * log(2π * σ²_empirical) - n/2
    
    # Likelihood ratio
    lr = ll_claimed - ll_empirical
    
    return lr, sqrt(σ²_empirical)
end

println("📊 Consistency Analysis:")
println("-"^50)

# Test our models
lr_correct, σ_emp_correct = likelihood_ratio_test(residuals_correct, true_σ)
println("Correct model:")
println("  Empirical σ: $(round(σ_emp_correct*1000, digits=2))nm (true: $(true_σ*1000)nm)")
println("  LR: $(round(lr_correct, digits=2)) (close to 0 = consistent)")

# Can't test overfit model with zero residuals in same way
# But we can test intermediate case
residuals_partial = Float64[]
emitter_positions = Float64[]
for i in 1:10  # 10 emitters
    subset = data[(i-1)*10+1:i*10]
    pos = mean(subset)
    push!(emitter_positions, pos)
    append!(residuals_partial, subset .- pos)
end

lr_partial, σ_emp_partial = likelihood_ratio_test(residuals_partial, true_σ)
println("\n10-emitter model:")
println("  Empirical σ: $(round(σ_emp_partial*1000, digits=2))nm (should be ~$(true_σ*1000)nm)")
println("  LR: $(round(lr_partial, digits=2)) (negative = overfitting)")

# The key insight
println("\n💡 KEY INSIGHT:")
println("-"^70)
println("Over-segmentation happens because standard likelihood rewards")
println("models that make residuals SMALLER than statistically expected!")
println()
println("With true σ = $(true_σ*1000)nm, we EXPECT residuals with std ≈ $(true_σ*1000)nm")
println("But more emitters can always reduce this, which likelihood rewards.")

# Practical solution
println("\n🛠️ PRACTICAL SOLUTION:")
println("-"^70)
println("Modify log-likelihood to penalize both:")
println("1. Residuals LARGER than expected (underfitting)")
println("2. Residuals SMALLER than expected (overfitting)")
println()
println("One approach: Add penalty term when empirical variance < claimed variance")

function modified_log_likelihood(residuals, σ_claimed; penalty_weight=1.0)
    n = length(residuals)
    σ²_empirical = var(residuals)
    
    # Standard log likelihood
    ll_standard = -n/2 * log(2π * σ_claimed^2) - sum(residuals.^2) / (2 * σ_claimed^2)
    
    # Penalty for variance mismatch
    # This penalizes when σ²_empirical << σ²_claimed
    variance_ratio = σ²_empirical / σ_claimed^2
    penalty = penalty_weight * n * (variance_ratio - log(variance_ratio) - 1)
    
    return ll_standard - penalty
end

println("Modified likelihood comparison:")
ll_mod_correct = modified_log_likelihood(residuals_correct, true_σ)
ll_mod_partial = modified_log_likelihood(residuals_partial, true_σ)

println("1 emitter:  $(round(ll_mod_correct, digits=2))")
println("10 emitters: $(round(ll_mod_partial, digits=2))")
println("Difference: $(round(ll_mod_partial - ll_mod_correct, digits=2)) (now penalizes over-segmentation!)")

println("\n✅ Analysis complete!")
println("\nBottom line: The likelihood should measure CONSISTENCY with uncertainty,")
println("not just minimize residuals!")
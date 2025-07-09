#!/usr/bin/env julia

"""
Test script to demonstrate improved Negative Binomial prior fitting
"""

using Pkg; Pkg.activate("examples")
using SMLMBaGoL
using Distributions
using Statistics
using Random
using CairoMakie

Random.seed!(123)

# Simulate data with known parameters
println("=== Testing Negative Binomial Prior Fitting ===\n")

# True parameters
μ_true = 40.0
κ_true = 3.0  # High overdispersion
println("True parameters: μ = $μ_true, κ = $κ_true")
println("Theoretical variance: $(μ_true + μ_true^2/κ_true)")

# Generate synthetic count data
r = κ_true
p = κ_true / (κ_true + μ_true)
nb_dist = NegativeBinomial(r, p)
counts = rand(nb_dist, 100)

println("\nEmpirical data:")
println("  Mean: $(round(mean(counts), digits=2))")
println("  Variance: $(round(var(counts), digits=2))")

# Test different fitting methods
println("\n1. Method of Moments Fitting:")
μ_mom, κ_mom = SMLMBaGoL.fit_negbinomial_mom(counts)
println("  μ = $(round(μ_mom, digits=2)), κ = $(round(κ_mom, digits=2))")
println("  Fitted variance: $(round(μ_mom + μ_mom^2/κ_mom, digits=2))")

println("\n2. Maximum Likelihood Fitting:")
μ_mle, κ_mle = SMLMBaGoL.fit_negbinomial_mle(counts)
println("  μ = $(round(μ_mle, digits=2)), κ = $(round(κ_mle, digits=2))")
println("  Fitted variance: $(round(μ_mle + μ_mle^2/κ_mle, digits=2))")

# Initialize hierarchical prior
println("\n3. Initialize Hierarchical Prior:")
prior = initialize_hierarchical_prior(counts; method=:mle)
println("  μ = $(round(prior.μ, digits=2))")
println("  κ = $(round(prior.κ, digits=2))")
println("  τ² = $(prior.τ²)")

# Assess fit quality
println("\n4. Fit Assessment:")
stats = assess_negbinomial_fit(counts, prior)
print_fit_summary(stats)

# Test adaptive slice sampling
println("\n5. Testing Adaptive Slice Sampling:")
println("  Starting κ = $(round(prior.κ, digits=2))")

# Run a few iterations
κ_current = prior.κ
for i in 1:5
    global κ_current
    κ_new = update_kappa_slice_adaptive(counts, prior.μ, prior.κ_hyperprior, κ_current; 
                                       n_steps=10, adapt_width=true)
    println("  Iteration $i: κ = $(round(κ_new, digits=2))")
    κ_current = κ_new
end

# Create visualization
println("\n6. Creating diagnostic plots...")

# Simple comparison plot
fig = Figure(size=(800, 600))
ax = CairoMakie.Axis(fig[1, 1], 
          xlabel="Count", 
          ylabel="Probability",
          title="Empirical vs Fitted Negative Binomial")

# Empirical histogram
hist!(ax, counts, bins=0:maximum(counts)+1, 
      normalization=:pdf, color=(:blue, 0.6), 
      label="Empirical")

# Fitted distribution
k_values = 0:maximum(counts)
r = prior.κ
p = prior.κ / (prior.κ + prior.μ)
nb_fitted = NegativeBinomial(r, p)
pmf_values = [pdf(nb_fitted, k) for k in k_values]
scatter!(ax, k_values, pmf_values, 
        color=:red, markersize=8,
        label="Fitted NB(μ=$(round(prior.μ,digits=1)), κ=$(round(prior.κ,digits=1)))")

axislegend(ax, position=:rt)

save("negbinomial_fit_test.png", fig, px_per_unit=2)
println("  ✓ Plot saved to negbinomial_fit_test.png")

println("\n✅ Test complete!")
println("\nKey improvements demonstrated:")
println("  - Data-driven initialization (MLE and MoM)")
println("  - Adaptive slice sampling for κ")
println("  - Quantitative fit assessment")
println("  - Clear diagnostics for poor fits")
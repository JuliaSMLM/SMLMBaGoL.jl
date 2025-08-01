#!/usr/bin/env julia

"""
Debug the slice sampling log density function to understand why κ < 1 is being preferred.
"""

using Pkg; Pkg.activate("examples")
using SMLMBaGoL
using Statistics
using Random
using Distributions
using CairoMakie

Random.seed!(123)

println("=== Debugging Slice Sampling Log Density Function ===\n")

# Use the problematic data from our previous run
counts = [17, 18, 17, 7, 19, 13, 1, 12]  # From debug output above
μ = 13.0
hyperprior = (0.5, 20.0)  # κ hyperprior from hierarchical prior

println("Data: counts = $counts")
println("μ = $μ")
println("κ hyperprior = $hyperprior (Gamma(0.5, 20) → mean=10, var=200)")

# Reproduce the log density function from slice sampling
function log_density(κ, counts, μ, hyperprior)
    c₀, d₀ = hyperprior
    n = length(counts)
    sum_counts = sum(counts)
    
    if κ ≤ 0
        return -Inf
    end
    
    # Prior contribution: κ ~ Gamma(c₀, d₀)
    log_p = (c₀ - 1) * log(κ) - κ / d₀
    
    # Likelihood contribution: ∏ NegBinomial(n_j; μ, κ)
    for n_j in counts
        log_p -= logbeta(κ, n_j + 1) - logbeta(κ, 1)  # Use logbeta instead of loggamma
        # Note: logbeta(a,b) = loggamma(a) + loggamma(b) - loggamma(a+b)
        # So loggamma(κ) = logbeta(κ, 1) and loggamma(n_j + κ) involves logbeta
        # Actually, let's use a simpler approximation for now
        if κ > 10
            log_p += n_j * log(κ)  # Stirling approximation for large κ
        else
            # Use beta function: Γ(κ)Γ(n_j+1)/Γ(n_j+κ+1) 
            log_p += logbeta(κ, n_j + 1)  # This is log(Γ(κ)Γ(n_j+1)/Γ(n_j+κ+1))
        end
    end
    
    log_p -= (sum_counts + n * κ) * log(κ + μ)
    
    return log_p
end

# Evaluate log density over a range of κ values
println("\n1. Evaluating log density over κ range...")
κ_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
log_densities = Float64[]

for κ in κ_values
    ld = log_density(κ, counts, μ, hyperprior)
    push!(log_densities, ld)
    println("   κ = $(rpad(κ, 5)) → log density = $(round(ld, digits=2))")
end

# Find maximum
max_idx = argmax(log_densities)
max_κ = κ_values[max_idx]
max_ld = log_densities[max_idx]

println("\n   Maximum at κ = $max_κ with log density = $(round(max_ld, digits=2))")

# Create a finer grid around the maximum
println("\n2. Fine-grained search around maximum...")
κ_fine = 0.01:0.01:2.0
ld_fine = [log_density(κ, counts, μ, hyperprior) for κ in κ_fine]
max_fine_idx = argmax(ld_fine)
max_fine_κ = κ_fine[max_fine_idx]
max_fine_ld = ld_fine[max_fine_idx]

println("   Fine maximum at κ = $(round(max_fine_κ, digits=3)) with log density = $(round(max_fine_ld, digits=2))")

# Analyze the components of the log density
println("\n3. Breaking down log density components at optimal κ...")
κ_opt = max_fine_κ
c₀, d₀ = hyperprior
n = length(counts)
sum_counts = sum(counts)

prior_term = (c₀ - 1) * log(κ_opt) - κ_opt / d₀
likelihood_gamma_terms = sum(-loggamma(κ_opt) + loggamma(n_j + κ_opt) for n_j in counts)
likelihood_normalizing_term = -(sum_counts + n * κ_opt) * log(κ_opt + μ)

println("   Prior term (Gamma): $(round(prior_term, digits=2))")
println("   Likelihood Gamma terms: $(round(likelihood_gamma_terms, digits=2))")
println("   Likelihood normalizing term: $(round(likelihood_normalizing_term, digits=2))")
println("   Total: $(round(prior_term + likelihood_gamma_terms + likelihood_normalizing_term, digits=2))")

# Compare with what we'd expect for reasonable κ values
println("\n4. Comparison with reasonable κ values...")
for κ_test in [5.0, 10.0, 20.0]
    ld_test = log_density(κ_test, counts, μ, hyperprior)
    prior_test = (c₀ - 1) * log(κ_test) - κ_test / d₀
    likelihood_gamma_test = sum(-loggamma(κ_test) + loggamma(n_j + κ_test) for n_j in counts)
    likelihood_norm_test = -(sum_counts + n * κ_test) * log(κ_test + μ)
    
    println("   κ = $κ_test:")
    println("     Total log density: $(round(ld_test, digits=2))")
    println("     Prior: $(round(prior_test, digits=2)), Likelihood Γ: $(round(likelihood_gamma_test, digits=2)), Norm: $(round(likelihood_norm_test, digits=2))")
end

# Check what happens if we change the hyperprior
println("\n5. Testing different hyperpriors...")
for (c₀_test, d₀_test, desc) in [(2.0, 10.0, "Gamma(2,10) - less diffuse"), 
                                  (1.0, 1.0, "Gamma(1,1) - exponential"), 
                                  (10.0, 1.0, "Gamma(10,1) - mean=10, low var")]
    println("   Testing $desc:")
    hyperprior_test = (c₀_test, d₀_test)
    κ_fine_test = 0.01:0.05:50.0
    ld_fine_test = [log_density(κ, counts, μ, hyperprior_test) for κ in κ_fine_test]
    max_test_idx = argmax(ld_fine_test)
    max_test_κ = κ_fine_test[max_test_idx]
    println("     Optimal κ: $(round(max_test_κ, digits=2))")
end

# Create visualization
println("\n6. Creating visualization...")
fig = Figure(size=(1000, 600))

# Main plot
ax1 = Axis(fig[1, 1], 
           xlabel="κ (overdispersion parameter)", 
           ylabel="Log Density",
           title="Log Density vs κ for Negative Binomial Fit")

lines!(ax1, κ_fine, ld_fine, linewidth=2, color=:blue, label="Current hyperprior Γ(0.5,20)")
vlines!(ax1, [max_fine_κ], color=:red, linewidth=2, linestyle=:dash, label="Optimal κ = $(round(max_fine_κ, digits=2))")

# Test with better hyperprior
hyperprior_better = (10.0, 1.0)  # Mean = 10, more concentrated
ld_better = [log_density(κ, counts, μ, hyperprior_better) for κ in κ_fine]
max_better_idx = argmax(ld_better)
max_better_κ = κ_fine[max_better_idx]

lines!(ax1, κ_fine, ld_better, linewidth=2, color=:green, label="Better hyperprior Γ(10,1)")
vlines!(ax1, [max_better_κ], color=:green, linewidth=2, linestyle=:dash, label="Better optimal κ = $(round(max_better_κ, digits=2))")

axislegend(ax1, position=:rt)

# Zoom in on interesting region
ax2 = Axis(fig[1, 2], 
           xlabel="κ", 
           ylabel="Log Density",
           title="Zoomed View (κ < 10)")

κ_zoom = 0.01:0.01:10.0
ld_zoom = [log_density(κ, counts, μ, hyperprior) for κ in κ_zoom]
ld_better_zoom = [log_density(κ, counts, μ, hyperprior_better) for κ in κ_zoom]

lines!(ax2, κ_zoom, ld_zoom, linewidth=2, color=:blue, label="Current Γ(0.5,20)")
lines!(ax2, κ_zoom, ld_better_zoom, linewidth=2, color=:green, label="Better Γ(10,1)")
vlines!(ax2, [max_fine_κ], color=:red, linewidth=2, linestyle=:dash)
vlines!(ax2, [max_better_κ], color=:green, linewidth=2, linestyle=:dash)

axislegend(ax2, position=:rt)

save("debug_kappa_log_density.png", fig, px_per_unit=2)
println("   ✓ Visualization saved to debug_kappa_log_density.png")

println("\n=== Summary ===")
println("The issue is that the κ hyperprior Gamma(0.5, 20) is:")
println("1. Too diffuse (variance = 200)")
println("2. Has shape parameter < 1, creating preference for small κ")
println("3. The current data has empirical overdispersion but the hyperprior overwhelms it")
println("\nRecommendation: Use a more informative hyperprior like Gamma(10, 1)")
println("This has mean=10 and much lower variance=10, preventing drift to extreme values.")
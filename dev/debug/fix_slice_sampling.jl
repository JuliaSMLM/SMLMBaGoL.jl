#!/usr/bin/env julia

"""
Fix the slice sampling numerical issues for κ when data is under-dispersed.
"""

using Pkg; Pkg.activate("examples")
using SMLMBaGoL
using Statistics
using Random
using Distributions

Random.seed!(123)

println("=== Investigating Slice Sampling Numerical Issues ===\n")

# Use under-dispersed data (similar to our debug output)
counts = [18, 19, 17, 19, 18, 13]  # Variance < Mean
μ = mean(counts)  # 17.33
hyperprior = (5.0, 2.0)  # New hyperprior

println("Data: $counts")
println("Mean: $(round(μ, digits=2))")
println("Variance: $(round(var(counts), digits=2))")
println("Ratio: $(round(var(counts)/μ, digits=2)) (< 1 = under-dispersed)")

# Test the log density function at various κ values
function log_density_safe(κ, counts, μ, hyperprior)
    c₀, d₀ = hyperprior
    n = length(counts)
    sum_counts = sum(counts)
    
    if κ ≤ 0
        return -Inf
    end
    
    # Prior contribution: κ ~ Gamma(c₀, d₀)
    log_p = (c₀ - 1) * log(κ) - κ / d₀
    
    # Likelihood contribution using safer numerical computation
    # For NegBinomial(n_j; μ, κ): P(X=n_j) ∝ Γ(n_j+κ)/Γ(κ) * (κ/(κ+μ))^κ * (μ/(κ+μ))^n_j
    
    # Use loggamma from NegativeBinomial distribution directly
    for n_j in counts
        r = κ
        p = κ / (κ + μ) 
        # log P(X = n_j) for NegativeBinomial(r, p)
        try
            nb_dist = NegativeBinomial(r, p)
            log_p += logpdf(nb_dist, n_j)
        catch e
            # Fallback for numerical issues
            return -Inf
        end
    end
    
    return log_p
end

# Test over a wide range of κ values
println("\n1. Testing log density over wide κ range...")
κ_range = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0]

global max_log_density = -Inf
global best_κ = 0.0

for κ in κ_range
    ld = log_density_safe(κ, counts, μ, hyperprior)
    if ld > max_log_density
        global max_log_density = ld
        global best_κ = κ
    end
    println("   κ = $(rpad(κ, 6)) → log density = $(round(ld, digits=2))")
end

println("\n   Best κ found: $best_κ with log density $(round(max_log_density, digits=2))")

# Fine search around the best
println("\n2. Fine search around κ = $best_κ...")
if best_κ < 100
    κ_fine = (best_κ * 0.1):0.1:(best_κ * 10)
else
    κ_fine = (best_κ * 0.1):10:(best_κ * 10)
end

global fine_max_ld = -Inf
global fine_best_κ = 0.0

for κ in κ_fine
    ld = log_density_safe(κ, counts, μ, hyperprior)
    if ld > fine_max_ld
        global fine_max_ld = ld
        global fine_best_κ = κ
    end
end

println("   Fine search best κ: $(round(fine_best_κ, digits=1)) with log density $(round(fine_max_ld, digits=2))")

# Compare with theoretical expectation
println("\n3. Theoretical analysis...")
emp_mean = mean(counts)
emp_var = var(counts)

if emp_var ≤ emp_mean
    println("   Data is under-dispersed or Poisson-like")
    println("   Theoretical κ should be very large (→ ∞)")
    println("   NegBinomial approaches Poisson as κ → ∞")
else
    κ_theory = emp_mean^2 / (emp_var - emp_mean)
    println("   Method of moments κ: $(round(κ_theory, digits=1))")
end

# Test what happens with the existing slice sampler
println("\n4. Testing existing slice sampler...")
initial_κ = 10.0
try
    κ_slice = SMLMBaGoL.update_kappa_slice_adaptive(
        counts, μ, hyperprior, initial_κ; 
        n_steps=50, adapt_width=true
    )
    println("   Slice sampler result: κ = $(round(κ_slice, digits=2))")
    
    # Check log density at this point
    ld_slice = log_density_safe(κ_slice, counts, μ, hyperprior)
    println("   Log density at slice result: $(round(ld_slice, digits=2))")
    println("   Difference from optimal: $(round(ld_slice - fine_max_ld, digits=2))")
    
catch e
    println("   Slice sampler failed: $e")
end

# Test improved initialization strategy
println("\n5. Testing improved slice sampling with better initialization...")

# When data is under-dispersed, start with very large κ
if emp_var ≤ emp_mean * 1.1  # If nearly Poisson or under-dispersed
    better_start = 1000.0
else
    better_start = max(10.0, emp_mean^2 / max(1.0, emp_var - emp_mean))
end

println("   Better starting κ: $(round(better_start, digits=1))")

try
    κ_better = SMLMBaGoL.update_kappa_slice_adaptive(
        counts, μ, hyperprior, better_start; 
        n_steps=50, adapt_width=true
    )
    println("   Better initialized result: κ = $(round(κ_better, digits=2))")
    
    ld_better = log_density_safe(κ_better, counts, μ, hyperprior)
    println("   Log density: $(round(ld_better, digits=2))")
    println("   Difference from optimal: $(round(ld_better - fine_max_ld, digits=2))")
    
catch e
    println("   Better slice sampler failed: $e")
end

println("\n=== Conclusions ===")
println("1. Optimal κ ≈ $(round(fine_best_κ, digits=1)) for this under-dispersed data")
println("2. Slice sampling may have issues with very large κ values")  
println("3. Need better initialization when variance ≤ mean")
println("4. May need bounds or different sampling strategy for under-dispersed case")
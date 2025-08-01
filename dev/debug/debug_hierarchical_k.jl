#!/usr/bin/env julia

"""
Debug script to investigate why hierarchical K parameter updates are finding κ < 1
when data should show no overdispersion.
"""

using Pkg; Pkg.activate("examples")
using SMLMBaGoL
using Statistics
using Random
using Distributions

Random.seed!(123)

println("=== Debugging Hierarchical K Parameter Updates ===\n")

# Generate small test case with known Poisson-like behavior
println("1. Generate synthetic data with expected Poisson-like behavior...")
localizations, spatial_prior, count_prior = simulate_n_mer(
    n=6,
    diameter=0.100,
    photons=1000,
    localizations_per_emitter_mean=20,  # Lower variance 
    localizations_per_emitter_variance=5,  # Much less variance than mean -> low overdispersion
    tau=0.001
)

println("   ✓ Created $(length(localizations)) localizations for 6 emitters")
println("   ✓ Expected: Poisson-like distribution (variance ≈ mean)")

# Run short BaGoL analysis to get initial data
println("\n2. Running short BaGoL analysis to collect initial emitter count data...")
chains_result = run_bagol(localizations; 
    n_iterations=5000,
    burn_in=1000,
    enable_hierarchical=true,
    hierarchical_interval=500,  # More frequent updates for diagnosis
    tau_mean=0.001,
    partition_data=false
)

chains = isa(chains_result, Vector) ? chains_result : [chains_result]

# Extract emitter counts from current state
println("\n3. Analyzing emitter count data from MCMC samples...")

# Get allocation counts from several samples
all_counts = SMLMBaGoL.collect_emitter_counts(chains)
println("   ✓ Collected $(length(all_counts)) emitter count observations")

if !isempty(all_counts)
    emp_mean = mean(all_counts)
    emp_var = var(all_counts)
    emp_std = std(all_counts)
    
    println("   • Empirical statistics:")
    println("     - Mean: $(round(emp_mean, digits=2))")
    println("     - Variance: $(round(emp_var, digits=2))")
    println("     - Std dev: $(round(emp_std, digits=2))")
    println("     - Variance/Mean ratio: $(round(emp_var/emp_mean, digits=2))")
    
    # For Poisson: var = mean, so ratio = 1
    # For NegBinomial: var = μ + μ²/κ, so ratio = 1 + μ/κ
    # If ratio ≈ 1, then κ should be very large (no overdispersion)
    # If ratio > 1, then κ = μ/(ratio - 1)
    
    println("\n   • Expected κ from data:")
    if emp_var > emp_mean
        expected_κ = emp_mean^2 / (emp_var - emp_mean)
        println("     - Method of moments κ: $(round(expected_κ, digits=2))")
    else
        println("     - Data is under-dispersed (var < mean), κ should be very large")
    end
    
    # Check what the hierarchical fitting found
    if !isempty(chains[1].hierarchical_history)
        last_update = chains[1].hierarchical_history[end]
        println("\n   • Hierarchical fit found:")
        println("     - μ: $(round(last_update.μ, digits=2))")
        println("     - κ: $(round(last_update.κ, digits=2))")
        println("     - Theoretical var: $(round(last_update.μ + last_update.μ^2/last_update.κ, digits=2))")
        
        # Check if κ is unreasonably small
        if last_update.κ < 1.0
            println("     ⚠ WARNING: κ < 1 indicates extreme overdispersion!")
            println("       This suggests the fitting algorithm is finding local minimum")
        end
    end
    
    # Test direct fitting methods
    println("\n4. Testing direct fitting methods on the same data...")
    
    # Method of moments
    println("   • Method of moments:")
    try
        μ_mom, κ_mom = SMLMBaGoL.fit_negbinomial_mom(all_counts)
        println("     - μ: $(round(μ_mom, digits=2)), κ: $(round(κ_mom, digits=2))")
    catch e
        println("     - Failed: $e")
    end
    
    # Maximum likelihood  
    println("   • Maximum likelihood:")
    try
        μ_mle, κ_mle = SMLMBaGoL.fit_negbinomial_mle(all_counts)
        println("     - μ: $(round(μ_mle, digits=2)), κ: $(round(κ_mle, digits=2))")
    catch e
        println("     - Failed: $e")
    end
    
    # Test slice sampling directly
    println("\n5. Testing slice sampling for κ directly...")
    μ_est = mean(all_counts)
    hyperprior = (0.5, 20.0)  # Same as used in hierarchical prior
    
    println("   • Testing different starting points for κ:")
    for start_κ in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
        try
            new_κ = SMLMBaGoL.update_kappa_slice_adaptive(
                all_counts, μ_est, hyperprior, start_κ; 
                n_steps=20, adapt_width=true
            )
            println("     - Start κ=$(start_κ) → Final κ=$(round(new_κ, digits=2))")
        catch e
            println("     - Start κ=$(start_κ) → Failed: $e")
        end
    end
    
    # Test multi-start sampling
    println("\n   • Multi-start slice sampling:")
    try
        κ_multistart = SMLMBaGoL.update_kappa_multistart(
            all_counts, μ_est, hyperprior, 1.0; 
            n_starts=5, n_steps=10, diagnostic=true
        )
        println("     - Multi-start result: κ=$(round(κ_multistart, digits=2))")
    catch e
        println("     - Multi-start failed: $e")
    end
end

println("\n6. Examining allocation patterns...")
# Look at the actual allocation counts in a sample
if !isempty(chains) && !isempty(chains[1].samples)
    sample = chains[1].samples[end]  # Last sample
    emitter_counts = SMLMBaGoL.count_allocations(sample)
    println("   • Last sample allocation counts: $emitter_counts")
    println("   • Number of emitters with allocations: $(count(x -> x > 0, emitter_counts))")
    println("   • Emitter count statistics:")
    println("     - Min: $(minimum(emitter_counts))")
    println("     - Max: $(maximum(emitter_counts))")
    println("     - Mean: $(round(mean(emitter_counts), digits=2))")
    println("     - Std: $(round(std(emitter_counts), digits=2))")
    
    # Check if there are many emitters with 0 allocations
    zero_count = count(x -> x == 0, emitter_counts)
    if zero_count > 0
        println("   ⚠ Found $zero_count emitters with 0 allocations")
        println("     This could bias the variance calculation upward")
    end
end

println("\n=== Analysis Complete ===")
println("\nKey insights:")
println("• If κ < 1 is consistently found, this suggests:")
println("  1. The slice sampling is getting stuck in local minima")
println("  2. The hyperprior on κ is too restrictive")  
println("  3. There may be zero-allocation emitters biasing the variance")
println("  4. The data collection includes spurious count patterns")
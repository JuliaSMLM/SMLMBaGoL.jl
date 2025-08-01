#!/usr/bin/env julia

"""
Diagnostic test: Run BaGoL with hyperparameters fixed at TRUE values.

This tests whether the algorithm can recover the correct K when given perfect
prior information. If this fails, we have a fundamental problem beyond just
parameter drift.
"""

using Pkg; Pkg.activate("../..")
using SMLMBaGoL
using Statistics
using Random
using Printf

# Configuration
const TRUE_K = 6
const N_ITERATIONS = 20000
const BURN_IN = 2000
const N_TRIALS = 5

# Generate test data and run with different prior configurations
function run_trial_with_prior_config(trial_num::Int, seed::Int, prior_config::Symbol)
    Random.seed!(seed)
    
    # Generate synthetic data
    localizations, _, _ = SMLMBaGoL.simulate_n_mer(
        n=TRUE_K,
        diameter=0.1,
        photons=1000.0,
        sigma_psf=0.13,
        min_photons=300.0,
        localizations_per_emitter_mean=50,
        localizations_per_emitter_variance=50,
        tau=1.0e-3  # 1 nm
    )
    
    n_locs = length(localizations)
    true_μ = n_locs / TRUE_K  # True mean localizations per emitter
    
    println("    Generated $(n_locs) localizations, true μ = $(round(true_μ, digits=1))")
    
    # Create spatial prior
    spatial_prior = SMLMBaGoL.create_spatial_prior_from_localizations(localizations, 0.2)
    
    # Create count prior based on configuration
    if prior_config == :fixed_true
        # Fixed at TRUE values - this SHOULD work perfectly
        count_prior = SMLMBaGoL.HierarchicalNegBinomialPrior(
            true_μ,      # TRUE μ
            10.0,        # Reasonable κ
            1.0e-6,      # τ² = (1nm)²
            (2.0, 5.0),  # μ hyperprior (ignored when not updating)
            (5.0, 2.0),  # κ hyperprior (ignored when not updating)
            (1.1, 1.0e-7)  # τ² hyperprior (ignored when not updating)
        )
    elseif prior_config == :fixed_wrong
        # Fixed at WRONG values - should struggle
        count_prior = SMLMBaGoL.HierarchicalNegBinomialPrior(
            10.0,        # WRONG μ (too low)
            10.0,        # κ
            1.0e-6,      # τ²
            (2.0, 5.0),  # μ hyperprior (ignored)
            (5.0, 2.0),  # κ hyperprior (ignored)
            (1.1, 1.0e-7)  # τ² hyperprior (ignored)
        )
    elseif prior_config == :hierarchical
        # Standard hierarchical (what we've been using)
        count_prior = SMLMBaGoL.HierarchicalNegBinomialPrior(
            10.0,        # Initial μ (will update)
            10.0,        # Initial κ (will update)
            1.0e-6,      # Initial τ² (will update)
            (2.0, 5.0),  # μ hyperprior
            (5.0, 2.0),  # κ hyperprior
            (1.1, 1.0e-7)  # τ² hyperprior
        )
    else
        error("Unknown prior config: $prior_config")
    end
    
    # Run BaGoL
    enable_updates = (prior_config == :hierarchical)
    
    chains = SMLMBaGoL.run_bagol(
        localizations,
        n_iterations=N_ITERATIONS,
        burn_in=BURN_IN,
        spatial_prior=spatial_prior,
        count_prior=count_prior,
        enable_hierarchical=enable_updates,  # KEY: Only update if hierarchical
        hierarchical_interval=2000,
        partition_data=false,
        enable_threading=false
    )
    
    # Extract results
    chain = chains[1]
    mapn_result = SMLMBaGoL.estimate_mapn(chain, burn_in=BURN_IN)
    mapn_k = length(mapn_result.emitters)
    
    # Get K trajectory
    k_samples = [length(state.emitters) for state in chain.samples[(BURN_IN+1):end]]
    k_posterior_mean = mean(k_samples)
    k_mode = mode(k_samples)
    
    # Get final hyperparameters
    final_μ = chain.count_prior.μ
    final_κ = chain.count_prior.κ
    
    return (
        mapn_k=mapn_k,
        k_mode=k_mode,
        k_posterior_mean=k_posterior_mean,
        final_μ=final_μ,
        true_μ=true_μ,
        n_locs=n_locs,
        error_pct=abs(mapn_k - TRUE_K) / TRUE_K * 100
    )
end

# Helper function for mode
function mode(x)
    counts = Dict{eltype(x), Int}()
    for val in x
        counts[val] = get(counts, val, 0) + 1
    end
    return argmax(counts)
end

# Run tests
println("🔬 DIAGNOSTIC TEST: Fixed Hyperparameters")
println("=========================================")
println("True K = $TRUE_K")
println("Testing whether algorithm recovers correct K with perfect priors\n")

# Test configurations
configs = [
    (:fixed_true, "Fixed at TRUE μ (should work perfectly)"),
    (:fixed_wrong, "Fixed at WRONG μ=10 (should fail)"),
    (:hierarchical, "Standard hierarchical updates")
]

results_by_config = Dict{Symbol, Vector{Any}}()

for (config, description) in configs
    println("\n📊 Configuration: $description")
    println(repeat("-", 60))
    
    results = []
    seeds = [4000 + i for i in 1:N_TRIALS]
    
    for trial in 1:N_TRIALS
        print("  Trial $trial: ")
        result = run_trial_with_prior_config(trial, seeds[trial], config)
        push!(results, result)
        
        @printf("MAPN K=%d (%.0f%% error), μ: %.1f", result.mapn_k, result.error_pct, result.final_μ)
        
        # Success indicator
        if result.mapn_k == TRUE_K
            print(" ✓")
        elseif abs(result.mapn_k - TRUE_K) <= 1
            print(" ~")
        else
            print(" ✗")
        end
        println()
    end
    
    results_by_config[config] = results
    
    # Summary statistics
    mapn_ks = [r.mapn_k for r in results]
    exact_matches = count(k -> k == TRUE_K, mapn_ks)
    mean_error = mean([r.error_pct for r in results])
    
    println("\n  Summary:")
    @printf("  - Exact matches: %d/%d (%.0f%%)\n", exact_matches, N_TRIALS, (exact_matches/N_TRIALS*100))
    @printf("  - Mean error: %.1f%%\n", mean_error)
    @printf("  - MAPN K values: %s\n", join(mapn_ks, ", "))
end

# Comparative analysis
println("\n" * repeat("=", 70))
println("🎯 COMPARATIVE ANALYSIS")
println(repeat("=", 70))

# Extract key metrics
fixed_true_results = results_by_config[:fixed_true]
fixed_wrong_results = results_by_config[:fixed_wrong]
hierarchical_results = results_by_config[:hierarchical]

fixed_true_accuracy = count(r -> r.mapn_k == TRUE_K, fixed_true_results) / N_TRIALS * 100
fixed_wrong_accuracy = count(r -> r.mapn_k == TRUE_K, fixed_wrong_results) / N_TRIALS * 100
hierarchical_accuracy = count(r -> r.mapn_k == TRUE_K, hierarchical_results) / N_TRIALS * 100

println("Exact Match Accuracy:")
@printf("  Fixed TRUE μ:    %.0f%%\n", fixed_true_accuracy)
@printf("  Fixed WRONG μ:   %.0f%%\n", fixed_wrong_accuracy)
@printf("  Hierarchical:    %.0f%%\n", hierarchical_accuracy)

println("\n🔍 DIAGNOSIS:")

if fixed_true_accuracy >= 80
    println("✅ Algorithm works correctly with true hyperparameters!")
    println("   → The problem is parameter drift/convergence, not fundamental")
    
    if hierarchical_accuracy < 40
        println("   → Hierarchical updates are converging to wrong values")
        println("   → Need to fix hyperprior parameters or initialization")
    end
else
    println("❌ Algorithm FAILS even with perfect hyperparameters!")
    println("   → This indicates a FUNDAMENTAL problem in:")
    println("     • Likelihood calculation")
    println("     • Birth/death move balance")
    println("     • Allocation algorithm")
    println("     • Prior formulation")
    println("\n   → Need to debug the core algorithm, not just hyperparameters")
end

# Check if wrong μ behaves as expected
if fixed_wrong_accuracy < fixed_true_accuracy
    println("\n✅ Wrong μ performs worse, as expected")
else
    println("\n⚠️ Wrong μ doesn't hurt performance - prior might be too weak")
end

println("\n📝 RECOMMENDATIONS:")
if fixed_true_accuracy >= 80 && hierarchical_accuracy < 40
    println("1. Focus on fixing hyperparameter convergence")
    println("2. Consider better initialization strategies")
    println("3. Tune hyperpriors to prevent drift")
    println("4. Maybe increase hierarchical update interval")
elseif fixed_true_accuracy < 80
    println("1. Debug core MCMC algorithm")
    println("2. Check likelihood calculations")
    println("3. Verify birth/death/move/allocate implementations")
    println("4. Test with even simpler scenarios")
else
    println("1. Algorithm is working reasonably well")
    println("2. Minor tuning of hyperparameters may help")
end

println("\n🎉 Diagnostic test complete!")
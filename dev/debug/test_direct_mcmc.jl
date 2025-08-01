#!/usr/bin/env julia

"""
Direct MCMC test with fixed hyperparameters to diagnose the fundamental issue.
"""

using Pkg; Pkg.activate("../..")
using SMLMBaGoL
using Statistics
using Random
using Printf

# Configuration
const TRUE_K = 6
const N_ITERATIONS = 10000
const N_TRIALS = 3

# Generate test data
function generate_test_data(seed::Int)
    Random.seed!(seed)
    
    # Simple n-mer simulation  
    localizations, spatial_prior, _ = SMLMBaGoL.simulate_n_mer(
        n=TRUE_K,
        diameter=0.1,
        photons=1000.0,
        sigma_psf=0.13,
        min_photons=300.0,
        localizations_per_emitter_mean=50,
        localizations_per_emitter_variance=50,
        tau=1.0e-3
    )
    
    n_locs = length(localizations)
    true_μ = n_locs / TRUE_K
    
    return localizations, spatial_prior, true_μ, n_locs
end

# Create count prior with specified μ
function create_fixed_prior(μ::Float64, update_enabled::Bool)
    κ = 10.0
    τ² = 1.0e-6
    
    prior = SMLMBaGoL.HierarchicalNegBinomialPrior(
        μ, κ, τ²,
        (2.0, 5.0),  # μ hyperprior
        (5.0, 2.0),  # κ hyperprior  
        (3.0, 3.0e-6)  # τ² hyperprior
    )
    
    # HACK: If updates disabled, we'll just not call update_hyperparameters
    return prior, update_enabled
end

# Run single MCMC trial
function run_mcmc_trial(localizations, spatial_prior, count_prior, enable_updates::Bool)
    # Initialize chain
    initial_K = max(1, length(localizations) ÷ 10)
    chain = SMLMBaGoL.initialize_chain(
        localizations,
        SMLMBaGoL.Emitter2D{Float64},
        spatial_prior,
        count_prior;
        initial_K=initial_K,
        burn_in=2000,
        thin=1
    )
    
    # Run MCMC with proper sample storage
    if enable_updates
        # Run with hierarchical updates
        for block in 1:5  # 5 blocks of 2000 iterations
            SMLMBaGoL.run_rjmcmc!(chain, 2000)
            if block > 1  # Skip first block (burn-in)
                SMLMBaGoL.update_hyperparameters!(chain)
            end
        end
    else
        # Run without updates (fixed hyperparameters)
        SMLMBaGoL.run_rjmcmc!(chain, N_ITERATIONS)
    end
    
    # Extract K trajectory (samples are already post-burn-in due to chain settings)
    k_samples = [length(state.emitters) for state in chain.samples]
    
    # Calculate statistics
    k_mode = mode(k_samples)
    k_mean = mean(k_samples)
    final_μ = chain.count_prior.μ
    
    return (
        k_mode=k_mode,
        k_mean=k_mean,
        final_μ=final_μ,
        k_samples=k_samples
    )
end

# Mode helper
function mode(x)
    counts = Dict()
    for val in x
        counts[val] = get(counts, val, 0) + 1
    end
    return argmax(counts)
end

# Main test
println("🔬 DIRECT MCMC TEST: Fixed vs Updating Hyperparameters")
println("="^60)
println("True K = $TRUE_K")
println("MCMC iterations = $N_ITERATIONS per trial\n")

# Test scenarios
scenarios = [
    ("Fixed TRUE μ", :fixed_true),
    ("Fixed WRONG μ=10", :fixed_wrong),
    ("Updating μ (standard)", :updating)
]

results_summary = Dict()

for (desc, scenario) in scenarios
    println("\n📊 Scenario: $desc")
    println("-"^50)
    
    trial_results = []
    
    for trial in 1:N_TRIALS
        # Generate data
        localizations, spatial_prior, true_μ, n_locs = generate_test_data(5000 + trial)
        
        # Set up count prior based on scenario
        if scenario == :fixed_true
            count_prior, enable_updates = create_fixed_prior(true_μ, false)
        elseif scenario == :fixed_wrong
            count_prior, enable_updates = create_fixed_prior(10.0, false)
        else  # :updating
            count_prior, enable_updates = create_fixed_prior(10.0, true)
        end
        
        # Run MCMC
        result = run_mcmc_trial(localizations, spatial_prior, count_prior, enable_updates)
        push!(trial_results, result)
        
        # Print result
        error = abs(result.k_mode - TRUE_K) / TRUE_K * 100
        status = result.k_mode == TRUE_K ? "✓" : "✗"
        
        @printf("  Trial %d: K=%d (%.0f%% error) μ=%.1f [N=%d] %s\n", 
                trial, result.k_mode, error, result.final_μ, n_locs, status)
    end
    
    results_summary[scenario] = trial_results
    
    # Summary statistics
    k_modes = [r.k_mode for r in trial_results]
    exact_matches = count(k -> k == TRUE_K, k_modes)
    
    println("\n  Summary: $(exact_matches)/$N_TRIALS exact matches")
    println("  K modes: $(k_modes)")
end

# Final analysis
println("\n" * "="^60)
println("🎯 ANALYSIS")
println("="^60)

fixed_true_results = results_summary[:fixed_true]
fixed_wrong_results = results_summary[:fixed_wrong]
updating_results = results_summary[:updating]

fixed_true_accuracy = count(r -> r.k_mode == TRUE_K, fixed_true_results) / N_TRIALS * 100
fixed_wrong_accuracy = count(r -> r.k_mode == TRUE_K, fixed_wrong_results) / N_TRIALS * 100
updating_accuracy = count(r -> r.k_mode == TRUE_K, updating_results) / N_TRIALS * 100

@printf("Accuracy: Fixed TRUE μ: %.0f%%, Fixed WRONG μ: %.0f%%, Updating: %.0f%%\n",
        fixed_true_accuracy, fixed_wrong_accuracy, updating_accuracy)

println("\n🔍 DIAGNOSIS:")
if fixed_true_accuracy >= 66  # 2/3 trials
    println("✅ Algorithm works with correct μ!")
    println("   → Problem is parameter drift, not fundamental")
else
    println("❌ Algorithm FAILS even with perfect μ!")
    println("   → FUNDAMENTAL problem in algorithm")
    
    # Print K distribution for diagnosis
    println("\n   K distributions with true μ:")
    for (i, result) in enumerate(fixed_true_results)
        k_counts = Dict()
        for k in result.k_samples
            k_counts[k] = get(k_counts, k, 0) + 1
        end
        total = length(result.k_samples)
        
        println("   Trial $i:")
        for k in sort(collect(keys(k_counts)))
            pct = k_counts[k] / total * 100
            bar = repeat("█", Int(round(pct/5)))
            marker = k == TRUE_K ? " ← TRUE" : ""
            @printf("     K=%d: %.1f%% %s%s\n", k, pct, bar, marker)
        end
    end
end

println("\n🎉 Direct MCMC test complete!")
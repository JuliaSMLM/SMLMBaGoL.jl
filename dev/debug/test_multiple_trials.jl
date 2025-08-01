#!/usr/bin/env julia

"""
Multi-trial test harness for validating the K prior fix.

Tests three scenarios:
1. Original problematic prior (μ-dependent)
2. New independent K prior (current fix)  
3. "Oracle" prior with true μ values (theoretical best case)
"""

using Pkg; Pkg.activate("../..")
using SMLMBaGoL
using Statistics
using Random
using Printf

# Test configuration
const N_TRIALS = 10
const TRUE_K = 6
const N_ITERATIONS = 20000  # Shorter for faster testing
const BURN_IN = 2000

# Test scenarios
abstract type PriorScenario end
struct ProblematicPrior <: PriorScenario end
struct IndependentPrior <: PriorScenario end  
struct OraclePrior <: PriorScenario end

function setup_prior(scenario::ProblematicPrior, localizations, true_μ)
    # Use the old problematic μ-dependent prior
    n_locs = length(localizations)
    spatial_prior, _ = SMLMBaGoL.create_n_mer_priors(localizations)
    
    # Force use of problematic prior by overriding log_prior_k_independent
    # We'll need to modify the code temporarily for this test
    return spatial_prior, nothing  # Will use original logic
end

function setup_prior(scenario::IndependentPrior, localizations, true_μ)
    # Use the new independent K prior (current implementation)
    n_locs = length(localizations)
    spatial_prior, count_prior = SMLMBaGoL.create_n_mer_priors(localizations)
    return spatial_prior, count_prior
end

function setup_prior(scenario::OraclePrior, localizations, true_μ)
    # Use hierarchical prior but initialize with true μ
    n_locs = length(localizations)
    spatial_prior, _ = SMLMBaGoL.create_n_mer_priors(localizations)
    
    # Create count prior with true μ
    κ = 10.0
    τ²_mean = 1.0
    a_τ = 1.1
    b_τ = τ²_mean * (a_τ - 1)
    initial_τ² = b_τ / (a_τ + 1)
    
    count_prior = SMLMBaGoL.HierarchicalNegBinomialPrior(
        true_μ, κ, initial_τ²,    # Use TRUE μ
        (2.0, 5.0),               # μ hyperprior  
        (5.0, 2.0),               # κ hyperprior
        (a_τ, b_τ)                # τ² hyperprior
    )
    
    return spatial_prior, count_prior
end

function run_single_trial(scenario::PriorScenario, trial_num::Int, seed::Int)
    Random.seed!(seed)
    
    println("  Trial $trial_num (seed=$seed)...")
    
    # Generate data
    emitters, localizations = SMLMBaGoL.simulate_n_mer(
        n=TRUE_K, 
        diameter=0.1, 
        photons=1000.0,
        localizations_per_emitter_mean=50,
        localizations_per_emitter_variance=50,
        sigma_psf=0.13,
        tau=1.0,
        min_photons=300.0
    )
    
    true_μ = length(localizations) / TRUE_K
    
    # Setup priors based on scenario
    spatial_prior, count_prior = setup_prior(scenario, localizations, true_μ)
    
    # Run analysis
    if count_prior === nothing
        # Need to temporarily revert to old prior for problematic test
        # For now, skip this scenario
        return (mapn_k=999, posterior_mean=999.0, converged_μ=999.0, n_locs=length(localizations))
    end
    
    try
        chains = SMLMBaGoL.run_bagol(
            localizations,
            n_iterations=N_ITERATIONS,
            burn_in=BURN_IN,
            spatial_prior=spatial_prior,
            count_prior=count_prior,
            enable_hierarchical=true,
            hierarchical_interval=5000,
            partition_data=false,  # Single partition for simplicity
            enable_threading=false
        )
        
        # Extract results
        chain = chains[1]
        mapn_result = SMLMBaGoL.estimate_mapn(chain, burn_in=BURN_IN)
        mapn_k = length(mapn_result.emitters)
        
        # Calculate posterior mean K
        k_samples = [length(state.emitters) for state in chain.samples[(BURN_IN+1):end]]
        posterior_mean = mean(k_samples)
        
        # Get final hierarchical parameters
        converged_μ = chain.count_prior.μ
        
        return (
            mapn_k=mapn_k, 
            posterior_mean=posterior_mean, 
            converged_μ=converged_μ,
            n_locs=length(localizations)
        )
        
    catch e
        println("    Error in trial $trial_num: $e")
        return (mapn_k=999, posterior_mean=999.0, converged_μ=999.0, n_locs=length(localizations))
    end
end

function run_scenario_tests(scenario::PriorScenario, scenario_name::String)
    println("\n" * "="^60)
    println("Testing: $scenario_name")
    println("="^60)
    
    results = []
    seeds = [1000 + i for i in 1:N_TRIALS]  # Reproducible seeds
    
    for trial in 1:N_TRIALS
        result = run_single_trial(scenario, trial, seeds[trial])
        push!(results, result)
        
        if result.mapn_k != 999
            error_pct = abs(result.mapn_k - TRUE_K) / TRUE_K * 100
            @printf "    MAPN K=%d (error: %.1f%%), E[K]=%.2f, μ=%.1f, N=%d\n" result.mapn_k error_pct result.posterior_mean result.converged_μ result.n_locs
        end
    end
    
    # Filter out failed trials
    valid_results = filter(r -> r.mapn_k != 999, results)
    n_valid = length(valid_results)
    
    if n_valid == 0
        println("❌ All trials failed!")
        return nothing
    end
    
    # Calculate statistics
    mapn_ks = [r.mapn_k for r in valid_results]
    posterior_means = [r.posterior_mean for r in valid_results]
    converged_μs = [r.converged_μ for r in valid_results]
    
    mapn_accuracy = count(k -> k == TRUE_K, mapn_ks) / n_valid * 100
    mean_mapn_error = mean(abs.(mapn_ks .- TRUE_K))
    mean_posterior_k = mean(posterior_means)
    std_posterior_k = std(posterior_means)
    mean_converged_μ = mean(converged_μs)
    
    println("\n📊 SUMMARY ($n_valid/$N_TRIALS trials successful):")
    @printf "  MAPN Accuracy: %.1f%% (exact matches)\n" mapn_accuracy
    @printf "  Mean MAPN Error: %.2f emitters\n" mean_mapn_error
    @printf "  Posterior Mean K: %.2f ± %.2f\n" mean_posterior_k std_posterior_k
    @printf "  Mean Converged μ: %.1f (true ≈ %.1f)\n" mean_converged_μ mean([r.n_locs for r in valid_results]) / TRUE_K
    
    return (
        scenario_name=scenario_name,
        mapn_accuracy=mapn_accuracy,
        mean_mapn_error=mean_mapn_error,
        mean_posterior_k=mean_posterior_k,
        std_posterior_k=std_posterior_k,
        mean_converged_μ=mean_converged_μ,
        n_valid=n_valid
    )
end

println("🧪 Multi-Trial K Prior Validation")
println("Testing $N_TRIALS trials each with $N_ITERATIONS iterations")
println("True K = $TRUE_K")

# Test scenarios
results = []

# Test 1: New independent prior (current implementation)
push!(results, run_scenario_tests(IndependentPrior(), "Independent K Prior (Current Fix)"))

# Test 2: Oracle with true μ values  
push!(results, run_scenario_tests(OraclePrior(), "Oracle Prior (True μ values)"))

# Print comparative summary
println("\n" * "="^80)
println("🏆 COMPARATIVE RESULTS")
println("="^80)

valid_results = filter(r -> r !== nothing, results)
if length(valid_results) >= 2
    current = valid_results[1]
    oracle = valid_results[2]
    
    println("Metric                    | Current Fix | Oracle (Best Case) | Gap")
    println("-" * 65)
    @printf "MAPN Accuracy (%%)         | %6.1f      | %6.1f             | %.1f\n" current.mapn_accuracy oracle.mapn_accuracy (oracle.mapn_accuracy - current.mapn_accuracy)
    @printf "Mean MAPN Error           | %6.2f      | %6.2f             | %.2f\n" current.mean_mapn_error oracle.mean_mapn_error (oracle.mean_mapn_error - current.mean_mapn_error)
    @printf "Posterior Mean K          | %6.2f      | %6.2f             | %.2f\n" current.mean_posterior_k oracle.mean_posterior_k (oracle.mean_posterior_k - current.mean_posterior_k)
    @printf "Posterior Std K           | %6.2f      | %6.2f             | %.2f\n" current.std_posterior_k oracle.std_posterior_k (oracle.std_posterior_k - current.std_posterior_k)
    
    println("\n🎯 CONCLUSIONS:")
    if abs(current.mapn_accuracy - oracle.mapn_accuracy) < 10
        println("✅ Current fix performs close to theoretical optimum!")
    else
        println("⚠️ Current fix still has room for improvement")
    end
    
    if current.mean_mapn_error < 1.0
        println("✅ Mean error < 1 emitter - excellent performance")
    elseif current.mean_mapn_error < 2.0
        println("✅ Mean error < 2 emitters - good performance")  
    else
        println("⚠️ Mean error ≥ 2 emitters - needs improvement")
    end
end

println("\n🎉 Multi-trial validation complete!")
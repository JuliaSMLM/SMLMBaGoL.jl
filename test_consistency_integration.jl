#!/usr/bin/env julia

"""
Test the consistency likelihood integration with run_bagol
"""

using Pkg
Pkg.activate(".")
using SMLMBaGoL
using Statistics
using StatsBase
using Random
using Printf

Random.seed!(42)

println("🧪 TESTING CONSISTENCY LIKELIHOOD INTEGRATION")
println("="^70)

# Generate test data: 6 emitters
localizations, _, _ = SMLMBaGoL.simulate_n_mer(
    n=6,
    diameter=0.1,
    photons=1000.0,
    sigma_psf=0.13,
    min_photons=300.0,
    localizations_per_emitter_mean=40,
    localizations_per_emitter_variance=20,
    tau=1.0e-3
)

n_locs = length(localizations)
println("Generated $(n_locs) localizations from 6 emitters")

# Test 1: Standard likelihood
println("\n📊 Test 1: Standard Likelihood (baseline)")
println("-"^70)

chain_standard = SMLMBaGoL.run_bagol(
    localizations;
    n_iterations=2000,
    burn_in=500,
    thin=1,
    initial_K=10,
    partition_data=false,  # Single partition for clarity
    enable_hierarchical=false,
    likelihood_config=SMLMBaGoL.StandardLikelihood()
)

k_samples_standard = [length(state.emitters) for state in chain_standard.samples]
k_mode_standard = argmax(countmap(k_samples_standard))
k_mean_standard = mean(k_samples_standard)

println("Standard likelihood results:")
println("  Mode K = $k_mode_standard")
println("  Mean K = $(round(k_mean_standard, digits=1))")

# Test 2: Consistency likelihood with α=10
println("\n📊 Test 2: Consistency Likelihood (α=10)")
println("-"^70)

chain_consistency_10 = SMLMBaGoL.run_bagol(
    localizations;
    n_iterations=2000,
    burn_in=500,
    thin=1,
    initial_K=10,
    partition_data=false,
    enable_hierarchical=false,
    likelihood_config=SMLMBaGoL.ConsistencyLikelihood(10.0)
)

k_samples_consistency_10 = [length(state.emitters) for state in chain_consistency_10.samples]
k_mode_consistency_10 = argmax(countmap(k_samples_consistency_10))
k_mean_consistency_10 = mean(k_samples_consistency_10)

println("Consistency likelihood (α=10) results:")
println("  Mode K = $k_mode_consistency_10")
println("  Mean K = $(round(k_mean_consistency_10, digits=1))")

# Test 3: Consistency likelihood with α=20
println("\n📊 Test 3: Consistency Likelihood (α=20)")
println("-"^70)

chain_consistency_20 = SMLMBaGoL.run_bagol(
    localizations;
    n_iterations=2000,
    burn_in=500,
    thin=1,
    initial_K=10,
    partition_data=false,
    enable_hierarchical=false,
    likelihood_config=SMLMBaGoL.ConsistencyLikelihood(20.0)
)

k_samples_consistency_20 = [length(state.emitters) for state in chain_consistency_20.samples]
k_mode_consistency_20 = argmax(countmap(k_samples_consistency_20))
k_mean_consistency_20 = mean(k_samples_consistency_20)

println("Consistency likelihood (α=20) results:")
println("  Mode K = $k_mode_consistency_20")
println("  Mean K = $(round(k_mean_consistency_20, digits=1))")

# Test 4: Adaptive consistency likelihood
println("\n📊 Test 4: Adaptive Consistency Likelihood")
println("-"^70)

chain_adaptive = SMLMBaGoL.run_bagol(
    localizations;
    n_iterations=2000,
    burn_in=500,
    thin=1,
    initial_K=10,
    partition_data=false,
    enable_hierarchical=false,
    likelihood_config=SMLMBaGoL.AdaptiveConsistencyLikelihood()
)

k_samples_adaptive = [length(state.emitters) for state in chain_adaptive.samples]
k_mode_adaptive = argmax(countmap(k_samples_adaptive))
k_mean_adaptive = mean(k_samples_adaptive)

println("Adaptive consistency likelihood results:")
println("  Mode K = $k_mode_adaptive")
println("  Mean K = $(round(k_mean_adaptive, digits=1))")

# Summary
println("\n🎯 SUMMARY:")
println("="^70)
println("True K = 6")
println("Standard:         K = $k_mode_standard ($(round(abs(k_mode_standard - 6)/6 * 100))% error)")
println("Consistency α=10: K = $k_mode_consistency_10 ($(round(abs(k_mode_consistency_10 - 6)/6 * 100))% error)")
println("Consistency α=20: K = $k_mode_consistency_20 ($(round(abs(k_mode_consistency_20 - 6)/6 * 100))% error)")
println("Adaptive:         K = $k_mode_adaptive ($(round(abs(k_mode_adaptive - 6)/6 * 100))% error)")

println("\n✅ Integration test complete!")


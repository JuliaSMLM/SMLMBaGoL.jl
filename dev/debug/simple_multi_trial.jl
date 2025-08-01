#!/usr/bin/env julia

"""
Simple multi-trial test to validate the K prior fix.
"""

using Pkg; Pkg.activate("../..")
using SMLMBaGoL
using Statistics
using Random
using Printf

# Test configuration
const N_TRIALS = 5  # Start smaller for debugging
const TRUE_K = 6
const N_ITERATIONS = 10000  # Shorter iterations
const BURN_IN = 1000

# N-mer simulation parameters (from nmer_demo.jl)
const DIAMETER = 0.1                      # μm
const PHOTONS_PER_LOC = 1000              # Mean photons per localization
const LOCS_PER_EMITTER_MEAN = 50          # Mean localizations per emitter
const LOCS_PER_EMITTER_VAR = 50           # Variance in localizations per emitter  
const PSF_SIGMA = 0.13                    # μm
const MIN_PHOTONS = 300                   # Minimum photon threshold
const TAU = 0.001                         # Systematic noise (μm)

function run_single_trial(trial_num::Int, seed::Int)
    Random.seed!(seed)
    
    println("  Trial $trial_num (seed=$seed)...")
    
    try
        # Generate data exactly like nmer_demo
        localizations = SMLMBaGoL.generate_n_mer_data(
            n=TRUE_K, 
            diameter=DIAMETER,
            n_frames=1000,
            photons_per_localization=PHOTONS_PER_LOC,
            photons_per_localization_std=200,
            localizations_per_emitter_mean=LOCS_PER_EMITTER_MEAN,
            localizations_per_emitter_variance=LOCS_PER_EMITTER_VAR,
            sigma_psf=PSF_SIGMA,
            min_photons=MIN_PHOTONS,
            tau=TAU,
            rng=Random.MersenneTwister(seed)
        )
        
        # Run analysis with current (fixed) implementation
        chains = SMLMBaGoL.run_bagol(
            localizations,
            n_iterations=N_ITERATIONS,
            burn_in=BURN_IN,
            initial_K=max(1, length(localizations) ÷ 10),
            enable_hierarchical=true,
            hierarchical_interval=5000,
            partition_data=false,  # Single partition
            enable_threading=false
        )
        
        # Extract results
        chain = chains[1]
        mapn_result = SMLMBaGoL.estimate_mapn(chain, burn_in=BURN_IN)
        mapn_k = length(mapn_result.emitters)
        
        # Calculate posterior statistics
        k_samples = [length(state.emitters) for state in chain.samples[(BURN_IN+1):end]]
        posterior_mean = mean(k_samples)
        posterior_mode = mode(k_samples)
        
        # Get final hierarchical parameters  
        final_μ = chain.count_prior.μ
        final_κ = chain.count_prior.κ
        
        n_locs = length(localizations)
        true_μ = n_locs / TRUE_K
        
        return (
            success=true,
            mapn_k=mapn_k,
            posterior_mean=posterior_mean,
            posterior_mode=posterior_mode,
            final_μ=final_μ,
            final_κ=final_κ,
            n_locs=n_locs,
            true_μ=true_μ
        )
        
    catch e
        println("    ❌ Error in trial $trial_num: $e")
        return (success=false, mapn_k=999, posterior_mean=999.0)
    end
end

function mode(x)
    # Simple mode calculation
    counts = Dict{eltype(x), Int}()
    for val in x
        counts[val] = get(counts, val, 0) + 1
    end
    return argmax(counts)
end

println("🧪 Simple Multi-Trial K Prior Validation")
println("Testing $N_TRIALS trials with $N_ITERATIONS iterations each")
println("True K = $TRUE_K, using independent K prior (current fix)")
println()

# Run trials
results = []
seeds = [2000 + i for i in 1:N_TRIALS]

for trial in 1:N_TRIALS
    result = run_single_trial(trial, seeds[trial])
    push!(results, result)
    
    if result.success
        error_pct = abs(result.mapn_k - TRUE_K) / TRUE_K * 100
        @printf "    ✅ MAPN K=%d (%.1f%% error), E[K]=%.1f, Mode K=%d, μ=%.1f→%.1f, N=%d\n" result.mapn_k error_pct result.posterior_mean result.posterior_mode result.true_μ result.final_μ result.n_locs
    end
end

# Analyze results
valid_results = filter(r -> r.success, results)
n_valid = length(valid_results)

if n_valid == 0
    println("❌ All trials failed!")
    exit(1)
end

println("\n" * "="^70)
println("📊 AGGREGATED RESULTS ($n_valid/$N_TRIALS successful trials)")
println("="^70)

# MAPN accuracy
mapn_ks = [r.mapn_k for r in valid_results]
exact_matches = count(k -> k == TRUE_K, mapn_ks)
mapn_accuracy = exact_matches / n_valid * 100

close_matches = count(k -> abs(k - TRUE_K) <= 1, mapn_ks)  # Within ±1
close_accuracy = close_matches / n_valid * 100

mean_mapn_error = mean(abs.(mapn_ks .- TRUE_K))
mapn_distribution = [(k, count(==(k), mapn_ks)) for k in sort(unique(mapn_ks))]

# Posterior statistics
posterior_means = [r.posterior_mean for r in valid_results]
mean_posterior = mean(posterior_means)
std_posterior = std(posterior_means)

# Parameter evolution
final_μs = [r.final_μ for r in valid_results]
true_μs = [r.true_μ for r in valid_results]
mean_final_μ = mean(final_μs)
mean_true_μ = mean(true_μs)
μ_bias = mean_final_μ - mean_true_μ

println("🎯 MAPN Performance:")
@printf "  Exact accuracy (K=%d): %.1f%% (%d/%d trials)\n" TRUE_K mapn_accuracy exact_matches n_valid
@printf "  Close accuracy (±1):   %.1f%% (%d/%d trials)\n" close_accuracy close_matches n_valid
@printf "  Mean absolute error:   %.2f emitters\n" mean_mapn_error

println("\n📈 MAPN Distribution:")
for (k, count) in mapn_distribution
    pct = count / n_valid * 100
    bar = "█" ^ max(1, round(Int, pct / 5))
    marker = k == TRUE_K ? " ← TRUE" : ""
    @printf "  K=%d: %2d trials (%.1f%%) %s%s\n" k count pct bar marker
end

println("\n📊 Posterior Statistics:")
@printf "  Mean E[K]: %.2f ± %.2f\n" mean_posterior std_posterior
@printf "  True K:    %d\n" TRUE_K
if abs(mean_posterior - TRUE_K) < 0.5
    println("  ✅ Posterior mean very close to truth!")
elseif abs(mean_posterior - TRUE_K) < 1.0
    println("  ✅ Posterior mean close to truth")
else
    println("  ⚠️ Posterior mean biased away from truth")
end

println("\n🔧 Parameter Evolution:")
@printf "  Mean μ: %.1f → %.1f (true: %.1f)\n" mean_true_μ mean_final_μ mean_true_μ
@printf "  μ bias: %.1f (%.1f%% of true value)\n" μ_bias (μ_bias/mean_true_μ * 100)

println("\n🏆 CONCLUSIONS:")
if mapn_accuracy >= 80
    println("✅ Excellent: >80% exact accuracy achieved!")
elseif close_accuracy >= 80
    println("✅ Good: >80% within ±1 emitter")  
elseif mean_mapn_error < 1.5
    println("✅ Acceptable: Mean error < 1.5 emitters")
else
    println("⚠️ Poor: High error rate, needs further improvement")
end

if abs(μ_bias) < 0.1 * mean_true_μ
    println("✅ μ parameter convergence excellent (<10% bias)")
elseif abs(μ_bias) < 0.2 * mean_true_μ  
    println("✅ μ parameter convergence good (<20% bias)")
else
    println("⚠️ μ parameter still has significant bias")
end

println("\n🎉 Multi-trial analysis complete!")
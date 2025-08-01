#!/usr/bin/env julia

using Pkg; Pkg.activate("../..")
using SMLMBaGoL
using Random

Random.seed!(1234)

println("🔍 Minimal test to isolate the length() error...")

try
    # Generate data
    println("1. Generating data...")
    emitters, localizations = SMLMBaGoL.simulate_n_mer(
        n=6, 
        diameter=0.1, 
        photons=1000.0,
        localizations_per_emitter_mean=50,
        localizations_per_emitter_variance=50,
        sigma_psf=0.13,
        tau=1.0,
        min_photons=300.0
    )
    println("   ✅ Generated $(length(localizations)) localizations")
    
    # Try to run analysis
    println("2. Running minimal analysis...")
    chains = SMLMBaGoL.run_bagol(
        localizations,
        n_iterations=1000,  # Very short
        burn_in=100,
        initial_K=5,
        enable_hierarchical=false,  # Disable hierarchical for now
        partition_data=false,
        enable_threading=false
    )
    println("   ✅ RJMCMC completed")
    
    # Try to extract results
    println("3. Extracting results...")
    chain = chains[1]
    println("   ✅ Got chain with $(length(chain.samples)) samples")
    
    # This is where the error likely occurs
    println("4. Computing MAPN...")
    mapn_result = SMLMBaGoL.estimate_mapn(chain, burn_in=100)
    println("   ✅ MAPN completed")
    
    mapn_k = length(mapn_result.emitters)
    println("   ✅ MAPN K = $mapn_k")
    
catch e
    println("❌ Error occurred: $e")
    println("Stacktrace:")
    Base.show_backtrace(stdout, catch_backtrace())
end
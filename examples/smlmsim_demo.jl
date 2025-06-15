#!/usr/bin/env julia

"""
SMLMSim Integration Demo: Complete Workflow with Threading and Visualization

This comprehensive example demonstrates the SMLMSim integration in SMLMBaGoL:
1. Generate realistic SMLM data using SMLMSim's sophisticated noise models
2. Analyze with BaGoL using multiple spatial partitions for efficient processing
3. Utilize threading for parallel partition processing (requires julia --threads=N)
4. Extract MAPN estimates with uncertainty quantification
5. Assess chain quality with comprehensive diagnostics
6. Generate super-resolution images (localizations, emitters, posterior)
7. Compare SMLMSim vs native BaGoL simulation approaches

USAGE:
  julia --threads=16 --project=. smlmsim_demo.jl   # Recommended: 16 threads
  julia --threads=auto --project=. smlmsim_demo.jl # Use all available threads
  julia --project=. smlmsim_demo.jl                # Single-threaded mode

The demo is fully configurable and generates publication-quality visualizations.
"""

using Pkg; Pkg.activate("examples")
using SMLMBaGoL
using Statistics

#=============================================================================
USER PARAMETERS - Configure your simulation here
=============================================================================#

# SMLMSim simulation parameters
const DENSITY = 0.1                     # Emitters per μm² 
const PSF_WIDTH = 0.13                  # PSF width (130 nm)
const MIN_PHOTONS = 300                 # Minimum photon threshold
const N_FRAMES = 2000                   # Number of frames
const FRAMERATE = 100.0                 # Frames per second

# Analysis parameters  
const N_ITERATIONS = 50000              # RJMCMC iterations
const BURN_IN = 10000                   # Burn-in period
const ENABLE_PARTITIONING = true        # Use multiple partitions for efficiency
const PARTITION_RADIUS = 0.5            # Partition radius in μm (4x avg uncertainty)
const ENABLE_THREADING = true           # Use threading for parallel partition processing
const ENABLE_HIERARCHICAL = false       # Use hierarchical updates
const HIERARCHICAL_INTERVAL = 5000      # Hierarchical update interval

# Visualization parameters
const PIXEL_SIZE = 0.002                # μm per pixel (2 nm super-resolution)
const SAVE_IMAGES = true                # Generate PNG files

#=============================================================================
SETUP & SIMULATION
=============================================================================#

# Create output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("=== SMLMSim Integration Demo ===\n")
println("Configuration:")
println("• SMLMSim density: $DENSITY emitters/μm²")
println("• PSF width: $(PSF_WIDTH*1000) nm")
println("• Minimum photons: $MIN_PHOTONS")
println("• Simulation time: $(N_FRAMES/FRAMERATE) seconds")
println("• RJMCMC iterations: $N_ITERATIONS (burn-in: $BURN_IN)")
println("• Spatial partitioning: $(ENABLE_PARTITIONING ? "enabled" : "disabled")")
if ENABLE_PARTITIONING
    println("• Partition radius: $(PARTITION_RADIUS*1000) nm")
end
println("• Threading: $(ENABLE_THREADING ? "enabled" : "disabled") ($(Threads.nthreads()) threads available)")
println("• Hierarchical updates: $(ENABLE_HIERARCHICAL ? "enabled (interval: $HIERARCHICAL_INTERVAL)" : "disabled")")
println("• Image pixel size: $(PIXEL_SIZE*1000) nm")
println()

#=============================================================================
1. SMLMSim Simulation
=============================================================================#

println("1. Creating SMLMSim simulation with realistic noise...")

# Create SMLMSim static simulation with proper blinking rates
smld = simulate_static_smlm(
    density=DENSITY,
    σ_psf=PSF_WIDTH,
    minphotons=MIN_PHOTONS,
    nframes=N_FRAMES,
    framerate=FRAMERATE
)

println("   ✓ Created SMLD with $(length(smld.emitters)) noisy localizations")
println("   ✓ PSF width: $(smld.metadata["σ_psf"]*1000) nm")
println("   ✓ Simulation type: $(smld.metadata["simulation_type"])")

# Calculate some statistics
if length(smld.emitters) > 0
    photon_counts = [e.photons for e in smld.emitters]
    uncertainties_x = [e.σ_x for e in smld.emitters]
    
    println("   ✓ Photon statistics: mean=$(round(mean(photon_counts), digits=1)), std=$(round(std(photon_counts), digits=1))")
    println("   ✓ Localization precision: mean=$(round(mean(uncertainties_x)*1000, digits=1)) nm, std=$(round(std(uncertainties_x)*1000, digits=1)) nm")
end

#=============================================================================
2. BaGoL Analysis with Partitioning
=============================================================================#

println()
println("2. Running BaGoL analysis with spatial partitioning...")

# Run BaGoL directly on SMLD data - automatic conversion happens internally
chains = run_bagol(smld; 
    n_iterations=N_ITERATIONS,
    burn_in=BURN_IN,
    partition_data=ENABLE_PARTITIONING,
    partition_radius=PARTITION_RADIUS,
    enable_threading=ENABLE_THREADING,
    enable_hierarchical=ENABLE_HIERARCHICAL,
    hierarchical_interval=HIERARCHICAL_INTERVAL
)

println("   ✓ Analysis completed with $(length(chains)) spatial partition(s)")
total_samples = sum(length(chain.samples) for chain in chains)
println("   ✓ Total samples collected: $total_samples")

# Partition statistics
if ENABLE_PARTITIONING && length(chains) > 1
    partition_sizes = [length(chain.current_state.localizations) for chain in chains]
    println("   ✓ Partition sizes: min=$(minimum(partition_sizes)), max=$(maximum(partition_sizes)), mean=$(round(mean(partition_sizes), digits=1))")
end

#=============================================================================
3. Results Analysis & Diagnostics
=============================================================================#

println()
println("3. Analyzing results and chain quality...")

# Extract MAPN estimates
mapn_results = estimate_mapn(chains)

println("   ✓ Original localizations: $(length(smld.emitters))")
println("   ✓ Estimated emitters: $(length(mapn_results.emitters))")
recovery_rate = length(mapn_results.emitters) / length(smld.emitters) * 100
println("   ✓ Recovery rate: $(round(recovery_rate, digits=1))%")

# Comprehensive chain diagnostics
println("   ✓ Running chain diagnostics...")
diagnostic_results = diagnose_chains(chains)
println("     - Burn-in assessment: $(diagnostic_results.burn_in_adequate ? "adequate" : "insufficient")")
println("     - Chain length: $(diagnostic_results.chain_length_adequate ? "adequate" : "insufficient")")
println("     - Overall quality: $(diagnostic_results.overall_quality)")

#=============================================================================
4. Visualization & Image Generation
=============================================================================#

println()
println("4. Generating super-resolution images...")

if SAVE_IMAGES
    # Convert SMLD to localizations for visualization
    localizations = smld_to_localizations(smld)
    
    # Generate localization image
    println("   ✓ Creating localizations image...")
    loc_image = gen_sr_image(localizations; pixel_size=PIXEL_SIZE, 
                            image_type=:localizations, save_path=joinpath(output_dir, "smlmsim_localizations_sr.png"))
    
    # Generate MAPN emitter image  
    println("   ✓ Creating MAPN emitters image...")
    mapn_image = gen_sr_image(localizations; pixel_size=PIXEL_SIZE, 
                             emitters=mapn_results.emitters, image_type=:emitters,
                             save_path=joinpath(output_dir, "smlmsim_mapn_emitters_sr.png"))
    
    # Generate posterior uncertainty image
    println("   ✓ Creating posterior uncertainty image...")
    post_image = gen_sr_image(localizations; pixel_size=PIXEL_SIZE, 
                             emitters=mapn_results.emitters, uncertainties=mapn_results.uncertainties,
                             image_type=:posterior, save_path=joinpath(output_dir, "smlmsim_posterior_uncertainty.png"))
    
    println("   ✓ All images saved to: $output_dir")
else
    println("   ⚠ Image generation disabled (SAVE_IMAGES = false)")
end

#=============================================================================
5. Comparison with Native BaGoL Simulation
=============================================================================#

println()
println("5. Comparing with native BaGoL simulation...")

# Estimate equivalent native parameters
estimated_locs_per_emitter = length(smld.emitters) / length(mapn_results.emitters)
estimated_photons = length(smld.emitters) > 0 ? mean([e.photons for e in smld.emitters]) : 1000

println("   ✓ Running equivalent native simulation...")
native_locs, native_prior = simulate_n_mer(
    n=6, 
    diameter=0.050,
    photons=estimated_photons,
    localizations_per_emitter_mean=estimated_locs_per_emitter,
    localizations_per_emitter_variance=estimated_locs_per_emitter
)

println("   ✓ Native simulation: $(length(native_locs)) localizations")
println("   ✓ SMLMSim simulation: $(length(smld.emitters)) localizations")
println("   ✓ Complexity ratio: $(round(length(smld.emitters) / length(native_locs), digits=1))x more localizations")

#=============================================================================
6. Advanced Usage Examples
=============================================================================#

println()
println("6. Advanced usage examples...")

# Example 1: High-density simulation
println("   Example A: High-density simulation")
dense_smld = simulate_static_smlm(density=0.3, minphotons=MIN_PHOTONS, nframes=1000)
println("     ✓ High-density: $(length(dense_smld.emitters)) localizations")

# Example 2: Low-noise, high-photon simulation
println("   Example B: High-photon simulation")
bright_smld = simulate_static_smlm(density=0.05, minphotons=1000, nframes=1000)
if length(bright_smld.emitters) > 0
    bright_precision = mean([e.σ_x for e in bright_smld.emitters]) * 1000
    println("     ✓ High-photon precision: $(round(bright_precision, digits=1)) nm")
end

# Example 3: Direct SMLMSim usage with custom fluorophore
println("   Example C: Custom fluorophore parameters")
custom_fluor = SMLMSim.GenericFluor(photons=5e4, k_off=2.0, k_on=1.0)
params = SMLMSim.StaticSMLMParams(density=0.05, σ_psf=0.10, minphotons=200, 
                                  ndatasets=1, nframes=1000, framerate=100.0, 
                                  ndims=2, zrange=[-0.5, 0.5])
smld_true, smld_model, smld_noisy = SMLMSim.simulate(params; molecule=custom_fluor)
println("     ✓ Custom simulation: $(length(smld_noisy.emitters)) localizations")

# Quick analysis of custom data
if length(smld_noisy.emitters) > 100
    custom_chains = run_bagol(smld_noisy; n_iterations=5000, burn_in=1000, partition_data=false)
    println("     ✓ Custom analysis: $(length(custom_chains)) chains completed")
end

#=============================================================================
Summary & Key Insights
=============================================================================#

println()
println("=== Summary & Key Insights ===")
println()
println("SMLMSim Integration Benefits:")
println("• Realistic photophysical noise models")
println("• Sophisticated blinking kinetics (k_off/k_on rates)")
println("• Photon-dependent localization uncertainties")
println("• Seamless BaGoL compatibility via run_bagol(smld)")
println("• No workflow changes needed")
println()

println("Spatial Partitioning Results:")
if ENABLE_PARTITIONING
    println("• $(length(chains)) partitions processed in parallel")
    if length(chains) > 1
        all_partition_sizes = [length(chain.current_state.localizations) for chain in chains]
        println("• Partition sizes: $(minimum(all_partition_sizes))-$(maximum(all_partition_sizes)) localizations")
        println("• Computational efficiency: O(n²) → O($(maximum(all_partition_sizes))²) per partition")
    end
    println("• Improved computational efficiency for large datasets")
    println("• Maintained analysis quality across partitions")
else
    println("• Single partition used (suitable for smaller datasets)")
end
println()

println("Performance Metrics:")
println("• Data complexity: $(length(smld.emitters)) localizations")
println("• Emitter recovery: $(round(recovery_rate, digits=1))%")
println("• Chain quality: $(diagnostic_results.overall_quality)")
println("• Analysis efficiency: $(ENABLE_PARTITIONING ? "enhanced with partitioning" : "standard single-partition")")
println()

if SAVE_IMAGES
    println("Output Files Generated:")
    println("• smlmsim_localizations_sr.png - Raw localization data")
    println("• smlmsim_mapn_emitters_sr.png - Estimated emitter positions")
    println("• smlmsim_posterior_uncertainty.png - Position uncertainties")
    println("• All files saved to: $output_dir")
end

println()
println("=== Demo Complete ===")
println("SMLMSim integration provides realistic SMLM simulation")
println("with seamless BaGoL analysis and publication-quality visualization.")
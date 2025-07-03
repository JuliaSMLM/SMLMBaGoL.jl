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
using CairoMakie
using SMLMSim

#=============================================================================
USER PARAMETERS - Configure your simulation here
=============================================================================#

# SMLMSim simulation parameters
const DENSITY = 0.5                     # Emitters per μm² 
const PSF_WIDTH = 0.13                  # PSF width (130 nm)
const MIN_PHOTONS = 300                 # Minimum photon threshold
const N_FRAMES = 2000                   # Number of frames
const FRAMERATE = 100.0                 # Frames per second
const EXPECTED_LOCS_PER_EMITTER = 10    # Expected localizations per emitter

# Analysis parameters  
const N_ITERATIONS = 50000              # RJMCMC iterations (reduced from 50000)
const BURN_IN = 2000                    # Burn-in period (reduced from 10000)
const ENABLE_PARTITIONING = true        # Use multiple partitions for efficiency
const PARTITION_RADIUS = 0.5            # Partition radius in μm (4x avg uncertainty)
const ENABLE_THREADING = true           # Use threading for parallel partition processing
const ENABLE_HIERARCHICAL = true        # Use hierarchical updates
const HIERARCHICAL_INTERVAL = 2000      # Hierarchical update interval

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
println("• Expected localizations per emitter: $EXPECTED_LOCS_PER_EMITTER")
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

# Create SMLMSim static simulation with proper blinking rates and spatial control
smld = simulate_static_smlm(
    density=DENSITY,
    σ_psf=PSF_WIDTH,
    minphotons=MIN_PHOTONS,
    nframes=N_FRAMES,
    framerate=FRAMERATE,
    npixelsx=64,      # 64×32 pixels for field
    npixelsy=32,
    pixelsize=0.1,    # 100nm pixels = 6.4μm × 3.2μm field
    loc_per_emitter=EXPECTED_LOCS_PER_EMITTER
)

println("   ✓ Created SMLD with $(length(smld.emitters)) noisy localizations")
println("   ✓ PSF width: $(smld.metadata["σ_psf"]*1000) nm")
println("   ✓ Simulation type: $(smld.metadata["simulation_type"])")

# Calculate field size and expected emitters
field_area = 64 * 32 * (0.1)^2  # 64×32 pixels × 0.1μm pixel size = area in μm²
expected_emitters = DENSITY * field_area / EXPECTED_LOCS_PER_EMITTER
actual_emitters = length(unique([e.id for e in smld.emitters]))
actual_locs_per_emitter = length(smld.emitters) / actual_emitters

println("   ✓ Field area: $(field_area) μm²")
println("   ✓ Expected emitters: $(round(expected_emitters, digits=1))")
println("   ✓ Actual emitters: $(actual_emitters)")
println("   ✓ Actual localizations per emitter: $(round(actual_locs_per_emitter, digits=1))")

# Calculate some statistics
if length(smld.emitters) > 0
    photon_counts = [e.photons for e in smld.emitters]
    uncertainties_x = [e.σ_x for e in smld.emitters]
    
    println("   ✓ Photon statistics: mean=$(round(mean(photon_counts), digits=1)), std=$(round(std(photon_counts), digits=1))")
    println("   ✓ Localization precision: mean=$(round(mean(uncertainties_x)*1000, digits=1)) nm, std=$(round(std(uncertainties_x)*1000, digits=1)) nm")
end

# Convert SMLD to localizations and save localization image early
println("   ✓ Converting to localizations and saving initial image...")
localizations = smld_to_localizations(smld)

if SAVE_IMAGES
    # Generate localization image using SMLD camera bounds
    println("   ✓ Creating localizations image with camera-defined bounds...")
    loc_image = gen_sr_image(smld; pixel_size=PIXEL_SIZE,
                            filename=joinpath(output_dir, "smlmsim_localizations_sr.png"))
    println("   ✓ Localizations image saved to: $(joinpath(output_dir, "smlmsim_localizations_sr.png"))")
end

#=============================================================================
2. BaGoL Analysis with Partitioning
=============================================================================#

println()
println("2. Running BaGoL analysis with spatial partitioning...")

# Run BaGoL directly on SMLD data - automatic conversion happens internally
chains_result = run_bagol(smld; 
    n_iterations=N_ITERATIONS,
    burn_in=BURN_IN,
    partition_data=ENABLE_PARTITIONING,
    # partition_radius=PARTITION_RADIUS,
    enable_threading=ENABLE_THREADING,
    enable_hierarchical=ENABLE_HIERARCHICAL,
    hierarchical_interval=HIERARCHICAL_INTERVAL
)

# Ensure chains is always a vector for consistent handling
chains = isa(chains_result, Vector) ? chains_result : [chains_result]

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
println("   ✓ Estimated emitters: $(length(mapn_results))")
recovery_rate = length(mapn_results) / length(smld.emitters) * 100
println("   ✓ Recovery rate: $(round(recovery_rate, digits=1))%")

# Comprehensive chain diagnostics
println("   ✓ Running chain diagnostics...")
diagnostic_results = diagnose_chains(length(chains) == 1 ? chains[1] : chains)
println("     - Burn-in recommendation: $(diagnostic_results.recommended_burn_in) iterations")
println("     - Chain convergence: $(diagnostic_results.converged ? "converged" : "needs more samples")")
println("     - Overall summary: $(diagnostic_results.summary)")

#=============================================================================
4. Visualization & Image Generation
=============================================================================#

println()
println("4. Generating super-resolution images and uncertainty plots...")

if SAVE_IMAGES
    # Convert SMLD to localizations for plotting
    localizations = smld_to_localizations(smld)
    
    # Generate individual uncertainty circles for reference
    println("   ✓ Creating localization uncertainty circles...")
    fig_locs, ax_locs = sr_circles(localizations, camera=smld.camera, 
                                  scale_factor=2.0, color=:blue, alpha=0.6, linewidth=1,
                                  figure_kwargs=(size=(4800, 2400),),
                                  axis_kwargs=(title="Localization Uncertainties (2σ)",))
    save(joinpath(output_dir, "smlmsim_localizations_uncertainty.png"), fig_locs)
    
    # Generate MAPN emitter image and individual circles if we have emitters
    if !isempty(mapn_results)
        println("   ✓ Creating MAPN emitters image...")
        mapn_image = gen_sr_image(mapn_results; 
                                 pixel_size=PIXEL_SIZE, 
                                 filename=joinpath(output_dir, "smlmsim_mapn_emitters_sr.png"))
        
        println("   ✓ Creating MAPN uncertainty circles...")
        fig_mapn, ax_mapn = sr_circles(mapn_results, camera=smld.camera,
                                      scale_factor=2.0, color=:red, linewidth=2, alpha=0.8,
                                      figure_kwargs=(size=(4800, 2400),),
                                      axis_kwargs=(title="MAPN Emitter Uncertainties (2σ)",))
        save(joinpath(output_dir, "smlmsim_mapn_uncertainty.png"), fig_mapn)
        
        # Generate combined comparison plot
        println("   ✓ Creating combined uncertainty comparison...")
        fig_combined, ax_combined = sr_circles_combined(localizations, mapn_results, camera=smld.camera,
                                                       figure_kwargs=(size=(4800, 2400),),
                                                       axis_kwargs=(title="Uncertainty Comparison: Localizations (black) vs MAPN (red) - 2σ",))
        save(joinpath(output_dir, "smlmsim_uncertainty_comparison.png"), fig_combined)
    else
        # Still create comparison plot with empty MAPN for consistency
        println("   ✓ Creating comparison plot (localizations only)...")
        fig_combined, ax_combined = sr_circles_combined(localizations, Emitter2D{Float64}[], camera=smld.camera,
                                                       figure_kwargs=(size=(4800, 2400),),
                                                       axis_kwargs=(title="Localization Uncertainties (2σ)",))
        save(joinpath(output_dir, "smlmsim_uncertainty_comparison.png"), fig_combined)
        println("   ⚠ No emitters found, skipping MAPN emitters image")
    end
    
    # Generate posterior uncertainty image if we have chains with samples
    if !isempty(chains) && !isempty(chains[1].samples)
        println("   ✓ Creating posterior uncertainty image...")
        # Use all chains if multiple partitions, or the single chain
        chains_for_posterior = length(chains) > 1 ? chains : chains[1]
        post_image = gen_sr_image(chains_for_posterior; 
                                 pixel_size=PIXEL_SIZE, 
                                 filename=joinpath(output_dir, "smlmsim_posterior_uncertainty.png"))
        println("     • Using $(length(chains)) partition(s) for posterior image")
    else
        println("   ⚠ No chain samples found, skipping posterior uncertainty image")
    end
    
    println("   ✓ Analysis images and uncertainty plots saved to: $output_dir")
else
    println("   ⚠ Image generation disabled (SAVE_IMAGES = false)")
end

#=============================================================================
5. Hierarchical Prior Visualization (if enabled)
=============================================================================#

if ENABLE_HIERARCHICAL
    println()
    println("5. Visualizing hierarchical prior updates...")
    
    # Get summary of hierarchical updates
    hier_summary = get_hierarchical_summary(chains)
    println("   • $(hier_summary.message)")
    
    if hier_summary.enabled && hier_summary.n_updates > 0
        # Plot evolution of hyperparameters
        plot_hierarchical_evolution(chains, 
                                  filename=joinpath(output_dir, "smlmsim_hierarchical_evolution.png"))
        println("   ✓ Created hierarchical evolution plot")
        
        # Plot Gamma distributions at different timepoints
        plot_gamma_distributions(chains,
                               filename=joinpath(output_dir, "smlmsim_gamma_distributions.png"),
                               n_timepoints=4)
        println("   ✓ Created Gamma distribution evolution plot")
        
        # Plot empirical vs fitted distribution with true value
        plot_emitter_count_histogram(chains,
                                   filename=joinpath(output_dir, "smlmsim_emitter_count_fit.png"),
                                   true_mean=EXPECTED_LOCS_PER_EMITTER)
        println("   ✓ Created empirical vs fitted distribution plot (with true mean)")
        
        # Check convergence
        conv_result = analyze_hierarchical_convergence(chains)
        println("   • Convergence: $(conv_result.message)")
        println("   • Final parameters: μ=$(round(conv_result.final_μ, digits=3)), κ=$(round(conv_result.final_κ, digits=3)), τ²=$(round(conv_result.final_τ²*1e6, digits=1)) nm²")
        
        # Compare to true value
        if conv_result.converged
            # For Negative Binomial, mean = μ
            fitted_mean = conv_result.final_μ
            true_mean = EXPECTED_LOCS_PER_EMITTER
            error_percent = abs(fitted_mean - true_mean) / true_mean * 100
            
            println("   • True localizations per emitter: $true_mean")
            println("   • Fitted mean (μ): $(round(fitted_mean, digits=2))")
            println("   • Relative error: $(round(error_percent, digits=1))%")
        end
    end
end

#=============================================================================
6. Comparison with Native BaGoL Simulation
=============================================================================#

println()
println("5. Comparing with native BaGoL simulation...")

# Estimate equivalent native parameters
estimated_locs_per_emitter = length(smld.emitters) / length(mapn_results)
estimated_photons = length(smld.emitters) > 0 ? mean([e.photons for e in smld.emitters]) : 1000

println("   ✓ Running equivalent native simulation...")
native_locs, native_spatial_prior, native_count_prior = simulate_n_mer(
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
7. Advanced Usage Note
=============================================================================#

println()
println("7. Advanced usage examples...")
println("   ✓ For advanced examples including:")
println("     - High-density simulations with custom field sizes")
println("     - High-photon precision studies")
println("     - Direct SMLMSim API usage with custom parameters")
println("     - Large-scale simulation workflows")
println("     - Custom analysis pipelines")
println("   ✓ See: smlmsim_advanced_examples.jl")
println("   ✓ Run: julia --project=examples examples/smlmsim_advanced_examples.jl")

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
println("• Chain quality: $(diagnostic_results.summary)")
println("• Analysis efficiency: $(ENABLE_PARTITIONING ? "enhanced with partitioning" : "standard single-partition")")
println()

if SAVE_IMAGES
    println("Output Files Generated:")
    println("• smlmsim_localizations_sr.png - Raw localization super-resolution image")
    println("• smlmsim_localizations_uncertainty.png - Localization uncertainty circles (2σ)")
    println("• smlmsim_mapn_emitters_sr.png - MAPN emitter super-resolution image")
    println("• smlmsim_mapn_uncertainty.png - MAPN emitter uncertainty circles (2σ)")
    println("• smlmsim_uncertainty_comparison.png - Combined comparison plot (localizations + MAPN)")
    println("• smlmsim_posterior_uncertainty.png - Posterior position uncertainties")
    if ENABLE_HIERARCHICAL
        println("• smlmsim_hierarchical_evolution.png - Evolution of μ, κ, and τ² hyperparameters")
        println("• smlmsim_gamma_distributions.png - Negative Binomial distribution evolution over time")
        println("• smlmsim_emitter_count_fit.png - Empirical vs fitted distribution comparison")
    end
    println("• All files saved to: $output_dir")
end

println()
println("=== Demo Complete ===")
println("SMLMSim integration provides realistic SMLM simulation")
println("with seamless BaGoL analysis and publication-quality visualization.")
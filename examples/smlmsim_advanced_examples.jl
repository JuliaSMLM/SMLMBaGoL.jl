#!/usr/bin/env julia

"""
SMLMSim Advanced Examples: Specialized Use Cases and Custom Configurations

This script demonstrates advanced SMLMSim integration patterns:
1. High-density simulations with custom field sizes
2. High-photon, low-noise simulations for precision studies
3. Direct SMLMSim API usage with custom fluorophore parameters
4. Large-scale simulation workflows
5. Custom analysis pipelines

USAGE:
  julia --threads=auto --project=. smlmsim_advanced_examples.jl

These examples showcase the flexibility of the SMLMSim integration
and demonstrate how to configure simulations for specific research needs.
"""

using Pkg; Pkg.activate("examples")
using SMLMBaGoL
using Statistics
using CairoMakie
import SMLMSim

println("=== SMLMSim Advanced Examples ===\\n")

#=============================================================================
Example 1: High-density simulation with custom field size
=============================================================================#

println("Example 1: High-density simulation")
println("   Creating high-density simulation (0.3 emitters/μm²)...")

# High-density simulation with square field
dense_smld = simulate_static_smlm(
    density=0.3,                    # 3x higher density
    minphotons=300,
    nframes=1000, 
    npixelsx=128, 
    npixelsy=128,                   # Square field
    pixelsize=0.05                  # Smaller pixels = 6.4μm × 6.4μm field
)

println("   ✓ High-density field: 6.4μm × 6.4μm")
println("   ✓ Generated: $(length(dense_smld.emitters)) localizations")

if length(dense_smld.emitters) > 0
    # Calculate density statistics
    field_area = 6.4 * 6.4  # μm²
    actual_density = length(dense_smld.emitters) / field_area
    println("   ✓ Actual localization density: $(round(actual_density, digits=1)) loc/μm²")
    
    # Photon statistics
    photons = [e.photons for e in dense_smld.emitters]
    println("   ✓ Photon range: $(minimum(photons)) - $(maximum(photons))")
    println("   ✓ Mean photons: $(round(mean(photons), digits=1))")
end

# Optional: Generate visualization for high-density case
if length(dense_smld.emitters) < 2000  # Only if manageable size
    println("   ✓ Generating high-density visualization...")
    gen_sr_image(dense_smld; 
                pixel_size=0.001,  # 1nm super-resolution
                filename=joinpath(@__DIR__, "output", "high_density_sr.png"))
end

println()

#=============================================================================
Example 2: High-photon, low-noise simulation for precision studies
=============================================================================#

println("Example 2: High-photon precision simulation")
println("   Creating high-photon simulation for precision studies...")

# High-photon simulation with large field
bright_smld = simulate_static_smlm(
    density=0.05,                   # Lower density for precision focus
    minphotons=1000,               # High photon threshold
    nframes=1000,
    npixelsx=512, 
    npixelsy=512,                  # Large field
    pixelsize=0.1                  # 51.2μm × 51.2μm field
)

println("   ✓ Large field: 51.2μm × 51.2μm")
println("   ✓ Generated: $(length(bright_smld.emitters)) localizations")

if length(bright_smld.emitters) > 0
    # Precision analysis
    uncertainties_x = [e.σ_x for e in bright_smld.emitters]
    uncertainties_y = [e.σ_y for e in bright_smld.emitters]
    
    mean_precision_x = mean(uncertainties_x) * 1000  # Convert to nm
    mean_precision_y = mean(uncertainties_y) * 1000
    
    println("   ✓ Mean X precision: $(round(mean_precision_x, digits=1)) nm")
    println("   ✓ Mean Y precision: $(round(mean_precision_y, digits=1)) nm")
    println("   ✓ Precision std: $(round(std(uncertainties_x)*1000, digits=1)) nm")
    
    # Photon statistics
    photons = [e.photons for e in bright_smld.emitters]
    println("   ✓ Mean photons: $(round(mean(photons), digits=1))")
    println("   ✓ High-photon fraction (>2000): $(round(sum(photons .> 2000)/length(photons)*100, digits=1))%")
end

println()

#=============================================================================
Example 3: Direct SMLMSim API usage with custom parameters
=============================================================================#

println("Example 3: Custom fluorophore and camera parameters")
println("   Using direct SMLMSim API for maximum control...")

# Custom fluorophore with specific photophysical properties
custom_fluor = SMLMSim.GenericFluor(
    photons=5e4,      # Very bright fluorophore
    k_off=2.0,        # Custom off-rate
    k_on=1.0          # Custom on-rate
)

# Custom simulation parameters
params = SMLMSim.StaticSMLMParams(
    density=0.05,                   # Moderate density
    σ_psf=0.10,                    # 100nm PSF
    minphotons=200,                # Lower threshold for bright fluorophore
    ndatasets=1,
    nframes=1000,
    framerate=100.0,
    ndims=2,
    zrange=[-0.5, 0.5]
)

# Custom camera with high resolution
custom_camera = SMLMSim.IdealCamera(200, 200, 0.08)  # 16μm × 16μm field

# Direct SMLMSim simulation call
println("   ✓ Running direct SMLMSim.simulate()...")
smld_true, smld_model, smld_noisy = SMLMSim.simulate(params; 
                                                     molecule=custom_fluor, 
                                                     camera=custom_camera)

println("   ✓ Custom field: 16μm × 16μm")
println("   ✓ True positions: $(length(smld_true.emitters))")
println("   ✓ Noisy localizations: $(length(smld_noisy.emitters))")

if length(smld_noisy.emitters) > 0
    # Compare true vs noisy
    println("   ✓ Noise factor: $(round(length(smld_noisy.emitters)/length(smld_true.emitters), digits=1))x localizations")
    
    # Precision from custom fluorophore
    uncertainties = [e.σ_x for e in smld_noisy.emitters]
    println("   ✓ Custom fluorophore precision: $(round(mean(uncertainties)*1000, digits=1)) nm")
end

# Quick analysis of custom data (if reasonable size)
if length(smld_noisy.emitters) > 100 && length(smld_noisy.emitters) < 1000
    println("   ✓ Running quick BaGoL analysis on custom data...")
    custom_chains = run_bagol(smld_noisy; 
                             n_iterations=5000, 
                             burn_in=1000, 
                             partition_data=false)
    
    if isa(custom_chains, Vector)
        println("   ✓ Custom analysis: $(length(custom_chains)) chains completed")
    else
        println("   ✓ Custom analysis: single chain completed")
    end
end

println()

#=============================================================================
Example 4: Large-scale simulation workflow
=============================================================================#

println("Example 4: Large-scale simulation considerations")
println("   Demonstrating scalability patterns...")

# Simulate progressively larger datasets
field_sizes = [64, 128, 256]
densities = [0.1, 0.2, 0.3]

println("   Testing scalability across field sizes and densities:")
println("   Field Size (px) | Density (em/μm²) | Localizations | Field Area (μm²)")
println("   ----------------|------------------|---------------|------------------")

for (i, field_size) in enumerate(field_sizes)
    density = densities[min(i, length(densities))]
    
    # Create simulation
    test_smld = simulate_static_smlm(
        density=density,
        npixelsx=field_size,
        npixelsy=field_size,
        pixelsize=0.1,          # 0.1μm pixels
        nframes=500,            # Shorter for speed
        minphotons=300
    )
    
    field_area = (field_size * 0.1)^2
    n_locs = length(test_smld.emitters)
    
    println("   $(lpad(field_size, 15)) | $(lpad(density, 16)) | $(lpad(n_locs, 13)) | $(lpad(round(field_area, digits=1), 18))")
    
    # Recommend analysis strategy based on size
    if n_locs < 500
        println("                   → Recommended: Single partition, standard analysis")
    elseif n_locs < 2000
        println("                   → Recommended: Spatial partitioning with 2-4 partitions")
    else
        println("                   → Recommended: Aggressive partitioning, reduced iterations")
    end
end

println()

#=============================================================================
Example 5: Custom analysis pipeline
=============================================================================#

println("Example 5: Custom analysis pipeline")
println("   Demonstrating specialized analysis workflows...")

# Create a moderate-size dataset for analysis
analysis_smld = simulate_static_smlm(
    density=0.15,
    npixelsx=96,
    npixelsy=96, 
    pixelsize=0.08,     # 7.68μm × 7.68μm field
    nframes=1500,
    minphotons=400
)

println("   ✓ Analysis dataset: $(length(analysis_smld.emitters)) localizations")

if length(analysis_smld.emitters) > 50 && length(analysis_smld.emitters) < 1500
    println("   ✓ Running custom analysis pipeline...")
    
    # Step 1: Quick diagnostic run
    quick_chains = run_bagol(analysis_smld; 
                            n_iterations=5000, 
                            burn_in=1000,
                            partition_data=true)
    
    # Step 2: Assess quality
    diagnostics = diagnose_chains(quick_chains)
    println("   ✓ Quick analysis convergence: $(diagnostics.converged ? "converged" : "needs more samples")")
    
    # Step 3: Full analysis if needed
    if diagnostics.converged
        println("   ✓ Quick analysis sufficient - extracting results...")
        quick_mapn = estimate_mapn(quick_chains)
        println("   ✓ MAPN emitters: $(length(quick_mapn))")
        
        # Generate comparison images
        mkpath(joinpath(@__DIR__, "output"))
        gen_sr_image(analysis_smld; 
                    filename=joinpath(@__DIR__, "output", "custom_localizations.png"))
        gen_sr_image(quick_mapn; 
                    filename=joinpath(@__DIR__, "output", "custom_mapn.png"))
        println("   ✓ Saved custom analysis images")
    else
        println("   ✓ Quick analysis suggests longer chains needed")
        println("   ✓ Recommended: $(diagnostics.recommended_burn_in) burn-in, 2-3x more iterations")
    end
end

#=============================================================================
Summary
=============================================================================#

println()
println("=== Advanced Examples Summary ===")
println()
println("Key Integration Patterns Demonstrated:")
println("• High-density simulations: Increased emitter density with appropriate field sizing")
println("• Precision studies: High-photon simulations for ultra-precise localization")
println("• Direct API access: Custom fluorophore and camera parameter control")
println("• Scalability testing: Performance considerations across dataset sizes")
println("• Custom pipelines: Adaptive analysis workflows based on data characteristics")
println()
println("Performance Considerations:")
println("• <500 localizations: Single partition, standard iterations")
println("• 500-2000 localizations: Spatial partitioning recommended")
println("• >2000 localizations: Aggressive partitioning, consider reduced iterations")
println("• Very large datasets: Consider preprocessing or region-of-interest analysis")
println()
println("=== Advanced Examples Complete ===")
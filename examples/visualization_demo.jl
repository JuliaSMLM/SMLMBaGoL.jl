#!/usr/bin/env julia

"""
SMLMBaGoL Visualization Demo

This example demonstrates the new super-resolution image generation capabilities:
1. Generate SR image from localizations (using precision)
2. Generate SR image from MAPN emitters (using uncertainty estimates)  
3. Generate posterior uncertainty image from RJMCMC chain

The gen_sr_image() function uses multiple dispatch to handle different data types automatically.
"""

using Pkg; Pkg.activate("examples")
using SMLMBaGoL

# Create output directory if it doesn't exist
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("=== SMLMBaGoL Visualization Demo ===\n")

# Step 1: Generate synthetic data
println("1. Generating synthetic 6-mer data...")
localizations, prior = simulate_n_mer(
    n = 6,                                        # 6-mer complex
    diameter = 0.1,                               # 0.3 μm diameter  
    localizations_per_emitter_mean = 8.0,        # ~8 localizations per emitter
    localizations_per_emitter_variance = 8.0,    # Poisson-like variation
    photons = 800,                                # 800 photons average
    prior_K_mean = 6.0,                          # Expect ~6 emitters  
    prior_K_variance = 2.0                       # Low uncertainty
)

println("   Generated $(length(localizations)) total localizations")

# Step 2: Run BaGoL analysis
println("\n2. Running BaGoL analysis...")
result = run_bagol(
    localizations,
    prior = prior,
    n_iterations = 5000,
    burn_in = 1000,
    partition_data = false  # Single partition for demo
)

# Step 3: Get MAPN estimates
println("\n3. Computing MAPN estimates...")
mapn_emitters = estimate_mapn([result])
println("   MAPN estimated $(length(mapn_emitters)) emitters")

# Step 4: Generate super-resolution images
println("\n4. Generating super-resolution images...")

# SR image from localizations (uses localization precision)
println("   - Localizations SR image...")
loc_image = gen_sr_image(localizations, 
                        pixel_size = 0.001,              # 5 nm pixels
                        filename = joinpath(output_dir, "localizations_sr.png"))
println("     Saved: output/localizations_sr.png ($(size(loc_image)) pixels)")

# SR image from MAPN emitters (uses uncertainty estimates)
if !isempty(mapn_emitters)
    println("   - MAPN emitters SR image...")
    emitter_image = gen_sr_image(mapn_emitters,
                                pixel_size = 0.001,
                                filename = joinpath(output_dir, "mapn_emitters_sr.png"))
    println("     Saved: output/mapn_emitters_sr.png ($(size(emitter_image)) pixels)")
    
    # Show emitter uncertainties
    println("   - MAPN emitter uncertainties:")
    for (i, emitter) in enumerate(mapn_emitters[1:min(3, end)])
        println("     Emitter $i: σ = ($(round(emitter.σx*1000, digits=1)), $(round(emitter.σy*1000, digits=1))) nm")
    end
end

# Posterior uncertainty image (pixel counting histogram)
println("   - Posterior uncertainty image...")
posterior_image = gen_sr_image(result,
                              pixel_size = 0.001,
                              mode = :posterior,
                              filename = joinpath(output_dir, "posterior_uncertainty.png"))
println("     Saved: output/posterior_uncertainty.png ($(size(posterior_image)) pixels)")
println("     Max counts per pixel: $(round(maximum(posterior_image), digits=1))")

# Step 5: Image statistics
println("\n5. Image Statistics:")
println("   Localizations image: range [$(round(minimum(loc_image), digits=3)), $(round(maximum(loc_image), digits=3))]")
if !isempty(mapn_emitters)
    println("   MAPN emitters image: range [$(round(minimum(emitter_image), digits=3)), $(round(maximum(emitter_image), digits=3))]")
end
println("   Posterior image: range [$(round(minimum(posterior_image), digits=1)), $(round(maximum(posterior_image), digits=1))] counts")

# Step 6: Comparison with ground truth
println("\n6. Ground Truth Comparison:")
println("   True 6-mer positions (radius = 0.15 μm):")
true_radius = 0.15
for i in 1:6
    angle = 2π * (i-1) / 6
    true_x = true_radius * cos(angle)
    true_y = true_radius * sin(angle)
    println("     True $i: ($(round(true_x, digits=3)), $(round(true_y, digits=3))) μm")
end

println("\n=== Visualization Demo Complete ===")
println("\nGenerated files in examples/output/:")
println("  - localizations_sr.png: Super-resolution image from raw localizations")
if !isempty(mapn_emitters)
    println("  - mapn_emitters_sr.png: Super-resolution image from MAPN estimates")
end
println("  - posterior_uncertainty.png: Posterior uncertainty heatmap from RJMCMC chain")
println("\nUse any image viewer to examine the results!")
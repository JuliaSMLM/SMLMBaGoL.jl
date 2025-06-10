#!/usr/bin/env julia

"""
SMLMBaGoL Complete Demo: Simulation, Analysis & Visualization

This comprehensive example demonstrates the complete SMLMBaGoL workflow:
1. Generate synthetic n-mer complex data with realistic photon noise
2. Analyze with BaGoL algorithm using RJMCMC sampling
3. Extract MAPN estimates with uncertainty quantification
4. Generate super-resolution images (localizations, emitters, posterior)
5. Compare results to ground truth

The demo is fully configurable via parameters at the top of the script.
"""

using Pkg; Pkg.activate("examples")
using SMLMBaGoL

#=============================================================================
USER PARAMETERS - Configure your simulation here
=============================================================================#

# N-mer simulation parameters
const N_EMITTERS = 6                    # Number of emitters in complex
const NMER_DIAMETER = 0.1               # μm - diameter of circular complex
const PHOTONS_MEAN = 800                # Average photons per localization
const LOCS_PER_EMITTER_MEAN = 8.0       # Mean localizations per emitter
const LOCS_PER_EMITTER_VAR = 8.0        # Variance in localizations per emitter

# Analysis parameters  
const N_ITERATIONS = 5000               # RJMCMC iterations
const BURN_IN = 1000                    # Burn-in period
const ENABLE_PARTITIONING = false       # Single partition for simple demo

# Visualization parameters
const PIXEL_SIZE = 0.001                # μm per pixel (1 nm super-resolution)
const SAVE_IMAGES = true                # Generate PNG files

#=============================================================================
SIMULATION & ANALYSIS
=============================================================================#

# Create output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("=== SMLMBaGoL Complete Demo ===\n")
println("Configuration:")
println("  N-mer size: $N_EMITTERS emitters")
println("  Diameter: $NMER_DIAMETER μm")
println("  Photons: $PHOTONS_MEAN average")
println("  Localizations/emitter: $LOCS_PER_EMITTER_MEAN ± $(sqrt(LOCS_PER_EMITTER_VAR))")
println("  Analysis: $N_ITERATIONS iterations, $BURN_IN burn-in")
println("  Pixel size: $(PIXEL_SIZE*1000) nm\n")

# Step 1: Generate synthetic data
println("1. Generating synthetic $(N_EMITTERS)-mer data...")
localizations, prior = simulate_n_mer(
    n = N_EMITTERS,
    diameter = NMER_DIAMETER,
    localizations_per_emitter_mean = LOCS_PER_EMITTER_MEAN,
    localizations_per_emitter_variance = LOCS_PER_EMITTER_VAR,
    photons = PHOTONS_MEAN
    # Using default K prior - let BaGoL discover emitter count from data
)

println("   Generated $(length(localizations)) total localizations")
println("   K prior: uninformative (mean=5.0, variance=5.0)")

println("\n   Localization summary:")
show(stdout, MIME("text/plain"), localizations)
println()

# Step 2: Run BaGoL analysis
println("\n2. Running BaGoL analysis...")
result = run_bagol(
    localizations,
    prior = prior,
    n_iterations = N_ITERATIONS,
    burn_in = BURN_IN,
    partition_data = ENABLE_PARTITIONING
)

# Step 3: Extract final state results
println("\n3. Analysis Results:")
final_state = result.current_state
n_estimated = length(final_state.emitters)
println("   Estimated emitter count: $n_estimated (true: $N_EMITTERS)")

println("\n   Final state summary:")
show(stdout, MIME("text/plain"), final_state)
println()

# Step 4: Get MAPN estimates
println("\n4. Computing MAPN estimates...")
mapn_emitters = estimate_mapn([result])
println("   MAPN estimated $(length(mapn_emitters)) emitters")

if !isempty(mapn_emitters)
    println("\n   MAPN emitters with uncertainties:")
    show(stdout, MIME("text/plain"), mapn_emitters)
    println()
end

# Step 5: Generate super-resolution images
if SAVE_IMAGES
    println("\n5. Generating super-resolution images...")
    
    # SR image from localizations (uses localization precision)
    println("   - Localizations SR image...")
    loc_image = gen_sr_image(localizations, 
                            pixel_size = PIXEL_SIZE,
                            filename = joinpath(output_dir, "localizations_sr.png"))
    println("     Saved: output/localizations_sr.png ($(size(loc_image)) pixels)")
    
    # SR image from MAPN emitters (uses uncertainty estimates)
    if !isempty(mapn_emitters)
        println("   - MAPN emitters SR image...")
        emitter_image = gen_sr_image(mapn_emitters,
                                    pixel_size = PIXEL_SIZE,
                                    filename = joinpath(output_dir, "mapn_emitters_sr.png"))
        println("     Saved: output/mapn_emitters_sr.png ($(size(emitter_image)) pixels)")
    end
    
    # Posterior uncertainty image (pixel counting histogram)
    println("   - Posterior uncertainty image...")
    posterior_image = gen_sr_image(result,
                                  pixel_size = PIXEL_SIZE,
                                  mode = :posterior,
                                  filename = joinpath(output_dir, "posterior_uncertainty.png"))
    println("     Saved: output/posterior_uncertainty.png ($(size(posterior_image)) pixels)")
    println("     Max counts per pixel: $(round(maximum(posterior_image), digits=1))")
    
    # Image statistics
    println("\n   Image Statistics:")
    println("     Localizations image: range [$(round(minimum(loc_image), digits=3)), $(round(maximum(loc_image), digits=3))]")
    if !isempty(mapn_emitters)
        println("     MAPN emitters image: range [$(round(minimum(emitter_image), digits=3)), $(round(maximum(emitter_image), digits=3))]")
    end
    println("     Posterior image: range [$(round(minimum(posterior_image), digits=1)), $(round(maximum(posterior_image), digits=1))] counts")
else
    println("\n5. Image generation skipped (SAVE_IMAGES = false)")
end

# Step 6: Ground Truth Comparison
println("\n6. Ground Truth Comparison:")
println("   True $(N_EMITTERS)-mer positions (radius = $(NMER_DIAMETER/2) μm):")
true_radius = NMER_DIAMETER / 2
true_positions = []
for i in 1:N_EMITTERS
    angle = 2π * (i-1) / N_EMITTERS
    true_x = true_radius * cos(angle)
    true_y = true_radius * sin(angle)
    push!(true_positions, (true_x, true_y))
    println("     True $i: ($(round(true_x, digits=3)), $(round(true_y, digits=3))) μm")
end

# Calculate position errors if we have MAPN estimates
if !isempty(mapn_emitters) && length(mapn_emitters) == N_EMITTERS
    println("\n   Position accuracy (closest pairing):")
    # Simple closest-pair matching for error calculation
    errors = Float64[]
    used_true = falses(N_EMITTERS)
    
    for emitter in mapn_emitters
        min_dist = Inf
        best_idx = 1
        for (i, (tx, ty)) in enumerate(true_positions)
            if !used_true[i]
                dist = sqrt((emitter.x - tx)^2 + (emitter.y - ty)^2)
                if dist < min_dist
                    min_dist = dist
                    best_idx = i
                end
            end
        end
        used_true[best_idx] = true
        push!(errors, min_dist)
    end
    
    mean_error = mean(errors) * 1000  # Convert to nm
    max_error = maximum(errors) * 1000
    println("     Mean position error: $(round(mean_error, digits=1)) nm")
    println("     Max position error: $(round(max_error, digits=1)) nm")
end

# Step 7: Summary statistics
println("\n7. Summary:")
println("   Total RJMCMC samples: $(length(result.samples))")
println("   Final log-likelihood: $(round(final_state.log_likelihood, digits=1))")

# Recovery metrics
count_recovery = abs(n_estimated - N_EMITTERS) <= 1
mapn_count_recovery = !isempty(mapn_emitters) && abs(length(mapn_emitters) - N_EMITTERS) <= 1

println("   Count recovery (final state): $(count_recovery ? "✓" : "✗") (within ±1 emitter)")
println("   Count recovery (MAPN): $(mapn_count_recovery ? "✓" : "✗") (within ±1 emitter)")

if !isempty(mapn_emitters) && length(mapn_emitters) == N_EMITTERS
    position_recovery = mean(errors) < 0.05  # 50 nm threshold
    println("   Position recovery: $(position_recovery ? "✓" : "✗") (mean error < 50 nm)")
end

println("\n=== Demo Complete ===")

if SAVE_IMAGES
    println("\nGenerated files in examples/output/:")
    println("  - localizations_sr.png: Super-resolution image from raw localizations")
    if !isempty(mapn_emitters)
        println("  - mapn_emitters_sr.png: Super-resolution image from MAPN estimates")
    end
    println("  - posterior_uncertainty.png: Posterior uncertainty heatmap from RJMCMC chain")
    println("\nUse any image viewer to examine the results!")
end

println("\n💡 Tip: Modify parameters at the top of this script to explore different scenarios!")
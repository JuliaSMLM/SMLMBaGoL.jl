#!/usr/bin/env julia
"""
Simple BaGoL Demo Script

A lightweight demo showing SMLMBaGoL.bagol() with synthetic data from SMLMSim.
Optimized for quick execution with fewer frames and localizations.
"""

# Activate the dev environment automatically
using Pkg
Pkg.activate("dev")

using SMLMSim
using SMLMBaGoL
using Statistics
using Random
using CairoMakie

println("=== Simple BaGoL Demo ===\n")

# Set random seed for reproducibility
Random.seed!(123)

println("1. Creating synthetic SMLM data...")

# Smaller simulation for faster execution
params = StaticSMLMParams(
    density = 10,          # Moderate density: 0.5 patterns per μm²
    σ_psf = 0.13,           # 130nm PSF width
    minphotons = 50,        # Lower threshold: 50 minimum photons
    ndatasets = 1,          # single dataset
    nframes = 200,          # 200 frames
    framerate = 50.0,       # 50 fps
    ndims = 2               # 2D simulation
)

# Simple dimer pattern 
pattern = Nmer2D(n=2, d=0.1)  # 100nm diameter dimer

# Fluorophore with higher on-rate and reasonable blinking
# Calculate k_on and k_off using frames to give about 5 localizations per emitter
# emitters stay on for 1 frame on average, so 
k_off = params.framerate  # 1 frame off time
k_on = 5.0 / (params.nframes/params.framerate)  # 5 localizations per emitter


fluor = GenericFluor(photons=50000.0, k_off=k_off, k_on=k_on)

println("   - Pattern: $(pattern.n)-mer with $(pattern.d*1000) nm diameter")
println("   - Density: $(params.density) patterns/μm²")
println("   - Frames: $(params.nframes)")

# Run simulation
smld_true, smld_model, smld_noisy = simulate(
    params;
    pattern=pattern,
    molecule=fluor,
    camera=IdealCamera(1:16, 1:16, 0.1)
)

n_true = length(smld_true.emitters)
n_localizations = length(smld_noisy.emitters)

println("   - Generated $(n_true) true emitters")
println("   - Generated $(n_localizations) localizations")

println("\n2. Running BaGoL clustering...")

# Run BaGoL with faster settings
result = bagol(
    smld_noisy.emitters;
    mcmc_steps=1000,        # Fewer MCMC steps
    burnin=200,             # Shorter burn-in
    n_threads=1             # Single thread for simplicity
)

n_mapn = length(result.mapn_emitters)
println("   - Found $(n_mapn) MAP-N emitters")
println("   - Compression ratio: $(round(n_localizations/n_mapn, digits=1)):1")

println("\n3. Results comparison:")
println("   Input localizations: $(n_localizations)")
println("   Output emitters: $(n_mapn)")

if n_mapn > 0
    # Show first few emitter positions
    println("\n   MAP-N emitter positions:")
    for (i, e) in enumerate(result.mapn_emitters[1:min(5, n_mapn)])
        println("     $(i): ($(round(e.x*1000, digits=1)), $(round(e.y*1000, digits=1))) nm, $(round(e.photons, digits=0)) photons")
    end
    if n_mapn > 5
        println("     ... and $(n_mapn-5) more")
    end
end

output_dir = joinpath("dev", "output")
mkpath(output_dir)

# Create circle plot showing localizations, MAP-N results, and true emitters
fig2 = plot_circles(smld_noisy.emitters; 
                   figsize=(900, 700))
save(joinpath(output_dir, "circle_plot_localizations.png"), fig2)

println("\n4. Creating circle plots...")

# Create circle plot showing localizations, MAP-N results, and true emitters
fig = plot_circles(smld_noisy.emitters; 
                   mapn_results=result.mapn_emitters,
                   figsize=(900, 700))

println("   - Circle plot created with:")
println("     • Gray circles: $(length(smld_noisy.emitters)) localizations")
println("     • Green circles: $(length(smld_true.emitters)) true emitters")
println("     • Red circles: $(length(result.mapn_emitters)) MAP-N emitters")

println("\n5. Saving images...")

# Save results to dev/output directory


# Save the circle plot
circle_plot_file = joinpath(output_dir, "circle_plot.png")
save(circle_plot_file, fig)
println("   - Circle plot saved to: $(circle_plot_file)")

# Save posterior image as PNG with nice colormap
if !isnothing(result.posterior) && !isempty(result.posterior)
    # Save as colorful PNG using viridis colormap
    png_file = joinpath(output_dir, "posterior_viridis.png")
    save_posterior_image(result.posterior, png_file; colormap=:viridis)
    
    # Also save as hot colormap for comparison
    hot_file = joinpath(output_dir, "posterior_hot.png")
    save_posterior_image(result.posterior, hot_file; colormap=:hot)
    
    # Save grayscale version
    gray_file = joinpath(output_dir, "posterior_gray.png")
    save_posterior_image(result.posterior, gray_file; colormap=:gray)
    
    println("   - Posterior images saved:")
    println("     • Viridis colormap: $(png_file)")
    println("     • Hot colormap: $(hot_file)")
    println("     • Grayscale: $(gray_file)")
    println("   - Image size: $(size(result.posterior, 1)) x $(size(result.posterior, 2)) pixels")
else
    println("   - No posterior image available to save")
end

println("\n✓ Demo completed! All images saved to $(output_dir)")
println("✓ BaGoL successfully processed synthetic SMLM data")
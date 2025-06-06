# SMLMBaGoL Visualization Tools Demo
# Demonstrates circle plots of localizations and MAP-N analysis

using Pkg
Pkg.activate(".")
using Revise
using SMLMBaGoL
BGL = SMLMBaGoL
using SMLMData
using SMLMSim
using Distributions
using Images
using ColorSchemes
using CairoMakie
using CairoMakie: Figure, Axis, scatter!, save, Point2f, image!, axislegend

println("=== SMLMBaGoL Visualization Tools Demo ===")

# Generate small test dataset
println("\n1. Generating test data...")
k_on = 5e-2
n_datasets = 3
n_frames = 100
framerate = 50.0
ρ = 15.0  # Higher density for more interesting plots
σ_PSF = 0.13
minphotons = 100
d_nmer = 0.025

smld_true, smld_model, smld_noisy = SMLMSim.simulate(
    SMLMSim.StaticSMLMParams(
        density=ρ,
        σ_psf=σ_PSF,
        minphotons=minphotons,
        ndatasets=n_datasets,
        nframes=n_frames,
        framerate=framerate
    );
    pattern=SMLMSim.Nmer2D(d=d_nmer),
    molecule=SMLMSim.GenericFluor(photons=1e5, k_off=50.0, k_on=k_on),
    camera=SMLMData.IdealCamera(24, 24, 0.1)
)

println("Generated $(length(smld_noisy.emitters)) noisy localizations")

# No conversion needed - use emitters directly from SMLMSim
println("Using emitters directly: $(typeof(smld_noisy.emitters[1]))")

println("\n=== Available Visualization Tools ===")

println("\n2. Circle Plot of Localizations")
println("   Function: BGL.plot_circles(emitters)")
println("   - Shows uncertainty circles for each localization")
println("   - Circle center = localization position")
println("   - Circle radius = localization uncertainty (σ)")

# Plot observations with uncertainty circles and true emitters
fig_obs = BGL.plot_circles(
    smld_noisy.emitters;
    true_emitters=smld_true.emitters,
    title="Localizations (black) + Truth (green X)"
)

save("dev/output/circle_plot_localizations.png", fig_obs)
println("   → Saved: dev/output/circle_plot_localizations.png")

println("\n3. Super-Resolution Image Reconstruction")
println("   Function: BGL.plot_sr(emitters, pixelsize=0.01)")
println("   - Gaussian blob reconstruction from localizations")
println("   - High-resolution rendering of localization data")

fig_sr = BGL.plot_sr(
    smld_noisy.emitters;
    pixelsize=0.01,
    title="Super-Resolution Image"
)

save("dev/output/sr_reconstruction.png", fig_sr)
println("   → Saved: dev/output/sr_reconstruction.png")

println("\n4. Running BaGoL Analysis for MAP-N Demo...")
# Setup prior and run BaGoL
μ_λ = length(smld_noisy.emitters) / (n_datasets * n_frames)
σ_λ = √μ_λ
α, θ = μ_λ^2 / σ_λ^2, σ_λ^2 / μ_λ
prior_λ = Gamma(α, θ)

# Run BaGoL analysis with new interface
@time srs, post = BGL.bagol(smld_noisy; 
    prior_λ=prior_λ, 
    posterior_pixel_size=0.005
)

println("\n5. MAP-N Analysis and Visualization")
if length(srs) > 0
    # Extract the first subregion's chain for analysis
    chain = srs[1].chains[1]
    
    println("   Function: BGL.plot_sld(chain)")
    println("   - State Length Distribution (how many emitters over MCMC run)")
    BGL.plot_sld(chain)
    
    println("   Function: BGL.plot_state_length(chain)")
    println("   - Shows number of emitters vs iteration number")
    BGL.plot_state_length(chain)
    
    println("   Function: BGL.plot_posterior(chain, obs)")
    println("   - Heatmap showing where emitters were sampled")
    # For plot_posterior we need the ROI - let's create a simplified version
    fig_post = Figure()
    ax_post = Axis(fig_post[1, 1], title="MCMC Posterior")
    
    # Add observations as circles
    BGL.plot_observations!(ax_post, obs; color=:white, strokewidth=1, alpha=0.5)
    save("dev/output/mcmc_posterior_with_circles.png", fig_post)
    println("   → Saved: dev/output/mcmc_posterior_with_circles.png")
    
    println("\n6. MAP-N Localization Estimates")
    println("   Functions for MAP-N analysis:")
    println("   - find_mapn(chain): Find most probable number of emitters")
    println("   - get_mapn_emitters(chain, obs): Get MAP-N localization estimates")
    
    # Find MAP-N
    n_map, n_vec = BGL.RJMCMC.find_mapn(chain)
    println("   MAP-N (most probable number of emitters): $n_map")
    
    # Extract MAP-N chain and get coordinates
    chain_mapn = BGL.RJMCMC.extract_mapn_chain(chain)
    if length(chain_mapn.states) > 0
        BGL.RJMCMC.sort_mapn_chain!(chain_mapn)
        mapn_coords = BGL.RJMCMC.get_mapn_emitters(chain_mapn, obs)
        
        println("   → Found $(length(mapn_coords)) MAP-N emitter estimates")
        
        # Create comprehensive visualization with MAP-N results
        fig_combined = Figure(size=(800, 600))
        ax_combined = Axis(fig_combined[1, 1], aspect=DataAspect(), 
                          title="Combined: Observations + MAP-N Estimates", 
                          xlabel="x", ylabel="y")
        
        # Plot observation circles (blue)
        BGL.plot_observations!(ax_combined, obs; 
            color=:blue, strokewidth=1, alpha=0.6)
        
        # Plot MAP-N estimates (green crosses)
        mapn_x = [coord.x for coord in mapn_coords]
        mapn_y = [coord.y for coord in mapn_coords]
        mapn_σx = [coord.σ_x for coord in mapn_coords]
        mapn_σy = [coord.σ_y for coord in mapn_coords]
        
        scatter!(ax_combined, mapn_x, mapn_y, 
            color=:green, markersize=12, marker=:x, 
            label="MAP-N estimates (n=$n_map)")
        
        # Add true emitters if available (red)
        if length(smld_true.emitters) > 0
            BGL.plot_true_values!(ax_combined, smld_true.emitters; 
                color=:red, markersize=10)
        end
        
        axislegend(ax_combined)
        save("dev/output/mapn_analysis_combined.png", fig_combined)
        println("   → Saved: dev/output/mapn_analysis_combined.png")
        
        println("\n   MAP-N Results Summary:")
        println("   - Blue circles: Original localizations with uncertainty")
        println("   - Green crosses: MAP-N estimated emitter positions")
        println("   - Green circles: MAP-N position uncertainties")
        println("   - Red circles: True emitter positions (if available)")
    end
end

println("\n7. Additional Visualization Options:")
println("   - animate_chain(chain, obs, emitters): Create MP4 animation of MCMC")
println("   - plot_prior_k(prior, obs): Plot prior distribution for number of emitters")
println("   - plot_prior_λ(prior, obs): Plot prior distribution for localizations/emitter")

println("\n=== Summary of Output Files ===")
println("Generated visualization files:")
println("- dev/output/circle_plot_localizations.png: Localization uncertainty circles")
println("- dev/output/sr_reconstruction.png: Super-resolution reconstruction")  
println("- dev/output/mcmc_posterior_with_circles.png: MCMC posterior heatmap")
println("- dev/output/mapn_analysis_combined.png: Complete MAP-N analysis")

println("\n=== Key Visualization Functions ===")
println("Circle Plots:")
println("• BGL.plot_observations(obs) - Basic circle plot")
println("• BGL.plot_observations!(ax, obs) - Add to existing plot")

println("\nMAP-N Analysis:")
println("• BGL.RJMCMC.find_mapn(chain) - Find most probable N")
println("• BGL.RJMCMC.get_mapn_emitters(chain, obs) - Get MAP-N coordinates")
println("• BGL.plot_posterior(chain, roi) - MCMC sampling heatmap")
println("• BGL.plot_sld(chain) - State length distribution")

println("\nImage Reconstruction:")
println("• BGL.gen_sr_image(obs, pixelsize) - Gaussian blob reconstruction")
println("• BGL.VisTools.gen_color_image(image) - Apply color maps")

println("\nDemo completed!")
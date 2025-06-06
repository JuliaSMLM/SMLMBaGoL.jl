# SMLMSim to BaGoL workflow
# Generates SMLD data using SMLMSim and processes it with BaGoL

using Pkg
Pkg.activate(".")  # Activate main project environment
using Revise
using SMLMBaGoL
BGL = SMLMBaGoL
using SMLMData
using SMLMSim
using Distributions
using Images
using ColorSchemes
using ProgressMeter
using CairoMakie

println("n_threads: ", Threads.nthreads())

# Simulation Parameters
k_on = 5e-2
n_datasets = 5
n_frames = 500
framerate = 50.0
ρ = 10.0  # density
σ_PSF = 0.13  # PSF width
minphotons = 100
d_nmer = 0.025  # n-mer separation distance

println("Simulation parameters:")
println("  Datasets: $n_datasets")
println("  Frames: $n_frames") 
println("  Framerate: $framerate Hz")
println("  Density: $ρ")
println("  PSF width: $σ_PSF")
println("  Min photons: $minphotons")
println("  N-mer distance: $d_nmer")

# Generate synthetic data using SMLMSim
@info "Generating synthetic SMLD data with SMLMSim"
@time smld_true, smld_model, smld_noisy = SMLMSim.simulate(
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
    camera=SMLMData.IdealCamera(32, 32, 0.1)
)

println("Generated SMLD data:")
println("  True localizations: $(length(smld_true.emitters))")
println("  Model localizations: $(length(smld_model.emitters))")
println("  Noisy localizations: $(length(smld_noisy.emitters))")

# Visualization parameters (adjusted for micron coordinates)
pixelsize = 0.1  # Analysis pixel size in microns (matches camera pixel size)
zoom_pixelsize = pixelsize / 100  # 10x zoom pixel size for all outputs (0.01 μm)

# Generate observations data structure
@info "Converting SMLD data to observations"
emitter_type = SMLMBaGoL.Emitter2D
y = [emitter.y for emitter in smld_noisy.emitters]
x = [emitter.x for emitter in smld_noisy.emitters]
σ_y = [emitter.σ_y for emitter in smld_noisy.emitters]
σ_x = [emitter.σ_x for emitter in smld_noisy.emitters]

obs = BGL.gen_observations(emitter_type, vcat(y', x'), vcat(σ_y', σ_x'))

# Generate and save observation image using traditional image approach
@info "Generating observation image"
@time bglim = BGL.gen_obs_image(obs, zoom_pixelsize)
save("dev/output/smlmsim_observations.png", bglim.data)
println("  → Saved observation image: dev/output/smlmsim_observations.png")

# Generate interactive circle plot using new VisTools
@info "Creating interactive observation plot"
fig_obs = BGL.plot_observations(obs; color=:blue, alpha=0.6)

# Add true emitter positions if available
if length(smld_true.emitters) > 0
    # Get the axis from the figure content
    ax_obs = fig_obs.content[1]
    BGL.plot_true_values!(ax_obs, smld_true.emitters; color=:red, markersize=8)
    ax_obs.title = "Observations (blue circles) vs True Emitters (red dots)"
end

save("dev/output/smlmsim_interactive_observations.png", fig_obs)
println("  → Saved interactive plot: dev/output/smlmsim_interactive_observations.png")

# Generate super-resolution image
@info "Generating super-resolution image"
@time srim = BGL.gen_sr_image(obs, zoom_pixelsize)
srim_color = BGL.VisTools.gen_color_image(srim; max_quantile=0.99)
save("dev/output/smlmsim_sr.png", srim_color)
println("  → Saved super-resolution image: dev/output/smlmsim_sr.png")

# Setup prior distribution for λ (mean number of localizations per emitter)
# λ represents the expected number of localizations per emitter
# Estimate this from the simulation parameters
total_localizations = length(smld_noisy.emitters)
total_true_emitters = length(smld_true.emitters)
μ_λ = total_localizations / total_true_emitters  # Mean localizations per emitter
σ_λ = √μ_λ  # Assume Poisson-like variance for the number of localizations
α, θ = μ_λ^2 / σ_λ^2, σ_λ^2 / μ_λ  # Method of moments for Gamma distribution
prior_λ = Gamma(α, θ)

println("Prior parameters:")
println("  μ_λ (mean localizations per emitter): $μ_λ")
println("  σ_λ (std localizations per emitter): $σ_λ") 
println("  α (Gamma shape): $α")
println("  θ (Gamma scale): $θ")
println("  Total localizations: $total_localizations")
println("  Total true emitters: $total_true_emitters")

# Run BaGoL analysis
@info "Running BaGoL analysis on SMLMSim data"
@time srs, post = bagol(smld_noisy; 
    prior_λ=prior_λ, 
    pixelsize=pixelsize,
    posterior_pixel_size=zoom_pixelsize
)

# Generate and save posterior image
@info "Generating posterior image"
img = deepcopy(post.post_arr)
BGL.VisTools.quantile_stretch!(img; max_quantile=0.95)
colormap = ColorSchemes.inferno
posterior_color = get(colormap, img)
save("dev/output/smlmsim_posterior.png", posterior_color)
println("  → Saved posterior image: dev/output/smlmsim_posterior.png")

# Analysis summary
n_subregions = length(srs)
total_emitters_found = sum(length(sr.chains[1].states[end].emitters) for sr in srs)

println("\nBaGoL Analysis Results:")
println("  Number of subregions: $n_subregions")
println("  Total emitters found: $total_emitters_found")
println("  Input emitters: $(length(smld_noisy.emitters))")
println("  Recovery rate: $(round(total_emitters_found/length(smld_noisy.emitters)*100, digits=1))%")

# Compare with ground truth
if length(smld_true.emitters) > 0
    true_count = length(smld_true.emitters)
    println("  True emitters: $true_count")
    println("  True recovery rate: $(round(total_emitters_found/true_count*100, digits=1))%")
end

# Chain analysis using new VisTools
if n_subregions > 0
    @info "Analyzing RJMCMC chains with new visualization tools"
    
    # Get the largest subregion's chain for analysis
    largest_sr_idx = argmax([length(sr.obs.ŷ) for sr in srs])
    chain = srs[largest_sr_idx].chains[1]
    
    println("  Analyzing subregion $largest_sr_idx with $(length(srs[largest_sr_idx].obs.ŷ)) observations")
    
    # Plot state length distribution
    fig_sld = BGL.plot_sld(chain)
    save("dev/output/smlmsim_state_length_dist.png", fig_sld)
    println("  → Saved state length distribution: dev/output/smlmsim_state_length_dist.png")
    
    # Plot state length over iterations
    fig_state = BGL.plot_state_length(chain)
    save("dev/output/smlmsim_state_length_trace.png", fig_state)
    println("  → Saved state length trace: dev/output/smlmsim_state_length_trace.png")
    
    # Create animation of RJMCMC chain (sample every 50 iterations for speed)
    @info "Creating RJMCMC chain animation"
    animation_file = BGL.animate_chain(chain, srs[largest_sr_idx].obs; 
        filename="dev/output/smlmsim_chain_animation.mp4", 
        fps=10, 
        skip=50)
    println("  → Saved chain animation: $animation_file")
    
    # Find MAP-N and create comprehensive visualization
    n_map, n_vec = BGL.RJMCMC.find_mapn(chain)
    println("  MAP-N (most probable number of emitters): $n_map")
    
    # Extract MAP-N chain and get coordinates
    chain_mapn = BGL.RJMCMC.extract_mapn_chain(chain)
    if length(chain_mapn.states) > 0
        BGL.RJMCMC.sort_mapn_chain!(chain_mapn)
        mapn_coords = BGL.RJMCMC.get_mapn_emitters(chain_mapn, srs[largest_sr_idx].obs)
        
        # Create comprehensive MAP-N analysis plot
        fig_mapn = Figure(size=(800, 600))
        ax_mapn = Axis(fig_mapn[1, 1], aspect=DataAspect(), 
                      title="MAP-N Analysis for Largest Subregion", 
                      xlabel="x (μm)", ylabel="y (μm)")
        
        # Plot observations (blue circles)
        BGL.plot_observations!(ax_mapn, srs[largest_sr_idx].obs; 
            color=:blue, alpha=0.4, strokewidth=1)
        
        # Plot MAP-N estimates (green crosses)
        if length(mapn_coords) > 0
            mapn_x = [coord.x for coord in mapn_coords]
            mapn_y = [coord.y for coord in mapn_coords]
            scatter!(ax_mapn, mapn_x, mapn_y, 
                color=:green, markersize=10, marker=:x, 
                label="MAP-N estimates (n=$n_map)")
        end
        
        # Add true emitters if available (red dots)
        if length(smld_true.emitters) > 0
            BGL.plot_true_values!(ax_mapn, smld_true.emitters; 
                color=:red, markersize=6)
        end
        
        axislegend(ax_mapn)
        save("dev/output/smlmsim_mapn_analysis.png", fig_mapn)
        println("  → Saved MAP-N analysis: dev/output/smlmsim_mapn_analysis.png")
    end
end

println("\nOutput files saved:")
println("  - dev/output/smlmsim_observations.png (raw observation image)")
println("  - dev/output/smlmsim_interactive_observations.png (interactive plot with truth)")
println("  - dev/output/smlmsim_sr.png (super-resolution reconstruction)")
println("  - dev/output/smlmsim_posterior.png (Bayesian posterior)")
if n_subregions > 0
    println("  - dev/output/smlmsim_state_length_dist.png (MCMC state length distribution)")
    println("  - dev/output/smlmsim_state_length_trace.png (MCMC convergence trace)")
    println("  - dev/output/smlmsim_chain_animation.mp4 (RJMCMC chain animation)")
    println("  - dev/output/smlmsim_mapn_analysis.png (MAP-N analysis)")
end

println("\n=== New VisTools Capabilities Demonstrated ===")
println("📊 Interactive Plotting:")
println("  • plot_observations() - Circle plots with uncertainty")
println("  • plot_true_values!() - Overlay ground truth emitters")
println("📈 Chain Analysis:")
println("  • plot_sld() - State length distribution")
println("  • plot_state_length() - Convergence monitoring")
println("🎬 Animations:")
println("  • animate_chain() - MP4 visualization of MCMC evolution")
println("🔬 MAP-N Analysis:")
println("  • Comprehensive visualization combining observations, estimates, and truth")

@info "SMLMSim to BaGoL workflow completed successfully with new VisTools"
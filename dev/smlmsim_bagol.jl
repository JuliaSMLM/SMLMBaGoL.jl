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

# Generate and save observation image
@info "Generating observation image"
emitter_type = SMLMBaGoL.Emitter2D
y = [emitter.y for emitter in smld_noisy.emitters]
x = [emitter.x for emitter in smld_noisy.emitters]
σ_y = [emitter.σ_y for emitter in smld_noisy.emitters]
σ_x = [emitter.σ_x for emitter in smld_noisy.emitters]

obs = BGL.gen_observations(emitter_type, vcat(y', x'), vcat(σ_y', σ_x'))
@time bglim = BGL.gen_obs_image(obs, zoom_pixelsize)
display(bglim.data)
save("dev/output/smlmsim_observations.png", bglim.data)

# Generate super-resolution image
@info "Generating super-resolution image"
@time srim = BGL.gen_sr_image(obs, zoom_pixelsize)
srim_color = BGL.VisTools.gen_color_image(srim; max_quantile=0.99)
display(srim_color)
save("dev/output/smlmsim_sr.png", srim_color)

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
display(posterior_color)
save("dev/output/smlmsim_posterior.png", posterior_color)

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

println("\nOutput files saved:")
println("  - dev/output/smlmsim_observations.png (raw observations)")
println("  - dev/output/smlmsim_sr.png (super-resolution reconstruction)")
println("  - dev/output/smlmsim_posterior.png (Bayesian posterior)")

@info "SMLMSim to BaGoL workflow completed successfully"
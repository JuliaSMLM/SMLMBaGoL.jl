# SMLMSim to BaGoL workflow using new simplified VisTools
# Uses only the 4 core plot functions: plot_circles, plot_sr, plot_mapn, plot_posterior

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
using Statistics

println("n_threads: ", Threads.nthreads())

# Simulation Parameters
k_on = 5e-2
n_datasets = 5
n_frames = 500
framerate = 50.0
ρ = 10.0  # density
σ_PSF = 0.13  # PSF width
minphotons = 100
d_nmer = 0.050  # n-mer separation distance

println("Simulation parameters:")
println("  Datasets: $n_datasets")
println("  Frames: $n_frames")
println("  Framerate: $framerate Hz")
println("  Density: $ρ")
println("  PSF width: $σ_PSF")
println("  Min photons: $minphotons")
println("  N-mer distance: $d_nmer")

# Generate SMLD data using SMLMSim
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

# Convert SMLD data to observations (use existing gen_observations function)
@info "Converting SMLD data to observations"
emitter_type = BGL.Emitters.Emitter2D
y = [emitter.y for emitter in smld_noisy.emitters]
x = [emitter.x for emitter in smld_noisy.emitters]
σ_y = [emitter.σ_y for emitter in smld_noisy.emitters]
σ_x = [emitter.σ_x for emitter in smld_noisy.emitters]

obs = BGL.gen_observations(emitter_type, vcat(y', x'), vcat(σ_y', σ_x'))

# Generate visualizations using new simplified API

# 1. Circle plot with localizations and true emitters
@info "Creating circle plot with localizations and truth"
fig_circles = BGL.plot_circles(
    obs.ŷ;  # Vector of observations
    true_emitters=smld_true.emitters,  # Vector of true emitters
    title="Localizations (black) + Truth (green X)"
)
save("dev/output/smlmsim_circles.png", fig_circles)
println("  → Saved circle plot: dev/output/smlmsim_circles.png")

# 2. Super-resolution image
@info "Creating super-resolution image"
pixelsize_sr = 0.01  # Fine pixel size for SR
fig_sr = BGL.plot_sr(
    obs.ŷ;  # Vector of observations
    pixelsize=pixelsize_sr,
    title="Super-Resolution Image"
)
save("dev/output/smlmsim_sr_new.png", fig_sr)
println("  → Saved SR image: dev/output/smlmsim_sr_new.png")

# 3. Posterior image (histogram-based)
@info "Creating posterior image"
pixelsize_post = 1.0  # Coarser pixel size for posterior
fig_posterior = BGL.plot_posterior(
    obs.ŷ;  # Vector of observations
    pixelsize=pixelsize_post,
    title="Posterior Distribution"
)
save("dev/output/smlmsim_posterior_new.png", fig_posterior)
println("  → Saved posterior image: dev/output/smlmsim_posterior_new.png")

# Run BaGoL analysis
@info "Running BaGoL analysis"
μ_λ = length(smld_noisy.emitters) / length(smld_true.emitters)
σ_λ = sqrt(μ_λ)
α = μ_λ
θ = 1.0
prior_λ = Gamma(α, θ)

println("Prior parameters:")
println("  μ_λ (mean localizations per emitter): $μ_λ")
println("  σ_λ (std localizations per emitter): $σ_λ")
println("  α (Gamma shape): $α")
println("  θ (Gamma scale): $θ")

# Run BaGoL
@time srs, post = BGL.bagol(smld_noisy; prior_λ=prior_λ, pixelsize=1.0)

println("\\nBaGoL Analysis Results:")
println("  Number of subregions: $(length(srs))")

# Collect MAP-N estimates from all subregions
all_mapn_coords = []
for sr in srs
    if length(sr.chains) > 0 && length(sr.chains[1].states) > 0
        n_map, _ = BGL.RJMCMC.find_mapn(sr.chains[1])
        chain_mapn = BGL.RJMCMC.extract_mapn_chain(sr.chains[1])
        
        if length(chain_mapn.states) > 0
            BGL.RJMCMC.sort_mapn_chain!(chain_mapn)
            mapn_coords = BGL.RJMCMC.get_mapn_emitters(chain_mapn, sr.obs)
            append!(all_mapn_coords, mapn_coords)
        end
    end
end

println("  Total MAP-N emitters: $(length(all_mapn_coords))")

# 4. Combined analysis with all data
if !isempty(all_mapn_coords)
    @info "Creating combined analysis plot"
    fig_combined = BGL.plot_circles(
        obs.ŷ;  # All observations
        mapn_emitters=all_mapn_coords,  # All MAP-N estimates
        true_emitters=smld_true.emitters,  # All true emitters
        title="Combined: Localizations (black) + MAP-N (red) + Truth (green)"
    )
    save("dev/output/smlmsim_circles_mapn_true.png", fig_combined)
    println("  → Saved combined analysis: dev/output/smlmsim_circles_mapn_true.png")
    
    # MAP-N only image
    @info "Creating MAP-N image"
    fig_mapn = BGL.plot_mapn(
        all_mapn_coords;
        pixelsize=pixelsize_sr,
        title="MAP-N Estimates"
    )
    save("dev/output/smlmsim_mapn_new.png", fig_mapn)
    println("  → Saved MAP-N image: dev/output/smlmsim_mapn_new.png")
end

println("\\nOutput files saved:")
println("  - dev/output/smlmsim_circles.png (circle plot with truth)")
println("  - dev/output/smlmsim_sr_new.png (super-resolution image)")
println("  - dev/output/smlmsim_posterior_new.png (posterior distribution)")
println("  - dev/output/smlmsim_circles_mapn_true.png (combined analysis)")
println("  - dev/output/smlmsim_mapn_new.png (MAP-N estimates)")

println("\\n=== New Simplified VisTools Demonstrated ===")
println("✅ Only 4 core functions used:")
println("  • plot_circles() - Circle plots with optional overlays")
println("  • plot_sr() - Super-resolution Gaussian blob images")
println("  • plot_mapn() - MAP-N Gaussian blob images")
println("  • plot_posterior() - Posterior histogram images")
println("✅ Clean vector-based API - no complex parameters")
println("✅ Consistent data-coordinate system throughout")
println("✅ Minimal, maintainable codebase")
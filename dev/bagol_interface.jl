using Pkg 
Pkg.activate("dev")
using Revise
using SMLMBaGoL
BGL = SMLMBaGoL
using Images
using ColorSchemes

println("n_threads: ", Threads.nthreads() )

include("gen_nmers.jl")
# include("gen_nmers_big.jl")
println("Total Localizations: ", length(smld_noisy.emitters))


# Generate visualizations using new plot functions
@info "Creating circle plot"
fig_circles = BGL.plot_circles(
    smld_noisy.emitters;
    title="Localizations from N-mers"
)
save("dev/output/observations_nmer.png", fig_circles)

@info "Creating super-resolution image"
fig_sr = BGL.plot_sr(
    smld_noisy.emitters;
    pixelsize=0.01,
    title="Super-Resolution N-mers"
)
save("dev/output/sr_nmer.png", fig_sr)

@info "Creating posterior image"
fig_posterior = BGL.plot_posterior(
    smld_noisy.emitters;
    pixelsize=0.05,
    title="Posterior Distribution"
)
save("dev/output/posterior_initial.png", fig_posterior)

# setup prior 
μ_λ = μ
σ_λ = μ
α, θ = μ_λ^2 / σ_λ^2, σ_λ^2 / μ_λ
prior_λ = Gamma(α, θ)

@info "running bagol"
@time srs, post = BGL.bagol(smld_noisy; prior_λ=prior_λ, posterior_pixel_size=0.005)
# @profview srs, post = BGL.bagol(smld_noisy; prior_λ=prior_λ, posterior_pixel_size=0.005)

# Collect MAP-N estimates from all subregions
all_mapn_coords = []
for sr in srs
    if length(sr.chains) > 0 && length(sr.chains[1].states) > 0
        chain_mapn = BGL.RJMCMC.extract_mapn_chain(sr.chains[1])
        if length(chain_mapn.states) > 0
            BGL.RJMCMC.sort_mapn_chain!(chain_mapn)
            mapn_coords = BGL.RJMCMC.get_mapn_emitters(chain_mapn, sr.obs)
            append!(all_mapn_coords, mapn_coords)
        end
    end
end

println("Total MAP-N emitters: ", length(all_mapn_coords))

# Create final visualization
if !isempty(all_mapn_coords)
    @info "Creating MAP-N visualization"
    fig_mapn = BGL.plot_mapn(
        all_mapn_coords;
        pixelsize=0.01,
        title="MAP-N Estimates"
    )
    save("dev/output/mapn_nmer.png", fig_mapn)
    
    @info "Creating combined analysis plot"
    fig_combined = BGL.plot_circles(
        smld_noisy.emitters;
        mapn_emitters=all_mapn_coords,
        title="Combined: Localizations + MAP-N"
    )
    save("dev/output/combined_nmer.png", fig_combined)
end





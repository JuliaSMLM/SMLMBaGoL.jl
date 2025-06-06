# Load and process a SMITE dataset
using Pkg
Pkg.activate("dev")
using Revise
using SMLMData
using SMLMBaGoL
BGL = SMLMBaGoL
using Images 
using Distributions

filepath = "C:\\Data"
filename = "Data_2024-4-5-17-6-12_E345R_Results.mat"
# filename = "Data_2023-9-11-12-48-23_Results.mat"
smd = SMLMData.SMITEsmd(filepath::String,filename::String)

smld = SMLMData.SMLD(smd)  # Use new SMLD format
println("Total Localizations: ", length(smld.emitters))

# Create visualizations using new plot functions
@info "Creating circle plot"
fig_circles = BGL.plot_circles(
    smld.emitters;
    title="SMITE Data Localizations"
)
save("dev/output/smite_circles.png", fig_circles)

@info "Creating super-resolution image"
fig_sr = BGL.plot_sr(
    smld.emitters;
    pixelsize=0.01,
    title="SMITE Super-Resolution"
)
save("dev/output/smite_sr.png", fig_sr)

# setup prior (need to define μ properly)
n_localizations = length(smld.emitters)
# Estimate μ_λ based on data characteristics or use default
μ_λ = 3.0  # Mean localizations per emitter (adjust based on data)
σ_λ = sqrt(μ_λ)
α, θ = μ_λ^2 / σ_λ^2, σ_λ^2 / μ_λ
prior_λ = Gamma(α, θ)

@info "Running BaGoL analysis"
@time srs, post = BGL.bagol(smld; prior_λ=prior_λ, posterior_pixel_size=0.005)

println("BaGoL completed with $(length(srs)) subregions")



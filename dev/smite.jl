# Load and process a SMITE dataset
using Pkg
Pkg.activate("dev")
using Revise
using SMLMData
using SMLMBaGoL
BGL = SMLMBaGoL
using Images 

filepath = "C:\\Data"
filename = "Data_2024-4-5-17-6-12_E345R_Results.mat"
# filename = "Data_2023-9-11-12-48-23_Results.mat"
smd = SMLMData.SMITEsmd(filepath::String,filename::String)

smld = SMLMData.SMLD2D(smd)
println("Total Localizations: ", length(smld.x))

# make a figure 
x = smld.x
y = smld.y
σ_x = smld.σ_x
σ_y = smld.σ_y
emitter_type = BGL.Emitter2D
obs = BGL.gen_observations(emitter_type,vcat(y', x'), vcat(σ_y', σ_x'))

fig, = BGL.RJMCMC.plot_observations(obs; size=(500,500))
save("observations_nmer.png", fig)

# setup prior 
μ_λ = μ
σ_λ = μ
α, θ = μ_λ^2 / σ_λ^2, σ_λ^2 / μ_λ
prior_λ = Gamma(α, θ)

@time post = bagol(smld_noisy; prior_λ, pixel_size = 0.1)



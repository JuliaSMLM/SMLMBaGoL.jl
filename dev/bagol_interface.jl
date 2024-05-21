using Pkg 
Pkg.activate("dev")
using Revise
using SMLMBaGoL
BGL = SMLMBaGoL
using Images
using ColorSchemes


include("gen_nmers.jl")
# include("gen_nmers_big.jl")
println("Total Localizations: ", length(smld_noisy.x))


# make a figure 
x = smld_noisy.x
y = smld_noisy.y
σ_x = smld_noisy.σ_x
σ_y = smld_noisy.σ_y
emitter_type = BGL.Emitter2D

obs = BGL.gen_observations(emitter_type,vcat(y', x'), vcat(σ_y', σ_x'))
@time bglim = BGL.gen_obs_image(obs, 0.001)
display(bglim.data)
save("observations_nmer.png", bglim.data)

@time srim = BGL.gen_sr_image(obs, 0.001)
BGL.VisTools.quantile_stretch!(srim.data; max_quantile=0.95)
display(srim.data)
save("sr_nmer.png", srim.data)


# setup prior 
μ_λ = μ
σ_λ = μ
α, θ = μ_λ^2 / σ_λ^2, σ_λ^2 / μ_λ
prior_λ = Gamma(α, θ)

@time srs, post = bagol(smld_noisy; prior_λ, pixel_size = 0.1)

img = deepcopy(post.post_arr)
BGL.VisTools.quantile_stretch!(img; max_quantile=0.95)
colormap = ColorSchemes.inferno
color_img = get(colormap, img)

save("posterior_nmer.png", color_img)
display(color_img)





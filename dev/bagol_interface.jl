using Revise
using SMLMBaGoL
BGL = SMLMBaGoL
using Images
using ColorSchemes

include("gen_nmers.jl")

# make a figure 
x = smld_noisy.x
y = smld_noisy.y
σ_x = smld_noisy.σ_x
σ_y = smld_noisy.σ_y
emitter_type = BGL.Emitter2D
obs = BGL.gen_observations(emitter_type,vcat(y', x'), vcat(σ_y', σ_x'))
fig, = BGL.RJMCMC.plot_observations(obs)
save("observations_nmer.png", fig)

# setup prior 
μ_λ = μ
σ_λ = μ
α, θ = μ_λ^2 / σ_λ^2, σ_λ^2 / μ_λ
prior_λ = Gamma(α, θ)

post = bagol(smld_noisy; prior_λ, pixel_size = 0.1)

gray_img = Gray.(post.post_arr./maximum(post.post_arr))
colormap = ColorSchemes.inferno

color_img = get(colormap,post.post_arr./maximum(post.post_arr) )

save("posterior_nmer.png", color_img)
display(color_img)





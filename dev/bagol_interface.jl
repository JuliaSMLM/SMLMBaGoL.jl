using Revise
using SMLMBaGoL
BGL = SMLMBaGoL
using Images

include("gen_nmers.jl")

# make a figure 
x = smld_noisy.x
y = smld_noisy.y
σ_x = smld_noisy.σ_x
σ_y = smld_noisy.σ_y
emitter_type = BGL.Emitter2D
obs = BGL.gen_observations(emitter_type,vcat(y', x'), vcat(σ_y', σ_x'))
BGL.RJMCMC.plot_observations(obs)
BGL.RJMCMC.plot_sr(obs)


# setup prior 
μ_λ = μ
σ_λ = μ
α, θ = μ_λ^2 / σ_λ^2, σ_λ^2 / μ_λ
prior_λ = Gamma(α, θ)

post = bagol(smld_noisy; prior_λ, pixel_size = 0.1)

im = Gray.(post.post_arr./maximum(post.post_arr))
save("posterior_nmer.png", im)
display(im)





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
println("Total Localizations: ", length(smld_noisy.x))


# make a figure 
x = smld_noisy.x
y = smld_noisy.y
σ_x = smld_noisy.σ_x
σ_y = smld_noisy.σ_y
emitter_type = BGL.Emitter2D
pixelsize = 1.0 # units are in pixels

@info "gen circle image"
obs = BGL.gen_observations(emitter_type,vcat(y', x'), vcat(σ_y', σ_x'))
@time bglim = BGL.gen_obs_image(obs, 1/100)
display(bglim.data)
save("observations_nmer.png", bglim.data)

@info "gen sr image"
@time srim = BGL.gen_sr_image(obs,pixelsize/50)
# @profview srim = BGL.gen_sr_image(obs, pixelsize/50)
@info "gen color sr image and save"
srim_color = BGL.VisTools.gen_color_image(srim; max_quantile=0.99)
@info "display sr image "
display(srim_color)
@info "save sr image"
@time save("sr_nmer.png", srim_color)

# setup prior 
μ_λ = μ
σ_λ = μ
α, θ = μ_λ^2 / σ_λ^2, σ_λ^2 / μ_λ
prior_λ = Gamma(α, θ)

@info "running bagol"
@time srs, post = bagol(smld_noisy; prior_λ, pixelsize = 1.0)
# @profview srs, post = bagol(smld_noisy; prior_λ, pixel_size = 0.1)

@info "gen color posterior image and save"
img = deepcopy(post.post_arr)
BGL.VisTools.quantile_stretch!(img; max_quantile=0.95)
colormap = ColorSchemes.inferno
color_img = get(colormap, img)

save("posterior_nmer.png", color_img)
display(color_img)





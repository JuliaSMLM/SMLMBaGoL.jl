module VisTools 

using SMLMBaGoL
BGL = SMLMBaGoL
using CairoMakie
using Images 
using ColorSchemes
using ImageDraw 
using Statistics

# Include existing functionality
include("types.jl")
include("plot.jl")
include("tools.jl")
include("images.jl")
include("blobs.jl")

# Include new submodules
include("Drawing.jl")
include("Analysis.jl")
include("Animation.jl")
include("Combined.jl")

# Re-export submodules for easier access
using .Analysis
using .Drawing  
using .Animation
using .Combined

# Export main image generation functions
export gen_obs_image, gen_sr_image, gen_color_image

# Export new unified API functions
export Analysis, Drawing, Animation, Combined

# Re-export key analysis functions for convenience
export plot_observations, plot_observations!
export plot_posterior, plot_sr
export plot_true_values!
export plot_prior_k, plot_prior_λ
export plot_state_length, plot_sld
export animate_chain

# Re-export drawing functions (for pixel arrays)
export draw_circle_on_image!, draw_x_on_image!
export draw_observations_on_image!, draw_emitters_on_image!
export draw_true_on_image!

# Export existing BGLImage2D drawing functions (legacy)
export draw_circle!, draw_x!, draw_observations!, draw_emitters!, draw_true!

# Re-export combined analysis functions
export plot_combined_analysis, plot_mapn_with_uncertainty

end
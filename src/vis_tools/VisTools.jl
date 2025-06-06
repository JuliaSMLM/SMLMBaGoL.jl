module VisTools

using ..SMLMBaGoL
using ..SMLMBaGoL.Emitters
using CairoMakie: Figure, Axis, DataAspect, heatmap!, xlims!, ylims!, save, lines!, poly!, Point2f
using Images
using ColorSchemes
using Statistics: mean, median, quantile

# Include submodules
include("primitives.jl")
include("images.jl") 
include("plots.jl")

# Export primitive functions
export draw_circle!, draw_x!

# Export image generation functions
export gauss_blob_image, histogram_image

# Export main plot functions
export plot_circles, plot_sr, plot_mapn, plot_posterior

# Export utilities
export quantile_stretch!, gen_color_image

end # module VisTools
module VisTools 

using SMLMBaGoL
BGL = SMLMBaGoL
using CairoMakie
using Images 
using ColorSchemes
using ImageDraw 

include("types.jl")
include("plot.jl")
include("images.jl")

export gen_obs_image

end
module VisTools 

using SMLMBaGoL
BGL = SMLMBaGoL
using CairoMakie
using Images 
using ColorSchemes
using ImageDraw 
using Statistics

include("types.jl")
include("plot.jl")
include("tools.jl")
include("images.jl")
include("blobs.jl")

export gen_obs_image, gen_sr_image

end
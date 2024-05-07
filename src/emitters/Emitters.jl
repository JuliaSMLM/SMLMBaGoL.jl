module Emitters

using Distributions
using StatsBase
using SpecialFunctions
using LinearAlgebra
using ..SMLMBaGoL

# These are the methods that must be implemented for each emitter type
import ..SMLMBaGoL.Observations
import ..SMLMBaGoL.Allocations
import ..SMLMBaGoL.log_p_z_given_y
import ..SMLMBaGoL.move!
import ..SMLMBaGoL.build_prior_y
import ..SMLMBaGoL.gen_emitter!
import ..SMLMBaGoL.gen_emitters
import ..SMLMBaGoL.merge_emitters
import ..SMLMBaGoL.gen_observations

include("types.jl")
include("emitters2D.jl")

# This is exported 
export AbstractEmitter


# These should be exported by the emitter type
export Emitter2D
export Localization2D


end
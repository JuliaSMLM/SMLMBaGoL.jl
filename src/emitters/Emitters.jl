module Emitters

using Distributions
using StatsBase
using SpecialFunctions
using LinearAlgebra
using SMLMData
using ..SMLMBaGoL

# Import emitter types from SMLMData
using SMLMData: Emitter2D, Emitter2DFit, Emitter3D, Emitter3DFit, AbstractEmitter

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

# Re-export SMLMData emitter types 
export AbstractEmitter
export Emitter2D, Emitter2DFit, Emitter3D, Emitter3DFit
export Localization2D

end
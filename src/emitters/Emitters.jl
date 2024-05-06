module Emitters

import ..SMLMBaGoL
include("types.jl")
include("emitters2D.jl")

# These are the methods that must be implemented for each emitter type
export log_p_z_given_y
export move!
export build_prior_y
export merge 

# These should be exported by the emitter type
export Emitter2D
export Localization2D


end
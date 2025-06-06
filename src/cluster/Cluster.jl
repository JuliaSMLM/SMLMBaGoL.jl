module Cluster

using Clustering
using Distributions 

# Import from SMLMBaGoL
using ..SMLMBaGoL 
using ..SMLMBaGoL.RJMCMC
using SMLMData: Emitter2DFit

import ..SMLMBaGoL.AbstractEmitter
import ..SMLMBaGoL.Observations
import ..SMLMBaGoL.Allocations
import ..SMLMBaGoL.Params
import ..SMLMBaGoL.log_p_z_given_y
import ..SMLMBaGoL.move!
import ..SMLMBaGoL.build_prior_y
import ..SMLMBaGoL.gen_emitter!
import ..SMLMBaGoL.gen_emitters
import ..SMLMBaGoL.merge_emitters
import ..SMLMBaGoL.gen_observations
import ..SMLMBaGoL.Emitters.create_minimal_emitter2dfit


include("types.jl")
include("hierarchical_bayes.jl")
include("cluster_tools.jl")
export Subregion, gen_subregions

end


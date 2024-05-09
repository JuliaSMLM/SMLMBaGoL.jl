module RJMCMC

using Distributions
using StatsBase 
using SpecialFunctions
using Hungarian
using CairoMakie 
using CairoMakie: Point2f0
using ..SMLMBaGoL
using ..Emitters

# Import from SMLMBaGoL
import ..SMLMBaGoL.Observations
import ..SMLMBaGoL.Allocations
import ..SMLMBaGoL.Params
import ..SMLMBaGoL.log_p_z_given_y
import ..SMLMBaGoL.move!
import ..SMLMBaGoL.build_prior_y
import ..SMLMBaGoL.merge_emitters
import ..SMLMBaGoL.gen_emitter!


include("types.jl")
include("move.jl")
include("allocate.jl")
include("add-remove.jl")
include("split-merge.jl")
include("interface.jl")
# include("emitters2D.jl")
include("buildchain.jl")
include("mapn.jl")
include("plottools.jl")

export rjmcmc
export RJMCMC_Chain

# Export plot tools
export plot_posterior, plot_observations, plot_observations!, plot_sr, plot_sld,
    plot_true_values!, plot_prior_λ, plot_prior_k, plot_state_length, animate_chain

end

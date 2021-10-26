using SMLMData
using Distributions

"""
    moveemitters(smld::SMLMData.SMLD2D, 
                 z::Vector{Int}, 
                 nloc::Int,
                 k::Int)

Sample the `k` emitter positions from the normal distribution.

# Description
This function samples new positions for the `k` emitter positions from the
normal distribution defined by the MLE position of the localizations in `smld`
(allocated to emitters by `z`).

# Inputs
-`smld`: SMLMData.SMLD2D data structure containing localizations.
-`z`: Allocations of the localizations in `smld` to emitters associated with
      the indices `1:k`. (length `nloc` integer array)
-`nloc`: Total number of localizations in `smld`.
-`k`: Total number of emitters to which the `nloc` localizations are allocated.
"""
function moveemitters(smld::SMLMData.SMLD2D, z::Vector{Int}, 
        nloc::Int = SMLMData.length(smld), k::Int = maximum(z))
    # Loop through the `k` emitters and sample new positions based on the
    # allocations of localizations in `smld` defined by `z`.
    μ = Matrix{Float64}(undef, nloc, 2)
    for ii = 1:k
        currentbool = z .== ii
        μ[ii, 1] = posterior_emitterpos(smld.x[currentbool], 
                                        smld.σ_x[currentbool]) 
        μ[ii, 2] = posterior_emitterpos(smld.y[currentbool], 
                                        smld.σ_y[currentbool])
    end

    return μ
end

"""
    posterior_emitterpos(x::Vector{Float64}, 
                         σ_x::Vector{Float64})

Construct a posterior distribution of emitter position.

# Description
This function constructs a posterior distribution for the position of the
emitter which generated the localizations `x`.

# Inputs
-`x`: Coordinate of a 2D localization. (pixels)([x; y])
-`σ_x`: Standard error of the localization `x`. (pixels)([x; y])
"""
function posterior_emitterpos(x::Vector{Float64}, 
                              σ_x::Vector{Float64})
    # Estimate the location of the `kID`-th emitter based on the allocated
    # localizations defined by `x` and `σ_x`.  `x_mle` is the MLE of the true
    # emitter position sampled by the length(x) Gaussians with mean `x` and 
    # standard deviation `σ_x`. `σ_fisher` is the square root of the inverse
    # Fisher information for `x_mle`.
    x_num = sum(x ./ (σ_x.^2))
    x_denom = sum(1.0 ./ (σ_x.^2))
    x_mle = x_num / x_denom
    σ_fisher = sqrt(1 / x_denom)

    # Sample a new position of the `kID`-th emitter from the Gaussian defined
    # by `x_mle` and `σ_fisher`.
    return rand(Distributions.Normal(x_mle, σ_fisher))
end
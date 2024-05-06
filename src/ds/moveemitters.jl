using SMLMData
using Distributions
using LinearAlgebra

# This file contains functions/methods related to moving emitters.

"""
    μ = moveemitters(smld::SMLMData.SMLD2D, 
                     z::Vector{Int}, 
                     k::Int = maximum(z))

Sample the `k` emitter positions from the normal distribution.

# Description
This function samples new positions for the `k` emitter positions from the
normal distribution defined by the MLE position of the localizations in `smld`
(allocated to emitters by `z`).

# Inputs
- `smld`: SMLMData.SMLD2D data structure containing localizations.
- z`: Allocations of the localizations in `smld` to emitters associated with
       the indices `1:k`. (length `nloc` integer array)
- `k`: Total number of emitters to which the `nloc` localizations are allocated.

# Outputs
- `μ`: New set of emitter positions. ([y x])
"""
function moveemitters(smld::SMLMData.SMLD2D, z::Vector{Int}, k::Int = maximum(z))
    # Loop through the `k` emitters and sample new positions based on the
    # allocations of localizations in `smld` defined by `z`.
    μ = Matrix{Float64}(undef, k, 2)
    for ii = 1:k
        currentbool = z .== ii
        μ[ii, 1] = SMLMBaGoL.posterior_emitterpos(smld.y[currentbool], 
                                                  smld.σ_y[currentbool])
        μ[ii, 2] = SMLMBaGoL.posterior_emitterpos(smld.x[currentbool], 
                                                  smld.σ_x[currentbool])
    end

    return μ
end

"""
    moveemitters(smld::SMLMData.SMLD2D, 
                 z::Vector{Int}, 
                 σ_a::Float64,
                 k::Int = maximum(z))

Sample the `k` emitter positions from the normal distribution.

# Description
This function samples new positions for the `k` emitter positions from the
normal distribution defined by the MLE position of the localizations in `smld`
(allocated to emitters by `z`).

# Inputs
- `smld`: SMLMData.SMLD2D data structure containing localizations.
- `z`: Allocations of the localizations in `smld` to emitters associated with
       the indices `1:k`. (length `nloc` integer array)
- `σ_a`: Standard deviation of the drift velocity. (same for each dimension)
- `k`: Total number of emitters to which the `nloc` localizations are allocated.

# Outputs
- `μ`: New set of emitter positions. ([y x])
"""
function moveemitters(smld::SMLMData.SMLD2D, 
                      z::Vector{Int}, 
                      σ_a::Float64, 
                      k::Int = maximum(z))
    # If σ_a isn't positive (e.g., 0.0) we should dispatch on the non-drift
    # method of moveemitters.
    if σ_a <= 0.0
        return moveemitters(smld, z, k), zeros(Float64, k, 2)
    end

    # Loop through the `k` emitters and sample new positions based on the
    # allocations of localizations in `smld` defined by `z`.
    μ = Matrix{Float64}(undef, k, 2)
    a = Matrix{Float64}(undef, k, 2)
    for ii = 1:k
        currentbool = z .== ii
        μ[ii, 1], a[ii, 1] = SMLMBaGoL.posterior_emitterpos(smld.y[currentbool], 
            smld.σ_y[currentbool], smld.framenum[currentbool], σ_a)
        μ[ii, 2], a[ii, 2] = SMLMBaGoL.posterior_emitterpos(smld.x[currentbool], 
            smld.σ_x[currentbool], smld.framenum[currentbool], σ_a)
    end

    return μ, a
end

"""
    sample = posterior_emitterpos(y::Vector{Float64}, 
                                  σ_y::Vector{Float64})

Sample a posterior distribution of emitter position along one dimension.

# Description
This function samples a posterior distribution for the position of the
emitter which generated the one dimensional localization coordinates `y`.

# Inputs
- `y`: Coordinate of a localization along one dimension. (pixels)(nlocx1)
- `σ_y`: Standard error of the localization `x`. (pixels)(nlocx1)

# Outputs
- `sample`: Sample from the posterior emitter distribution.
"""
function posterior_emitterpos(y::Vector{Float64}, 
                              σ_y::Vector{Float64})
    # Estimate the location of the `kID`-th emitter based on the allocated
    # localizations defined by `y` and `σ_y`.  `μ_mle` is the MLE of the true
    # emitter position sampled by the length(y) Gaussians with mean `y` and 
    # standard deviation `σ_y`. `σ_fisher` is the square root of the inverse
    # Fisher information for `μ_mle`.
    μ_num = sum(y ./ (σ_y.^2))
    μ_denom = sum(1.0 ./ (σ_y.^2))
    μ_mle = μ_num / μ_denom
    σ_fisher = sqrt(1 / μ_denom)

    # Sample a new position of the `kID`-th emitter from the Gaussian defined
    # by `μ_mle` and `σ_fisher`.
    return Distributions.rand(Distributions.Normal(μ_mle, σ_fisher))
end

"""
    sample = posterior_emitterpos(y::Vector{Float64}, 
                                  σ_y::Vector{Float64},
                                  t::Float64,
                                  σ_a::Float64)

Sample the posterior of emitter position and drift along one dimension.

# Description
This function samples a posterior distribution for the position of the
emitter and its drift which generated the localizations `y`.

# Inputs
- `y`: Coordinate of a localizations along one dimension. (pixels)(nlocx1)
- `σ_y`: Standard error of the localizations `y`. (pixels)(nlocx1)
- `t`: Time of observation of localizations `y`. (frame)(nlocx1)
- `σ_a`: Standard deviation of the drift velocity. (pixels/frame)

# Outputs
- `sample`: Sample from the posterior emitter and drift velocity
            distribution. ([μ_sample; a_sample])
"""
function posterior_emitterpos(y::Vector{Float64},
                              σ_y::Vector{Float64},
                              t::Vector{Int},
                              σ_a::Float64)
    # Estimate the location of the `k-th` emitter based on the allocated
    # localizations defined by `y` and `σ_y`.  `μ` is the MLE of the true
    # emitter position sampled by the length(y) Gaussians with mean `y` and 
    # standard deviation `σ_y`. `Ξ` is the inverse of the Fisher information
    # for the estimate of `μ`.
    var_y = σ_y .^ 2
    A = sum(y ./ var_y)
    B = sum(t ./ var_y)
    C = sum(1.0 ./ var_y)
    D = sum(t.^2 ./ var_y)
    a = sum((C*y.-A)./(var_y./t)) / ((C/σ_a) + sum((C*t.-B)/(var_y./t)))
    μ = (A-a*B) / C
    Ξ = LinearAlgebra.pinv([A B; B D + 1.0./σ_a^2])

    # Sample a new position of the `k-th` emitter from the Gaussian defined
    # by `μ` and `Ξ`.  Note that I'm forcing Ξ to be Hermitian, as it's often
    # non-Hermitian due to floating-point errors.
    return Distributions.rand(Distributions.MvNormal([μ; a], 
                              Matrix(LinearAlgebra.Hermitian(Ξ))))
end
using SMLMData
using Base

# This file contains misc. math helpers/common distributions/statistics that
# find usage in specific contexts in the BaGoL analysis, so having BaGoL
# specific helpers (as opposed to using, e.g., the Distributions package) might
# be useful.

## Likelihoods of emitters given localizations.

"""
    likelihood = emitterlikelihood2D(smld::SMLMData.SMLD2D,
                                     μ::Matrix{Float64},
                                     a::Matrix{Float64},
                                     z::Vector{Int})

Compute the likelihood of the emitters `μ` given localizations in `smld`.

# Description
This function computes the log-likelihood that the localizations stored in 
`smld` arose from emitters at locations `μ` with standard deviations `σ` and
drift velocities `a`.  I.e., likelihood for k emitters along 2 dimensions.

# Inputs
-`y`: 1D coordinate of a set of localizations. (pixels)(nlocx1)
-`μ`: Location of the emitters. (pixels)(kx1)
-`a`: Drift velocities of each emitter. (pixels/frame)(kx2)([y x])
-`z`: Allocations of the `nloc` localizations to the `k` emittters. (nlocx1)

# Outputs
-`likelihood`: Likelihood of the emitters given the localizations.
"""
function emitterlikelihood2D(smld::SMLMData.SMLD2D,
    μ::Matrix{Float64},
    a::Matrix{Float64},
    z::Vector{Int})
    # Compute the likelihood that the (2D) localization coordinates in `y`
    # arose from the emitters located at `μ`.
    return emitterlikelihood2D([smld.y smld.x], [smld.σ_y smld.σ_x],
        Float64.(smld.framenum), μ, a, z)
end

"""
    likelihood = emitterlikelihood2D(y::Matrix{Float64},
                                     σ::Matrix{Float64},
                                     t::Vector{Float64},
                                     μ::Matrix{Float64},
                                     a::Matrix{Float64},
                                     z::Vector{Int})

Compute the likelihood of the emitters `μ` given localizations in `y`.

# Description
This function computes the likelihood that the localizations stored in 
`y` arose from emitters at locations `μ` with standard deviations `σ` and
drift velocities `a`.  This function is a wrapper for emitterlikelihood1D() 
which multiplies the likelihoods over the 2 spatial dimensions.  I.e., 
likelihood for k emitters along 2 dimensions.

# Inputs
-`y`: 2D coordinates of a set of localizations. (pixels)(nlocx2)([y x])
-`σ`: Standard deviations of the observation distributions. 
      (pixels)(nlocx2)([y x])
-`t`: Observation time corresponding to localizations in `y`. (frames)(nlocx1)
-`μ`: Location of the emitters. (pixels)(kx2)([y x])
-`a`: Drift velocities of each emitter. (pixels/frame)(kx2)([y x])
-`z`: Allocations of the `nloc` localizations to the `k` emittters. (nlocx1)

# Outputs
-`likelihood`: Likelihood of the emitters given the localizations.
"""
function emitterlikelihood2D(y::Matrix{Float64},
    σ::Matrix{Float64},
    t::Vector{Float64},
    μ::Matrix{Float64},
    a::Matrix{Float64},
    z::Vector{Int})
    # Compute the likelihood that the (2D) localization coordinates in `y`
    # arose from the emitters located at `μ`.
    return emitterlikelihood1D(y[:, 1], σ[:, 1], t, μ[:, 1], a[:, 1], z) *
           emitterlikelihood1D(y[:, 2], σ[:, 2], t, μ[:, 2], a[:, 2], z)
end

"""
    likelihood = emitterlikelihood2D(y::Matrix{Float64},
                                     σ::Matrix{Float64},
                                     t::Vector{Float64},
                                     μ::Vector{Float64},
                                     a::Vector{Float64})

Compute the likelihood of the emitters `μ` given localizations in `y`.

# Description
This function computes the likelihood that the localizations stored in 
`y` arose from an emitter at location `μ` with standard deviations `σ` and
drift velocities `a`.  This function is a wrapper for emitterlikelihood1D()
that multiplies the likelihoods over the 2 spatial dimensions.  I.e., 
likelihood for 1 emitter along 2 dimensions.

# Inputs
-`y`: 2D coordinates of a set of localizations. (pixels)(nlocx2)([y x])
-`σ`: Standard deviations of the observation distributions. 
      (pixels)(nlocx2)([y x])
-`t`: Observation time corresponding to localizations in `y`. (frames)(nlocx1)
-`μ`: Location of the emitters. (pixels)(kx2)([y x])
-`a`: Drift velocities of each emitter. (pixels/frame)(kx2)([y x])

# Outputs
-`likelihood`: Likelihood of the emitter given the localizations.
"""
function emitterlikelihood2D(y::Matrix{Float64},
    σ::Matrix{Float64},
    t::Vector{Float64},
    μ::Vector{Float64},
    a::Vector{Float64})
    # Compute the likelihood that the (2D) localization coordinates in `y`
    # arose from the emitter located at `μ`.
    return emitterlikelihood1D(y[:, 1], σ[:, 1], μ[1] .+ a[1] * t) *
           emitterlikelihood1D(y[:, 2], σ[:, 2], μ[2] .+ a[2] * t)
end

"""
    likelihood = emitterlikelihood2D(y::Vector{Float64},
                                     σ::Vector{Float64},
                                     t::Float64,
                                     μ::Vector{Float64},
                                     a::Vector{Float64})

Compute the likelihood of the emitters `μ` given localizations in `y`.

# Description
This function computes the likelihood that the localization stored in 
`y` arose from an emitter at location `μ` with standard deviation `σ` and
drift velocities `a`.  This function is a wrapper for emitterlikelihood1D()
that multiplies the likelihoods over the 2 spatial dimensions.  I.e., 
likelihood for 1 localization of 1 emitter along 2 dimensions.

# Inputs
-`y`: 2D coordinates of a localization. (pixels)(2x1)([y; x])
-`σ`: Standard deviations of the observation distribution. 
      (pixels)(2x1)([y; x])
-`t`: Observation time corresponding to localization `y`. (frames)
-`μ`: Location of the emitter. (pixels)(2x1)([y; x])
-`a`: Drift velocities of each emitter. (pixels/frame)(2x1)([y; x])

# Outputs
-`likelihood`: Likelihood of the emitter given the localization.
"""
function emitterlikelihood2D(y::Vector{Float64},
    σ::Vector{Float64},
    t::Float64,
    μ::Vector{Float64},
    a::Vector{Float64})
    # Compute the likelihood that the (2D) localization coordinate in `y`
    # arose from the emitter located at `μ`.
    return emitterlikelihood1D([y[1]], [σ[1]], [μ[1] + a[1] * t]) *
           emitterlikelihood1D([y[2]], [σ[2]], [μ[2] + a[2] * t])
end

"""
    likelihood = emitterlikelihood1D(y::Vector{Float64},
                                     σ::Vector{Float64},
                                     t::Vector{Float64},
                                     μ::Vector{Float64},
                                     a::Vector{Float64},
                                     z::Vector{Int})

Compute the likelihood of the emitters `μ` given localizations `y`.

# Description
This function computes the likelihood that the localizations `y` arose from
the emitters they were allocated to (governed by `z`) at the locations `μ`.
I.e., likelihood for k emitters along 1 dimension.

# Inputs
-`y`: 1D coordinate of a set of localizations. (pixels)(nlocx1)
-`σ`: Standard deviations of the observation distributions. (pixels)(nlocx1)
-`t`: Observation time corresponding to localizations in `y`. (frames)(nlocx1)
-`μ`: 1D location of the emitters. (pixels)(kx1)
-`a`: Drift velocities of each emitter. (pixels/frame)(kx1)
-`z`: Allocations of the `nloc` localizations to the `k` emittters. (nlocx1)

# Outputs
-`likelihood`: Likelihood of the emitters given the localizations.
"""
function emitterlikelihood1D(y::Vector{Float64},
    σ::Vector{Float64},
    t::Vector{Float64},
    μ::Vector{Float64},
    a::Vector{Float64},
    z::Vector{Int})
    # Compute the likelihood that the (1D) localization coordinates in `y`
    # arose from the emitters located at `μ` with standard deviation of
    # observations given by `σ`.
    likelihood = 1.0
    k = Base.length(μ)
    for ee = 1:k
        # Determine which of `y` were allocated to emitter `ee`.
        currentbool = z .== ee

        # Compute the likelihood of this emitter given the allocated
        # localizations.
        likelihood *= emitterlikelihood1D(y[currentbool],
            σ[currentbool],
            μ[ee] .+ a[ee] * t[currentbool])
    end

    return likelihood
end

"""
    likelihood = emitterlikelihood1D(y::Vector{Float64}, 
                                     σ::Vector{Float64},
                                     μt::Vector{Float64})

Compute the likelihood of the emitter `μt` given localizations `y`.

# Description
This function computes the likelihood that the localizations `y` arose from
the emitter at position `μt(t)`, assuming that observations of the emitter are
normally distributed. I.e., likelihood for 1 emitter along 1 dimension.

# Inputs
-`y`: 1D coordinate of a set of localizations. (pixels)(nlocx1)
-`σ`: Standard deviations of the observation distribution. (pixels)
-`μt`: Location of the emitter over time. (pixels)(nframesx1)

# Outputs
-`likelihood`: Likelihood of the emitter given the localizations.
"""
function emitterlikelihood1D(y::Vector{Float64},
    σ::Vector{Float64},
    μt::Vector{Float64})
    # Compute the likelihood that the (1D) localization coordinates in `y`
    # arose from the emitter located at `μ(t)`.
    likelihood = 1.0
    for ii = 1:Base.length(y)
        likelihood *= (1 / (σ[ii] * sqrt(2.0 * pi))) * exp(-0.5 * (y[ii] - μt[ii])^2 / (σ[ii]^2))
    end

    return likelihood
end


## Log-likelihoods of emitters given localizations.

"""
    logL = emitterlogL2D(smld::SMLMData.SMLD2D,
                         μ::Matrix{Float64},
                         a::Matrix{Float64},
                         z::Vector{Int})

Compute the log-likelihood of the emitters `μ` given localizations in `smld`.

# Description
This function computes the log-likelihood that the localizations stored in 
`smld` arose from emitters at locations `μ` with standard deviations `σ` and
drift velocities `a`.  I.e., log-likelihood for k emitters along 2 dimensions.

# Inputs
-`y`: 1D coordinate of a set of localizations. (pixels)(nlocx1)
-`μ`: Location of the emitters. (pixels)(kx1)
-`a`: Drift velocities of each emitter. (pixels/frame)(kx2)([y x])
-`z`: Allocations of the `nloc` localizations to the `k` emittters. (nlocx1)

# Outputs
-`logL`: Log-likelihood of the emitters given the localizations.
"""
function emitterlogL2D(smld::SMLMData.SMLD2D,
    μ::Matrix{Float64},
    a::Matrix{Float64},
    z::Vector{Int})
    # Compute the log-likelihood that the (2D) localization coordinates in `y`
    # arose from the emitters located at `μ`.
    return emitterlogL2D([smld.y smld.x], [smld.σ_y smld.σ_x],
        Float64.(smld.framenum), μ, a, z)
end

"""
    logL = emitterlogL2D(y::Matrix{Float64},
                         σ::Matrix{Float64},
                         t::Vector{Float64},
                         μ::Matrix{Float64},
                         a::Matrix{Float64},
                         z::Vector{Int})

Compute the log-likelihood of the emitters `μ` given localizations in `y`.

# Description
This function computes the log-likelihood that the localizations stored in 
`y` arose from emitters at locations `μ` with standard deviations `σ` and
drift velocities `a`.  This function is a wrapper for emitterlogL1D() which 
sums the likelihoods over the 2 spatial dimensions.  I.e., log-likelihood for
k emitters along 2 dimensions.

# Inputs
-`y`: 2D coordinates of a set of localizations. (pixels)(nlocx2)([y x])
-`σ`: Standard deviations of the observation distributions. 
      (pixels)(nlocx2)([y x])
-`t`: Observation time corresponding to localizations in `y`. (frames)(nlocx1)
-`μ`: Location of the emitters. (pixels)(kx2)([y x])
-`a`: Drift velocities of each emitter. (pixels/frame)(kx2)([y x])
-`z`: Allocations of the `nloc` localizations to the `k` emittters. (nlocx1)

# Outputs
-`logL`: Log-likelihood of the emitters given the localizations.
"""
function emitterlogL2D(y::Matrix{Float64},
    σ::Matrix{Float64},
    t::Vector{Float64},
    μ::Matrix{Float64},
    a::Matrix{Float64},
    z::Vector{Int})
    # Compute the log-likelihood that the (2D) localization coordinates in `y`
    # arose from the emitters located at `μ`.
    return emitterlogL1D(y[:, 1], σ[:, 1], t, μ[:, 1], a[:, 1], z) +
           emitterlogL1D(y[:, 2], σ[:, 2], t, μ[:, 2], a[:, 2], z)
end

"""
    logL = emitterlogL2D(y::Matrix{Float64},
                         σ::Matrix{Float64},
                         t::Vector{Float64},
                         μ::Vector{Float64},
                         a::Vector{Float64})

Compute the log-likelihood of the emitters `μ` given localizations in `y`.

# Description
This function computes the log-likelihood that the localizations stored in 
`y` arose from an emitter at location `μ` with standard deviations `σ` and
drift velocities `a`.  This function is a wrapper for emitterlogL1D() which 
sums the likelihoods over the 2 spatial dimensions.  I.e., log-likelihood for
1 emitter along 2 dimensions.

# Inputs
-`y`: 2D coordinates of a set of localizations. (pixels)(nlocx2)([y x])
-`σ`: Standard deviations of the observation distributions. 
      (pixels)(nlocx2)([y x])
-`t`: Observation time corresponding to localizations in `y`. (frames)(nlocx1)
-`μ`: Location of the emitters. (pixels)(2x1)([y; x])
-`a`: Drift velocities of each emitter. (pixels/frame)(2x1)([y; x])

# Outputs
-`logL`: Log-likelihood of the emitter given the localizations.
"""
function emitterlogL2D(y::Matrix{Float64},
    σ::Matrix{Float64},
    t::Vector{Float64},
    μ::Vector{Float64},
    a::Vector{Float64})
    # Compute the log-likelihood that the (2D) localization coordinates in `y`
    # arose from the emitter located at `μ`.
    return emitterlogL1D(y[:, 1], σ[:, 1], μ[1] .+ a[1] * t) +
           emitterlogL1D(y[:, 2], σ[:, 2], μ[2] .+ a[2] * t)
end

"""
    logL = emitterlogL2D(y::Vector{Float64},
                         σ::Vector{Float64},
                         t::Float64,
                         μ::Vector{Float64},
                         a::Vector{Float64})

Compute the log-likelihood of the emitters `μ` given localizations in `y`.

# Description
This function computes the log-likelihood that the localizations stored in 
`y` arose from an emitter at location `μ` with standard deviations `σ` and
drift velocities `a`.  This function is a wrapper for emitterlogL1D() which 
sums the likelihoods over the 2 spatial dimensions.  I.e., log-likelihood for
1 localization of 1 emitter along 2 dimensions.

# Inputs
-`y`: 2D coordinates of a localization. (pixels)(2x1)([y; x])
-`σ`: Standard deviations of the observation distributions. 
        (pixels)(2x1)([y; x])
-`t`: Observation time corresponding to the localization in `y`. (frames)
-`μ`: Location of the emitters. (pixels)(2x1)([y; x])
-`a`: Drift velocities of each emitter. (pixels/frame)(2x1)([y; x])

# Outputs
-`logL`: Log-likelihood of the emitter given the localization.
"""
function emitterlogL2D(y::Vector{Float64},
    σ::Vector{Float64},
    t::Float64,
    μ::Vector{Float64},
    a::Vector{Float64})
    # Compute the log-likelihood that the (2D) localization in `y`
    # arose from the emitter located at `μ`.
    return emitterlogL1D(y[1], σ[1], μ[1] .+ a[1] * t) +
           emitterlogL1D(y[2], σ[2], μ[2] .+ a[2] * t)
end

"""
    logL = emitterlogL1D(y::Vector{Float64},
                         σ::Vector{Float64},
                         t::Vector{Float64},
                         μ::Vector{Float64},
                         a::Vector{Float64},
                         z::Vector{Int})

Compute the log-likelihood of the emitters `μ` given localizations `y`.

# Description
This function computes the log-likelihood that the localizations `y` arose from
the emitters they were allocated to (governed by `z`) at the locations `μ`.
I.e., log-likelihood for k emitters along 1 dimension.

# Inputs
-`y`: 1D coordinate of a set of localizations. (pixels)(nlocx1)
-`σ`: Standard deviations of the observation distributions. (pixels)(nlocx1)
-`t`: Observation time corresponding to localizations in `y`. (frames)(nlocx1)
-`μ`: Location of the emitters. (pixels)(kx1)
-`a`: Drift velocities of each emitter. (pixels/frame)(kx1)
-`z`: Allocations of the `nloc` localizations to the `k` emittters. (nlocx1)

# Outputs
-`logL`: Log-likelihood of the emitters given the localizations.
"""
function emitterlogL1D(y::Vector{Float64},
    σ::Vector{Float64},
    t::Vector{Float64},
    μ::Vector{Float64},
    a::Vector{Float64},
    z::Vector{Int})
    # Compute the log-likelihood that the (1D) localization coordinates in `y`
    # arose from the emitters located at `μ` with standard deviation of
    # observations given by `σ`.
    logL = 0.0
    k = Base.length(μ)
    for ee = 1:k
        # Determine which of `y` were allocated to emitter `ee`.
        currentbool = z .== ee

        # Compute the likelihood of this emitter given the allocated
        # localizations.
        logL += emitterlogL1D(y[currentbool],
            σ[currentbool],
            μ[ee] .+ a[ee] * t[currentbool])
    end

    return logL
end

"""
    logL = emitterlogL1D(y::Vector{Float64}, 
                         σ::Vector{Float64},
                         μt::Vector{Float64})

Compute the log-likelihood of the emitter `μt` given localizations `y`.

# Description
This function computes the log-likelihood that the localizations `y` arose from
the emitter at position `μt(t)`, assuming that observations of the emitter are
normally distributed. I.e., log-likelihood for 1 emitter along 1 dimension.

# Inputs
-`y`: 1D coordinate of a set of localizations. (pixels)(nlocx1)
-`σ`: Standard deviations of the observation distribution. (pixels)
-`μt`: Location of the emitter over time. (pixels)(nframesx1)

# Outputs
-`logL`: Log-likelihood of the emitter given the localizations.
"""
function emitterlogL1D(y::Vector{Float64},
    σ::Vector{Float64},
    μt::Vector{Float64})
    # Compute the log-likelihood that the (1D) localization coordinates in `y`
    # arose from the emitter located at `μ(t)`.
    logL = 0.0
    for ii = 1:Base.length(y)
        logL += emitterlogL1D(y[ii], σ[ii], μt[ii])
    end

    return logL
end

"""
    logL = emitterlogL1D(y::Float64, 
                         σ::Float64,
                         μt::Float64}

Compute the log-likelihood of the emitter `μt` given localization `y`.

# Description
This function computes the log-likelihood that the localization `y` arose from
the emitter at position `μt(t)`, assuming that observations of the emitter are
normally distributed. I.e., log-likelihood for 1 localization of 1 emitter 
along 1 dimension.

# Inputs
-`y`: 1D coordinate of a set of localizations. (pixels)
-`σ`: Standard deviations of the observation distribution. (pixels)
-`μt`: Location of the emitter over time. (pixels)

# Outputs
-`logL`: Log-likelihood of the emitter given the localization.
"""
function emitterlogL1D(y::Float64,
    σ::Float64,
    μt::Float64)
    return -0.5 * (log(2.0 * pi * σ^2) + ((y - μt)^2) / (σ^2))
end

"""
    rndmatrix = gammarnd(α::Float64, β::Vector{Float64}, n::Int)

Make `n` samples of the Gamma PDF for the length(α) Gamma distributions.

# Inputs
-`α`: Vector of shape parameters for the Gamma distribution.
-`β`: Scale parameter for the Gamma distribution.
-`n`: Number of samples to make.

# Outputs
-`x`: Matrix of the random samples. (length(α)xn)
"""
function gammarnd(α::Float64, β::Vector{Float64}, n::Int)
    x = Matrix{Float64}(undef, length(β), n)
    for ii = 1:length(β)
        x[ii, :] = Distributions.rand(Gamma(α, β[ii]), n)
    end

    return x
end

"""
    rndmatrix = gammarnd(α::Vector{Float64}, β::Float64, n::Int)

Make `n` samples of the Gamma PDF for the length(α) Gamma distributions.

# Inputs
-`α`: Vector of shape parameters for the Gamma distribution.
-`β`: Scale parameter for the Gamma distribution.
-`n`: Number of samples to make.

# Outputs
-`x`: Matrix of the random samples. (length(α)xn)
"""
function gammarnd(α::Vector{Float64}, β::Float64, n::Int)
    x = Matrix{Float64}(undef, length(α), n)
    for ii = 1:length(α)
        x[ii, :] = Distributions.rand(Gamma(α[ii], β), n)
    end

    return x
end

"""
    pdf = gammapdf(α::Vector{Float64}, β::Float64, x::Vector{Float64})

Compute the Gamma PDF for the length(α) Gamma distributions at points `x`.

# Inputs
-`α`: Vector of shape parameters for the Gamma distribution.
-`β`: Scale parameter for the Gamma distribution.
-`x`: Sampling point for each `α` (length(α) vector)

# Outputs
-`pdf`: Vector of the pdf samples. (length(α)x1)
"""
function gammapdf(α::Vector{Float64}, β::Float64, x::Vector{Float64})
    pdf = Vector{Float64}(undef, length(α))
    for ii = 1:length(α)
        pdf[ii] = Distributions.pdf(Gamma(α[ii], β), x[ii])
    end

    return pdf
end

"""
    pdf = gammapdf(α::Float64, β::Vector{Float64}, x::Vector{Float64})

Compute the Gamma PDF for the length(β) Gamma distributions at points `x`.

# Inputs
-`α`: Shape parameter for the Gamma distribution.
-`β`: Vector of scale parameters for the Gamma distribution.
-`x`: Sampling point for each `α` (length(β) vector)

# Outputs
-`pdf`: Vector of the pdf samples. (length(β)x1)
"""
function gammapdf(α::Float64, β::Vector{Float64}, x::Vector{Float64})
    pdf = Vector{Float64}(undef, length(β))
    for ii = 1:length(α)
        pdf[ii] = Distributions.pdf(Gamma(α, β[ii]), x[ii])
    end

    return pdf
end

"""
    compressrange!(ints::Vector{Int})

Compress the set of `ints` to consist of integers 1:length(unique(ints))

# Inputs
-`ints`: Vector of integers to be compressed.

# Example 
ints = [2; 4; 4; 7; 4; 11]
compressrange!(ints)
    -> ints = [1; 2; 2; 3; 2; 4]
"""
function compressrange!(ints::Vector{Int})
    # Loop through entries of `ints` and modify as needed.
    intsunique = unique(ints)
    for nn = 1:length(intsunique)
        ints[ints.==intsunique[nn]] .= nn
    end

    return ints
end

"""
    intscomp, intsunique = compressrange(ints::Vector{Int})

Compress the set of `ints` to consist of integers 1:length(unique(ints))

# Inputs
-`ints`: Vector of integers.

# Outputs
-`intscomp`: Input `ints` with entries modified to consist of the integers
             1:length(unique(ints)).
-`intsunique`: Equivalent to `unique(ints)`, returned for convenience.
"""
function compressrange(ints::Vector{Int})
    # Loop through entries of `ints` and modify as needed.
    intsunique = unique(ints)
    intscomp = Vector{Int}(undef, length(ints))
    for nn = 1:length(intsunique)
        intscomp[ints.==intsunique[nn]] .= nn
    end

    return intscomp, intsunique
end
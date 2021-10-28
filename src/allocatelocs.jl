using SMLMData
using Distributions
using NearestNeighbors

"""
    allocatelocs(smld::SMLMData.SMLD2D, 
                 μ::Matrix{Float64}, 
                 a::Vector{Float64} = [0.0; 0.0])

Allocate localizations to emitters.

# Description
This function allocates localizations in `smld` to the emitters located at
the positions `μ` at time t=0.

# Inputs
-`smld`: SMLMData.SMLD2D data structure containing localizations.
-`μ`: Coordinates of the proposed emitter positions. (pixels)([x y])
-`a`: Drift velocity of the localizations in `smld`. (pixels/frame)
"""
function allocatelocs(smld::SMLMData.SMLD2D, 
                      μ::Matrix{Float64}, 
                      a::Vector{Float64} = [0.0; 0.0])
    # Loop through localizations in `smld` and allocate to emitters using 
    # Gibbs sampling.
    nlocs = SMLMData.length(smld)
    zprime = Vector{Int}(undef, nlocs)
    for nn = 1:nlocs
        posterior = SMLMBaGoL.posterior_allocations([smld.x[nn]; smld.y[nn]], 
                                                    [smld.σ_x[nn]; smld.σ_y[nn]], 
                                                    Float64(smld.framenum[nn]),
                                                    μ, 
                                                    a)
        zprime[nn] = rand(posterior)
    end

    return zprime
end

"""
    posterior_allocations(y::Vector{Float64}, 
                          σ_y::Vector{Float64}, 
                          t::Float64, 
                          μ::Matrix{Float64},
                          a::Vector{Float64})

Construct a posterior distribution of allocations of a localization.

# Description
This function constructs a posterior distribution of the allocation of
the localization defined by `y`, `σ_x`, and `t` to the emitters at
positions `μ` at time t=0.

# Inputs
-`y`: Coordinates of a 2D localization. (pixels)([x; y])
-`σ_y`: Standard error of the localization `y`. (pixels)([x; y])
-`t`: Time of observation of localization `y`. (frame)
-`μ`: Coordinates of the proposed emitter positions. (pixels)([x y])
-`a`: Drift velocity of the localizations in `smld`. (pixels/frame)([α_x; α_y])
"""
function posterior_allocations(y::Vector{Float64}, 
                               σ_y::Vector{Float64}, 
                               t::Float64, 
                               μ::Matrix{Float64},
                               a::Vector{Float64})
    # Construct a normalized posterior for the allocations that we can sample
    # from.
    pmf = (1/sqrt(2*pi*σ_y[1]^2)) .* exp.(-(y[1].-μ[:, 1].-a[1]*t).^2 / (2*σ_y[1]^2)) .*
          (1/sqrt(2*pi*σ_y[2]^2)) .* exp.(-(y[2].-μ[:, 2].-a[2]*t).^2 / (2*σ_y[2]^2))
    pmf = pmf ./ sum(pmf)
    
    # If any of `pmf` is NaN, it was not normalizable numerically so we'll just
    # allocate the localization to its nearest-neighbor emitter.
    if any(isnan.(pmf))
        kdtree = NearestNeighbors.KDTree(μ')
        nnindex, _ = NearestNeighbors.knn(kdtree, y, 1, true)
        pmf = zeros(Float64, length(pmf))
        pmf[nnindex[1]] = 1.0
    end

    return Distributions.DiscreteNonParametric(1:length(pmf), pmf)
end





## Log-likelihood kernels of allocations.
"""
    logLalloc_kernel(y::Matrix{Float64},
                     σ::Matrix{Float64},
                     t::Vector{Float64},
                     μ::Matrix{Float64},
                     a::Matrix{Float64},
                     w::Vector{Float64})

Compute the unnormalized log-likelihood of allocating `y` to emitters `μ`.

# Description
This method computes the unnormalized log-likelihood of allocating the
localizations (`y`, `σ`, `t`) to the emitters at location `μ+at`.  I.e., this
method computes the log of the kernel of the distribution P_alloc(Z).

# Inputs
-`y`: 2D coordinates of localizations. (pixels)(nlocx2)([x y])
-`σ`: Standard deviations of the observation distributions. 
      (pixels)(nlocx2)([x y])
-`t`: Observation times corresponding to localizations `y`. (frames)(nlocx1)
-`μ`: Location of the emitter. (pixels)(kx2)([x y])
-`a`: Drift velocities of each emitter. (pixels/frame)(kx2)([x y])
-`w`: Relative weighting of the emitters. (kx1)
"""
function logLalloc_kernel(y::Matrix{Float64},
                          σ::Matrix{Float64},
                          t::Vector{Float64},
                          μ::Matrix{Float64},
                          a::Matrix{Float64},
                          w::Vector{Float64})
    # Compute the unnormalized probabilites of allocating localizations to 
    # the provided emitters.
    p_kernel = Vector{Float64}(undef, size(y, 1))
    for jj = 1:length(w)
        p_kernel += palloc_kernel(y, σ, t, μ[jj, :], a[jj, :], w[jj])
    end

    # Return the log-likelihood of the given allocations.
    return sum(log.(p_kernel))
end





## Probability kernels of allocations.
"""
    palloc_kernel(y::Matrix{Float64},
                  σ::Matrix{Float64},
                  t::Vector{Float64},
                  μ::Matrix{Float64},
                  a::Matrix{Float64},
                  w::Vector{Float64})

Compute the unnormalized probability of allocating `y` to emitters `μ`.

# Description
This method computes the unnormalized probability of allocating the
localizations (`y`, `σ`, `t`) to the emitters at location `μ+at`.  I.e., this
method computes the kernel of the distribution P_alloc(Z).

# Inputs
-`y`: 2D coordinates of localizations. (pixels)(nlocx2)([x y])
-`σ`: Standard deviations of the observation distributions. 
      (pixels)(nlocx2)([x y])
-`t`: Observation times corresponding to localizations `y`. (frames)(nlocx1)
-`μ`: Location of the emitter. (pixels)(kx2)([x y])
-`a`: Drift velocities of each emitter. (pixels/frame)(kx2)([x y])
-`w`: Relative weighting of the emitters. (kx1)
"""
function palloc_kernel(y::Matrix{Float64},
                       σ::Matrix{Float64},
                       t::Vector{Float64},
                       μ::Matrix{Float64},
                       a::Matrix{Float64},
                       w::Vector{Float64})
    # Compute the unnormalized probability of allocating localizations to the 
    # provided emitters.
    p_kernel = Vector{Float64}(undef, size(y, 1))
    for jj = 1:length(w)
        p_kernel += palloc_kernel(y, σ, t, μ[jj, :], a[jj, :], w[jj])
    end

    return prod(p_kernel)
end

"""
    palloc_kernel(y::Matrix{Float64},
                  σ::Matrix{Float64},
                  t::Vector{Float64},
                  μ::Vector{Float64},
                  a::Vector{Float64},
                  w::Float64)

Compute the unnormalized probability of allocating `y` to emitter `μ`.

# Description
This method computes the unnormalized probabilities of allocating the
localizations (`y`, `σ`, `t`) to the emitter at location `μ+at`.  I.e., this
method computes the kernel of the distribution P_alloc(Z_i|j).

# Inputs
-`y`: 2D coordinates of localizations. (pixels)(nlocx2)([x y])
-`σ`: Standard deviations of the observation distributions. 
      (pixels)(nlocx2)([x y])
-`t`: Observation times corresponding to localizations `y`. (frames)(nlocx1)
-`μ`: Location of the emitter. (pixels)(2x1)([x; y])
-`a`: Drift velocities of each emitter. (pixels/frame)(2x1)([x; y])
-`w`: Relative weighting of the emitter.
"""
function palloc_kernel(y::Matrix{Float64},
                       σ::Matrix{Float64},
                       t::Vector{Float64},
                       μ::Vector{Float64},
                       a::Vector{Float64},
                       w::Float64)
    # Compute the unnormalized probability of allocating localizations to the 
    # provided emitter.
    nloc = size(y, 1)
    p_kernel = Vector{Float64}(undef, nloc)
    for ii = 1:nloc
        p_kernel[ii] = palloc_kernel(y[ii, :], σ[ii, :], t[ii], μ, a, w)
    end

    return p_kernel
end

"""
    palloc_kernel(y::Vector{Float64},
                  σ::Vector{Float64},
                  t::Float64,
                  μ::Vector{Float64},
                  a::Vector{Float64},
                  w::Float64)

Compute the unnormalized probability of allocating `y` to emitter `μ`.

# Description
This method computes the unnormalized probability of allocating the
localization (`y`, `σ`, `t`) to the emitter at location `μ+at`.  I.e., this
method computes the kernel of the distribution P_alloc(Z_i=j) (the 
normalization factor is the same for {j=1:k | P_alloc(Z_i=j)}).

# Inputs
-`y`: 2D coordinates of a localization. (pixels)(2x1)([x; y])
-`σ`: Standard deviations of the observation distribution. 
      (pixels)(2x1)([x; y])
-`t`: Observation time corresponding to localization `y`. (frames)
-`μ`: Location of the emitter. (pixels)(2x1)([x; y])
-`a`: Drift velocities of each emitter. (pixels/frame)(2x1)([x; y])
-`w`: Relative weighting of the emitter.
"""
function palloc_kernel(y::Vector{Float64},
                       σ::Vector{Float64},
                       t::Float64,
                       μ::Vector{Float64},
                       a::Vector{Float64},
                       w::Float64)
    # Compute the unnormalized probability of allocating this localization to
    # the provided emitter.
    p_kernel = w * emitterlikelihood2D(y, σ, t, μ, a)

    return p_kernel
end








## Likelihoods of emitters given localizations.

"""
    emitterlikelihood2D(smld::SMLMData.SMLD2D,
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
-`a`: Drift velocities of each emitter. (pixels/frame)(kx2)([x y])
-`z`: Allocations of the `nloc` localizations to the `k` emittters. (nlocx1)
"""
function emitterlikelihood2D(smld::SMLMData.SMLD2D,
                             μ::Matrix{Float64},
                             a::Matrix{Float64},
                             z::Vector{Int})
    # Compute the likelihood that the (2D) localization coordinates in `y`
    # arose from the emitters located at `μ`.
    return emitterlikelihood2D([smld.x smld.y], [smld.σ_x smld.σ_y], 
                               Float64.(smld.framenum), μ, a, z)
end

"""
    emitterlikelihood2D(y::Matrix{Float64},
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
-`y`: 2D coordinates of a set of localizations. (pixels)(nlocx2)([x y])
-`σ`: Standard deviations of the observation distributions. 
      (pixels)(nlocx2)([x y])
-`t`: Observation time corresponding to localizations in `y`. (frames)(nlocx1)
-`μ`: Location of the emitters. (pixels)(kx2)([x y])
-`a`: Drift velocities of each emitter. (pixels/frame)(kx2)([x y])
-`z`: Allocations of the `nloc` localizations to the `k` emittters. (nlocx1)
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
    emitterlikelihood2D(y::Matrix{Float64},
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
-`y`: 2D coordinates of a set of localizations. (pixels)(nlocx2)([x y])
-`σ`: Standard deviations of the observation distributions. 
      (pixels)(nlocx2)([x y])
-`t`: Observation time corresponding to localizations in `y`. (frames)(nlocx1)
-`μ`: Location of the emitters. (pixels)(kx2)([x y])
-`a`: Drift velocities of each emitter. (pixels/frame)(kx2)([x y])
"""
function emitterlikelihood2D(y::Matrix{Float64},
                             σ::Matrix{Float64},
                             t::Vector{Float64},
                             μ::Vector{Float64},
                             a::Vector{Float64})
    # Compute the likelihood that the (2D) localization coordinates in `y`
    # arose from the emitter located at `μ`.
    return emitterlikelihood1D(y[:, 1], σ[:, 1], μ[1] .+ a[1]*t) *
        emitterlikelihood1D(y[:, 2], σ[:, 2], μ[2] .+ a[2]*t)
end

"""
    emitterlikelihood2D(y::Vector{Float64},
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
-`y`: 2D coordinates of a localization. (pixels)(2x1)([x; y])
-`σ`: Standard deviations of the observation distribution. 
      (pixels)(2x1)([x; y])
-`t`: Observation time corresponding to localization `y`. (frames)
-`μ`: Location of the emitter. (pixels)(2x1)([x; y])
-`a`: Drift velocities of each emitter. (pixels/frame)(2x1)([x; y])
"""
function emitterlikelihood2D(y::Vector{Float64},
                             σ::Vector{Float64},
                             t::Float64,
                             μ::Vector{Float64},
                             a::Vector{Float64})
    # Compute the likelihood that the (2D) localization coordinate in `y`
    # arose from the emitter located at `μ`.
    return emitterlikelihood1D([y[1]], [σ[1]], [μ[1] + a[1]*t]) *
        emitterlikelihood1D([y[2]], [σ[2]], [μ[2] + a[2]*t])
end

"""
    emitterlikelihood1D(y::Vector{Float64},
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
-`μ`: Location of the emitters. (pixels)(kx1)
-`a`: Drift velocities of each emitter. (pixels/frame)(kx2)([x y])
-`z`: Allocations of the `nloc` localizations to the `k` emittters. (nlocx1)
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
    k = length(μ)
    for ee = 1:k
        # Determine which of `y` were allocated to emitter `ee`.
        currentbool = z .== ee

        # Compute the likelihood of this emitter given the allocated
        # localizations.
        likelihood *= emitterlikelihood1D(y[currentbool], 
                                          σ[currentbool],
                                          μ[ee] .+ a[ee]*t[currentbool])
    end

    return likelihood
end

"""
    emitterlikelihood1D(y::Vector{Float64}, 
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
"""
function emitterlikelihood1D(y::Vector{Float64}, 
                             σ::Vector{Float64},
                             μt::Vector{Float64})
    # Compute the likelihood that the (1D) localization coordinates in `y`
    # arose from the emitter located at `μ(t)`.
    likelihood = 1.0
    for ii = 1:length(y)
        likelihood *= (1/(σ[ii]*sqrt(2.0*pi))) * exp(-0.5*(y[ii]-μt[ii])^2 / (σ[ii]^2))
    end

    return likelihood
end








## Log-likelihoods of emitters given localizations.

"""
    emitterlogL2D(smld::SMLMData.SMLD2D,
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
-`a`: Drift velocities of each emitter. (pixels/frame)(kx2)([x y])
-`z`: Allocations of the `nloc` localizations to the `k` emittters. (nlocx1)
"""
function emitterlogL2D(smld::SMLMData.SMLD2D,
                       μ::Matrix{Float64},
                       a::Matrix{Float64},
                       z::Vector{Int})
    # Compute the log-likelihood that the (2D) localization coordinates in `y`
    # arose from the emitters located at `μ`.
    return emitterlogL2D([smld.x smld.y], [smld.σ_x smld.σ_y], 
                         Float64.(smld.framenum), μ, a, z)
end

"""
    emitterlogL2D(y::Matrix{Float64},
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
-`y`: 2D coordinates of a set of localizations. (pixels)(nlocx2)([x y])
-`σ`: Standard deviations of the observation distributions. 
      (pixels)(nlocx2)([x y])
-`t`: Observation time corresponding to localizations in `y`. (frames)(nlocx1)
-`μ`: Location of the emitters. (pixels)(kx2)([x y])
-`a`: Drift velocities of each emitter. (pixels/frame)(kx2)([x y])
-`z`: Allocations of the `nloc` localizations to the `k` emittters. (nlocx1)
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
    emitterlogL2D(y::Matrix{Float64},
                  σ::Matrix{Float64},
                  t::Vector{Float64},
                  μ::Matrix{Float64},
                  a::Matrix{Float64})

Compute the log-likelihood of the emitters `μ` given localizations in `y`.

# Description
This function computes the log-likelihood that the localizations stored in 
`y` arose from an emitter at location `μ` with standard deviations `σ` and
drift velocities `a`.  This function is a wrapper for emitterlogL1D() which 
sums the likelihoods over the 2 spatial dimensions.  I.e., log-likelihood for
1 emitter along 2 dimensions.

# Inputs
-`y`: 2D coordinates of a set of localizations. (pixels)(nlocx2)([x y])
-`σ`: Standard deviations of the observation distributions. 
      (pixels)(nlocx2)([x y])
-`t`: Observation time corresponding to localizations in `y`. (frames)(nlocx1)
-`μ`: Location of the emitters. (pixels)(2x1)([x; y])
-`a`: Drift velocities of each emitter. (pixels/frame)(2x1)([x; y])
"""
function emitterlogL2D(y::Matrix{Float64},
                       σ::Matrix{Float64},
                       t::Vector{Float64},
                       μ::Vector{Float64},
                       a::Vector{Float64})
    # Compute the log-likelihood that the (2D) localization coordinates in `y`
    # arose from the emitter located at `μ`.
    return emitterlogL1D(y[:, 1], σ[:, 1], μ[1] .+ a[1]*t) +
        emitterlogL1D(y[:, 2], σ[:, 2], μ[2] .+ a[2]*t)
end

"""
# Description
This function computes the log-likelihood that the localizations stored in 
`y` arose from an emitter at location `μ` with standard deviations `σ` and
drift velocities `a`.  This function is a wrapper for emitterlogL1D() which 
sums the likelihoods over the 2 spatial dimensions.  I.e., log-likelihood for
1 localizations of 1 emitter along 2 dimensions.

# Inputs
-`y`: 2D coordinates of a localization. (pixels)(2x1)([x; y])
-`σ`: Standard deviations of the observation distributions. 
        (pixels)(2x1)([x; y])
-`t`: Observation time corresponding to the localization in `y`. (frames)
-`μ`: Location of the emitters. (pixels)(2x1)([x; y])
-`a`: Drift velocities of each emitter. (pixels/frame)(2x1)([x; y])
"""
function emitterlogL2D(y::Vector{Float64},
                       σ::Vector{Float64},
                       t::Float64,
                       μ::Vector{Float64},
                       a::Vector{Float64})
    # Compute the log-likelihood that the (2D) localization in `y`
    # arose from the emitter located at `μ`.
    return emitterlogL1D(y[1], σ[1], μ[1] .+ a[1]*t) +
        emitterlogL1D(y[2], σ[2], μ[2] .+ a[2]*t)
end

"""
    emitterlogL1D(y::Vector{Float64},
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
-`a`: Drift velocities of each emitter. (pixels/frame)(kx2)([x y])
-`z`: Allocations of the `nloc` localizations to the `k` emittters. (nlocx1)
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
    k = length(μ)
    for ee = 1:k
        # Determine which of `y` were allocated to emitter `ee`.
        currentbool = z .== ee

        # Compute the likelihood of this emitter given the allocated
        # localizations.
        logL += emitterlogL1D(y[currentbool], 
                              σ[currentbool],
                              μ[ee] .+ a[ee]*t[currentbool])
    end

    return logL
end

"""
    emitterlogL1D(y::Vector{Float64}, 
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
"""
function emitterlogL1D(y::Vector{Float64}, 
                       σ::Vector{Float64},
                       μt::Vector{Float64})
    # Compute the log-likelihood that the (1D) localization coordinates in `y`
    # arose from the emitter located at `μ(t)`.
    logL = 0.0
    for ii = 1:length(y)
        logL += emitterlogL1D(y[ii], σ[ii], μt[ii])
    end

    return logL
end

"""
    emitterlogL1D(y::Float64, 
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
"""
function emitterlogL1D(y::Float64, 
                       σ::Float64,
                       μt::Float64)
    return -0.5 * (log(2.0*pi*σ^2) + ((y-μt)^2)/(σ^2))
end
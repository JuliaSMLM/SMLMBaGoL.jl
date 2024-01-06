using SMLMData
using Distributions
using NearestNeighbors

# This file contains functions related to the allocation of localizations to
# emitters.

"""
    zvalid, zunique, k = validifyallocs(z::Vector{Int})

Compress the range of `z` to consist only of the integers `1:k`.

# Inputs
- `z`: Set of allocations to emitters associated to integer indices.

# Outputs
- `zvalid`: Set of allocations to emitters where integer indices are 
            compressed to the range `1:k`.
- `zunique`: Equivalent to `unique(z)`, returned for convenience since it is
             computed internally but often used outside of this method.
- `k`: Number of emitters with valid allocations.
"""
function validifyallocs(z::Vector{Int})
    # Ensure that the allocations in `z` are "complete", i.e., each of the
    # `k` emitters was allocated at least one localization.
    zvalid, zunique = SMLMBaGoL.compressrange(z)
    k = maximum(zvalid)

    return zvalid, zunique, k
end

"""
    zvalid, μvalid, avalid, k = isolateuseful(z::Vector{Int}, 
                                              μ::Matrix{<:Real}, 
                                              a::Matrix{<:Real})

Keep only those emitters in `μ` and `a` with allocated localizations.

# Inputs
- `z`: Set of allocations to emitters associated to integer indices.
- `μ`: Positions of emitters indexed by entries of `z`. ([y x])
- `a`: Drift velocities of emitters indexed by entries of `z`. ([v_y v_x])

# Outputs
- `zvalid`: Set of allocations to emitters where integer indices are 
            compressed to the range `1:k`.
- `μvalid`: Positions of emitters with localizations allocated to them.
- `avalid`: Drift velocities of emitters with localizations allocated to them.
- `k`: Number of emitters with valid allocations.
"""
function isolateuseful(z::Vector{Int}, μ::Matrix{<:Real}, a::Matrix{<:Real})
    # Determine which emitters should be kept (i.e., which ones have 
    # localizations allocated to them).
    zvalid, zunique, k = SMLMBaGoL.validifyallocs(z)

    return zvalid, μ[zunique, :], a[zunique, :], k
end

"""
    statevalid = isolateuseful(state::SMLMBaGoL.BaGoLState2D)

Keep only those emitters in `state` with allocated localizations.

# Inputs
- `state`: State with fields `μ`, `z`, and `a`.

# Outputs
- `statevalid`: Set of emitters from the input `state` which had allocations.
"""
function isolateuseful(state::SMLMBaGoL.BaGoLState2D)
    # Determine which emitters should be kept (i.e., which ones have 
    # localizations allocated to them).
    zvalid, μvalid, avalid, k = SMLMBaGoL.isolateuseful(state.z, state.μ, state.a)

    return SMLMBaGoL.BaGoLState2D(k, zvalid, μvalid, avalid)
end

"""
    isolateuseful!(state::SMLMBaGoL.BaGoLState2D)

Keep only those emitters in `state` with allocated localizations.

# Inputs
- `state`: State with fields `μ`, `z`, and `a`.
"""
function isolateuseful!(state::SMLMBaGoL.BaGoLState2D)
    # Determine which emitters should be kept (i.e., which ones have 
    # localizations allocated to them).
    state.z, state.μ, state.a, state.k = SMLMBaGoL.isolateuseful(
        state.z, state.μ, state.a)
end

"""
    zprime = allocatelocs(smld::SMLMData.SMLD2D, 
                          μ::Matrix{<:Real}, 
                          a::Matrix{<:Real})

Allocate localizations to emitters.

# Description
This function allocates localizations in `smld` to the emitters located at
the positions `μ` at time t=0.

# Inputs
- `smld`: SMLMData.SMLD2D data structure containing localizations.
- `μ`: Coordinates of the proposed emitter positions. (pixels)([y x])
- `a`: Drift velocity of the emitters `μ`. (pixels/frame)([a_y a_x])

# Outputs
- `zprime`: Array of emitter indices defining the allocations of `smld` 
            localizations.  For example, if `zprime[n] = k`, the `n-th`
            localization of `smld` was allocated to the `k-th` emitter
            (i.e., row `k` of `μ`).
"""
function allocatelocs(smld::SMLMData.SMLD2D, 
                      μ::Matrix{<:Real}, 
                      a::Matrix{<:Real})
    # Loop through localizations in `smld` and allocate to emitters using 
    # Gibbs sampling.
    nlocs = Base.length(smld)
    zprime = Vector{Int}(undef, nlocs)
    for nn = 1:nlocs
        posterior = SMLMBaGoL.posterior_allocations([smld.y[nn]; smld.x[nn]], 
                                                    [smld.σ_y[nn]; smld.σ_x[nn]], 
                                                    Float32(smld.framenum[nn]),
                                                    μ, 
                                                    a)
        zprime[nn] = Distributions.rand(posterior)
    end

    return zprime
end

"""
    posterior = posterior_allocations(y::Vector{<:Real}, 
                                      σ_y::Vector{<:Real}, 
                                      t::Real, 
                                      μ::Matrix{<:Real},
                                      a::Matrix{<:Real})

Construct a posterior distribution of allocations of a localization.

# Description
This function constructs a posterior distribution of the allocation of
the localization defined by `y`, `σ_x`, and `t` to the emitters at
positions `μ` at time t=0.

# Inputs
- `y`: Coordinates of a 2D localization. (pixels)([y; x])
- `σ_y`: Standard error of the localization `y`. (pixels)([y; x])
- `t`: Time of observation of localization `y`. (frame)
- `μ`: Coordinates of the proposed emitter positions. (pixels)([y x])
- `a`: Drift velocity of the emitters `μ`. (pixels/frame)([a_y a_x])

# Outputs
- `posterior`: A Distributions.Distribution defining the allocation posterior.
"""
function posterior_allocations(y::Vector{<:Real}, 
                               σ_y::Vector{<:Real}, 
                               t::Real, 
                               μ::Matrix{<:Real},
                               a::Matrix{<:Real})
    # Construct a normalized posterior for the allocations that we can sample
    # from.
    pmf = (1/sqrt(2*pi*σ_y[1]^2)) .* exp.(-(y[1].-μ[:, 1].-a[:, 1]*t).^2 / (2*σ_y[1]^2)) .*
          (1/sqrt(2*pi*σ_y[2]^2)) .* exp.(-(y[2].-μ[:, 2].-a[:, 2]*t).^2 / (2*σ_y[2]^2))
    pmf = pmf ./ sum(pmf)
    
    # If any of `pmf` is NaN, we'll just allocate the localization to its
    # nearest-neighbor emitter.  If the `pmf` is not normalizable, we'll 
    # instead...? not sure yet!
    if any(isnan.(pmf))
        kdtree = NearestNeighbors.KDTree(μ')
        nnindex, _ = NearestNeighbors.knn(kdtree, y, 1, true)
        pmf = zeros(Float32, Base.length(pmf))
        pmf[nnindex[1]] = 1.0
    end

    return Distributions.DiscreteNonParametric(1:Base.length(pmf), pmf)
end





## Log-likelihood kernels of allocations.
"""
    logL = logLalloc_kernel(y::Matrix{<:Real},
                            σ::Matrix{<:Real},
                            t::Vector{<:Real},
                            μ::Matrix{<:Real},
                            a::Matrix{<:Real},
                            w::Vector{<:Real})

Compute the unnormalized log-likelihood of allocating `y` to emitters `μ`.

# Description
This method computes the unnormalized log-likelihood of allocating the
localizations (`y`, `σ`, `t`) to the emitters at location `μ+at`.  I.e., this
method computes the log of the kernel of the distribution P_alloc(Z).

# Inputs
- `y`: 2D coordinates of localizations. (pixels)(nlocx2)([y x])
- `σ`: Standard deviations of the observation distributions. 
       (pixels)(nlocx2)([y x])
- `t`: Observation times corresponding to localizations `y`. (frames)(nlocx1)
- `μ`: Location of the emitter. (pixels)(kx2)([y x])
- `a`: Drift velocities of each emitter. (pixels/frame)(kx2)([y x])
- `w`: Relative weighting of the emitters. (kx1)

# Outputs
- `logL`: Log-likelihood kernel of the allocation distribution.
"""
function logLalloc_kernel(y::Matrix{<:Real},
                          σ::Matrix{<:Real},
                          t::Vector{<:Real},
                          μ::Matrix{<:Real},
                          a::Matrix{<:Real},
                          w::Vector{<:Real})
    # Compute the unnormalized probabilites of allocating localizations to 
    # the provided emitters.
    p_kernel = Vector{Float32}(undef, size(y, 1))
    for jj = 1:Base.length(w)
        p_kernel += palloc_kernel(y, σ, t, μ[jj, :], a[jj, :], w[jj])
    end

    # Return the log-likelihood of the given allocations.
    return sum(log.(p_kernel))
end





## Probability kernels of allocations.
"""
    p = palloc_kernel(y::Matrix{<:Real},
                      σ::Matrix{<:Real},
                      t::Vector{<:Real},
                      μ::Matrix{<:Real},
                      a::Matrix{<:Real},
                      w::Vector{<:Real})

Compute the unnormalized probability of allocating `y` to emitters `μ`.

# Description
This method computes the unnormalized probability of allocating the
localizations (`y`, `σ`, `t`) to the emitters at location `μ+at`.  I.e., this
method computes the kernel of the distribution P_alloc(Z).

# Inputs
- `y`: 2D coordinates of localizations. (pixels)(nlocx2)([y x])
- `σ`: Standard deviations of the observation distributions. 
       (pixels)(nlocx2)([y x])
- `t`: Observation times corresponding to localizations `y`. (frames)(nlocx1)
- `μ`: Location of the emitter. (pixels)(kx2)([y x])
- `a`: Drift velocities of each emitter. (pixels/frame)(kx2)([y x])
- `w`: Relative weighting of the emitters. (kx1)

# Outputs
- `p`: Probability kernel of the allocation probability.
"""
function palloc_kernel(y::Matrix{<:Real},
                       σ::Matrix{<:Real},
                       t::Vector{<:Real},
                       μ::Matrix{<:Real},
                       a::Matrix{<:Real},
                       w::Vector{<:Real})
    # Compute the unnormalized probability of allocating localizations to the 
    # provided emitters.
    p_kernel = Vector{Float32}(undef, size(y, 1))
    for jj = 1:Base.length(w)
        p_kernel += palloc_kernel(y, σ, t, μ[jj, :], a[jj, :], w[jj])
    end

    return prod(p_kernel)
end

"""
    p = palloc_kernel(y::Matrix{<:Real},
                      σ::Matrix{<:Real},
                      t::Vector{<:Real},
                      μ::Vector{<:Real},
                      a::Vector{<:Real},
                      w::Real)

Compute the unnormalized probability of allocating `y` to emitter `μ`.

# Description
This method computes the unnormalized probabilities of allocating the
localizations (`y`, `σ`, `t`) to the emitter at location `μ+at`.  I.e., this
method computes the kernel of the distribution P_alloc(Z_i|j).

# Inputs
- `y`: 2D coordinates of localizations. (pixels)(nlocx2)([y x])
- `σ`: Standard deviations of the observation distributions. 
       (pixels)(nlocx2)([y x])
- `t`: Observation times corresponding to localizations `y`. (frames)(nlocx1)
- `μ`: Location of the emitter. (pixels)(2x1)([y; x])
- `a`: Drift velocities of each emitter. (pixels/frame)(2x1)([y; x])
- `w`: Relative weighting of the emitter.

# Outputs
-`p`: Probability kernel of the allocation probability.
"""
function palloc_kernel(y::Matrix{<:Real},
                       σ::Matrix{<:Real},
                       t::Vector{<:Real},
                       μ::Vector{<:Real},
                       a::Vector{<:Real},
                       w::Real)
    # Compute the unnormalized probability of allocating localizations to the 
    # provided emitter.
    nloc = size(y, 1)
    p_kernel = Vector{Float32}(undef, nloc)
    for ii = 1:nloc
        p_kernel[ii] = palloc_kernel(y[ii, :], σ[ii, :], t[ii], μ, a, w)
    end

    return p_kernel
end

"""
    p = palloc_kernel(y::Vector{<:Real},
                      σ::Vector{<:Real},
                      t::Real,
                      μ::Vector{<:Real},
                      a::Vector{<:Real},
                      w::Real)

Compute the unnormalized probability of allocating `y` to emitter `μ`.

# Description
This method computes the unnormalized probability of allocating the
localization (`y`, `σ`, `t`) to the emitter at location `μ+at`.  I.e., this
method computes the kernel of the distribution P_alloc(Z_i=j) (the 
normalization factor is the same for {j=1:k | P_alloc(Z_i=j)}).

# Inputs
- `y`: 2D coordinates of a localization. (pixels)(2x1)([y; x])
- `σ`: Standard deviations of the observation distribution. 
       (pixels)(2x1)([y; x])
- `t`: Observation time corresponding to localization `y`. (frames)
- `μ`: Location of the emitter. (pixels)(2x1)([y; x])
- `a`: Drift velocities of each emitter. (pixels/frame)(2x1)([y; x])
- `w`: Relative weighting of the emitter.

# Outputs
- `p`: Probability kernel of the allocation probability.
"""
function palloc_kernel(y::Vector{<:Real},
                       σ::Vector{<:Real},
                       t::Real,
                       μ::Vector{<:Real},
                       a::Vector{<:Real},
                       w::Real)
    # Compute the unnormalized probability of allocating this localization to
    # the provided emitter.
    p_kernel = w * emitterlikelihood2D(y, σ, t, μ, a)

    return p_kernel
end
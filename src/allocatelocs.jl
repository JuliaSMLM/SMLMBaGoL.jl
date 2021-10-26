using SMLMData
using Distributions
using NearestNeighbors

"""
    allocatelocs(smld::SMLMData.SMLD2D, 
                 μ::Matrix{Float64}, 
                 α::Vector{Float64} = [0.0; 0.0])

Allocate localizations to emitters.

# Description
This function allocates localizations in `smld` to the emitters located at
the positions `μ` at time t=0.

# Inputs
-`smld`: SMLMData.SMLD2D data structure containing localizations.
-`μ`: Coordinates of the proposed emitter positions. (pixels)
-`α`: Drift velocity of the localizations in `smld`. (pixels/frame)
"""
function allocatelocs(smld::SMLMData.SMLD2D, 
                      μ::Matrix{Float64}, 
                      α::Vector{Float64} = [0.0; 0.0])
    # Loop through localizations in `smld` and allocate to emitters using 
    # Gibbs sampling.
    nlocs = SMLMData.length(smld)
    zprime = Vector{Int}(undef, nlocs)
    for nn = 1:nlocs
        posterior = SMLMBaGoL.posterior_allocations([smld.x[nn]; smld.y[nn]], 
                                                    [smld.σ_x[nn]; smld.σ_y[nn]], 
                                                    Float64(smld.framenum[nn]),
                                                    μ, 
                                                    α)
        zprime[nn] = rand(posterior)
    end

    return zprime
end

"""
    posterior_allocations(x::Vector{Float64}, 
                          σ_x::Vector{Float64}, 
                          t::Float64, 
                          μ::Matrix{Float64},
                          α::Vector{Float64})

Construct a posterior distribution of allocations of a localization.

# Description
This function constructs a posterior distribution of the allocation of
the localization defined by `x`, `σ_x`, and `t` to the emitters at
positions `μ` at time t=0.

# Inputs
-`x`: Coordinate of a 2D localization. (pixels)([x; y])
-`σ_x`: Standard error of the localization `x`. (pixels)([x; y])
-`t`: Time of observation of localization `x`. (frame)
-`μ`: Coordinates of the proposed emitter positions. (pixels)([x y])
-`α`: Drift velocity of the localizations in `smld`. (pixels/frame)([α_x; α_y])
"""
function posterior_allocations(x::Vector{Float64}, 
                               σ_x::Vector{Float64}, 
                               t::Float64, 
                               μ::Matrix{Float64},
                               α::Vector{Float64})
    # Construct a normalized posterior for the allocations that we can sample
    # from.
    pmf = (1/sqrt(2*pi*σ_x[1]^2)) .* exp.(-(x[1].-μ[:, 1].-α[1]*t).^2 / (2*σ_x[1]^2)) .*
          (1/sqrt(2*pi*σ_x[2]^2)) .* exp.(-(x[2].-μ[:, 2].-α[2]*t).^2 / (2*σ_x[2]^2))
    pmf = pmf ./ sum(pmf)
    
    # If any of `pmf` is NaN, it was not normalizable numerically so we'll just
    # allocate the localization to its nearest-neighbor emitter.
    if any(isnan.(pmf))
        kdtree = NearestNeighbors.KDTree(μ')
        nnindex, _ = NearestNeighbors.knn(kdtree, x, 1, true)
        pmf = zeros(Float64, length(pmf))
        pmf[nnindex[1]] = 1.0
    end

    return Distributions.DiscreteNonParametric(1:length(pmf), pmf)
end
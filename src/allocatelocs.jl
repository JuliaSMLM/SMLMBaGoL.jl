using SMLMData
using Distributions
using NearestNeighbors 

function allocatelocs(smld::SMLMData.SMLD2D)
    # Loop through localizations in `smld` and allocate to emitters using 
    # Gibbs sampling.
    nlocs = SMLMData.length(smld)
    allocations = Vector{Float64}(undef, nlocs)
    for nn = 1:nlocs
        posterior = SMLMBaGoL.posterior_allocations([smld.x[nn]; smld.y[nn]], 
                                                    [smld.σ_x[nn]; smld.σ_y[nn]], 
                                                    Float64(smld.framenum[nn]),
                                                    [0.0; 0.0],
                                                    [0.0 0.0; 1.0 1.0])
        allocations[nn] = rand(posterior)
    end

    return allocations
end

function posterior_allocations(x::Vector{Float64}, 
                               σ_x::Vector{Float64}, 
                               t::Float64, 
                               α::Vector{Float64},
                               μ::Matrix{Float64})
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
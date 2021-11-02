using Distributions
using Base

# This file contains type definitions for types used in SMLMBaGoL.


## Parameter structures.
"""
    SubregionParams

Abstract type defining subregion splitting parameter structures.
"""
abstract type SubregionParams
end

"""
    SubregionParams2D(roisize, roioverlap)

Structure of parameters defining subregion splitting of data.

# Description
The SubregionParams2D structure organizes parameters related to subregion
splitting of data.

# Fields
-`roisize`: Size of each subregion (region of interest, or ROI). 
            (Pixels)(Default = 5.0)
-`roioverlap`: Size of the overlap between subregions. (Default = 1.0)(Pixels)
"""
mutable struct SubregionParams2D <: SubregionParams
    roisize
    roioverlap
end
function SubregionParams2D()
    return SubregionParams2D(5.0, 1.0)
end

"""
    PreThreshParams

Abstract type defining preprocessing threshold parameter structures.
"""
abstract type PreThreshParams
end

"""
    PreThreshParams2D(maxsigmadev_photons::Float64, n_min::Int, r::Float64)

Structure of parameters defining preprocessing thresholds.

# Description
The PreThreshParams2D structure organizes parameters related to thresholds
applied to data during preprocessing.

# Fields
-`maxsigmadev_photons`: Maximum number of standard deviations above the mean
                        photons allowed. (Default = Inf64)
-`n_min`: Minimum number of nearest neighbors required within `r`. 
          (Pixels)(Default = 1)
-`r`: Distance defining the region within which `n_min` nearest neighbor 
      localizations must be present. (Pixels)(Default = 10.0)
"""
mutable struct PreThreshParams2D <: PreThreshParams
    maxsigmadev_photons::Float64
    n_min::Int
    r::Float64
end
function PreThreshParams2D()
    return PreThreshParams2D(Inf64, 0, 10.0)
end

"""
    PreclusterParams

Abstract type defining preclustering parameter structures.
"""
abstract type PreclusterParams
end

"""
    PreclusterParams2D(maxdist::Float64)

Structure of parameters defining preclustering of localizations in subregions.

# Description
The PreclusterParams2D structure organizes parameters related to the
preclustering of localizations within each subregion.

# Fields
-`maxdist::Float64`: Maximum distance from one localization to its 
                     nearest-neighbor allowed in each precluster. 
                     (Default = 0.15)(Pixels)
"""
mutable struct PreclusterParams2D <: PreclusterParams
    maxdist::Float64
end
function PreclusterParams2D()
    return PreclusterParams2D(0.15)
end

"""
    MCParams

Abstract type defining RJMCMC parameters.
"""
abstract type MCParams
end

"""
    MCParams2D

RJMCMC type structure specific to BaGoL.

# Description
The MCParams structure organizes parameters, distributions, or other info.
related to RJMCMC.
"""
mutable struct MCParams2D <: MCParams
    α::Float64
    β::Float64
    σ_a::Float64
    n_chain::Int
    n_burnin::Int
    p_jump::Vector{Float64}
    srmag::Float64
    srimsize::Vector{Int}
    roi::Vector{Float64}
    nsigma::Float64
    area::Float64
    jumpdistrib::Distributions.Distribution
    imdistrib::Distributions.Distribution
    priork::Distributions.Distribution
    priorz # we probably can't return the full distribution for this prior
    priorμ::Vector{Distributions.Distribution}
    priora::Vector{Distributions.Distribution}
    MCParams2D() = new()
end

"""
    BaGoLParams

Abstract type defining RJMCMC parameters.
"""
abstract type BaGoLParams
end

"""
    BaGoLParams()

Structure of parameters defining the BaGoL workflow.

# Description
This structure organizes the parameter structures used in a typical BaGoL
analysis.  The intention is that this structure plus the data represents a 
complete description of the BaGoL analyses/results.
"""
mutable struct BaGoLParams2D <: BaGoLParams
    subregion::SubregionParams
    prethresholds::PreThreshParams
    preclustering::PreclusterParams
    mcparams::MCParams
end
function BaGoLParams2D()
    return BaGoLParams2D(SubregionParams2D(), 
                         PreThreshParams2D(), 
                         PreclusterParams2D(), 
                         MCParams2D())
end


## Data structures.

abstract type MarkovChain
end

mutable struct BaGoLChain <: MarkovChain
    k::Vector{Int}
    μ::Vector{Matrix{Float64}}
    a::Vector{Matrix{Float64}}
    z::Vector{Vector{Int}}
end
function BaGoLChain(n_chain::Int)
    # Initialize a chain structure of length `n_chain`.
    return BaGoLChain(Vector{Int}(undef, n_chain), 
        Vector{Matrix{Float64}}(undef, n_chain),
        Vector{Matrix{Float64}}(undef, n_chain),
        Vector{Vector{Int}}(undef, n_chain))
end
function BaGoLChain(k::Int, 
                    μ::Matrix{Float64},
                    a::Matrix{Float64},
                    z::Vector{Int})
    return BaGoLChain([k], [μ], [a], [z])
end
length(chain::BaGoLChain) = Base.length(chain.k)
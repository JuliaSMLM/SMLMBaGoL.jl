# This file contains type definitions for types used in SMLMBaGoL.


## Parameter types.
"""
    ParamStruct()
Abstract type that parameter structures should inherit from.
"""
abstract type ParamStruct 
end

"""
    SubregionParams(roisize, roioverlap)

Structure of parameters defining subregion splitting of data.

# Description
The SubregionParams structure organizes parameters related to subregion
splitting of data.

# Fields
-`roisize`: Size of each subregion (region of interest, or ROI). 
            (Pixels)(Default = 5.0)
-`roioverlap`: Size of the overlap between subregions. (Default = 1.0)(Pixels)
"""
mutable struct SubregionParams <: ParamStruct
    roisize
    roioverlap
end
function SubregionParams()
    return SubregionParams(5.0, 1.0)
end

"""
    PreThreshParams(maxsigmadev_photons::Float64, n_min::Int, r::Float64)

Structure of parameters defining preprocessing thresholds.

# Description
The PreThreshParams structure organizes parameters related to thresholds
applied to data during preprocessing.

# Fields
-`maxsigmadev_photons`: Maximum number of standard deviations above the mean
                        photons allowed. (Default = Inf64)
-`n_min`: Minimum number of nearest neighbors required within `r`. 
          (Pixels)(Default = 1)
-`r`: Distance defining the region within which `n_min` nearest neighbor 
      localizations must be present. (Pixels)(Default = 10.0)
"""
mutable struct PreThreshParams <: ParamStruct
    maxsigmadev_photons::Float64
    n_min::Int
    r::Float64
end
function PreThreshParams()
    return PreThreshParams(Inf64, 0, 10.0)
end

"""
    PreclusterParams(maxdist::Float64)

Structure of parameters defining preclustering of localizations in subregions.

# Description
The PreclusterParams structure organizes parameters related to the
preclustering of localizations within each subregion.

# Fields
-`maxdist::Float64`: Maximum distance from one localization to its 
                     nearest-neighbor allowed in each precluster. 
                     (Default = 0.15)(Pixels)
"""
mutable struct PreclusterParams <: ParamStruct
    maxdist::Float64
end
function PreclusterParams()
    return PreclusterParams(0.15)
end

"""
    RJStruct()

RJMCMC type structure specific to BaGoL.

# Description
The RJStruct structure organizes parameters, distributions, or other info.
related to RJMCMC.
"""
mutable struct RJStruct <: ParamStruct
end

"""
    BaGoLParams()

Structure of parameters defining the BaGoL workflow.

# Description
This structure organizes the parameter structures used in a typical BaGoL
analysis.  The intention is that this structure plus the data represents a 
complete description of the BaGoL analyses/results.
"""
mutable struct BaGoLParams <: ParamStruct
    subregion::SubregionParams
    prethresholds::PreThreshParams
    preclustering::PreclusterParams
end
function BaGoLParams()
    return BaGoLParams(SubregionParams(), PreThreshParams(), PreclusterParams())
end


## Distribution types.
"""
    Prior()
Abstract type that definitions of priors should inherit from.
"""
abstract type Prior
end

"""
    PriorStruct1D()
Basic structure defining a prior distribution and associated information.

# Fields
-`pdf`: 1D Prior distribution evaluated in the range defined by `θ_start` and
        `θ_step`.
-`θ_start`: Start of parameter range across which `pdf` was evaluated.
-`θ_step`: Step size of the parameter `θ_start` defining the range spanned by
           `pdf`.
-`nsteps`: Number of steps made in the range of θ, stored for convenience since
           this number is often needed.
-`alias`: String describing the prior, e.g., "Gamma".  This should not be used
          by code and should only serve as a reference/reminder for the user.

"""
mutable struct PriorStruct1D <: Prior
    θ_start::Float64
    θ_step::Float64
    nsteps::Int
    alias::String
    pdf::Vector{Float64}
end
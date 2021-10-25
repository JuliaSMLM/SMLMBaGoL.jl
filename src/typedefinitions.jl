# This file contains type definitions for types used in SMLMBaGoL.


## Parameter types.
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
    RJStruct()

RJMCMC type structure specific to BaGoL.

# Description
The RJStruct structure organizes parameters, distributions, or other info.
related to RJMCMC.
"""
mutable struct RJStruct
end

"""
    BaGoLParams()

Structure of parameters defining the BaGoL workflow.

# Description
This structure organizes the parameter structures used in a typical BaGoL
analysis.  The intention is that this structure plus the data represents a 
complete description of the BaGoL analyses/results.
"""
mutable struct BaGoLParams
    subregion::SubregionParams
    prethresholds::PreThreshParams
    preclustering::PreclusterParams
end
function BaGoLParams()
    return BaGoLParams(SubregionParams2D(), 
                       PreThreshParams2D(), 
                       PreclusterParams2D())
end
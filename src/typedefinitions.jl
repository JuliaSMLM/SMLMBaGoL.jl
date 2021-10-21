# This file contains type definitions for types used in SMLMBaGoL.

abstract type ParamStruct 
end

"""
    SubregionParams(roisize, roioverlap)

Structure of parameters defining subregion splitting of data.

# Description
The SubregionParams structure organizes parameters related to subregion
splitting of data.
    roisize: Size of each subregion (region of interest, or ROI).
        (Pixels)(Default = 5.0)
    roioverlap: Size of the overlap between subregions. (Default = 1.0)(Pixels)
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
    maxsigmadev_photons: Maximum number of standard deviations above the mean
                         photons allowed. (Default = Inf64)
    n_min: Minimum number of nearest neighbors required within `r`. 
        (Pixels)(Default = 1)
    r: Distance defining the region within which `n_min` nearest neighbor 
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
    maxdist: Maximum distance allowed between a preclustered localization and
             its nearest neighbor in the cluster. (Default = 0.15)(Pixels)
"""
mutable struct PreclusterParams <: ParamStruct
    maxdist::Float64
end
function PreclusterParams()
    return PreclusterParams(0.15)
end

"""
    BaGoLParams()

Structure of parameters defining the BaGoL workflow.

# Description
"""
mutable struct BaGoLParams <: ParamStruct
    subregion::SubregionParams
    prethresholds::PreThreshParams
    preclustering::PreclusterParams
end
function BaGoLParams()
    return BaGoLParams(SubregionParams(), PreThreshParams(), PreclusterParams())
end
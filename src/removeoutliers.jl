using SMLMData
using Statistics
using NearestNeighbors
using Base

# This file contains functions/methods used to remove localizations deemed to
# be outliers.

"""
    smld_thresh = removeoutliers(smld::SMLMData.SMLD2D, 
                                 thresholds::PreThreshParams)

Remove outlier localizations based on the specified `thresholds`

# Description
This method removes localizations from `smld` that do not pass the thresholds
defined by `thresholds`.

# Inputs
- `smld`: SMLD2D structure containing the localization data. (see SMLMData.jl)
- `thresholds`: Structure of preprocessing threshold parameters.

# Outputs
- `smld_thresh`: Input `smld` with the outliers removed.
"""
function removeoutliers(smld::SMLMData.SMLD2D, thresholds::PreThreshParams)
    # Threshold localizations that are too bright (i.e., those which might be 
    # multiple emitters fit as one).
    smld_thresh = threshphotons(smld, thresholds.maxsigmadev_photons)

    # Remove isolated localizations (i.e., those which may be due to 
    # non-specific binding or false localizations).
    smld_thresh = removeisolated(smld_thresh::SMLMData.SMLD2D, 
                                 thresholds.n_min, thresholds.r)

    return smld_thresh
end

"""
    removeoutliers!(smld::SMLMData.SMLD2D, thresholds::PreThreshParams)

Remove outlier localizations based on the specified `thresholds`

# Description
This method removes localizations from `smld` that do not pass the thresholds
defined by `thresholds`.

# Inputs
- `smld`: SMLD2D structure containing the localization data. (see SMLMData.jl)
- `thresholds`: Structure of preprocessing threshold parameters.
"""
function removeoutliers!(smld::SMLMData.SMLD2D, thresholds::PreThreshParams)
    # Threshold localizations that are too bright (i.e., those which might be 
    # multiple emitters fit as one).
    threshphotons!(smld, thresholds.maxsigmadev_photons)

    # Remove isolated localizations (i.e., those which may be due to 
    # non-specific binding or false localizations).
    removeisolated!(smld, thresholds.n_min, thresholds.r)
end

"""
    smld_thresh = removeoutliers(smld::Matrix{SMLMData.SMLD2D}, 
                                 thresholds::PreThreshParams)

Remove outlier localizations based on the specified `thresholds`

# Description
This method removes localizations from `smld` that do not pass the thresholds
defined by `thresholds`.

# Inputs
- `smld`: A matrix of SMLD2D structures. (see SMLMData.jl)
- `thresholds`: Structure of preprocessing threshold parameters.

# Outputs
- `smld_thresh`: Input `smld` with the outliers removed.
"""
function removeoutliers(smld::Matrix{SMLMData.SMLD2D}, 
                        thresholds::PreThreshParams)
    # Threshold each of the SMLMData.SMLD2D structures in `smld`.
    smld_thresh = deepcopy(smld)
    for ii = 1:length(smld)
        smld_thresh[ii] = removeoutliers(smld[ii], thresholds)
    end

    return smld_thresh
end

"""
    removeoutliers!(smld::Matrix{SMLMData.SMLD2D}, thresholds::PreThreshParams)

Remove outlier localizations based on the specified `thresholds`

# Description
This method removes localizations from `smld` that do not pass the thresholds
defined by `thresholds`.

# Inputs
- `smld`: A matrix of SMLD2D structures. (see SMLMData.jl)
- `thresholds`: Structure of preprocessing threshold parameters.

# Notes
This method exists because I couldn't figure out how to get something like
removeoutliers.(smld, thresholds) to work...
"""
function removeoutliers!(smld::Matrix{SMLMData.SMLD2D}, 
                        thresholds::PreThreshParams)
    # Threshold each of the SMLMData.SMLD2D structures in `smld`.
    for ii = 1:length(smld)
        removeoutliers!(smld[ii], thresholds)
    end
end

"""
    smld_thresh = threshphotons(smld::SMLMData.SMLD2D, 
                                maxsigmadev_photons::Real)

Remove localizations that are `maxsigmadev_photons` st. dev. above the mean.

# Description
This method removes localizations from `smld` that are greater than 
`maxsigmadev_photons` standard deviations above the mean photon value.

# Inputs
- `smld`: SMLD2D structure containing the localization data. (see SMLMData.jl)
- `maxsigmadev_photons`: Maximum number of standard deviations above the mean 
                         photons allowed for valid localizations.
             
# Outputs
- `smld_thresh`: Input `smld` with the outliers removed.
"""
function threshphotons(smld::SMLMData.SMLD2D, maxsigmadev_photons::Real)
    # Threshold localizations that are too bright (i.e., those which might be 
    # multiple emitters fit as one).
    meanphotons = Statistics.mean(smld.photons)
    stdevphotons = Statistics.std(smld.photons)
    photons_thresh = meanphotons + maxsigmadev_photons*stdevphotons
    smld_thresh = SMLMData.isolatesmld(smld, smld.photons .<= photons_thresh)

    return smld_thresh
end

"""
    threshphotons!(smld::SMLMData.SMLD2D, maxsigmadev_photons::Real)

Remove localizations that are `maxsigmadev_photons` st. dev. above the mean.

# Description
This method removes localizations from `smld` that are greater than 
`maxsigmadev_photons` standard deviations above the mean photon value.

# Inputs
- `smld`: SMLD2D structure containing the localization data. (see SMLMData.jl)
- `maxsigmadev_photons`: Maximum number of standard deviations above the mean 
                         photons allowed for valid localizations.
"""
function threshphotons!(smld::SMLMData.SMLD2D, maxsigmadev_photons::Real)
    # Threshold localizations that are too bright (i.e., those which might be 
    # multiple emitters fit as one).
    meanphotons = Statistics.mean(smld.photons)
    stdevphotons = Statistics.std(smld.photons)
    photons_thresh = meanphotons + maxsigmadev_photons*stdevphotons
    smld = SMLMData.isolatesmld(smld, smld.photons .<= photons_thresh)
end

"""
    smld_thresh = removeisolated(smld::SMLMData.SMLD2D, 
                                 n_min::Int, 
                                 r::Real)

Remove localizations that don't have `n_min` nearest-neighbors within `r`.

# Description
This method removes localizations from `smld` that are isolated, i.e., those
localizations that have fewer than `n_min` localizations within `r` pixels.

# Inputs
- `smld`: SMLD2D structure containing the localization data. (see SMLMData.jl)
- `n_min`: Minimum number of nearest neighbors required within `r`. (Pixels)
- `r`: Distance defining the region within which `n_min` nearest neighbor 
       localizations must be present. (Pixels)

# Outputs
- `smld_thresh`: Input `smld` with the isolated localizations removed.
"""
function removeisolated(smld::SMLMData.SMLD2D, n_min::Int, r::Real)
    # If there are fewer than `n_min+1` localizations, we can return an empty
    # SMLD2D structure immediately.
    if length(smld) < (n_min+1)
        return SMLMData.SMLD2D()
    end

    # Find the nearest-neighbors to each localization.
    kdtree = NearestNeighbors.KDTree([smld.y smld.x]')
    _, nndist = NearestNeighbors.knn(kdtree, [smld.y smld.x]', n_min+1, true)
    keepbool = getindex.(nndist, n_min+1) .<= r
    smld_prethresh = SMLMData.isolatesmld(smld, keepbool)

    return smld_prethresh
end

"""
    removeisolated!(smld::SMLMData.SMLD2D, n_min::Int, r::Real)

Remove localizations that don't have `n_min` nearest-neighbors within `r`.

# Description
This method removes localizations from `smld` that are isolated, i.e., those
localizations that have fewer than `n_min` localizations within `r` pixels.

# Inputs
- `smld`: SMLD2D structure containing the localization data. (see SMLMData.jl)
- `n_min`: Minimum number of nearest neighbors required within `r`. (Pixels)
- `r`: Distance defining the region within which `n_min` nearest neighbor 
       localizations must be present. (Pixels)
"""
function removeisolated!(smld::SMLMData.SMLD2D, n_min::Int, r::Real)
    # If there are fewer than `n_min+1` localizations, we can return an empty
    # SMLD2D structure immediately.
    if length(smld) < (n_min+1)
        return SMLMData.SMLD2D()
    end

    # Find the nearest-neighbors to each localization.
    kdtree = NearestNeighbors.KDTree([smld.y smld.x]')
    _, nndist = NearestNeighbors.knn(kdtree, [smld.y smld.x]', n_min+1, true)
    keepbool = getindex.(nndist, n_min+1) .<= r
    smld = SMLMData.isolatesmld(smld, keepbool)
end
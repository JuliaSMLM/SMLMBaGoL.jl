using SMLMData

# This file contains functions/methods useful for splitting data into 
# subregions.

"""
    rois = genrois(datasize::Vector{Float64}, 
                   roisize::Float64 = 5.0, 
                   roioverlap::Float64 = 1.0)

Define subregion region of interests (`rois`).

# Description
This method generates `rois` splitting the frame defined by `datasize` into
subregions of (maximum) size `roisize` with overlaps of `roioverlap`.

# Inputs
- `datasize`: Size of the data frame. (pixels)([ysize; xsize])
- `roisize`: Nominal size of the square regions of interest (subregion). 
             (pixels)
- `roioverlap`: Size of the overlap region between neighboring subregions.
                (pixels)

# Outputs
- `rois`: Matrix of ROIs defining each subregion. ROIs are arranged as 
          [ystart, xstart, yend, xend].

# Notes 
The output `rois` has the regions defined as [ystart, xstart, yend, xend].
This method follows the convention that the center of a pixel is an integer
coordinate, e.g., pixel n spans (left edge to right edge) the coordinate range 
[n-0.5, n+0.5]. As such, an ROI of, e.g., [1, 6, 5, 10] covers the y range
[0.5, 5.5] and the x range [5.5, 10.5].
"""
function genrois(datasize::Vector{Float64}, 
                 roisize::Float64 = 5.0, 
                 roioverlap::Float64 = 1.0)
    # Validate some parameters.
    roisize = all(roisize.<=datasize) ? roisize : minimum(datasize)
    roioverlap = all(roioverlap<roisize) ? roioverlap : (roisize-1)

    # Compute some useful parameters.
    nsplit = Int.(ceil.((datasize.-roisize) / (roisize-roioverlap))) .+ 1
    ystart = min.(1 .+ (roisize-roioverlap)*collect(0:(nsplit[1]-1)), 
        datasize[1])
    yend = min.(ystart .+ roisize .- 1, datasize[1])
    xstart = min.(1 .+ (roisize-roioverlap)*collect(0:(nsplit[2]-1)), 
        datasize[2])
    xend = min.(xstart .+ roisize .- 1, datasize[2])
    
    # Define the subregions rois.
    rois = Matrix{Vector{Float64}}(undef, tuple(nsplit...))
    for ii = 1:nsplit[1], jj = 1:nsplit[2]
        # Define the splitting ROI for this subregion.
        rois[ii, jj] = [ystart[ii], xstart[jj], yend[ii], xend[jj]]
    end

    return rois
end

"""
    smld_subregions, rois, connectID = gensubregions(smld::SMLMData.SMLD2D, 
                                                     roisize::Float64 = 5.0, 
                                                     roioverlap::Float64 = 1.0)

Split the `smld` localizations into overlapping subregions.

# Description
Localizations in the input structure `smld` are split into a grid of subregions
of size `roisize` overlapping by `roioverlap` pixels.  The output 
`smld_subregions` is an array of the subregion SMLD structures.  The output 
`rois` is an array of the ROIs explicitly written out (corresponding to each of
`smld_subregions`) organized as [ystart, xstart, yend, xend].  The output 
`connectID` stores the connectIDs of each localization placed in a subregion.

# Inputs
- `smld`: SMLD2D structure containing the localization data. (see SMLMData.jl)
- `roisize`: Nominal size of the square regions of interest (subregion). 
             (Pixels)
- `roioverlap`: Size of the overlap region between neighboring subregions.
                (Pixels)

# Outputs
- `smld_subregions`: Matrix of SMLD2D structures corresponding to each 
                     subregion.
- `rois`: Matrix of ROIs of each subregion, with indexing matching that of
          `smld_subregions`. ROIs are arranged as [ystart, xstart, yend, xend].
- `inds`: Matrix of localization indices indicating which of `smld` were stored
          in which subregion.

# Notes 
The output `rois` has the regions defined as [ystart, xstart, yend, xend].
This method follows the convention that the center of a pixel is an integer
coordinate, e.g., pixel n spans (left edge to right edge) the coordinate range 
[n-0.5, n+0.5]. As such, an ROI of, e.g., [1, 6, 5, 10] covers the y range
[0.5, 5.5] and the x range [5.5, 10.5].
"""
function gensubregions(smld::SMLMData.SMLD2D, 
                       roisize::Float64, 
                       roioverlap::Float64)
    # Generate the subregion ROIs.
    datasize = Float64.(smld.datasize)
    rois = SMLMBaGoL.genrois(datasize, roisize, roioverlap)
    
    # Loop through `smld` and split into subregions.
    nsplit = Int.(ceil.((datasize.-roisize) / (roisize-roioverlap))) .+ 1
    smld_subregions = Matrix{SMLMData.SMLD2D}(undef, tuple(nsplit...))
    inds = Matrix{Vector{Int}}(undef, tuple(nsplit...))
    for ii = 1:nsplit[1], jj = 1:nsplit[2]
        # Store the current subregion (treating pixel 1 to be the center of the
        # first pixel).
        keepbool = 
            (smld.y.>=(rois[ii, jj][1]-0.5)) .* (smld.y.<=(rois[ii, jj][3]+0.5)) .*
            (smld.x.>=(rois[ii, jj][2]-0.5)) .* (smld.x.<=(rois[ii, jj][4]+0.5))
        smld_subregions[ii, jj] = SMLMData.isolatesmld(smld, keepbool)
        inds[ii, jj] = findall(keepbool)
    end

    return smld_subregions, rois, inds
end

"""
    smld_subregions, rois, connectID = gensubregions(smld::SMLMData.SMLD2D, 
                                                     roisize::Int = 5, 
                                                     roioverlap::Int = 1)

Split the `smld` localizations into overlapping subregions.

# Description
Localizations in the input structure `smld` are split into a grid of subregions
of size `roisize` overlapping by `roioverlap` pixels.  The output 
`smld_subregions` is an array of the subregion SMLD structures.  The output 
`rois` is an array of the ROIs explicitly written out (corresponding to each of
`smld_subregions`) organized as [ystart, xstart, yend, xend].  The output 
`connectID` stores the connectIDs of each localization placed in a subregion.

# Inputs
- `smld`: SMLD2D structure containing the localization data. (see SMLMData.jl)
- `roisize`: Nominal size of the square regions of interest (subregion). 
             (Pixels)
- `roioverlap`: Size of the overlap region between neighboring subregions.
                (Pixels)

# Outputs
- `smld_subregions`: Matrix of SMLD2D structures corresponding to each 
                     subregion.
- `rois`: Matrix of ROIs of each subregion, with indexing matching that of
          `smld_subregions`. ROIs are arranged as [ystart, xstart, yend, xend].
- `inds`: Matrix of localization indices indicating which of `smld` were stored
          in which subregion.

# Notes 
The output `rois` has the regions defined as [ystart, xstart, yend, xend].
This method follows the convention that the center of a pixel is an integer
coordinate, e.g., pixel n spans (left edge to right edge) the coordinate range 
[n-0.5, n+0.5]. As such, an ROI of, e.g., [1, 6, 5, 10] covers the y range
[0.5, 5.5] and the x range [5.5, 10.5].
"""
function gensubregions(smld::SMLMData.SMLD2D, 
                       roisize::Int = 5, 
                       roioverlap::Int = 1)
    # Grab/validate some parameters from the input `smld`.
    datasize = smld.datasize
    roisize = all(roisize.<=datasize) ? roisize : minimum(datasize)
    roioverlap = all(roioverlap<roisize) ? roioverlap : (roisize-1)

    # Compute some useful parameters.
    nsplit = Int.(ceil.((datasize.-roisize) / (roisize-roioverlap))) .+ 1
    
    # Loop through `smld` and split into subregions.
    smld_subregions = Matrix{SMLMData.SMLD2D}(undef, tuple(nsplit...))
    rois = Matrix{Vector{Float64}}(undef, tuple(nsplit...))
    inds = Matrix{Vector{Int}}(undef, tuple(nsplit...))
    ystart = 1 .+ (roisize-roioverlap)*collect(0:(nsplit[1]-1))
    yend = min.(ystart .+ roisize .- 1, datasize[1])
    xstart = 1 .+ (roisize-roioverlap)*collect(0:(nsplit[2]-1))
    xend = min.(xstart .+ roisize .- 1, datasize[2])
    for ii = 1:nsplit[1], jj = 1:nsplit[2]
        # Define the splitting ROI for this subregion.
        rois[ii, jj] = [ystart[ii], xstart[jj], yend[ii], xend[jj]]

        # Store the current subregion (treating pixel 1 to be the center of the
        # first pixel).
        keepbool = (smld.y.>=(ystart[ii]-0.5)) .* (smld.y.<=(yend[ii]+0.5)) .*
            (smld.x.>=(xstart[jj]-0.5)) .* (smld.x.<=(xend[jj]+0.5))
        smld_subregions[ii, jj] = SMLMData.isolatesmld(smld, keepbool)
        smld_subregions[ii, jj].datasize = [roisize; roisize]
        inds[ii, jj] = findall(keepbool)
    end

    return smld_subregions, rois, inds
end
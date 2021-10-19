using SMLMData

"""
    smld_subregions, rois = gensubregions(smld::SMLD2D, 
                                          roisize::Int, 
                                          roioverlap::Int)

Split the `smld` localizations into overlapping subregions.

# Description
Localizations in the input structure `smld` are split into a grid of subregions
of size `roisize` overlapping by `roioverlap` pixels.  The output 
`smld_subregions` is an array of the subregion SMLD structures.  The output 
`rois` is an array of the ROIs explicitly written out (corresponding to each of
`smld_subregions`) organized as [ystart, xstart, yend, xend].
"""
function gensubregions(smld::SMLMData.SMLD2D, roisize::Int=5, roioverlap::Int=1)
    # Grab/validate some parameters from the input `smld`.
    datasize = smld.datasize
    roisize = all(roisize.<=datasize) ? roisize : minimum(datasize)
    roioverlap = all(roioverlap<roisize) ? roioverlap : (roisize-1)

    # Compute some useful parameters.
    nsplit = Int.(ceil.((datasize.-roisize) / (roisize-roioverlap))) .+ 1
    
    # Loop through `smld` and split into subregions.
    smld_subregions = Matrix{SMLMData.SMLD2D}(undef, tuple(nsplit...))
    rois = Matrix{Vector{Int}}(undef, tuple(nsplit...))
    ystart = 1 .+ (roisize-roioverlap)*collect(0:(nsplit[1]-1))
    yend = minimum.(ystart .+ roisize)
    xstart = 1 .+ (roisize-roioverlap)*collect(0:(nsplit[2]-1))
    xend = minimum.(xstart .+ roisize)
    for ii = 1:nsplit[1], jj = 1:nsplit[2]
        # Define the splitting ROI for this subregion.
        rois[ii, jj] = [ystart[ii], xstart[jj], yend[ii], xend[jj]]

        # Store the current subregion (treating pixel 1 to be the center of the
        # first pixel).
        keepbool = (smld.y.>=(ystart[ii]-0.5)) .* (smld.y.<=(yend[ii]+0.5)) .*
            (smld.x.>=(xstart[jj]-0.5)) .* (smld.x.<=(xend[jj]+0.5))
        println(any(keepbool))
        smld_subregions[ii, jj] = SMLMData.isolatesubregion(smld, keepbool)
    end

    return smld_subregions
end
using Base

# This file contains functions for removing parts of an RJMCMC chain present
# within subregion overlaps (see SMLMBaGoL.gensubregions()).

"""
    findvalidemitters(μ::Matrix{Float64},
                      minval::Matrix{Float64},
                      maxval::Matrix{Float64}))

Find the indices of valid emitter locations in `μ`.

# Description
This function finds the indices of emitters in `μ` which fall within the
valid region, i.e., within the bounds defined by `minval` and `maxval`.

# Inputs
-`μ`: Matrix of emitter locations. ([y x])
-`minval`: Matrix defining the minimum allowed value of `μ`. ([ymin xmin])
-`maxval`: Matrix defining the maximum allowed value of `μ`. ([ymax xmax])

# Outputs
-`validind`: Row indices of input `μ` identifying emitters that fell within the
             prescribed boundaries.
"""
function findvalidemitters(μ::Matrix{Float64},
                           minval::Matrix{Float64},
                           maxval::Matrix{Float64})
    # Determine which emitters fall within the valid region (i.e., within the
    # boundaries defined by `minval` and `maxval`).
    valid = all(μ .>= repeat(minval, state.k), dims=2) .*
        all(μ .<= repeat(maxval, state.k), dims=2)
    
    # Convert `valid` to a set of emitter indices.
    # NOTE: There should certainly be a better way to do this... I just want an
    #       output of type Vector{Int}, but I'm struggling to find a nice way 
    #       to do this.
    validind = findall(vec(.!iszero.(valid)))

    return validind
end

"""
    removeoverlap(chain::SMLMBaGoL.BaGoLChain2D, 
                  minval::Matrix{Float64}, 
                  maxval::Matrix{Float64})

Find the emitters in `chain` that fall within `minval` and `maxval`.

# Description
This method finds the emitter coordinates within `chain` that fall within the
valid region defined by `minval` and `maxval`.

# Inputs
-`chain`: Chain of states, with each state having a field `state.μ`.
-`minval`: Matrix defining the minimum allowed value of `state.μ`. 
           ([ymin xmin])
-`maxval`: Matrix defining the maximum allowed value of `state.μ`. 
           ([ymax xmax])

# Outputs
-`μ_valid`: Matrix of emitter locations in `chain` that fell within the bounds.
"""
function removeoverlap(chain::SMLMBaGoL.BaGoLChain2D, 
                       minval::Matrix{Float64}, 
                       maxval::Matrix{Float64})
    # Loop over the states in `chain` and remove emitters in the overlap region.
    μ_valid = Matrix{Float64}(undef, 0, 2)
    for nn = 1:Base.length(chain.states)
        validind = SMLMBaGoL.findvalidemitters(chain.states[nn].μ, 
                                               minval, maxval)
        μ_valid = [μ_valid; chain.states[nn].μ[validind, :]]
    end

    return μ_valid
end

"""
    removeoverlap(chain::SMLMBaGoL.BaGoLChain2D,
                  datasize::Vector{Int},
                  roi::Vector{Float64},
                  roioverlap::Float64)

Find the emitters in `chain` that fall within the valid exploration region.

# Description
This method finds the emitter coordinates within `chain` that fall within the
valid region of exploration.  That is, emitter locations that fell in the outer 
`roioverlap/2` edge of the `roi+-0.5` will be discarded.  Emitters at the edge 
of the data boundary defined by `datasize` (e.g., if datasize = [128; 128], the 
data is 128 pixels wide, so the edges [0.5; 128.5] are the data boundaries) 
will be retained, as these regions do not overlap with any other chain rois.


# Inputs
-`chain`: Chain of states, with each state having a field `state.μ`.
-`datasize`: Size of the raw data matrix. ([ysize xsize])
-`roi`: Region of interest explored by `chain`. ([ystart xstart yend xend])
-`roioverlap`: Overlap of neighboring chain rois.

# Outputs
-`μ_valid`: Matrix of emitter locations in `chain` that fell within the bounds.
"""
function removeoverlap(chain::SMLMBaGoL.BaGoLChain2D,
                       datasize::Vector{Int},
                       roi::Vector{Float64},
                       roioverlap::Float64)
    # Remove any parts of the `chain` contained within `roioverlap/2` of the 
    # edge. If the chain is near an actual data boundary (e.g., if the raw
    # data is 128 pixels wide, the edges [0.5; 128.5] are the data boundaries),
    # we'll retain the chain along the edge (since that edge is not an overlap
    # region!).

    # Define the minimum and maximum allowable coordinates for this `roi`.
    ymin = roi[1]>1.0 ? roi[1]-0.5+roioverlap/2 : 0.5
    ymax = roi[3]<datasize[1] ? roi[3]+0.5-roioverlap/2 : datasize[1]+0.5
    xmin = roi[2]>1.0 ? roi[2]-0.5+roioverlap/2 : 0.5
    xmax = roi[4]<datasize[2] ? roi[4]+0.5-roioverlap/2 : datasize[2]+0.5
    minval = [ymin xmin]
    maxval = [ymax xmax]

    return SMLMBaGoL.removeoverlap(chain, minval, maxval)
end

"""
    removeoverlap(chain::Vector{SMLMBaGoL.BaGoLChain2D},
                  datasize::Vector{Int},
                  roi::Vector{Float64},
                  roioverlap::Float64)

Find the emitters in `chain` that fall within the valid exploration region.

# Description
This method finds the emitter coordinates within `chain` that fall within the
valid region of exploration.  That is, emitter locations that fell in the outer 
`roioverlap/2` edge of the `roi+-0.5` will be discarded.  Emitters at the edge 
of the data boundary defined by `datasize` (e.g., if datasize = [128; 128], the 
data is 128 pixels wide, so the edges [0.5; 128.5] are the data boundaries) 
will be retained, as these regions do not overlap with any other chain rois.
This method dispatches on removeoverlap(chain::SMLMBaGoL.BaGoLChain2D, ...)
for each chain in the vector, concatenating the resulting valid positions.

# Inputs
-`chain`: Vector of state chains, with each state having a field `state.μ`.
-`datasize`: Size of the raw data matrix. ([ysize xsize])
-`roi`: Region of interest explored by `chain`. ([ystart xstart yend xend])
-`roioverlap`: Overlap of neighboring chain rois.

# Outputs
-`μ_valid`: Matrix of emitter locations in `chain` that fell within the bounds.
"""
function removeoverlap(chain::Vector{SMLMBaGoL.BaGoLChain2D},
                       datasize::Vector{Int},
                       roi::Vector{Float64},
                       roioverlap::Float64)

    # Loop over the provided chains and create a matrix of valid emitter
    # coordinates.
    μ_valid = Matrix{Float64}(undef, 0, 2)
    for ii = 1:Base.length(chain)
        μ_valid = [μ_valid; 
            SMLMBaGoL.removeoverlap(chain[ii], datasize, roi, roioverlap)]
    end

    return μ_valid
end

"""
    removeoverlap(chain::Matrix{Vector{SMLMBaGoL.BaGoLChain2D}},
                  datasize::Vector{Int},
                  rois::Matrix{Vector{Float64}},
                  roioverlap::Float64)

Find the emitters in `chain` that fall within the valid exploration region.

# Description
This method finds the emitter coordinates within `chain` that fall within the
valid region of exploration.  That is, emitter locations that fell in the outer 
`roioverlap/2` edge of the `roi+-0.5` will be discarded.  Emitters at the edge 
of the data boundary defined by `datasize` (e.g., if datasize = [128; 128], the 
data is 128 pixels wide, so the edges [0.5; 128.5] are the data boundaries) 
will be retained, as these regions do not overlap with any other chain rois.
This method dispatches on
removeoverlap(chain::Vector{SMLMBaGoL.BaGoLChain2D}, ...) for each entry of the
matrix `chain`, concatenating the resulting valid positions.

# Inputs
-`chain`: Matrix of vectors of state chains, with each state having a field 
          `state.μ`.
-`datasize`: Size of the raw data matrix. ([ysize xsize])
-`rois`: Matrix of rois, with indexing matching that of `chain`.
-`roioverlap`: Overlap of neighboring chain rois.

# Outputs
-`μ_valid`: Matrix of emitter locations in `chain` that fell within the bounds.
"""
function removeoverlap(chain::Matrix{Vector{SMLMBaGoL.BaGoLChain2D}},
                       datasize::Vector{Int},
                       rois::Matrix{Vector{Float64}},
                       roioverlap::Float64)

    # Loop over the provided chains and create a matrix of valid emitter
    # coordinates.
    μ_valid = Matrix{Float64}(undef, 0, 2)
    for ii = 1:prod(size(chain))
        μ_valid = [μ_valid; 
            SMLMBaGoL.removeoverlap(chain[ii], datasize, rois[ii], roioverlap)]
    end

    return μ_valid
end
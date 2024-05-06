using Base

# This file contains functions for removing parts of an RJMCMC chain present
# within subregion overlaps (see SMLMBaGoL.gensubregions()).

"""
    validind, valid = findvalid(μ::Matrix{Float64},
                                minval::Matrix{Float64},
                                maxval::Matrix{Float64}))

Find the indices of valid emitter locations in `μ`.

# Description
This function finds the indices of emitters in `μ` which fall within the
valid region, i.e., within the bounds defined by `minval` and `maxval`.

# Inputs
- `μ`: Matrix of emitter locations. ([y x])
- `minval`: Matrix defining the minimum allowed value of `μ`. ([ymin xmin])
- `maxval`: Matrix defining the maximum allowed value of `μ`. ([ymax xmax])

# Outputs
- `validind`: Row indices of input `μ` identifying emitters that fell within the
              prescribed boundaries.
- `valid`: BitMatrix defining the valid emitters, returned for convenience since
           it can be used to find the invalid indices if desired.
"""
function findvalid(μ::Matrix{Float64}, 
                   minval::Matrix{Float64}, 
                   maxval::Matrix{Float64})
    # Determine which emitters fall within the valid region (i.e., within the
    # boundaries defined by `minval` and `maxval`).
    k = size(μ, 1)
    valid = all(μ .>= repeat(minval, k), dims=2) .*
        all(μ .<= repeat(maxval, k), dims=2)
    
    # Convert `valid` to a set of emitter indices.
    # NOTE: There should certainly be a better way to do this... I just want an
    #       output of type Vector{Int}, but I'm struggling to find a nice way 
    #       to do this.
    validind = findall(vec(.!iszero.(valid)))

    return validind, valid
end

"""
    invalidind, invalid = findinvalid(μ::Matrix{Float64},
                                      minval::Matrix{Float64},
                                      maxval::Matrix{Float64}))

Find the indices of emitter locations in `μ` that fall outisde the bounds.

# Description
This function finds the indices of emitters in `μ` which fall outside the
valid region, i.e., outside the bounds defined by `minval` and `maxval`.

# Inputs
- `μ`: Matrix of emitter locations. ([y x])
- `minval`: Matrix defining the minimum allowed value of `μ`. ([ymin xmin])
- `maxval`: Matrix defining the maximum allowed value of `μ`. ([ymax xmax])

# Outputs
- `invalidind`: Row indices of input `μ` identifying emitters that fell outside
                the prescribed boundaries.
- `invalid`: BitMatrix defining the invalid emitters, returned for convenience 
             since it can be used to find the valid indices if desired.
"""
function findinvalid(μ::Matrix{Float64}, 
                     minval::Matrix{Float64}, 
                     maxval::Matrix{Float64})
    # Determine which emitters fall outside the valid region (i.e., outside the
    # boundaries defined by `minval` and `maxval`).
    k = size(μ, 1)
    invalid = any(μ .< repeat(minval, k), dims=2) .+
        any(μ .> repeat(maxval, k), dims=2)
    
    # Convert `invalid` to a set of emitter indices.
    # NOTE: There should certainly be a better way to do this... I just want an
    #       output of type Vector{Int}, but I'm struggling to find a nice way 
    #       to do this.
    invalidind = findall(vec(.!iszero.(invalid)))

    return invalidind, invalid
end

"""
    removeoverlap!(chain::SMLMBaGoL.BaGoLChain2D, 
                   minval::Matrix{Float64}, 
                   maxval::Matrix{Float64})

Remove the emitters in `chain` that fall outside `minval` and `maxval`.

# Description
This method finds the emitters within `chain` that fall outside the valid 
region defined by `minval` and `maxval` and deletes them from the chain.
Note that the allocations in `state.z` are set to -1 for those emitters 
which were removed.

# Inputs
- `chain`: Chain of states, with each state having a field `state.μ`.
- `minval`: Matrix defining the minimum allowed value of `state.μ`. 
            ([ymin xmin])
- `maxval`: Matrix defining the maximum allowed value of `state.μ`. 
            ([ymax xmax])
"""
function removeoverlap!(chain::SMLMBaGoL.BaGoLChain2D, 
                        minval::Matrix{Float64}, 
                        maxval::Matrix{Float64})
    # Loop over the states in `chain` and remove emitters in the overlap region.
    for ii in Base.length(chain.states):-1:1
        # Determine which emitters should be kept.
        invalidind, _ = SMLMBaGoL.findinvalid(chain.states[ii].μ, 
                                              minval, maxval)

        # If no emitters are kept, delete the entire state.  Otherwise, just
        # delete the invalid emitters.
        if Base.length(invalidind) == chain.states[ii].k
            SMLMBaGoL.removestate!(chain, ii)
        else
            SMLMBaGoL.removeemitter!(chain.states[ii], invalidind)
        end
    end
end

"""
    chain_out = removeoverlap(chain::SMLMBaGoL.BaGoLChain2D, 
                              minval::Matrix{Float64}, 
                              maxval::Matrix{Float64})

Find the emitters in `chain` that fall within `minval` and `maxval`.

# Description
This method finds the emitters within `chain` that fall within the valid region
defined by `minval` and `maxval`.

# Inputs
- `chain`: Chain of states, with each state having a field `state.μ`.
- `minval`: Matrix defining the minimum allowed value of `state.μ`. 
            ([ymin xmin])
- `maxval`: Matrix defining the maximum allowed value of `state.μ`. 
            ([ymax xmax])

# Outputs
- `chain_out`: Chain with the invalid emitters removed.  Note that the 
               allocations in `state.z` are replaced with `-1` to to indicate
               allocations to emitters that were removed.
"""
function removeoverlap(chain::SMLMBaGoL.BaGoLChain2D, 
                       minval::Matrix{Float64}, 
                       maxval::Matrix{Float64})
    # Loop over the states in `chain` and remove emitters in the overlap region.
    chain_out = deepcopy(chain)
    for ii in Base.length(chain_out.states):-1:1
        # Determine which emitters should be kept.
        invalidind, _ = SMLMBaGoL.findinvalid(chain_out.states[ii].μ, 
                                              minval, maxval)

        # If no emitters are kept, delete the entire state.  Otherwise, just
        # delete the invalid emitters.
        if Base.length(invalidind) == chain_out.states[ii].k
            SMLMBaGoL.removestate!(chain_out, ii)
        else
            SMLMBaGoL.removeemitter!(chain_out.states[ii], invalidind)
        end
    end

    return chain_out
end

"""
    out = removeoverlap(chain::SMLMBaGoL.BaGoLChain2D,
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
- `chain`: Chain of states, with each state having a field `state.μ`.
- `datasize`: Size of the raw data matrix. ([ysize xsize])
- `roi`: Region of interest explored by `chain`. ([ystart xstart yend xend])
- `roioverlap`: Overlap of neighboring chain rois.

# Outputs
- `out`: see return value of removeoverlap(chain::SMLMBaGoL.BaGoLChain2D, 
                                           minval::Matrix{Float64}, 
                                           maxval::Matrix{Float64})
"""
function removeoverlap(chain::SMLMBaGoL.BaGoLChain2D,
                       datasize::Vector{Int},
                       roi::Vector{Float64},
                       roioverlap::Float64)
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
    chain_out = removeoverlap(chain::Vector{SMLMBaGoL.BaGoLChain2D},
                              datasize::Vector{Int},
                              roi::Vector{Float64},
                              roioverlap::Float64)

Remove the emitters in `chain` that fall outside the valid exploration region.

# Description
This method removes the emitters in `chain` that fall outside the valid region
of exploration.  That is, emitter locations that fell in the outer 
`roioverlap/2` edge of the `roi+-0.5` will be discarded.  Emitters at the edge 
of the data boundary defined by `datasize` (e.g., if datasize = [128; 128], the 
data is 128 pixels wide, so the edges [0.5; 128.5] are the data boundaries) 
will be retained, as these regions do not overlap with any other chain rois.
This method dispatches on removeoverlap(chain::SMLMBaGoL.BaGoLChain2D, ...)
for each chain in the vector.

# Inputs
- `chain`: Vector of state chains.
- `datasize`: Size of the raw data matrix. ([ysize xsize])
- `roi`: Region of interest explored by `chain`. ([ystart xstart yend xend])
- `roioverlap`: Overlap of neighboring chain rois.

# Outputs
- `chain_out`: Chains with the invalid emitters removed.  Note that the 
               allocations in `state.z` are replaced with `-1` to to indicate
               allocations to emitters that were removed.
"""
function removeoverlap(chain::Vector{SMLMBaGoL.BaGoLChain2D},
                       datasize::Vector{Int},
                       roi::Vector{Float64},
                       roioverlap::Float64)

    # Loop over the provided chains and concatenate the valid states.
    chain_out = Vector{SMLMBaGoL.BaGoLChain2D}(undef, length(chain))
    for ii = 1:length(chain)
        chain_out[ii] = SMLMBaGoL.removeoverlap(chain[ii], datasize, roi, roioverlap)
    end

    return chain_out
end

"""
    chain_out = removeoverlap(chain::Matrix{Vector{SMLMBaGoL.BaGoLChain2D}},
                              datasize::Vector{Int},
                              rois::Matrix{Vector{Float64}},
                              roioverlap::Float64)

Remove the emitters in `chain` that fall outside the valid exploration region.

# Description
This method removes the emitters in `chain` that fall outside the valid region
of exploration.  That is, emitter locations that fell in the outer 
`roioverlap/2` edge of the `roi+-0.5` will be discarded.  Emitters at the edge 
of the data boundary defined by `datasize` (e.g., if datasize = [128; 128], the 
data is 128 pixels wide, so the edges [0.5; 128.5] are the data boundaries) 
will be retained, as these regions do not overlap with any other chain rois.
This method dispatches on removeoverlap(chain::SMLMBaGoL.BaGoLChain2D, ...)
for each chain in the vector, concatenating the resulting valid positions.
This method dispatches on
removeoverlap(chain::Vector{SMLMBaGoL.BaGoLChain2D}, ...) for each entry of the
matrix `chain`.

# Inputs
- `chain`: Matrix of vectors of state chains.
- `datasize`: Size of the raw data matrix. ([ysize xsize])
- `rois`: Matrix of rois, with indexing matching that of `chain`.
- `roioverlap`: Overlap of neighboring chain rois.

# Outputs
- `chain_out`: Chains with the invalid emitters removed. Note that the 
               allocations in `state.z` are replaced with `-1` to to indicate
               allocations to emitters that were removed.
"""
function removeoverlap(chain::Matrix{Vector{SMLMBaGoL.BaGoLChain2D}},
                       datasize::Vector{Int},
                       rois::Matrix{Vector{Float64}},
                       roioverlap::Float64)

    # Loop over the provided chains and create a matrix of valid emitter
    # coordinates.
    chain_out = Matrix{Vector{SMLMBaGoL.BaGoLChain2D}}(undef, size(chain))
    for ii = 1:prod(size(chain))
        chain_out[ii] = SMLMBaGoL.removeoverlap(chain[ii], datasize, rois[ii], roioverlap)
    end

    return chain_out
end
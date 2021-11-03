# This file contains functions for removing parts of an RJMCMC chain present
# within subregion overlaps (see SMLMBaGoL.gensubregions()).

function findoverlap(state::SMLMBaGoL.BaGoLState2D,
                     minval::Matrix{Float64},
                     maxval::Matrix{Float64})
    # Find any emitters in the `state` that don't fall within the boundaries.
    overlapping = any(state.μ .<= repeat(minval, state.k), dims=2) .+
        any(state.μ .>= repeat(maxval, state.k), dims=2)

    return overlapping
end

function findoverlap(chain::SMLMBaGoL.BaGoLChain2D,
                     minval::Matrix{Float64},
                     maxval::Matrix{Float64})
    # Loop through each state in `chain` and search for emitters in the overlap
    # regions.
    overlapping = Vector{BitVector}(undef, chain.n)
    for nn = 1:chain.n
        overlapping[nn] = findoverlap(chain.states[nn], minval, maxval)
    end

    return overlapping
end

function removeoverlap!(chain::SMLMBaGoL.BaGoLChain2D,
                        datasize::Vector{Int},
                        roi::Vector{Float64},
                        roioverlap::Float64)
    # Remove any parts of the `chain` contained within `roioverlap/2` of the 
    # edge. If the chain is near an actual data boundary (e.g., if the raw
    # data is 128 pixels wide, the edges [0.5; 128.5] are the data boundaries),
    # we'll retain the chain along the edge (since that edge is not an overlap
    # region!).

    # Define the minimum and maximum allowable coordinates for this `roi`.
    ymin = roi[1]>1.0 ? roi[1]+roioverlap/2 : 0.5
    ymax = roi[3]<datasize[1] ? roi[3]-roioverlap/2 : datasize[1]+0.5
    xmin = roi[2]>1.0 ? roi[2]+roioverlap/2 : 0.5
    xmax = roi[4]<datasize[2] ? roi[4]-roioverlap/2 : datasize[2]+0.5

    # Remove emitters from `chain` that fall within the overlap regions.
    minval = [ymin xmin]
    maxval = [ymax xmax]
    SMLMBaGoL.removeoverlap!(chain, minval, maxval)
end

function removeoverlap!(chain::SMLMBaGoL.BaGoLChain2D, 
                        minval::Matrix{Float64}, 
                        maxval::Matrix{Float64})
    # Loop over the states in `chain` and remove emitters in the overlap region.
    for nn = 1:chain.n
        overlapping = SMLMBaGoL.findoverlap(chain.states[nn], minval, maxval)
        SMLMBaGoL.removeemitter!(chain.states[nn], overlapping)
    end
end
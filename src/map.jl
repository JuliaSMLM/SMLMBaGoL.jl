using SMLMData
using StatsBase
using Clustering

# This file contains functions/methods related to computing maximum a 
# posteriori estimates (MAP) of quantities related to BaGoL analysis.

"""
    μ, σ_μ, a, σ_a, nalloc = mapn(chain::SMLMBaGoL.BaGoLChain2D)

This method computes the MAPN emitter position estimates.

# Description
This method finds the mode of the number of emitters in `chain` and then
performs k-means clustering on the proposed emitter positions for all states
with `k=mode(k)`.  The resulting `k` clusters are each averaged to estimate 
the positions, drift velocities, and associated uncertainties of the `k` 
emitters.

# Inputs
- `chain`: Chain of Markov states.

# Outputs
- `μ`: Position estimates of the MAPN emitters. (kx2)([y x])
- `σ_μ`: Uncertainties in the positions `μ`. (kx2)([y x])
- `a`: Drift velocities of the MAPN emitters. (kx2)([y x])
- `σ_a`: Uncertainties in the velocities `a`. (kx2)([y x])
- `nalloc`: Number of localizations allocated to each emitter. (kx1)
"""
function mapn(chain::SMLMBaGoL.BaGoLChain2D)
    # Determine the mode number of emitters in this chain.
    k, _ = SMLMBaGoL.catfields(chain)
    if isempty(k)
        return Matrix{Float32}(undef, 0, 2),
        Matrix{Float32}(undef, 0, 2),
        Matrix{Float32}(undef, 0, 2),
        Matrix{Float32}(undef, 0, 2),
        Vector{Int}(undef, 0),
        Vector{Int}(undef, 0)
    end
    n = Int(StatsBase.mode(k))

    # Extract all states with `n` emitters.
    mapnbool = k .== n
    mapnstates = deepcopy(chain.states[mapnbool])
    _, zmapn, μmapn, amapn = SMLMBaGoL.catfields(mapnstates)

    # Perform k-means clustering on the states with `n` emitters.
    mapnresults = Clustering.kmeans(transpose(μmapn), n)

    # Estimate emitter positions from the kmeans results and determine how many
    # localizations were allocated to each emitter in each state.
    μout = Matrix{Float32}(undef, n, 2)
    σ_μout = Matrix{Float32}(undef, n, 2)
    aout = Matrix{Float32}(undef, n, 2)
    σ_aout = Matrix{Float32}(undef, n, 2)
    nalloc = Vector{Vector{Int}}(undef, n)
    for nn = 1:n
        nnmembers = mapnresults.assignments .== nn
        μout[nn, :] = StatsBase.mean(μmapn[nnmembers, :], dims=1)
        σ_μout[nn, :] = StatsBase.std(μmapn[nnmembers, :], dims=1)
        aout[nn, :] = StatsBase.mean(amapn[nnmembers, :], dims=1)
        σ_aout[nn, :] = StatsBase.std(amapn[nnmembers, :], dims=1)
        nalloc[nn] = Vector{Int}(undef, length(zmapn))
        for ii = 1:length(zmapn)
            nalloc[nn][ii] = Int(sum(zmapn[ii] .== nn))
        end
    end

    return μout, σ_μout, aout, σ_aout, nalloc, n
end

"""
    μ, σ_μ, a, σ_a, nalloc = mapn(chain::Vector{SMLMBaGoL.BaGoLChain2D})

This method computes the MAPN emitter position estimates.

# Description
This method loops through each entry of `chain` and dispatches on 
mapn(chain::SMLMBaGoL.BaGoLChain2D), concatenating the results across all
chains.

# Inputs
- `chain`: Vector of chains of Markov states.

# Outputs
- `μ`: Position estimates of the MAPN emitters. (kx2)([y x])
- `σ_μ`: Uncertainties in the positions `μ`. (kx2)([y x])
- `a`: Drift velocities of the MAPN emitters. (kx2)([y x])
- `σ_a`: Uncertainties in the velocities `a`. (kx2)([y x])
- `nalloc`: Number of localizations allocated to each emitter. (kx1)
"""
function mapn(chain::Vector{SMLMBaGoL.BaGoLChain2D})
    # Loop over entries in `chain` and dispatch on the single chain mapn().
    μout = Matrix{Float32}(undef, 0, 2)
    σ_μout = Matrix{Float32}(undef, 0, 2)
    aout = Matrix{Float32}(undef, 0, 2)
    σ_aout = Matrix{Float32}(undef, 0, 2)
    nallocout = Vector{Int}(undef, 0)
    nout = Vector{Int}(undef, 0)
    for ii = 1:length(chain)
        μ, σ_μ, a, σ_a, nalloc, n = SMLMBaGoL.mapn(chain[ii])
        μout = [μout; μ]
        σ_μout = [σ_μout; σ_μ]
        aout = [aout; a]
        σ_aout = [σ_aout; σ_a]
        nallocout = [nallocout; nalloc]
        nout = [nout; n]
    end

    return μout, σ_μout, aout, σ_aout, nallocout, nout
end

"""
    μ, σ_μ, a, σ_a, nalloc = mapn(chain::Matrix{Vector{SMLMBaGoL.BaGoLChain2D}})

This method computes the MAPN emitter position estimates.

# Description
This method loops through each entry of `chain` and dispatches on 
mapn(chain::Vector{SMLMBaGoL.BaGoLChain2D}), concatenating the results across 
all chains.

# Inputs
- `chain`: Matrix of vectors of chains of Markov states.

# Outputs
- `μ`: Position estimates of the MAPN emitters. (kx2)([y x])
- `σ_μ`: Uncertainties in the positions `μ`. (kx2)([y x])
- `a`: Drift velocities of the MAPN emitters. (kx2)([y x])
- `σ_a`: Uncertainties in the velocities `a`. (kx2)([y x])
- `nalloc`: Number of localizations allocated to each emitter. (kx1)
"""
function mapn(chain::Matrix{Vector{SMLMBaGoL.BaGoLChain2D}})
    # Loop over entries in `chain` and dispatch on the single chain mapn().
    μout = Matrix{Float32}(undef, 0, 2)
    σ_μout = Matrix{Float32}(undef, 0, 2)
    aout = Matrix{Float32}(undef, 0, 2)
    σ_aout = Matrix{Float32}(undef, 0, 2)
    nallocout = Vector{Int}(undef, 0)
    nout = Vector{Int}(undef, 0)
    for ii = 1:prod(size(chain))
        μ, σ_μ, a, σ_a, nalloc, n = SMLMBaGoL.mapn(chain[ii])
        μout = [μout; μ]
        σ_μout = [σ_μout; σ_μ]
        aout = [aout; a]
        σ_aout = [σ_aout; σ_a]
        nallocout = [nallocout; nalloc]
        nout = [nout; n]
    end

    return μout, σ_μout, aout, σ_aout, nallocout, nout
end

"""
    smld = convert_mapn_smld(mapnout::Tuple{Matrix{T}, Matrix{T}, Matrix{T}, Matrix{T}, Vector{Int}, Vector{Int}} where T<:AbstractFloat; 
        smld_in::SMLMData.SMLD2D = SMLMData.SMLD2D())

This method converts a tuple of outputs from mapn() into an SMLMData.SMLD2D structure.

# Inputs
- `mapnout`: Tuple of outputs from SMLMBaGoL.mapn() (e.g., 
             mapnout = SMLMBaGoL.mapn(chain)).
- `datasize`: See SMLMData.SMLD2D().  Default value defines the smallest square
              encompassing the localizations where the side length is a power 
              of 2.
- `nframes`: See SMLMData.SMLD2D()
- `ndatasets`: See SMLMData.SMLD2D()

# Outputs
- `smld`: SMLMData.SMLD2D structure containing the MAPN localizations defined
          by the input `mapnout`.
"""
function convert_mapn_smld(mapnout::Tuple{Matrix{T},Matrix{T},Matrix{T},Matrix{T},Vector{Any},Vector{Int}} where {T<:AbstractFloat};
    datasize::Vector{Int}=2^ceil(Int, log2(maximum(mapnout[1]))) * [1; 1],
    nframes::Int=1,
    ndatasets::Int=1)

    # Place the mapn localizations into an smld structure.
    smld = SMLMData.SMLD2D(size(mapnout[1], 1))
    smld.y = copy(mapnout[1][:, 1])
    smld.x = copy(mapnout[1][:, 2])
    smld.σ_y = copy(mapnout[2][:, 1])
    smld.σ_x = copy(mapnout[2][:, 2])

    # Populate some additional fields describing the data.
    smld.datasize = datasize
    smld.nframes = nframes
    smld.ndatasets = ndatasets

    return smld
end
# Spatial utilities for dimension-agnostic coordinate access

# Abstract model types (defined early for CollapsedState and dispatch)
"""Base type for spatial prior models (FlatSpatial, LocmixSpatial)."""
abstract type AbstractSpatialModel end
"""Base type for allocation models (DMAllocation, DecoupledAllocation)."""
abstract type AbstractAllocationModel end
"""Base type for cluster sufficient statistics (ClusterStats, MultiClusterStats)."""
abstract type AbstractClusterStats end
"""Base type for per-loc precomputed contributions (LocPrecision, MultiLocPrecision)."""
abstract type AbstractLocPrecision end

"""
    get_coords(loc)

Extract coordinates as a static vector. Works with any emitter type.
"""
get_coords(loc::SMLMData.Emitter2DFit) = SVector(loc.x, loc.y)
get_coords(loc::SMLMData.Emitter3DFit) = SVector(loc.x, loc.y, loc.z)

# Generic fallback for any AbstractEmitter with x, y fields
function get_coords(loc::SMLMData.AbstractEmitter)
    if hasproperty(loc, :z)
        return SVector(loc.x, loc.y, loc.z)
    else
        return SVector(loc.x, loc.y)
    end
end

"""
    get_sigma(loc)

Extract uncertainty as a static vector. Works with any emitter type.
"""
get_sigma(loc::SMLMData.Emitter2DFit) = SVector(loc.σ_x, loc.σ_y)
get_sigma(loc::SMLMData.Emitter3DFit) = SVector(loc.σ_x, loc.σ_y, loc.σ_z)

# Generic fallback
function get_sigma(loc::SMLMData.AbstractEmitter)
    if hasproperty(loc, :σ_z)
        return SVector(loc.σ_x, loc.σ_y, loc.σ_z)
    else
        return SVector(loc.σ_x, loc.σ_y)
    end
end

"""
    mean_sigma(loc)

Compute geometric mean of uncertainties for precision weighting.
"""
mean_sigma(loc) = sqrt(prod(get_sigma(loc)))

"""
    get_cov_xy(loc)

Extract x-y covariance. Returns 0.0 for emitter types without covariance field.
Uses compile-time field check for zero overhead when field is absent.
"""
@generated function get_cov_xy(loc::T) where T<:SMLMData.AbstractEmitter
    if hasfield(T, :σ_xy)
        return :(loc.σ_xy)
    else
        return :(0.0)
    end
end

"""
    coords_matrix(locs)

Convert vector of localizations to coordinate matrix (dims × n_locs).
Used for building KDTree.
"""
function coords_matrix(locs::Vector{<:SMLMData.AbstractEmitter})
    if isempty(locs)
        return zeros(2, 0)
    end
    n = length(locs)
    ndims = length(get_coords(locs[1]))
    mat = zeros(ndims, n)
    for (i, loc) in enumerate(locs)
        mat[:, i] = get_coords(loc)
    end
    return mat
end

"""
    pairwise_distance(p1, p2)

Euclidean distance between two coordinate vectors.
"""
pairwise_distance(p1::SVector, p2::SVector) = sqrt(sum((p1 .- p2).^2))
pairwise_distance(p1::AbstractVector, p2::AbstractVector) = sqrt(sum((p1 .- p2).^2))

"""
    precision_weighted_distance(loc_i, loc_j)

Compute precision-weighted distance between two localizations.
d_eff = ||p_i - p_j|| / (σ_i + σ_j) where σ is geometric mean of uncertainties.
"""
function precision_weighted_distance(
    loc_i::SMLMData.AbstractEmitter,
    loc_j::SMLMData.AbstractEmitter
)
    d = pairwise_distance(get_coords(loc_i), get_coords(loc_j))
    σ_sum = mean_sigma(loc_i) + mean_sigma(loc_j)
    return d / σ_sum
end

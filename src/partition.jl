# Spatial partitioning for large datasets

"""
    Partition{E<:SMLMData.AbstractEmitter}

A partition of localizations for parallel processing.

# Fields
- `id`: Unique partition identifier
- `locs`: Vector of localizations in this partition (typed for dispatch)
- `original_indices`: Indices mapping back to original input
- `is_boundary`: Flags for localizations near partition edge
- `parent_id`: 0 for original DBSCAN clusters, >0 if split from oversized
- `is_overlap`: Flags for localizations in an overlap strip (result emitters here are discarded)
- `core_bounds`: If split with overlap, the axis-aligned half-space defining the core region.
  `nothing` for unsplit DBSCAN partitions. For bisected partitions:
  `(axis::Vector{Float64}, threshold::Float64, side::Symbol)` where
  `side ∈ {:left, :right}` means core = `dot(pos, axis) ≤ threshold` or `> threshold`.
"""
struct Partition{E<:SMLMData.AbstractEmitter}
    id::Int
    locs::Vector{E}
    original_indices::Vector{Int}
    is_boundary::BitVector
    parent_id::Int
    is_overlap::BitVector
    core_bounds::Union{Nothing, NamedTuple{(:axis, :threshold, :side), Tuple{Vector{Float64}, Float64, Symbol}}}
end

# Backward-compatible constructor (no overlap)
function Partition{E}(id::Int, locs::Vector{E}, original_indices::Vector{Int},
                      is_boundary::BitVector, parent_id::Int) where E<:SMLMData.AbstractEmitter
    Partition{E}(id, locs, original_indices, is_boundary, parent_id,
                 falses(length(locs)), nothing)
end

"""
    partition_locs(locs; partition_sigma, min_size, max_size, skip_size, boundary_margin, overlap)

Partition localizations using precision-weighted DBSCAN.

Two localizations are neighbors if `||p_i - p_j|| / (σ_i + σ_j) < partition_sigma`.

# Arguments
- `locs`: Vector of localizations
- `partition_sigma=3.0`: DBSCAN threshold in sigma units (Inf = no partitioning)
- `min_size=0`: Minimum locs per partition (clusters below this are noise)
- `max_size=1000`: Split partitions larger than this
- `skip_size=typemax(Int)`: Skip partitions larger than this (Inf = never skip)
- `boundary_margin=0.0`: Distance from edge to flag as boundary (0 = auto: partition_sigma×median(σ))
- `overlap=:auto`: Overlap width for bisected oversized partitions. `:auto` uses
  `partition_sigma × median(σ)` (μm). Numeric value in μm. `0.0` disables overlap.

# Returns
- `partitions`: Vector of Partition for valid clusters
- `skipped`: Vector of Partition for skipped oversized clusters
"""
function partition_locs(
    locs::Vector{E};
    partition_sigma::Float64=3.0,
    min_size::Int=0,
    max_size::Int=1000,
    skip_size::Int=typemax(Int),
    boundary_margin::Float64=0.0,
    overlap::Union{Float64, Symbol}=:auto
) where E<:SMLMData.AbstractEmitter
    if isempty(locs)
        return Partition{E}[], Partition{E}[]
    end

    # Auto boundary margin: scale with partition_sigma (partition gap ≈ partition_sigma*(σ_i+σ_j))
    sigmas = [mean_sigma(loc) for loc in locs]
    med_sigma = median(sigmas)
    if boundary_margin <= 0.0
        boundary_margin = partition_sigma * med_sigma
    end

    # Resolve overlap for bisected partitions
    overlap_um = if overlap === :auto
        partition_sigma * med_sigma
    elseif overlap isa Float64
        overlap
    else
        error("overlap must be :auto or a Float64 in μm (got $overlap)")
    end

    # Run precision-weighted DBSCAN
    labels = precision_dbscan(locs, partition_sigma, min_size)

    # Group by cluster label
    cluster_ids = unique(labels)
    filter!(x -> x > 0, cluster_ids)  # Remove noise label (0)

    partitions = Partition{E}[]
    skipped = Partition{E}[]
    partition_id = 1

    for cluster_id in cluster_ids
        indices = findall(==(cluster_id), labels)
        cluster_locs = locs[indices]

        # Create partition
        is_boundary = mark_boundaries(cluster_locs, boundary_margin)
        partition = Partition{E}(partition_id, cluster_locs, indices, is_boundary, 0)

        n_locs = length(cluster_locs)
        if n_locs <= max_size
            # Small enough, keep as-is
            push!(partitions, partition)
            partition_id += 1
        elseif n_locs >= skip_size
            # Too large, skip entirely
            push!(skipped, partition)
            partition_id += 1
        else
            # Between max_size and skip_size: split recursively with overlap
            sub_partitions = split_partition(partition, max_size, boundary_margin, partition_id; overlap=overlap_um)
            for sp in sub_partitions
                push!(partitions, sp)
                partition_id += 1
            end
        end
    end

    return partitions, skipped
end

"""
    precision_dbscan(locs, nsigma, min_pts)

Precision-weighted DBSCAN clustering.

Two localizations are neighbors if d_eff(i, j) < nsigma where
d_eff = ||p_i - p_j|| / (σ_i + σ_j).

# Returns
Vector of cluster labels (0 = noise, 1..n = cluster IDs)
"""
function precision_dbscan(
    locs::Vector{<:SMLMData.AbstractEmitter},
    nsigma::Float64,
    min_pts::Int
)
    n = length(locs)
    if n == 0
        return Int[]
    end

    # Build KDTree for fast neighbor queries
    coords = coords_matrix(locs)
    tree = KDTree(coords)

    # Maximum physical distance for any pair to be neighbors
    # (happens when both have max sigma)
    max_sigma = maximum(mean_sigma(loc) for loc in locs)
    max_radius = nsigma * 2 * max_sigma

    labels = zeros(Int, n)  # 0 = unvisited/noise
    cluster_id = 0

    for i in 1:n
        labels[i] != 0 && continue  # Already assigned

        # Find precision-weighted neighbors
        neighbors = precision_neighbors(tree, locs, i, nsigma, max_radius)

        if length(neighbors) < min_pts
            labels[i] = 0  # Mark as noise (may be claimed later)
            continue
        end

        # Start new cluster
        cluster_id += 1
        labels[i] = cluster_id

        # Expand cluster
        seed_set = Set(neighbors)
        delete!(seed_set, i)

        while !isempty(seed_set)
            j = pop!(seed_set)

            if labels[j] == 0  # Was noise, now border point
                labels[j] = cluster_id
            end

            labels[j] != 0 && labels[j] != cluster_id && continue  # Already in another cluster

            labels[j] = cluster_id

            # Find neighbors of j
            j_neighbors = precision_neighbors(tree, locs, j, nsigma, max_radius)

            if length(j_neighbors) >= min_pts
                # Add new neighbors to seed set
                for k in j_neighbors
                    if labels[k] == 0
                        push!(seed_set, k)
                    end
                end
            end
        end
    end

    return labels
end

"""
    precision_neighbors(tree, locs, i, nsigma, max_radius)

Find all localizations that are precision-weighted neighbors of loc i.
Uses KDTree for initial range query, then filters by precision-weighted distance.
"""
function precision_neighbors(
    tree::KDTree,
    locs::Vector{<:SMLMData.AbstractEmitter},
    i::Int,
    nsigma::Float64,
    max_radius::Float64
)
    loc_i = locs[i]
    coords_i = get_coords(loc_i)

    # Range query for candidates
    candidates = inrange(tree, coords_i, max_radius)

    # Filter by precision-weighted distance
    neighbors = Int[]
    for j in candidates
        j == i && continue
        d_eff = precision_weighted_distance(loc_i, locs[j])
        if d_eff < nsigma
            push!(neighbors, j)
        end
    end

    return neighbors
end

"""
    split_partition(partition, max_size, boundary_margin, start_id; overlap)

Recursively split an oversized partition using principal axis bisection.

When `overlap > 0`, localizations within `overlap` μm of the cut plane are
included in BOTH sub-partitions (marked `is_overlap=true`). After RJMCMC,
emitters in the overlap strip are discarded — only core-region emitters are
kept. This ensures emitters near the cut have full neighbor context.

Returns vector of sub-partitions, each with core size ≤ max_size.
"""
function split_partition(
    partition::Partition{E},
    max_size::Int,
    boundary_margin::Float64,
    start_id::Int;
    overlap::Float64=0.0
) where E
    locs = partition.locs
    n = length(locs)

    if n <= max_size
        return [Partition{E}(start_id, locs, partition.original_indices,
                            partition.is_boundary, partition.id,
                            falses(n), nothing)]
    end

    # Find principal axis (direction of maximum variance)
    coords = coords_matrix(locs)
    centroid = vec(mean(coords, dims=2))
    centered = coords .- centroid

    # Covariance matrix
    cov_mat = (centered * centered') / (n - 1)

    # Principal axis is eigenvector with largest eigenvalue
    eigenvalues, eigenvectors = eigen(cov_mat)
    principal_axis = eigenvectors[:, argmax(eigenvalues)]

    # Project onto principal axis
    projections = vec(principal_axis' * centered)

    # Split at median
    median_proj = median(projections)

    # Core membership: which side of the cut each loc belongs to
    core_left = projections .<= median_proj
    core_right = .!core_left

    # Overlap strip: locs within `overlap` μm of the cut plane (included in both sides)
    # Project overlap distance onto the principal axis (axis is unit vector, so
    # distance along axis = physical distance when axis is normalized)
    axis_norm = sqrt(principal_axis[1]^2 + principal_axis[2]^2)
    overlap_proj = overlap / max(axis_norm, 1e-10)
    near_cut = abs.(projections .- median_proj) .< overlap_proj

    # Create sub-partitions
    sub_partitions = Partition{E}[]
    current_id = start_id

    for (side, core_mask) in [(:left, core_left), (:right, core_right)]
        # Include core locs + overlap locs from the other side
        include_mask = core_mask .| near_cut
        local_indices = findall(include_mask)
        isempty(local_indices) && continue

        sub_locs = locs[local_indices]
        sub_orig_indices = partition.original_indices[local_indices]

        # Mark boundary: near original boundary
        sub_is_boundary = BitVector(undef, length(sub_locs))
        for (i, idx) in enumerate(local_indices)
            sub_is_boundary[i] = partition.is_boundary[idx]
        end

        # Mark overlap: locs that are NOT in this side's core
        sub_is_overlap = BitVector(undef, length(sub_locs))
        for (i, idx) in enumerate(local_indices)
            sub_is_overlap[i] = !core_mask[idx]
        end

        # Core bounds for emitter filtering after RJMCMC
        bounds = (axis=collect(principal_axis), threshold=median_proj + centroid' * principal_axis, side=side)

        sub_partition = Partition{E}(current_id, sub_locs, sub_orig_indices,
                                     sub_is_boundary, partition.id,
                                     sub_is_overlap, bounds)

        # Recurse if core locs still oversized.
        # Guard: only recurse if the core actually got smaller than the input.
        # When overlap captures the entire cluster (tight data), the split doesn't
        # reduce size and recursion would be infinite.
        n_core = count(.!sub_is_overlap)
        n_parent_core = count(.!partition.is_overlap)
        if n_core > max_size && n_core < n_parent_core
            recursed = split_partition(sub_partition, max_size, boundary_margin, current_id; overlap)
            append!(sub_partitions, recursed)
            current_id += length(recursed)
        else
            push!(sub_partitions, sub_partition)
            current_id += 1
        end
    end

    return sub_partitions
end

"""
    mark_boundaries(locs, margin)

Mark localizations that are near the convex hull boundary.
Uses bounding box approximation for efficiency.
"""
function mark_boundaries(
    locs::Vector{<:SMLMData.AbstractEmitter},
    margin::Float64
)
    n = length(locs)
    if n == 0
        return BitVector()
    end

    is_boundary = falses(n)

    # Get coordinate bounds
    coords = coords_matrix(locs)
    ndims = size(coords, 1)

    for d in 1:ndims
        min_val = minimum(coords[d, :])
        max_val = maximum(coords[d, :])

        for i in 1:n
            if coords[d, i] - min_val < margin || max_val - coords[d, i] < margin
                is_boundary[i] = true
            end
        end
    end

    return is_boundary
end

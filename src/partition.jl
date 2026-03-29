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
"""
struct Partition{E<:SMLMData.AbstractEmitter}
    id::Int
    locs::Vector{E}
    original_indices::Vector{Int}
    is_boundary::BitVector
    parent_id::Int
end

"""
    partition_locs(locs; partition_sigma, min_size, max_size, skip_size, boundary_margin)

Partition localizations using precision-weighted DBSCAN.

Two localizations are neighbors if `||p_i - p_j|| / (σ_i + σ_j) < partition_sigma`.

# Arguments
- `locs`: Vector of localizations
- `partition_sigma=3.0`: DBSCAN threshold in sigma units (Inf = no partitioning)
- `min_size=0`: Minimum locs per partition (clusters below this are noise)
- `max_size=1000`: Split partitions larger than this
- `skip_size=typemax(Int)`: Skip partitions larger than this (Inf = never skip)
- `boundary_margin=0.0`: Distance from edge to flag as boundary (0 = auto: partition_sigma×median(σ))

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
    boundary_margin::Float64=0.0
) where E<:SMLMData.AbstractEmitter
    if isempty(locs)
        return Partition{E}[], Partition{E}[]
    end

    # Auto boundary margin: scale with partition_sigma (partition gap ≈ partition_sigma*(σ_i+σ_j))
    if boundary_margin <= 0.0
        sigmas = [mean_sigma(loc) for loc in locs]
        boundary_margin = partition_sigma * median(sigmas)
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
            # Between max_size and skip_size: split recursively
            sub_partitions = split_partition(partition, max_size, boundary_margin, partition_id)
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
    split_partition(partition, max_size, boundary_margin, start_id)

Recursively split an oversized partition using principal axis bisection.

Returns vector of sub-partitions, each with size <= max_size.
"""
function split_partition(
    partition::Partition{E},
    max_size::Int,
    boundary_margin::Float64,
    start_id::Int
) where E
    locs = partition.locs
    n = length(locs)

    if n <= max_size
        return [Partition{E}(start_id, locs, partition.original_indices,
                            partition.is_boundary, partition.id)]
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

    # Assign to sub-partitions
    left_mask = projections .<= median_proj
    right_mask = .!left_mask

    left_indices = findall(left_mask)
    right_indices = findall(right_mask)

    # Create sub-partitions
    sub_partitions = Partition{E}[]
    current_id = start_id

    for (local_indices, is_left) in [(left_indices, true), (right_indices, false)]
        isempty(local_indices) && continue

        sub_locs = locs[local_indices]
        sub_orig_indices = partition.original_indices[local_indices]

        # Mark boundary: locs near the split plane or near original boundary
        sub_is_boundary = BitVector(undef, length(sub_locs))
        for (i, idx) in enumerate(local_indices)
            # Near split plane?
            dist_to_plane = abs(projections[idx] - median_proj)
            near_split = dist_to_plane < boundary_margin

            # Already a boundary loc?
            was_boundary = partition.is_boundary[idx]

            sub_is_boundary[i] = near_split || was_boundary
        end

        sub_partition = Partition{E}(current_id, sub_locs, sub_orig_indices,
                                     sub_is_boundary, partition.id)

        # Recurse if still oversized
        if length(sub_locs) > max_size
            recursed = split_partition(sub_partition, max_size, boundary_margin, current_id)
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

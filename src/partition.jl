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
- `is_overlap`: Flags for localizations included from sibling sub-partitions
  for spatial context. Emitters primarily composed of overlap locs are discarded
  after MAP-N extraction.
"""
struct Partition{E<:SMLMData.AbstractEmitter}
    id::Int
    locs::Vector{E}
    original_indices::Vector{Int}
    is_boundary::BitVector
    parent_id::Int
    is_overlap::BitVector
end

# Backward-compatible constructor (no overlap)
function Partition{E}(id::Int, locs::Vector{E}, original_indices::Vector{Int},
                      is_boundary::BitVector, parent_id::Int) where E<:SMLMData.AbstractEmitter
    Partition{E}(id, locs, original_indices, is_boundary, parent_id,
                 falses(length(locs)))
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

# ============================================================================
# MST-based partition splitting
#
# For oversized DBSCAN clusters: build a precision-weighted kNN graph,
# extract MST via Kruskal's algorithm, then remove heaviest edges until
# all components have ≤ max_size nodes. Each component becomes a sub-partition.
# Overlap locs are added from sibling components via pointwise proximity.
# ============================================================================

# Union-Find for Kruskal's MST
mutable struct _UnionFind
    parent::Vector{Int}
    rank::Vector{Int}
    size::Vector{Int}
end

function _UnionFind(n::Int)
    _UnionFind(collect(1:n), zeros(Int, n), ones(Int, n))
end

function _uf_find(uf::_UnionFind, x::Int)
    while uf.parent[x] != x
        uf.parent[x] = uf.parent[uf.parent[x]]  # path halving
        x = uf.parent[x]
    end
    return x
end

function _uf_union!(uf::_UnionFind, a::Int, b::Int)
    ra, rb = _uf_find(uf, a), _uf_find(uf, b)
    ra == rb && return false
    if uf.rank[ra] < uf.rank[rb]
        ra, rb = rb, ra
    end
    uf.parent[rb] = ra
    uf.size[ra] += uf.size[rb]
    if uf.rank[ra] == uf.rank[rb]
        uf.rank[ra] += 1
    end
    return true
end

"""
    split_partition(partition, max_size, boundary_margin, start_id; overlap, k_neighbors)

Split an oversized partition using MST-based graph cutting.

1. Build kNN graph with precision-weighted distances
2. Extract MST via Kruskal's algorithm
3. Remove heaviest edges until all components ≤ max_size
4. Add overlap locs from sibling components (pointwise proximity)

Returns vector of sub-partitions with natural, data-driven boundaries.
"""
function split_partition(
    partition::Partition{E},
    max_size::Int,
    boundary_margin::Float64,
    start_id::Int;
    overlap::Float64=0.0,
    k_neighbors::Int=15
) where E
    locs = partition.locs
    n = length(locs)

    if n <= max_size
        return [Partition{E}(start_id, locs, partition.original_indices,
                            partition.is_boundary, partition.id,
                            falses(n))]
    end

    # --- Build kNN graph with precision-weighted distances ---
    coords = coords_matrix(locs)
    tree = KDTree(coords)
    max_sigma = maximum(mean_sigma(loc) for loc in locs)
    k = min(k_neighbors, n - 1)

    # Collect weighted edges (i, j, distance)
    edges = Tuple{Int, Int, Float64}[]
    for i in 1:n
        # Use physical radius for KDTree query, then filter by precision distance
        idxs, _ = knn(tree, coords[:, i], k + 1)  # +1 for self
        for j in idxs
            j == i && continue
            d = precision_weighted_distance(locs[i], locs[j])
            push!(edges, (i, j, d))
        end
    end

    # --- MST via Kruskal's ---
    sort!(edges, by=e -> e[3])
    uf = _UnionFind(n)
    mst_edges = Tuple{Int, Int, Float64}[]
    for (i, j, d) in edges
        if _uf_union!(uf, i, j)
            push!(mst_edges, (i, j, d))
            length(mst_edges) == n - 1 && break
        end
    end

    # --- Size-constrained cuts: remove heaviest MST edges until all components ≤ max_size ---
    # Sort MST edges by weight descending
    sort!(mst_edges, by=e -> -e[3])

    # Rebuild components after removing edges
    uf_cut = _UnionFind(n)
    # Add edges lightest-first, skipping the heaviest ones that would create oversized components
    # Simpler: iteratively remove heaviest edge if its component is oversized
    cut_edges = Set{Int}()  # indices into mst_edges to skip

    # Greedy: try adding all MST edges lightest-first, reject if would exceed max_size
    sorted_by_weight = sortperm(mst_edges, by=e -> e[3])
    uf_build = _UnionFind(n)
    for idx in sorted_by_weight
        i, j, d = mst_edges[idx]
        ri, rj = _uf_find(uf_build, i), _uf_find(uf_build, j)
        combined_size = uf_build.size[ri] + uf_build.size[rj]
        if combined_size <= max_size
            _uf_union!(uf_build, i, j)
        end
        # else: skip this edge (cut it), leaving the components separate
    end

    # --- Extract components ---
    component_map = Dict{Int, Vector{Int}}()
    for i in 1:n
        root = _uf_find(uf_build, i)
        if haskey(component_map, root)
            push!(component_map[root], i)
        else
            component_map[root] = [i]
        end
    end
    components = collect(values(component_map))

    # --- Build sub-partitions with pointwise sibling overlap ---
    # For each component, find locs in OTHER components within overlap distance
    sub_partitions = Partition{E}[]
    current_id = start_id

    # Component membership lookup
    loc_component = zeros(Int, n)
    for (ci, comp) in enumerate(components)
        for idx in comp
            loc_component[idx] = ci
        end
    end

    for (ci, core_indices) in enumerate(components)
        if overlap > 0.0
            # Find overlap locs: locs in sibling components within proximity
            overlap_indices = Int[]
            for core_idx in core_indices
                loc_i = locs[core_idx]
                σ_i = mean_sigma(loc_i)
                # Query nearby locs
                radius = overlap > 0 ? overlap : 3 * σ_i
                nearby = inrange(tree, coords[:, core_idx], radius)
                for j in nearby
                    if loc_component[j] != ci && j ∉ overlap_indices
                        push!(overlap_indices, j)
                    end
                end
            end
            unique!(overlap_indices)

            # Combine core + overlap
            all_indices = vcat(core_indices, overlap_indices)
            is_overlap = BitVector(vcat(falses(length(core_indices)),
                                         trues(length(overlap_indices))))
        else
            all_indices = core_indices
            is_overlap = falses(length(core_indices))
        end

        sub_locs = locs[all_indices]
        sub_orig = partition.original_indices[all_indices]
        sub_boundary = BitVector([partition.is_boundary[idx] for idx in all_indices])

        push!(sub_partitions, Partition{E}(current_id, sub_locs, sub_orig,
                                            sub_boundary, partition.id, is_overlap))
        current_id += 1
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

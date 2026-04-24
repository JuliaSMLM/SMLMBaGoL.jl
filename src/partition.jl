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
- `bridge_ratio=0.0`: Split DBSCAN clusters joined only by low-density bridges.
  Disabled by default. Values in (0, 1] prune points whose local DBSCAN-neighbor
  count is below this fraction of the cluster's median neighbor count.
- `min_split_size=3`: Minimum core component size retained by bridge refinement.

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
    overlap::Union{Float64, Symbol}=:auto,
    bridge_ratio::Float64=0.0,
    min_split_size::Int=3,
) where E<:SMLMData.AbstractEmitter
    bridge_ratio >= 0.0 ||
        throw(ArgumentError("bridge_ratio must be non-negative (got $bridge_ratio)"))
    min_split_size >= 1 ||
        throw(ArgumentError("min_split_size must be positive (got $min_split_size)"))
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

    # Run precision-weighted DBSCAN. `precision_neighbors` excludes the query
    # point, while public `min_size` is the total DBSCAN min-points convention.
    labels = precision_dbscan(locs, partition_sigma, max(min_size - 1, 0))
    if bridge_ratio > 0.0
        labels = refine_bridge_clusters(locs, labels, partition_sigma;
                                        bridge_ratio=bridge_ratio,
                                        min_split_size=min_split_size)
    end

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
    refine_bridge_clusters(locs, labels, nsigma; bridge_ratio, min_split_size)

Split DBSCAN clusters that are connected only by sparse bridge points.

DBSCAN's transitive closure can merge nearby structures through a thin chain of
localizations. This pass computes the within-cluster DBSCAN-neighbor count for
each point, removes low-count bridge candidates, finds connected components of
the remaining core graph, then assigns bridge points to the nearest retained
component. If refinement does not produce at least two components of
`min_split_size`, the original cluster is preserved.
"""
function refine_bridge_clusters(
    locs::Vector{<:SMLMData.AbstractEmitter},
    labels::Vector{Int},
    nsigma::Float64;
    bridge_ratio::Float64,
    min_split_size::Int=3,
)
    n = length(locs)
    n == 0 && return Int[]

    coords = coords_matrix(locs)
    tree = KDTree(coords)
    max_sigma = maximum(mean_sigma(loc) for loc in locs)
    max_radius = nsigma * 2 * max_sigma

    refined = zeros(Int, n)
    next_label = 0

    cluster_ids = sort!(filter(>(0), unique(labels)))
    for cid in cluster_ids
        cluster_indices = findall(==(cid), labels)
        m = length(cluster_indices)
        if m < 2 * min_split_size
            next_label += 1
            refined[cluster_indices] .= next_label
            continue
        end

        local_index = Dict{Int, Int}(gi => li for (li, gi) in enumerate(cluster_indices))
        neighbors = [Int[] for _ in 1:m]
        degrees = zeros(Int, m)

        for (li, gi) in enumerate(cluster_indices)
            for gj in precision_neighbors(tree, locs, gi, nsigma, max_radius)
                labels[gj] == cid || continue
                lj = local_index[gj]
                push!(neighbors[li], lj)
            end
            degrees[li] = length(neighbors[li])
        end

        median_degree = median(degrees)
        min_core_degree = max(1, ceil(Int, bridge_ratio * median_degree))
        is_core = degrees .>= min_core_degree

        # Connected components in the core-only neighbor graph.
        component_id = zeros(Int, m)
        components = Vector{Int}[]
        for li in 1:m
            is_core[li] || continue
            component_id[li] != 0 && continue

            push!(components, Int[])
            cid_local = length(components)
            stack = [li]
            component_id[li] = cid_local
            while !isempty(stack)
                u = pop!(stack)
                push!(components[cid_local], u)
                for v in neighbors[u]
                    is_core[v] || continue
                    component_id[v] == 0 || continue
                    component_id[v] = cid_local
                    push!(stack, v)
                end
            end
        end

        keep = [length(c) >= min_split_size for c in components]
        if count(keep) < 2
            next_label += 1
            refined[cluster_indices] .= next_label
            continue
        end

        kept_components = components[keep]
        new_labels = collect((next_label + 1):(next_label + length(kept_components)))
        local_to_new = zeros(Int, m)
        for (comp_idx, comp) in enumerate(kept_components)
            for li in comp
                local_to_new[li] = new_labels[comp_idx]
            end
        end

        # Attach pruned bridge points and undersized core fragments to the
        # nearest retained component, preserving all original localizations.
        for li in 1:m
            local_to_new[li] != 0 && continue
            local_to_new[li] = _nearest_component_label(
                li, kept_components, new_labels, cluster_indices, locs)
        end

        for li in 1:m
            refined[cluster_indices[li]] = local_to_new[li]
        end
        next_label += length(kept_components)
    end

    return refined
end

function _nearest_component_label(
    li::Int,
    components::Vector{Vector{Int}},
    labels::Vector{Int},
    cluster_indices::Vector{Int},
    locs::Vector{<:SMLMData.AbstractEmitter},
)
    gi = cluster_indices[li]
    best_label = labels[1]
    best_d = Inf
    for (comp_idx, comp) in enumerate(components)
        for lj in comp
            gj = cluster_indices[lj]
            d = precision_weighted_distance(locs[gi], locs[gj])
            if d < best_d
                best_d = d
                best_label = labels[comp_idx]
            end
        end
    end
    return best_label
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
# METIS-based partition splitting
#
# For oversized DBSCAN clusters: build a precision-weighted kNN graph,
# then use METIS multilevel k-way partitioning to split into balanced
# sub-partitions. METIS cuts at weak graph connections (density valleys)
# in non-uniform data, and produces compact balanced pieces in uniform data.
# Overlap locs are added from neighboring partitions via pointwise proximity.
# ============================================================================


"""
    split_partition(partition, max_size, boundary_margin, start_id; overlap, k_neighbors)

Split an oversized partition using METIS multilevel graph partitioning.

1. Build symmetric kNN graph with precision-weighted affinity weights
2. METIS k-way partition into ceil(N/target_size) balanced parts
3. Add overlap locs from neighboring partitions (pointwise proximity)

METIS cuts at weak graph connections (density valleys) in non-uniform data,
and produces compact balanced pieces in uniform data.

Returns vector of sub-partitions with balanced sizes and natural boundaries.
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

    # --- Build symmetric kNN graph with precision-weighted affinity ---
    coords = coords_matrix(locs)
    tree = KDTree(coords)
    k = min(k_neighbors, n - 1)

    # Collect edges: precision-weighted distance per pair, keep closest
    edge_dict = Dict{Tuple{Int,Int}, Float64}()
    for i in 1:n
        idxs, _ = knn(tree, coords[:, i], k + 1)
        for j in idxs
            j == i && continue
            d = precision_weighted_distance(locs[i], locs[j])
            key = i < j ? (i, j) : (j, i)
            if !haskey(edge_dict, key) || d < edge_dict[key]
                edge_dict[key] = d
            end
        end
    end

    # --- METIS k-way partition ---
    # Convert distances to integer affinities (closer = heavier = don't cut)
    max_d = maximum(values(edge_dict))
    I = Int[]; J = Int[]; W = Int[]
    for ((i, j), d) in edge_dict
        w = max(1, round(Int, 1000.0 * (max_d / max(d, 1e-10))))
        push!(I, i); push!(J, j); push!(W, w)
        push!(I, j); push!(J, i); push!(W, w)
    end
    adj = sparse(I, J, W, n, n)

    # Target ~90% of max_size per partition (leave room for halo)
    target_core = max(div(max_size * 9, 10), 1)
    nparts = max(2, cld(n, target_core))

    g = Metis.graph(adj)
    partition_vec = Metis.partition(g, nparts)

    # Extract components
    components = [Int[] for _ in 1:nparts]
    for i in 1:n
        push!(components[partition_vec[i]], i)
    end
    filter!(!isempty, components)

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

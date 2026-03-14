# Boundary deduplication for partitioned BaGoL

"""
    emitter_near_boundary(emitter, partition, margin)

Check if an emitter is near any boundary localization in the partition.
"""
function emitter_near_boundary(
    emitter::SMLMData.Emitter2DFit,
    partition::Partition,
    margin::Float64
)
    for (i, is_bound) in enumerate(partition.is_boundary)
        if is_bound
            loc = partition.locs[i]
            dx = emitter.x - loc.x
            dy = emitter.y - loc.y
            if sqrt(dx^2 + dy^2) < 2 * margin
                return true
            end
        end
    end
    return false
end

"""
    deduplicate_boundary_emitters(emitters, partition_ids, is_near_boundary, margin)

Use spatial indexing + Hungarian matching to identify and merge duplicate emitters
near partition boundaries. Only compares emitters from different partitions that
are within `margin` distance of each other.

Returns a new vector of Emitter2DFit with duplicates merged.
"""
function deduplicate_boundary_emitters(
    emitters::Vector{SMLMData.Emitter2DFit},
    partition_ids::Vector{Int},
    is_near_boundary::Vector{Bool},
    margin::Float64
)
    n = length(emitters)
    keep = trues(n)
    result = copy(emitters)

    # Collect boundary emitter indices
    boundary_idx = findall(is_near_boundary)
    isempty(boundary_idx) && return result

    # Build KDTree on boundary emitter positions
    coords = hcat([[emitters[i].x, emitters[i].y] for i in boundary_idx]...)
    tree = NearestNeighbors.KDTree(coords)

    # Find all pairs within margin distance
    # Group by (pid_i, pid_j) pair to do Hungarian per partition pair
    pair_map = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}()

    for (bi, gi) in enumerate(boundary_idx)
        keep[gi] || continue
        pid_i = partition_ids[gi]

        # Find neighbors within margin
        neighbors = NearestNeighbors.inrange(tree, coords[:, bi], margin)

        for bj in neighbors
            gj = boundary_idx[bj]
            gj <= gi && continue  # avoid duplicates and self
            keep[gj] || continue
            pid_j = partition_ids[gj]
            pid_i == pid_j && continue  # same partition, skip

            # Canonical order
            key = pid_i < pid_j ? (pid_i, pid_j) : (pid_j, pid_i)
            if !haskey(pair_map, key)
                pair_map[key] = Tuple{Int,Int}[]
            end
            push!(pair_map[key], (gi, gj))
        end
    end

    # For each partition pair with nearby boundary emitters, run Hungarian
    for ((pid_a, pid_b), pairs) in pair_map
        # Collect unique emitter indices per partition
        idx_a_set = Set{Int}()
        idx_b_set = Set{Int}()
        for (gi, gj) in pairs
            pa, pb = partition_ids[gi], partition_ids[gj]
            if pa == pid_a
                push!(idx_a_set, gi)
                push!(idx_b_set, gj)
            else
                push!(idx_a_set, gj)
                push!(idx_b_set, gi)
            end
        end

        idx_a = sort!(collect(idx_a_set))
        idx_b = sort!(collect(idx_b_set))

        # Filter to still-kept emitters
        filter!(i -> keep[i], idx_a)
        filter!(i -> keep[i], idx_b)
        (isempty(idx_a) || isempty(idx_b)) && continue

        # Build cost matrix
        cost = zeros(length(idx_a), length(idx_b))
        for (ii, ki) in enumerate(idx_a)
            for (jj, kj) in enumerate(idx_b)
                e_i, e_j = result[ki], result[kj]
                cost[ii, jj] = sqrt((e_i.x - e_j.x)^2 + (e_i.y - e_j.y)^2)
            end
        end

        assignment, _ = Hungarian.hungarian(cost)

        for (ii, jj) in enumerate(assignment)
            jj == 0 && continue
            ki, kj = idx_a[ii], idx_b[jj]
            e_i, e_j = result[ki], result[kj]

            # Merge threshold: distance must be within combined uncertainties
            σ_combined = sqrt(e_i.σ_x^2 + e_i.σ_y^2 + e_j.σ_x^2 + e_j.σ_y^2)
            threshold = 2.0 * σ_combined  # 2σ test
            if cost[ii, jj] < threshold
                # Precision-weighted merge
                det_i = e_i.σ_x^2 * e_i.σ_y^2 - e_i.σ_xy^2
                det_j = e_j.σ_x^2 * e_j.σ_y^2 - e_j.σ_xy^2
                w_i = 1.0 / (det_i + 1e-10)
                w_j = 1.0 / (det_j + 1e-10)
                w_total = w_i + w_j

                new_x = (w_i * e_i.x + w_j * e_j.x) / w_total
                new_y = (w_i * e_i.y + w_j * e_j.y) / w_total
                new_σ_x = sqrt(1.0 / (1.0/e_i.σ_x^2 + 1.0/e_j.σ_x^2 + 1e-10))
                new_σ_y = sqrt(1.0 / (1.0/e_i.σ_y^2 + 1.0/e_j.σ_y^2 + 1e-10))
                new_σ_xy = (w_i * e_i.σ_xy + w_j * e_j.σ_xy) / w_total

                result[ki] = SMLMData.Emitter2DFit(
                    new_x, new_y, e_i.photons + e_j.photons, 0.0,
                    new_σ_x, new_σ_y, new_σ_xy,
                    0.0, 0.0, 1, 1, 0, e_i.id
                )
                keep[kj] = false
            end
        end
    end

    return result[keep]
end

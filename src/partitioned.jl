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

Use Hungarian matching to identify and merge duplicate emitters near partition boundaries.
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
    # Make mutable copies for merging
    result = copy(emitters)

    unique_pids = unique(partition_ids)

    for i in 1:length(unique_pids)
        for j in (i+1):length(unique_pids)
            pid_i, pid_j = unique_pids[i], unique_pids[j]

            idx_i = findall(k -> partition_ids[k] == pid_i && is_near_boundary[k] && keep[k], 1:n)
            idx_j = findall(k -> partition_ids[k] == pid_j && is_near_boundary[k] && keep[k], 1:n)

            (isempty(idx_i) || isempty(idx_j)) && continue

            # Build cost matrix
            cost = zeros(length(idx_i), length(idx_j))
            for (ii, ki) in enumerate(idx_i)
                for (jj, kj) in enumerate(idx_j)
                    e_i, e_j = result[ki], result[kj]
                    cost[ii, jj] = sqrt((e_i.x - e_j.x)^2 + (e_i.y - e_j.y)^2)
                end
            end

            assignment, _ = Hungarian.hungarian(cost)

            for (ii, jj) in enumerate(assignment)
                jj == 0 && continue
                ki, kj = idx_i[ii], idx_j[jj]
                e_i, e_j = result[ki], result[kj]

                # Merge threshold: distance must be within combined uncertainties
                σ_combined = sqrt(e_i.σ_x^2 + e_i.σ_y^2 + e_j.σ_x^2 + e_j.σ_y^2)
                threshold = 2.0 * σ_combined  # 2σ test
                if cost[ii, jj] < threshold
                    # Precision-weighted merge using covariance determinant
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

                    # Create merged emitter (sum photons from both)
                    result[ki] = SMLMData.Emitter2DFit(
                        new_x, new_y, e_i.photons + e_j.photons, 0.0,
                        new_σ_x, new_σ_y, new_σ_xy,
                        0.0, 0.0, 1, 1, 0, e_i.id
                    )
                    keep[kj] = false
                end
            end
        end
    end

    return result[keep]
end

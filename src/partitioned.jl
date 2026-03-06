# Partitioned BaGoL execution for large datasets

"""
    PartitionedBaGoLResult{E<:SMLMData.AbstractEmitter}

Result container for partitioned BaGoL analysis.

# Fields
- `chains`: One RJMCMCChain per partition
- `partitions`: Partition definitions with boundary info
- `emitters`: Combined emitters after boundary deduplication
- `posterior_k`: Combined K posterior histogram
- `skipped`: Oversized clusters that were skipped (if any)
"""
struct PartitionedBaGoLResult{E<:SMLMData.AbstractEmitter}
    chains::Vector{RJMCMCChain}
    partitions::Vector{Partition{E}}
    emitters::Vector{SMLMData.Emitter2DFit}
    posterior_k::Vector{Int}
    skipped::Vector{Partition{E}}
end

"""
    run_bagol_partitioned(locs; kwargs...)

Run BaGoL with spatial partitioning for large datasets.

# Partitioning Arguments
- `nsigma=4.0`: DBSCAN threshold in sigma units
- `min_partition_size=10`: Minimum locs per partition
- `max_partition_size=1000`: Target max locs per partition
- `oversized=:split`: Strategy for large clusters (:split or :skip)
- `boundary_margin=0.0`: Distance from edge to flag (0 = auto: 5×median(σ))

# Standard BaGoL Arguments
All other kwargs are passed to `run_bagol` for each partition.

# Returns
`PartitionedBaGoLResult` with chains, partitions, and merged MAP-N result.
"""
function run_bagol_partitioned(
    locs::Vector{E};
    # Partitioning parameters
    nsigma::Float64=4.0,
    min_partition_size::Int=10,
    max_partition_size::Int=1000,
    oversized::Symbol=:split,
    boundary_margin::Float64=0.0,
    verbose::Bool=true,
    # Standard BaGoL parameters passed through
    kwargs...
) where E<:SMLMData.AbstractEmitter

    if verbose
        println("Partitioning $(length(locs)) localizations...")
    end

    # Auto boundary margin (scaled with nsigma)
    actual_margin = boundary_margin
    if actual_margin <= 0.0
        sigmas = [mean_sigma(loc) for loc in locs]
        actual_margin = nsigma * median(sigmas)
    end

    # Partition the data
    partitions, skipped = partition_locs(
        locs;
        nsigma=nsigma,
        min_size=min_partition_size,
        max_size=max_partition_size,
        oversized=oversized,
        boundary_margin=actual_margin
    )

    if verbose
        println("  Created $(length(partitions)) partitions")
        if !isempty(skipped)
            n_skipped_locs = sum(length(p.locs) for p in skipped)
            println("  Skipped $(length(skipped)) oversized clusters ($n_skipped_locs locs)")
        end
        sizes = [length(p.locs) for p in partitions]
        if !isempty(sizes)
            println("  Partition sizes: min=$(minimum(sizes)), median=$(Int(round(median(sizes)))), max=$(maximum(sizes))")
        end
    end

    if isempty(partitions)
        @warn "No valid partitions after clustering"
        return PartitionedBaGoLResult{E}(RJMCMCChain[], partitions, SMLMData.Emitter2DFit[], Int[], skipped)
    end

    # Run RJMCMC in parallel
    chains = run_partitions_parallel(partitions; verbose=verbose, kwargs...)

    # Merge results with boundary deduplication
    merged_emitters, posterior_k = merge_partition_results(chains, partitions, actual_margin)

    if verbose
        println("\nMerged result: $(length(merged_emitters)) emitters")
    end

    return PartitionedBaGoLResult{E}(chains, partitions, merged_emitters, posterior_k, skipped)
end

"""
    run_partitions_parallel(partitions; kwargs...)

Run BaGoL on each partition in parallel using threads.
"""
function run_partitions_parallel(
    partitions::Vector{Partition{E}};
    verbose::Bool=true,
    kwargs...
) where E

    n = length(partitions)
    chains = Vector{RJMCMCChain}(undef, n)

    if verbose
        println("\nRunning BaGoL on $n partitions ($(Threads.nthreads()) threads)...")
    end

    # Progress tracking (thread-safe)
    completed = Threads.Atomic{Int}(0)

    Threads.@threads for i in 1:n
        # Run BaGoL on this partition (verbose=false to avoid interleaved output)
        chains[i] = run_bagol(partitions[i].locs; verbose=false, kwargs...)

        # Update progress
        done = Threads.atomic_add!(completed, 1)
        if verbose && (done % max(1, n ÷ 10) == 0 || done == n)
            println("  Completed $done/$n partitions")
        end
    end

    return chains
end

"""
    merge_partition_results(chains, partitions, boundary_margin) -> (Vector{Emitter2DFit}, Vector{Int})

Merge MAP-N results from partitions with boundary deduplication.

1. Get MAP-N from each partition
2. Identify emitters near partition boundaries
3. Use Hungarian matching to find duplicates across boundaries
4. Merge matched pairs with precision-weighted averaging

Returns (merged_emitters, posterior_k).
"""
function merge_partition_results(
    chains::Vector{<:RJMCMCChain},
    partitions::Vector{Partition{E}},
    boundary_margin::Float64
) where E

    # Get MAP-N from each partition (now returns tuple)
    mapn_results = [estimate_mapn(chain) for chain in chains]

    # Collect all emitters with partition info
    all_emitters = SMLMData.Emitter2DFit[]
    all_posterior_ks = Vector{Int}[]
    partition_ids = Int[]
    is_near_boundary = Bool[]

    for (pid, (partition, (emitters, posterior_k))) in enumerate(zip(partitions, mapn_results))
        push!(all_posterior_ks, posterior_k)
        for emitter in emitters
            push!(all_emitters, emitter)
            push!(partition_ids, pid)
            near_boundary = emitter_near_boundary(emitter, partition, boundary_margin)
            push!(is_near_boundary, near_boundary)
        end
    end

    n_total = length(all_emitters)
    if n_total == 0
        return SMLMData.Emitter2DFit[], Int[]
    end

    # Find and merge boundary duplicates
    merged_emitters = deduplicate_boundary_emitters(
        all_emitters, partition_ids, is_near_boundary, boundary_margin
    )

    # Build posterior_k histogram (sum across partitions)
    max_k = maximum(length(pk) for pk in all_posterior_ks)
    posterior_k = zeros(Int, max_k)
    for pk in all_posterior_ks
        for (k, count) in enumerate(pk)
            if k <= max_k
                posterior_k[k] += count
            end
        end
    end

    return merged_emitters, posterior_k
end

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

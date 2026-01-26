# Partitioned BaGoL execution for large datasets

"""
    PartitionedBaGoLResult{E<:SMLMData.AbstractEmitter}

Result container for partitioned BaGoL analysis.

# Fields
- `chains`: One RJMCMCChain per partition
- `partitions`: Partition definitions with boundary info
- `merged_mapn`: Combined MAP-N result after boundary deduplication
- `skipped`: Oversized clusters that were skipped (if any)
"""
struct PartitionedBaGoLResult{E<:SMLMData.AbstractEmitter}
    chains::Vector{RJMCMCChain}
    partitions::Vector{Partition{E}}
    merged_mapn::MAPNResult
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

    # Auto boundary margin
    actual_margin = boundary_margin
    if actual_margin <= 0.0
        sigmas = [mean_sigma(loc) for loc in locs]
        actual_margin = 5.0 * median(sigmas)
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
        empty_result = MAPNResult(0, Tuple{Float64,Float64}[], Tuple{Float64,Float64}[], Int[])
        return PartitionedBaGoLResult{E}(RJMCMCChain[], partitions, empty_result, skipped)
    end

    # Run RJMCMC in parallel
    chains = run_partitions_parallel(partitions; verbose=verbose, kwargs...)

    # Merge results with boundary deduplication
    merged = merge_partition_results(chains, partitions, actual_margin)

    if verbose
        println("\nMerged result: $(merged.n_emitters) emitters")
    end

    return PartitionedBaGoLResult{E}(chains, partitions, merged, skipped)
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
    merge_partition_results(chains, partitions, boundary_margin)

Merge MAP-N results from partitions with boundary deduplication.

1. Get MAP-N from each partition
2. Identify emitters near partition boundaries
3. Use Hungarian matching to find duplicates across boundaries
4. Merge matched pairs with precision-weighted averaging
"""
function merge_partition_results(
    chains::Vector{RJMCMCChain},
    partitions::Vector{Partition{E}},
    boundary_margin::Float64
) where E

    # Get MAP-N from each partition
    mapn_results = [estimate_mapn(chain) for chain in chains]

    # Collect all emitters with partition info
    all_emitters = Tuple{Float64, Float64}[]  # (x, y)
    all_uncertainties = Tuple{Float64, Float64}[]  # (σ_x, σ_y)
    partition_ids = Int[]  # Which partition each emitter came from
    is_near_boundary = Bool[]  # Whether emitter is near partition boundary

    for (pid, (partition, mapn)) in enumerate(zip(partitions, mapn_results))
        for (pos, unc) in zip(mapn.emitters, mapn.uncertainties)
            push!(all_emitters, pos)
            push!(all_uncertainties, unc)
            push!(partition_ids, pid)

            # Check if emitter is near any boundary localization
            near_boundary = emitter_near_boundary(pos, partition, boundary_margin)
            push!(is_near_boundary, near_boundary)
        end
    end

    n_total = length(all_emitters)
    if n_total == 0
        return MAPNResult(0, Tuple{Float64,Float64}[], Tuple{Float64,Float64}[], Int[])
    end

    # Find and merge boundary duplicates
    merged_emitters, merged_uncertainties = deduplicate_boundary_emitters(
        all_emitters, all_uncertainties, partition_ids, is_near_boundary, boundary_margin
    )

    # Build posterior_k histogram (sum across partitions)
    max_k = maximum(length(mapn.posterior_k) for mapn in mapn_results)
    posterior_k = zeros(Int, max_k)
    for mapn in mapn_results
        for (k, count) in enumerate(mapn.posterior_k)
            if k <= max_k
                posterior_k[k] += count
            end
        end
    end

    return MAPNResult(length(merged_emitters), merged_emitters, merged_uncertainties, posterior_k)
end

"""
    emitter_near_boundary(pos, partition, margin)

Check if an emitter position is near any boundary localization in the partition.
"""
function emitter_near_boundary(
    pos::Tuple{Float64, Float64},
    partition::Partition,
    margin::Float64
)
    for (i, is_bound) in enumerate(partition.is_boundary)
        if is_bound
            loc = partition.locs[i]
            dx = pos[1] - loc.x
            dy = pos[2] - loc.y
            if sqrt(dx^2 + dy^2) < 2 * margin
                return true
            end
        end
    end
    return false
end

"""
    deduplicate_boundary_emitters(emitters, uncertainties, partition_ids, is_near_boundary, margin)

Use Hungarian matching to identify and merge duplicate emitters near partition boundaries.
"""
function deduplicate_boundary_emitters(
    emitters::Vector{Tuple{Float64, Float64}},
    uncertainties::Vector{Tuple{Float64, Float64}},
    partition_ids::Vector{Int},
    is_near_boundary::Vector{Bool},
    margin::Float64
)
    n = length(emitters)
    merged = trues(n)  # Track which emitters to keep

    # For each pair of different partitions
    unique_pids = unique(partition_ids)

    for i in 1:length(unique_pids)
        for j in (i+1):length(unique_pids)
            pid_i, pid_j = unique_pids[i], unique_pids[j]

            # Get boundary emitters from each partition
            idx_i = findall(k -> partition_ids[k] == pid_i && is_near_boundary[k], 1:n)
            idx_j = findall(k -> partition_ids[k] == pid_j && is_near_boundary[k], 1:n)

            isempty(idx_i) || isempty(idx_j) && continue

            # Build cost matrix
            cost = zeros(length(idx_i), length(idx_j))
            for (ii, ki) in enumerate(idx_i)
                for (jj, kj) in enumerate(idx_j)
                    pos_i = emitters[ki]
                    pos_j = emitters[kj]
                    cost[ii, jj] = sqrt((pos_i[1] - pos_j[1])^2 + (pos_i[2] - pos_j[2])^2)
                end
            end

            # Hungarian matching
            assignment, _ = Hungarian.hungarian(cost)

            # Merge matched pairs within threshold
            threshold = 2 * margin
            for (ii, jj) in enumerate(assignment)
                jj == 0 && continue
                if cost[ii, jj] < threshold
                    ki, kj = idx_i[ii], idx_j[jj]

                    # Precision-weighted merge
                    σ_i = uncertainties[ki]
                    σ_j = uncertainties[kj]

                    # Weights inversely proportional to variance
                    w_i = 1.0 / (σ_i[1]^2 + σ_i[2]^2 + 1e-10)
                    w_j = 1.0 / (σ_j[1]^2 + σ_j[2]^2 + 1e-10)
                    w_total = w_i + w_j

                    # Weighted average position
                    pos_i = emitters[ki]
                    pos_j = emitters[kj]
                    new_x = (w_i * pos_i[1] + w_j * pos_j[1]) / w_total
                    new_y = (w_i * pos_i[2] + w_j * pos_j[2]) / w_total

                    # Combined uncertainty (approximation)
                    new_σ_x = sqrt(1.0 / (1.0/σ_i[1]^2 + 1.0/σ_j[1]^2 + 1e-10))
                    new_σ_y = sqrt(1.0 / (1.0/σ_i[2]^2 + 1.0/σ_j[2]^2 + 1e-10))

                    # Update first emitter with merged values, mark second for removal
                    emitters[ki] = (new_x, new_y)
                    uncertainties[ki] = (new_σ_x, new_σ_y)
                    merged[kj] = false
                end
            end
        end
    end

    # Return only non-merged emitters
    keep_idx = findall(merged)
    return emitters[keep_idx], uncertainties[keep_idx]
end

function partition_localizations(localizations::Vector{<:AbstractLocalization}; 
                               radius::Real = 1.0,
                               min_partition_size::Int = 1)
    
    if length(localizations) < 1
        # Return empty if no data
        return Vector{Vector{eltype(localizations)}}()
    end
    
    # Extract positions for spatial partitioning
    positions = Matrix{Float64}(undef, 2, length(localizations))
    for (i, loc) in enumerate(localizations)
        positions[1, i] = loc.x
        positions[2, i] = loc.y
    end
    
    # Run DBSCAN clustering for spatial partitioning
    # Use minimal parameters to create smallest possible partitions for computational efficiency
    clusters = dbscan(positions, radius; min_neighbors=1, min_cluster_size=1)
    
    # Group localizations by spatial partition
    partitioned_localizations = Vector{Vector{eltype(localizations)}}()
    
    # Add core partitions
    for cluster in clusters.clusters
        all_indices = vcat(cluster.core_indices, cluster.boundary_indices)
        partition_locs = localizations[all_indices]
        push!(partitioned_localizations, partition_locs)
    end
    
    # Handle noise points - group them into a separate partition if substantial
    assigned_indices = Set{Int}()
    for cluster in clusters.clusters
        union!(assigned_indices, cluster.core_indices)
        union!(assigned_indices, cluster.boundary_indices)
    end
    noise_indices = setdiff(1:length(localizations), assigned_indices)
    
    if length(noise_indices) >= min_partition_size
        push!(partitioned_localizations, localizations[noise_indices])
    elseif !isempty(noise_indices) && !isempty(partitioned_localizations)
        # Add noise points to nearest partition
        append!(partitioned_localizations[1], localizations[noise_indices])
    elseif !isempty(noise_indices)
        # Only noise points, make them a partition
        push!(partitioned_localizations, localizations[noise_indices])
    end
    
    # Ensure we have at least one partition
    if isempty(partitioned_localizations)
        push!(partitioned_localizations, localizations)
    end
    
    return partitioned_localizations
end

function estimate_partitioning_radius(localizations::Vector{<:AbstractLocalization})
    if length(localizations) < 2
        return 1.0
    end
    
    # Calculate average localization uncertainty
    avg_sigma = 0.0
    for loc in localizations
        avg_sigma += sqrt(loc.σx^2 + loc.σy^2)
    end
    avg_sigma /= length(localizations)
    
    # Use 3-5x the average uncertainty as partitioning radius
    return 4.0 * avg_sigma
end

function print_partitioning_summary(partitioned_localizations::Vector{Vector{T}}) where T<:AbstractLocalization
    # Minimal partitioning info - details commented out for cleaner output
    if length(partitioned_localizations) > 1
        total_locs = sum(length(p) for p in partitioned_localizations)
        println("Data partitioned: $(length(partitioned_localizations)) partitions, $total_locs total localizations")
    end
    # Detailed output available by uncommenting:
    # for (i, partition) in enumerate(partitioned_localizations)
    #     if length(partition) > 0
    #         x_coords = [loc.x for loc in partition]
    #         y_coords = [loc.y for loc in partition]
    #         x_center = sum(x_coords) / length(x_coords)
    #         y_center = sum(y_coords) / length(y_coords)
    #         println("    Partition $i: $(length(partition)) localizations at ($(round(x_center, digits=2)), $(round(y_center, digits=2)))")
    #     end
    # end
end
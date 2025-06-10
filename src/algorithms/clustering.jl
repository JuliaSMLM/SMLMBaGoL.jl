function cluster_localizations(localizations::Vector{<:AbstractLocalization}; 
                             radius::Real = 1.0,
                             min_cluster_size::Int = 3)
    
    if length(localizations) < min_cluster_size
        # Single cluster if too few points
        return [localizations]
    end
    
    # Extract positions for clustering
    positions = Matrix{Float64}(undef, 2, length(localizations))
    for (i, loc) in enumerate(localizations)
        positions[1, i] = loc.x
        positions[2, i] = loc.y
    end
    
    # Run DBSCAN clustering  
    clusters = dbscan(positions, radius; min_neighbors=min_cluster_size-1, min_cluster_size=min_cluster_size)
    
    # Group localizations by cluster
    clustered_localizations = Vector{Vector{eltype(localizations)}}()
    
    # Add core clusters
    for cluster in clusters.clusters
        all_indices = vcat(cluster.core_indices, cluster.boundary_indices)
        cluster_locs = localizations[all_indices]
        push!(clustered_localizations, cluster_locs)
    end
    
    # Handle noise points - group them into a separate cluster if substantial
    assigned_indices = Set{Int}()
    for cluster in clusters.clusters
        union!(assigned_indices, cluster.core_indices)
        union!(assigned_indices, cluster.boundary_indices)
    end
    noise_indices = setdiff(1:length(localizations), assigned_indices)
    
    if length(noise_indices) >= min_cluster_size
        push!(clustered_localizations, localizations[noise_indices])
    elseif !isempty(noise_indices) && !isempty(clustered_localizations)
        # Add noise points to nearest cluster
        append!(clustered_localizations[1], localizations[noise_indices])
    elseif !isempty(noise_indices)
        # Only noise points, make them a cluster
        push!(clustered_localizations, localizations[noise_indices])
    end
    
    # Ensure we have at least one cluster
    if isempty(clustered_localizations)
        push!(clustered_localizations, localizations)
    end
    
    return clustered_localizations
end

function estimate_clustering_radius(localizations::Vector{<:AbstractLocalization})
    if length(localizations) < 2
        return 1.0
    end
    
    # Calculate average localization uncertainty
    avg_sigma = 0.0
    for loc in localizations
        avg_sigma += sqrt(loc.σx^2 + loc.σy^2)
    end
    avg_sigma /= length(localizations)
    
    # Use 3-5x the average uncertainty as clustering radius
    return 4.0 * avg_sigma
end

function print_clustering_summary(clustered_localizations::Vector{Vector{T}}) where T<:AbstractLocalization
    println("Clustering summary:")
    println("  Total clusters: $(length(clustered_localizations))")
    for (i, cluster) in enumerate(clustered_localizations)
        if length(cluster) > 0
            x_coords = [loc.x for loc in cluster]
            y_coords = [loc.y for loc in cluster]
            x_center = sum(x_coords) / length(x_coords)
            y_center = sum(y_coords) / length(y_coords)
            println("    Cluster $i: $(length(cluster)) localizations at ($(round(x_center, digits=2)), $(round(y_center, digits=2)))")
        end
    end
end


function median_precision(obs::Observations)

end

# break data into subregions using dbscan


function gen_subregions(positions, sigmas, emitter_type, min_pts::Int, ϵ::Float64)
    # data must be d x n array of points
    
    dbscan_result = dbscan(positions, ϵ)
    n_clusters = length(dbscan_result.clusters)
    
    subregions = Vector{Subregion}(undef, n_clusters)
    
    for i in 1:length(dbscan_result.clusters)
        cluster = dbscan_result.clusters[i]
        inx = vcat(cluster.core_indices, cluster.boundary_indices)
        positions_sub = positions[:, inx]
        sigmas_sub = sigmas[:, inx]
        obs = gen_observations(emitter_type, positions_sub, sigmas_sub)
        subregions[i] = Subregion(obs, 1)
    end

    return subregions
end






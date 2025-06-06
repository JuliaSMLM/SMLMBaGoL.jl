

function median_precision(obs::Observations)

end

# break data into subregions using dbscan


function gen_subregions(emitters::Vector{<:Emitter2DFit}, min_pts::Int, ϵ::Float64)
    # Extract positions for clustering - data must be d x n array of points
    positions = hcat([em.x for em in emitters], [em.y for em in emitters])'
    
    dbscan_result = dbscan(positions, ϵ)
    n_clusters = length(dbscan_result.clusters)
    
    subregions = Vector{Subregion}(undef, n_clusters)
    
    for i in 1:length(dbscan_result.clusters)
        cluster = dbscan_result.clusters[i]
        indices = vcat(cluster.core_indices, cluster.boundary_indices)
        cluster_emitters = emitters[indices]  # Direct subsetting of emitters
        obs = Observations(cluster_emitters)  # Direct constructor
        subregions[i] = Subregion(obs, 1)
    end

    return subregions
end

# Keep old function signature for backward compatibility (deprecated)
function gen_subregions(positions, sigmas, emitter_type, min_pts::Int, ϵ::Float64)
    @warn "gen_subregions with positions/sigmas arrays is deprecated. Use gen_subregions(emitters, min_pts, ϵ) instead."
    
    # Convert back to emitters and call new function
    y = positions[1, :]
    x = positions[2, :]
    σ_y = sigmas[1, :]
    σ_x = sigmas[2, :]
    
    emitters = create_minimal_emitter2dfit.(x, y, σ_x, σ_y)
    return gen_subregions(emitters, min_pts, ϵ)
end






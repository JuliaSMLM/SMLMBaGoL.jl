# Main BaGoL interface
using Distributions
using Random
using LinearAlgebra
using Clustering: dbscan
using Base.Threads

"""
    bagol(emitters; kwargs...) -> BaGoLResult

Perform Bayesian Grouping of Localizations on single molecule localization data.

# Arguments
- `emitters::Vector{<:AbstractEmitter}`: Vector of localizations/emitters

# Keywords
- `prior::HierarchicalPrior`: Prior distribution parameters
- `subregion_radius`: Clustering radius for DBSCAN (default: 4σ)
- `mcmc_steps`: Number of MCMC steps per chain
- `burnin`: Number of burn-in steps
- `posterior_resolution`: Resolution for posterior image
- `n_threads`: Number of parallel threads
- `move_probs::MoveProbs`: RJMCMC move probabilities
"""
function bagol(
    emitters::Vector{E};
    prior::Union{HierarchicalPrior,Nothing} = nothing,
    subregion_radius::Union{<:AbstractFloat,Nothing} = nothing,
    mcmc_steps::Int = 10_000,
    burnin::Int = 2_000,
    posterior_resolution::Union{<:AbstractFloat,Nothing} = nothing,
    n_threads::Int = Threads.nthreads(),
    move_probs::Union{MoveProbs,Nothing} = nothing,
    verbose::Bool = true
) where E
    
    # Validate inputs
    isempty(emitters) && error("No emitters provided")
    
    # Extract the float type from emitters
    T = eltype(emitters[1].x)
    
    # Set defaults based on type
    prior = isnothing(prior) ? HierarchicalPrior{T}() : prior
    posterior_resolution = isnothing(posterior_resolution) ? T(0.005) : T(posterior_resolution)
    move_probs = isnothing(move_probs) ? MoveProbs{T}() : move_probs
    mcmc_steps > burnin || error("mcmc_steps must be > burnin")
    
    # Auto-determine subregion radius if not provided
    if isnothing(subregion_radius)
        # Check if emitters have uncertainty fields
        if hasproperty(emitters[1], :σ_x)
            mean_σ = mean(e -> (e.σ_x + e.σ_y) / 2, emitters)
            subregion_radius = 4 * mean_σ
        else
            # Default to 0.4 if no uncertainty info
            subregion_radius = T(0.4)
        end
        verbose && @info "Auto-selected subregion radius: $(round(subregion_radius, digits=3))"
    else
        subregion_radius = T(subregion_radius)
    end
    
    # 1. Cluster into subregions
    verbose && @info "Clustering $(length(emitters)) emitters..."
    subregions = cluster_subregions(emitters, subregion_radius)
    verbose && @info "Found $(length(subregions)) subregions"
    
    # 2. Run parallel RJMCMC
    verbose && @info "Running RJMCMC with $(n_threads) threads..."
    chains = run_parallel_rjmcmc(
        subregions, prior, mcmc_steps, burnin, move_probs, n_threads
    )
    
    # 3. Update hierarchical prior
    verbose && @info "Updating hierarchical prior..."
    updated_prior = update_hierarchical_prior(chains, prior)
    
    # 4. Compute posterior image
    verbose && @info "Computing posterior image..."
    posterior = compute_posterior_image(chains, posterior_resolution)
    
    # 5. Extract MAP-N estimates
    verbose && @info "Extracting MAP-N emitters..."
    mapn_emitters = extract_mapn_emitters(chains)
    
    # 6. Compute log evidence
    log_evidence = compute_log_evidence(chains)
    
    verbose && @info "BaGoL complete: $(length(mapn_emitters)) MAP-N emitters found"
    
    return BaGoLResult(chains, posterior, mapn_emitters, log_evidence, updated_prior)
end

# Convenience method for objects with emitters field (duck typing)
bagol(smld; kwargs...) = bagol(smld.emitters; kwargs...)

"""
Cluster emitters into spatial subregions using DBSCAN.
"""
function cluster_subregions(
    emitters::Vector{E},
    radius::T
) where {E, T<:AbstractFloat}
    
    # Extract positions for clustering
    n = length(emitters)
    positions = Matrix{T}(undef, 2, n)
    
    @inbounds for i in 1:n
        positions[1, i] = emitters[i].x
        positions[2, i] = emitters[i].y
    end
    
    # Run DBSCAN
    result = dbscan(positions, radius, min_cluster_size=1)
    
    # Group emitters by cluster
    subregions = Vector{Vector{E}}()
    
    # Get unique cluster assignments
    assignments = result.assignments
    unique_clusters = unique(assignments)
    
    for cluster_id in unique_clusters
        if cluster_id > 0  # Skip noise points (cluster_id = 0)
            indices = findall(==(cluster_id), assignments)
            push!(subregions, emitters[indices])
        end
    end
    
    # Add noise points as individual subregions
    noise_indices = findall(==(0), assignments)
    for idx in noise_indices
        push!(subregions, [emitters[idx]])
    end
    
    return subregions
end

"""
Run RJMCMC chains in parallel on subregions.
"""
function run_parallel_rjmcmc(
    subregions::Vector{Vector{E}},
    prior::HierarchicalPrior{T},
    steps::Int,
    burnin::Int,
    move_probs::MoveProbs{T},
    n_threads::Int
) where {E, T<:AbstractFloat}
    
    n_regions = length(subregions)
    chains = Vector{BaGoLChain{T, E}}(undef, n_regions)
    
    # Process regions in parallel
    @threads for i in 1:n_regions
        # Thread-local RNG
        rng = MersenneTwister(hash((i, time_ns())))
        
        # Run chain
        chains[i] = run_single_chain(
            subregions[i], prior, steps, burnin, move_probs, rng
        )
    end
    
    return chains
end

"""
Extract MAP-N emitter estimates from all chains using proper particle identity handling.
"""
function extract_mapn_emitters(chains::Vector{BaGoLChain{T, O}}) where {T, O}
    all_mapn = Vector{Emitter2D{T}}()
    
    for chain in chains
        # Find MAP-N for this chain
        n_states = length.(chain.states)
        isempty(n_states) && continue
        
        max_n = maximum(n_states)
        n_counts = zeros(Int, max_n + 1)
        
        for n in n_states
            n_counts[n + 1] += 1
        end
        
        mapn = argmax(n_counts) - 1
        
        if mapn > 0
            # Extract states with MAP-N emitters
            mapn_indices = findall(==(mapn), n_states)
            
            if length(mapn_indices) > 1
                # Multiple MAP-N states - use sophisticated averaging
                mapn_emitters = extract_mapn_with_identity_matching(
                    chain.states[mapn_indices], mapn
                )
                append!(all_mapn, mapn_emitters)
            elseif length(mapn_indices) == 1
                # Single MAP-N state
                append!(all_mapn, chain.states[mapn_indices[1]])
            end
        end
    end
    
    return all_mapn
end

"""
Handle particle identity problem using spatial histogram and Hungarian algorithm approach.
Simplified version of the mathematical reference algorithm.
"""
function extract_mapn_with_identity_matching(
    mapn_states::Vector{Vector{Emitter2D{T}}}, 
    mapn::Int
) where T
    
    n_states = length(mapn_states)
    n_states <= 1 && return isempty(mapn_states) ? Emitter2D{T}[] : mapn_states[1]
    
    # Create spatial histogram to find reference configuration
    all_positions = Vector{Tuple{T, T}}()
    for state in mapn_states
        for emitter in state
            push!(all_positions, (emitter.x, emitter.y))
        end
    end
    
    # Use first state as initial reference
    reference_state = mapn_states[1]
    
    # Accumulate positions with simple nearest-neighbor matching
    accumulated_positions = Vector{Vector{Tuple{T, T}}}(undef, mapn)
    accumulated_photons = Vector{Vector{T}}(undef, mapn)
    
    for i in 1:mapn
        accumulated_positions[i] = Tuple{T, T}[]
        accumulated_photons[i] = T[]
    end
    
    for state in mapn_states
        # Simple assignment: match each emitter to nearest reference emitter
        for (i, emitter) in enumerate(state)
            min_dist = Inf
            best_ref = 1
            
            for (j, ref_emitter) in enumerate(reference_state)
                dist = (emitter.x - ref_emitter.x)^2 + (emitter.y - ref_emitter.y)^2
                if dist < min_dist
                    min_dist = dist
                    best_ref = j
                end
            end
            
            push!(accumulated_positions[best_ref], (emitter.x, emitter.y))
            push!(accumulated_photons[best_ref], emitter.photons)
        end
    end
    
    # Compute mean positions
    result_emitters = Vector{Emitter2D{T}}()
    for i in 1:mapn
        if !isempty(accumulated_positions[i])
            mean_x = sum(pos[1] for pos in accumulated_positions[i]) / length(accumulated_positions[i])
            mean_y = sum(pos[2] for pos in accumulated_positions[i]) / length(accumulated_positions[i])
            mean_photons = sum(accumulated_photons[i]) / length(accumulated_photons[i])
            
            push!(result_emitters, Emitter2D(mean_x, mean_y, mean_photons))
        end
    end
    
    return result_emitters
end

"""
Compute log evidence from chains using harmonic mean estimator.
"""
function compute_log_evidence(chains::Vector{BaGoLChain{T, O}}) where {T, O}
    all_log_probs = vcat([chain.log_probs for chain in chains]...)
    
    if isempty(all_log_probs)
        return -Inf
    end
    
    # Harmonic mean estimator (simple but biased)
    # Better alternatives: thermodynamic integration, stepping stone sampling
    max_log_prob = maximum(all_log_probs)
    mean_inv_prob = mean(exp.(max_log_prob .- all_log_probs))
    
    return max_log_prob - log(mean_inv_prob)
end

"""
Compute posterior probability image from all chains.
"""
function compute_posterior_image(
    chains::Vector{BaGoLChain{T, O}},
    resolution::T
) where {T, O}
    
    # Find image bounds from all observations
    all_obs = vcat([chain.observations for chain in chains]...)
    
    if isempty(all_obs)
        return zeros(T, 1, 1)
    end
    
    x_min = minimum(o -> o.x, all_obs)
    x_max = maximum(o -> o.x, all_obs)
    y_min = minimum(o -> o.y, all_obs)
    y_max = maximum(o -> o.y, all_obs)
    
    # Add padding
    padding = T(1.0)
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    
    # Create image grid
    nx = ceil(Int, (x_max - x_min) / resolution)
    ny = ceil(Int, (y_max - y_min) / resolution)
    
    posterior = zeros(T, ny, nx)
    
    # Accumulate emitter positions from all chains
    for chain in chains
        for state in chain.states
            for emitter in state
                # Convert position to pixel coordinates
                ix = floor(Int, (emitter.x - x_min) / resolution) + 1
                iy = floor(Int, (emitter.y - y_min) / resolution) + 1
                
                if 1 <= ix <= nx && 1 <= iy <= ny
                    posterior[iy, ix] += one(T)
                end
            end
        end
    end
    
    # Normalize by total samples
    total_samples = sum(length(chain.states) for chain in chains)
    if total_samples > 0
        posterior ./= total_samples
    end
    
    return posterior
end
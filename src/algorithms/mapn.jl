function euclidean_distance(emitter1::AbstractEmitter, emitter2::AbstractEmitter)
    return sqrt((emitter1.x - emitter2.x)^2 + (emitter1.y - emitter2.y)^2)
end

function fill_cost_matrix!(cost_matrix::Matrix{Float64}, emitters1::Vector{<:AbstractEmitter}, emitters2::Vector{<:AbstractEmitter})
    n = size(cost_matrix, 1)
    for i in 1:n
        for j in 1:n
            cost_matrix[i, j] = euclidean_distance(emitters1[i], emitters2[j])
        end
    end
    return nothing
end

function hungarian_assignment(ref_emitters::Vector{<:AbstractEmitter}, state_emitters::Vector{<:AbstractEmitter})
    n = length(ref_emitters)
    @assert n == length(state_emitters) "Emitter vectors must have same length"
    
    cost_matrix = zeros(Float64, n, n)
    fill_cost_matrix!(cost_matrix, ref_emitters, state_emitters)
    
    assignment, = hungarian(cost_matrix)
    return assignment
end

function extract_mapn_states(chain::RJMCMCChain)
    isempty(chain.samples) && return BaGoLState{eltype(chain.samples)}[]
    
    # Count emitter frequencies across chain samples
    emitter_counts = [length(state.emitters) for state in chain.samples]
    mapn = StatsBase.mode(emitter_counts)  # Most frequent count
    
    # Filter to states with MAPN count
    return [state for state in chain.samples if length(state.emitters) == mapn]
end

function build_spatial_histogram(mapn_states::Vector{<:BaGoLState}, n_bins::Int=20)
    # Get spatial bounds from all emitter positions
    all_x = Float64[]
    all_y = Float64[]
    
    for state in mapn_states
        for emitter in state.emitters
            push!(all_x, emitter.x)
            push!(all_y, emitter.y)
        end
    end
    
    isempty(all_x) && return (zeros(n_bins, n_bins), 0.0, 0.0, 0.0, 0.0)
    
    x_min, x_max = extrema(all_x)
    y_min, y_max = extrema(all_y)
    
    # Add small margin to avoid edge effects
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Handle degenerate cases where all emitters are at the same position
    if x_range == 0
        x_min -= 0.1
        x_max += 0.1
    else
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
    end
    
    if y_range == 0
        y_min -= 0.1
        y_max += 0.1
    else
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
    end
    
    # Build histogram
    histogram = zeros(Float64, n_bins, n_bins)
    
    for state in mapn_states
        for emitter in state.emitters
            # Find bin indices
            x_bin = min(n_bins, max(1, floor(Int, (emitter.x - x_min) / (x_max - x_min) * n_bins) + 1))
            y_bin = min(n_bins, max(1, floor(Int, (emitter.y - y_min) / (y_max - y_min) * n_bins) + 1))
            histogram[x_bin, y_bin] += 1.0
        end
    end
    
    return histogram, x_min, x_max, y_min, y_max
end

function score_state_against_histogram(state::BaGoLState, histogram::Matrix{Float64}, 
                                     x_min::Float64, x_max::Float64, y_min::Float64, y_max::Float64)
    n_bins = size(histogram, 1)
    score = 0.0
    
    for emitter in state.emitters
        # Find bin indices
        x_bin = min(n_bins, max(1, floor(Int, (emitter.x - x_min) / (x_max - x_min) * n_bins) + 1))
        y_bin = min(n_bins, max(1, floor(Int, (emitter.y - y_min) / (y_max - y_min) * n_bins) + 1))
        score += histogram[x_bin, y_bin]
    end
    
    return score
end

function find_reference_state(mapn_states::Vector{<:BaGoLState})
    isempty(mapn_states) && error("Cannot find reference state: no MAPN states available")
    length(mapn_states) == 1 && return mapn_states[1]
    
    # Build spatial histogram of all emitter positions
    histogram, x_min, x_max, y_min, y_max = build_spatial_histogram(mapn_states)
    
    # Score each state against histogram
    best_score = -Inf
    best_state = mapn_states[1]
    
    for state in mapn_states
        score = score_state_against_histogram(state, histogram, x_min, x_max, y_min, y_max)
        if score > best_score
            best_score = score
            best_state = state
        end
    end
    
    return best_state
end

function sort_mapn_states!(mapn_states::Vector{<:BaGoLState}, reference_state::BaGoLState)
    isempty(mapn_states) && return
    
    n_emitters = length(reference_state.emitters)
    
    for state in mapn_states
        length(state.emitters) != n_emitters && continue
        
        # Get Hungarian assignment
        assignment = hungarian_assignment(reference_state.emitters, state.emitters)
        
        # Reorder emitters in place
        state.emitters[:] = state.emitters[assignment]
    end
end

function refine_mapn_assignment!(mapn_states::Vector{<:BaGoLState}; n_iterations::Int=3)
    isempty(mapn_states) && return
    
    n_emitters = length(mapn_states[1].emitters)
    n_states = length(mapn_states)
    
    for iter in 1:n_iterations
        # Compute mean positions as new reference
        mean_emitters = Vector{eltype(mapn_states[1].emitters)}(undef, n_emitters)
        
        for i in 1:n_emitters
            x_mean = sum(state.emitters[i].x for state in mapn_states) / n_states
            y_mean = sum(state.emitters[i].y for state in mapn_states) / n_states
            
            # Create mean emitter for reference - use basic Emitter2D
            EmitterType = eltype(mapn_states[1].emitters)
            photons = 1000.0  # Default photon count
            mean_emitters[i] = EmitterType(x_mean, y_mean, photons)
        end
        
        # Create temporary reference state
        temp_ref = BaGoLState(mean_emitters, mapn_states[1].localizations, 
                             mapn_states[1].allocations, mapn_states[1].spatial_prior,
                             mapn_states[1].count_prior, mapn_states[1].τ², 0.0)
        
        # Re-sort using updated reference
        sort_mapn_states!(mapn_states, temp_ref)
    end
end

function compute_final_mapn_emitters(sorted_states::Vector{<:BaGoLState}, partition_id::Int)
    isempty(sorted_states) && return Emitter2DFit{Float64}[]
    
    n_emitters = length(sorted_states[1].emitters)
    n_states = length(sorted_states)
    # Always return Emitter2DFit for MAPN results
    mapn_emitters = Emitter2DFit{Float64}[]
    
    for emitter_idx in 1:n_emitters
        # Mean position across all MAPN states
        x_mean = sum(state.emitters[emitter_idx].x for state in sorted_states) / n_states
        y_mean = sum(state.emitters[emitter_idx].y for state in sorted_states) / n_states
        
        # Compute uncertainty as standard deviation across MAPN states
        if n_states > 1
            x_var = sum((state.emitters[emitter_idx].x - x_mean)^2 for state in sorted_states) / (n_states - 1)
            y_var = sum((state.emitters[emitter_idx].y - y_mean)^2 for state in sorted_states) / (n_states - 1)
            σx = sqrt(x_var)
            σy = sqrt(y_var)
        else
            # Default uncertainty if only one state
            σx = 0.01  # 10 nm default uncertainty
            σy = 0.01
        end
        
        # Create Emitter2DFit with uncertainty estimates
        # For MAPN results, we use sensible defaults for other fields
        photons = 1000.0  # Default photon count
        bg = 10.0         # Default background
        σ_photons = 50.0  # Default photon uncertainty
        σ_bg = 2.0        # Default background uncertainty
        frame = 1         # MAPN is aggregated over all frames
        dataset = 1       # Default dataset
        track_id = 0      # No tracking for MAPN results
        id = partition_id * 10000 + emitter_idx  # Unique ID combining partition and emitter index
        
        push!(mapn_emitters, Emitter2DFit(x_mean, y_mean, photons, bg, σx, σy, σ_photons, σ_bg, 
                                         frame, dataset, track_id, id))
    end
    
    return mapn_emitters
end

function estimate_mapn_single_partition(chain::RJMCMCChain, partition_id::Int)
    # Step 1: Extract MAPN states (most frequent emitter count)
    mapn_states = extract_mapn_states(chain)
    
    # Handle edge cases - always return Emitter2DFit for MAPN
    isempty(mapn_states) && return Emitter2DFit{Float64}[]
    
    # Step 2: Find reference state using spatial histogram
    reference_state = find_reference_state(mapn_states)
    
    # Step 3: Hungarian algorithm sorting with iterative refinement
    sort_mapn_states!(mapn_states, reference_state)
    refine_mapn_assignment!(mapn_states; n_iterations=3)
    
    # Step 4: Compute final emitter positions
    return compute_final_mapn_emitters(mapn_states, partition_id)
end

function estimate_mapn(chains::Vector{<:RJMCMCChain})
    # MAPN always returns Emitter2DFit with uncertainty estimates
    all_mapn_emitters = Emitter2DFit{Float64}[]
    
    for (partition_id, chain) in enumerate(chains)
        partition_emitters = estimate_mapn_single_partition(chain, partition_id)
        append!(all_mapn_emitters, partition_emitters)
    end
    
    return all_mapn_emitters
end
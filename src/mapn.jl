# MAP-N estimation for BaGoL
#
# Uses iterative Hungarian matching with median-based reference positions
# and MAD-based robust uncertainty estimation to handle label switching.

# ============================================================================
# Helper functions
# ============================================================================

"""
    _smoothed_map_n(posterior_k) -> (map_n, confidence)

Find MAP-N with 3-bin moving average smoothing and confidence metric.
Returns `(map_n, confidence)` where confidence is the ratio of the
best to second-best smoothed bin. Warns when confidence < 2.
"""
function _smoothed_map_n(posterior_k::Vector{Int})
    n = length(posterior_k)
    if n == 0
        return 0, 0.0
    end

    # Primary: raw argmax
    map_n = argmax(posterior_k) - 1  # index 1 = K=0

    # Confidence: ratio of best to second-best (raw counts)
    sorted_vals = sort(Float64.(posterior_k), rev=true)
    if length(sorted_vals) >= 2 && sorted_vals[2] > 0
        confidence = sorted_vals[1] / sorted_vals[2]
    else
        confidence = Inf
    end

    # Only use 3-bin smoothing to break ties/near-ties
    if confidence < 1.5
        smoothed = zeros(Float64, n)
        for i in 1:n
            lo = max(1, i - 1)
            hi = min(n, i + 1)
            smoothed[i] = sum(posterior_k[lo:hi]) / (hi - lo + 1)
        end
        map_n = argmax(smoothed) - 1
    end

    if confidence < 2.0
        @warn "Low MAP-N confidence" map_n confidence
    end

    return map_n, confidence
end

"""
MAD-based robust standard deviation estimate.

For data from Normal(μ, σ), returns σ. Robust to up to 50% outliers.
The scale factor 1.4826 ensures consistency with the normal distribution.
"""
function mad_sigma(x::Vector{Float64})
    if length(x) < 2
        return 0.0
    end
    med = median(x)
    mad = median(abs.(x .- med))
    return 1.4826 * mad
end

"""
Position-only Hungarian matching.

Returns assignment vector where assignment[i] = j means ref_i matches sample_j.
"""
function position_hungarian(
    ref_positions::Vector{Tuple{Float64, Float64}},
    sample_positions::Vector{Tuple{Float64, Float64}}
)
    K = length(ref_positions)
    @assert length(sample_positions) == K

    if K == 1
        return [1]
    end

    # Build cost matrix (squared distances)
    cost = zeros(K, K)
    for i in 1:K, j in 1:K
        cost[i, j] = (ref_positions[i][1] - sample_positions[j][1])^2 +
                     (ref_positions[i][2] - sample_positions[j][2])^2
    end

    assignment, _ = Hungarian.hungarian(cost)
    return assignment
end

# ============================================================================
# MAP-N estimation from collapsed chain assignment samples
# ============================================================================

"""
    _positions_from_assignments(assignments, locs) -> Vector{Tuple{Float64,Float64}}

Build ClusterStats from an assignment vector and extract posterior mean
positions for each unique cluster. Returns positions in cluster-label order.
"""
function _positions_from_assignments(
    assignments::Vector{Int16},
    locs::Vector{<:SMLMData.AbstractEmitter}
)
    # Find unique cluster labels (sorted)
    unique_labels = sort!(unique(assignments))
    K = length(unique_labels)

    positions = Vector{Tuple{Float64, Float64}}(undef, K)
    for (ki, lab) in enumerate(unique_labels)
        cs = ClusterStats()
        for i in eachindex(assignments)
            if assignments[i] == lab
                cs = add_loc(cs, locs[i])
            end
        end
        positions[ki] = posterior_mean(cs)
    end
    return positions
end

"""
    estimate_mapn_collapsed(samples, locs; n_refine=10) -> (Vector{Emitter2DFit}, Vector{Int})

Estimate MAP-N emitters from collapsed chain assignment samples.

Uses iterative Hungarian matching with median-based reference positions
to handle label switching. Positions are derived from ClusterStats posterior
means (deterministic given assignments).

# Algorithm
1. Count K per assignment sample
2. Find MAP-N (most common K)
3. Filter to samples with K = MAP-N
4. For each sample, build ClusterStats per cluster → posterior mean positions
5. Iterative Hungarian matching → update reference to median
6. Final positions: median (robust to label switching)
7. Final σ: ClusterStats posterior covariance (analytic)

# Arguments
- `samples`: Vector of assignment vectors from `PartitionSamples`
- `locs`: Original localizations
- `n_refine`: Number of iterative refinement steps (default 10)

# Returns
- `emitters`: Vector of Emitter2DFit with positions and uncertainties
- `posterior_k`: Histogram of K values (index k+1 = count of K=k)
"""
function estimate_mapn_collapsed(
    samples::Vector{Vector{Int16}},
    locs::Vector{<:SMLMData.AbstractEmitter};
    n_refine::Int = 10
)
    if isempty(samples)
        error("No assignment samples — run with PartitionSamples accumulator")
    end

    # Build K histogram
    ks = [length(unique(s)) for s in samples]
    k_max = maximum(ks)
    posterior_k = zeros(Int, k_max + 1)
    for k in ks
        posterior_k[k + 1] += 1
    end

    # MAP-N (smoothed mode)
    map_n, _ = _smoothed_map_n(posterior_k)
    if map_n == 0
        return SMLMData.Emitter2DFit[], posterior_k
    end

    # Filter to samples with K = MAP-N
    map_indices = findall(k -> k == map_n, ks)
    if isempty(map_indices)
        return SMLMData.Emitter2DFit[], posterior_k
    end

    # Extract positions from each MAP-N sample
    map_positions = [_positions_from_assignments(samples[i], locs) for i in map_indices]

    # Initialize reference from sample closest to overall centroid
    centroid_x = 0.0
    centroid_y = 0.0
    n_total = 0
    for positions in map_positions
        for (px, py) in positions
            centroid_x += px
            centroid_y += py
            n_total += 1
        end
    end
    centroid_x /= n_total
    centroid_y /= n_total

    best_ref = map_positions[1]
    best_dist = Inf
    for positions in map_positions
        sx = mean(p[1] for p in positions)
        sy = mean(p[2] for p in positions)
        d = (sx - centroid_x)^2 + (sy - centroid_y)^2
        if d < best_dist
            best_dist = d
            best_ref = positions
        end
    end
    ref_positions = copy(best_ref)

    # Iterative refinement: match → median → update reference
    for _ in 1:n_refine
        matched_positions = [Vector{Tuple{Float64, Float64}}() for _ in 1:map_n]

        for positions in map_positions
            assignment = position_hungarian(ref_positions, positions)
            for (i, j) in enumerate(assignment)
                if j <= length(positions)
                    push!(matched_positions[i], positions[j])
                end
            end
        end

        for i in 1:map_n
            if !isempty(matched_positions[i])
                xs = [p[1] for p in matched_positions[i]]
                ys = [p[2] for p in matched_positions[i]]
                ref_positions[i] = (median(xs), median(ys))
            end
        end
    end

    # Final pass
    matched_positions = [Vector{Tuple{Float64, Float64}}() for _ in 1:map_n]
    for positions in map_positions
        assignment = position_hungarian(ref_positions, positions)
        for (i, j) in enumerate(assignment)
            if j <= length(positions)
                push!(matched_positions[i], positions[j])
            end
        end
    end

    # Build emitters with median positions and ClusterStats posterior uncertainties.
    # MAD of posterior means is 0 when assignments are stable (common in collapsed
    # sampler), so we use the analytic posterior covariance from ClusterStats instead.
    # Use the modal (most common) assignment to build the reference ClusterStats.
    ref_sample = samples[map_indices[1]]  # any MAP-N sample works
    result_emitters = SMLMData.Emitter2DFit[]
    ref_labels = sort!(unique(ref_sample))

    for (id, positions) in enumerate(matched_positions)
        isempty(positions) && continue
        xs = [p[1] for p in positions]
        ys = [p[2] for p in positions]

        pos_x = median(xs)
        pos_y = median(ys)

        # Find which cluster label in the reference sample is closest to this
        # matched position, then use its ClusterStats posterior covariance
        σ_x = 0.0
        σ_y = 0.0
        σ_xy = 0.0
        n_locs_cluster = 0
        best_d = Inf
        for lab in ref_labels
            cs = ClusterStats()
            for i in eachindex(ref_sample)
                if ref_sample[i] == lab
                    cs = add_loc(cs, locs[i])
                end
            end
            mx, my = posterior_mean(cs)
            d = (mx - pos_x)^2 + (my - pos_y)^2
            if d < best_d
                best_d = d
                Σ_xx, Σ_xy_val, Σ_yy = posterior_cov(cs)
                σ_x = sqrt(max(Σ_xx, 0.0))
                σ_y = sqrt(max(Σ_yy, 0.0))
                σ_xy = Σ_xy_val
                n_locs_cluster = Int(cs.n)
            end
        end

        push!(result_emitters, SMLMData.Emitter2DFit(
            pos_x, pos_y,
            Float64(n_locs_cluster), 0.0,  # photons = n_locs in cluster
            σ_x, σ_y, σ_xy,
            0.0, 0.0, 1, 1, 0, id
        ))
    end

    return result_emitters, posterior_k
end

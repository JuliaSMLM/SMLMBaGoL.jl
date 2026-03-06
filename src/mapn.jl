# MAP-N estimation for BaGoL
#
# Uses iterative Hungarian matching with median-based reference positions
# and MAD-based robust uncertainty estimation to handle label switching.

# ============================================================================
# Helper functions
# ============================================================================

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

"""
Initialize reference positions from sample closest to overall centroid.
"""
function initialize_reference_positions(
    map_samples::Vector{<:BaGoLSample},
    K::Int
)
    # Pool all positions to find overall centroid
    all_positions = Tuple{Float64, Float64}[]
    for sample in map_samples
        for e in sample.emitters
            push!(all_positions, (Float64(e.x), Float64(e.y)))
        end
    end

    centroid_x = mean(p[1] for p in all_positions)
    centroid_y = mean(p[2] for p in all_positions)

    # Find sample whose centroid is closest to overall centroid
    best_sample = map_samples[1]
    best_dist = Inf
    for sample in map_samples
        sx = mean(e.x for e in sample.emitters)
        sy = mean(e.y for e in sample.emitters)
        d = (sx - centroid_x)^2 + (sy - centroid_y)^2
        if d < best_dist
            best_dist = d
            best_sample = sample
        end
    end

    return [(Float64(e.x), Float64(e.y)) for e in best_sample.emitters]
end

# ============================================================================
# Main MAP-N estimation
# ============================================================================

"""
    estimate_mapn(chain::RJMCMCChain; n_refine=10) -> (Vector{Emitter2DFit}, Vector{Int})

Estimate MAP-N (Maximum A Posteriori number) emitters from RJMCMC chain.

Uses iterative Hungarian matching with median-based reference positions to handle
label switching between nearby emitters. Position uncertainties are computed using
MAD (Median Absolute Deviation) which is robust to outliers from residual label
switching - this produces σ values that accurately represent the posterior width
for downstream analysis assuming normal distributions.

# Algorithm
1. Find MAP-N (most common K in posterior)
2. Filter to samples with K = MAP-N
3. Initialize reference positions from sample closest to centroid
4. Iteratively: Hungarian match all samples → update reference to median
5. Final positions: median of matched positions (robust to outliers)
6. Final σ: MAD-based estimate (robust, valid for normal assumption)

# Arguments
- `chain`: RJMCMCChain with post-burn-in samples
- `n_refine`: Number of iterative refinement steps (default 10)

# Returns
- `emitters`: Vector of Emitter2DFit with positions and uncertainties
- `posterior_k`: Histogram of K values across all samples (index k+1 = count of K=k)

# Example
```julia
chain = run_bagol_chain(locs; n_iterations=10000, burn_in=2000)
emitters, posterior_k = estimate_mapn(chain)
```
"""
function estimate_mapn(chain::RJMCMCChain; n_refine::Int = 10)
    samples = chain.samples

    if isempty(samples)
        error("No samples in chain - run more iterations or reduce burn_in")
    end

    # Build histogram of K
    ks = [length(s.emitters) for s in samples]
    k_max = maximum(ks)
    posterior_k = zeros(Int, k_max + 1)
    for k in ks
        posterior_k[k + 1] += 1
    end

    # Find MAP-N (most common K)
    map_n = argmax(posterior_k) - 1

    if map_n == 0
        return SMLMData.Emitter2DFit[], posterior_k
    end

    # Get samples with MAP-N emitters
    map_samples = filter(s -> length(s.emitters) == map_n, samples)

    if isempty(map_samples)
        return SMLMData.Emitter2DFit[], posterior_k
    end

    # Initialize reference from sample closest to centroid
    ref_positions = initialize_reference_positions(map_samples, map_n)

    # Iterative refinement: match → median → new reference → repeat
    for _ in 1:n_refine
        matched_positions = [Vector{Tuple{Float64, Float64}}() for _ in 1:map_n]

        for sample in map_samples
            sample_positions = [(Float64(e.x), Float64(e.y)) for e in sample.emitters]
            assignment = position_hungarian(ref_positions, sample_positions)

            for (i, j) in enumerate(assignment)
                if j <= length(sample.emitters)
                    push!(matched_positions[i], sample_positions[j])
                end
            end
        end

        # Update reference to MEDIAN (robust to label switching outliers)
        for i in 1:map_n
            if !isempty(matched_positions[i])
                xs = [p[1] for p in matched_positions[i]]
                ys = [p[2] for p in matched_positions[i]]
                ref_positions[i] = (median(xs), median(ys))
            end
        end
    end

    # Final pass to collect positions with converged reference
    matched_positions = [Vector{Tuple{Float64, Float64}}() for _ in 1:map_n]

    for sample in map_samples
        sample_positions = [(Float64(e.x), Float64(e.y)) for e in sample.emitters]
        assignment = position_hungarian(ref_positions, sample_positions)

        for (i, j) in enumerate(assignment)
            if j <= length(sample.emitters)
                push!(matched_positions[i], sample_positions[j])
            end
        end
    end

    # Compute position and uncertainty for each emitter
    result_emitters = SMLMData.Emitter2DFit[]

    for (id, positions) in enumerate(matched_positions)
        if isempty(positions)
            continue
        end
        xs = [p[1] for p in positions]
        ys = [p[2] for p in positions]

        # Median position (robust to outliers)
        pos_x = median(xs)
        pos_y = median(ys)

        # MAD-based uncertainty (robust, valid for normal assumption)
        n = length(xs)
        if n > 1
            σ_x = mad_sigma(xs)
            σ_y = mad_sigma(ys)
            # Covariance uses median as center
            σ_xy = sum((xs[i] - pos_x) * (ys[i] - pos_y) for i in 1:n) / (n - 1)
        else
            σ_x = 0.0
            σ_y = 0.0
            σ_xy = 0.0
        end

        push!(result_emitters, SMLMData.Emitter2DFit(
            pos_x, pos_y,
            0.0, 0.0,        # photons, bg (not applicable)
            σ_x, σ_y, σ_xy,  # position uncertainties
            0.0, 0.0,        # σ_photons, σ_bg
            1, 1, 0, id
        ))
    end

    return result_emitters, posterior_k
end

# ============================================================================
# Collapsed chain MAP-N estimation
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

Uses the same iterative Hungarian matching + median position approach as the
RJMCMC `estimate_mapn`, but derives positions from ClusterStats posterior
means (which are deterministic given assignments).

# Algorithm
1. Count K per assignment sample
2. Find MAP-N (most common K)
3. Filter to samples with K = MAP-N
4. For each sample, build ClusterStats per cluster → posterior mean positions
5. Iterative Hungarian matching → update reference to median
6. Final positions: median (robust to label switching)
7. Final σ: MAD-based (robust, valid for normal assumption)

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

    # MAP-N
    map_n = argmax(posterior_k) - 1
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

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

"""
Overlap-based Hungarian matching via contingency matrix.

Builds K_ref × K_sample contingency matrix C[i,j] = |{locs in ref cluster i ∩ sample cluster j}|
in O(N), then runs Hungarian to maximize total overlap.

Returns (assignment, total_overlap, cluster_overlaps) where assignment[i] = j means
ref cluster i matches sample cluster j, and cluster_overlaps[i] is the number of
shared localizations for that specific match.
"""
function overlap_hungarian(
    ref_assignments::Vector{Int16},
    sample_assignments::Vector{Int16},
    ref_labels::Vector{Int16},
    sample_labels::Vector{Int16}
)
    K_ref = length(ref_labels)
    K_sample = length(sample_labels)

    if K_ref == 1 && K_sample == 1
        n = Int(count(==(ref_labels[1]), ref_assignments))
        return [1], n, [n]
    end

    # Map labels to contiguous indices
    ref_idx = Dict{Int16, Int}()
    for (i, lab) in enumerate(ref_labels)
        ref_idx[lab] = i
    end
    sample_idx = Dict{Int16, Int}()
    for (i, lab) in enumerate(sample_labels)
        sample_idx[lab] = i
    end

    # Build contingency matrix in O(N)
    K = max(K_ref, K_sample)
    C = zeros(Int, K, K)
    @inbounds for i in eachindex(ref_assignments)
        ri = get(ref_idx, ref_assignments[i], 0)
        si = get(sample_idx, sample_assignments[i], 0)
        if ri > 0 && si > 0
            C[ri, si] += 1
        end
    end

    # Maximize overlap via Hungarian on negated cost
    assignment, _ = Hungarian.hungarian(-C)

    total_overlap = 0
    cluster_overlaps = zeros(Int, K_ref)
    @inbounds for i in 1:K_ref
        j = assignment[i]
        if j <= K_sample
            cluster_overlaps[i] = C[i, j]
            total_overlap += cluster_overlaps[i]
        end
    end

    return assignment[1:K_ref], total_overlap, cluster_overlaps
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
    n_refine::Int = 10,
    k_override::Union{Nothing, Int} = nothing
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

    # K selection: use override if provided, else histogram mode
    if k_override !== nothing
        map_n = k_override
    else
        map_n, _ = _smoothed_map_n(posterior_k)
    end
    if map_n == 0
        return SMLMData.Emitter2DFit{Float64}[], posterior_k
    end

    # Filter to samples with K = map_n
    map_indices = findall(k -> k == map_n, ks)
    if isempty(map_indices)
        # No samples with exact K — find closest available
        available_ks = sort!(unique(ks))
        _, idx = findmin(abs.(available_ks .- map_n))
        map_n = available_ks[idx]
        map_indices = findall(k -> k == map_n, ks)
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
    result_emitters = SMLMData.Emitter2DFit{Float64}[]
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

# ============================================================================
# Overlap-based MAP-N: Dahl template + overlap Hungarian matching
# ============================================================================

"""
    estimate_mapn_overlap(samples, locs, dahl_assignments; min_overlap_frac=0.5)
        -> (Vector{Emitter2DFit}, Vector{Int})

Fast MAP-N extraction using the Dahl partition as reference template.
Matches each MCMC sample to the Dahl partition via overlap (contingency matrix)
Hungarian matching, requiring only ONE O(K³) call per sample instead of 11
iterative position-based rounds.

# Algorithm
1. K from Dahl assignments
2. Filter samples to K = K_dahl
3. For each matching sample: overlap Hungarian → per-cluster overlap filter
4. Only include a sample's posterior mean for cluster i if its per-cluster
   overlap fraction >= min_overlap_frac (rejects label-switching mismatches)
5. Position: mean of well-matched posterior means
6. Uncertainty: Term 1 (Dahl analytic covariance) + Term 2 (allocation variance
   from well-matched samples only). This is the law of total variance applied
   to the collapsed posterior, equivalent to the marginal posterior variance
   an RJMCMC sampler would produce.
7. Fallback: if no samples match, use Dahl assignments directly

# Returns
- `emitters`: Vector of Emitter2DFit with positions and uncertainties
- `posterior_k`: Histogram of K values (index k+1 = count of K=k)
"""
function estimate_mapn_overlap(
    samples::Vector{Vector{Int16}},
    locs::Vector{<:SMLMData.AbstractEmitter},
    dahl_assignments::Vector{Int16};
    min_overlap_frac::Float64 = 0.5
)
    isempty(samples) && error("No assignment samples")

    # Dahl K and labels
    dahl_labels = sort!(unique(dahl_assignments))
    K_dahl = length(dahl_labels)

    # Build K histogram
    ks = [length(unique(s)) for s in samples]
    k_max = maximum(ks)
    posterior_k = zeros(Int, k_max + 1)
    for k in ks
        posterior_k[k + 1] += 1
    end

    if K_dahl == 0
        return SMLMData.Emitter2DFit{Float64}[], posterior_k
    end

    # Build Dahl ClusterStats and cluster sizes
    N = length(locs)
    dahl_cs = Dict{Int16, ClusterStats}()
    dahl_cluster_sizes = zeros(Int, K_dahl)
    for (ki, lab) in enumerate(dahl_labels)
        cs = ClusterStats()
        for i in eachindex(dahl_assignments)
            if dahl_assignments[i] == lab
                cs = add_loc(cs, locs[i])
            end
        end
        dahl_cs[lab] = cs
        dahl_cluster_sizes[ki] = Int(cs.n)
    end

    # Filter to samples with K = K_dahl
    map_indices = findall(k -> k == K_dahl, ks)
    if isempty(map_indices)
        # Fallback: no samples with K_dahl, use Dahl directly
        return _emitters_from_assignments(dahl_assignments, locs), posterior_k
    end

    # Match each sample to Dahl via overlap Hungarian with per-cluster filtering
    matched_positions = [Vector{Tuple{Float64, Float64}}() for _ in 1:K_dahl]
    n_used = 0

    for idx in map_indices
        s = samples[idx]
        sample_labels = sort!(unique(s))

        assignment, total_overlap, cluster_overlaps = overlap_hungarian(
            dahl_assignments, s, dahl_labels, sample_labels
        )

        # Skip entire sample if total overlap is very low
        total_overlap < div(N, 2) && continue
        n_used += 1

        # Build sample ClusterStats per cluster
        sample_cs = Dict{Int16, ClusterStats}()
        for lab in sample_labels
            cs = ClusterStats()
            for i in eachindex(s)
                if s[i] == lab
                    cs = add_loc(cs, locs[i])
                end
            end
            sample_cs[lab] = cs
        end

        # Per-cluster overlap gate: only include well-matched clusters
        for (ki, j) in enumerate(assignment)
            j > length(sample_labels) && continue
            # Check per-cluster overlap fraction
            dahl_cluster_sizes[ki] == 0 && continue
            overlap_frac = cluster_overlaps[ki] / dahl_cluster_sizes[ki]
            overlap_frac < min_overlap_frac && continue

            slab = sample_labels[j]
            if haskey(sample_cs, slab) && sample_cs[slab].n > 0
                push!(matched_positions[ki], posterior_mean(sample_cs[slab]))
            end
        end
    end

    # Fallback: no samples passed overlap filter
    if n_used == 0
        return _emitters_from_assignments(dahl_assignments, locs), posterior_k
    end

    # Build emitters: mean positions, total variance = analytic + allocation
    result_emitters = SMLMData.Emitter2DFit{Float64}[]
    for (ki, lab) in enumerate(dahl_labels)
        positions = matched_positions[ki]
        cs = dahl_cs[lab]
        cs.n == 0 && continue

        if isempty(positions)
            # No well-matched samples for this cluster, use Dahl directly
            mx, my = posterior_mean(cs)
        else
            xs = [p[1] for p in positions]
            ys = [p[2] for p in positions]
            mx = mean(xs)
            my = mean(ys)
        end

        # Term 1: E[Var(θ|Z)] — analytic posterior covariance (Dahl)
        Σ_xx, Σ_xy, Σ_yy = posterior_cov(cs)

        # Term 2: Cov[E(θ|Z)] — allocation covariance from well-matched samples
        # Bad matches excluded by per-cluster overlap gate, so var/cov is clean
        # Full 2×2 covariance (not just diagonal) for coordinate-system invariance
        if length(positions) >= 3
            xs = [p[1] for p in positions]
            ys = [p[2] for p in positions]
            Σ_xx += var(xs)
            Σ_yy += var(ys)
            Σ_xy += cov(xs, ys)
        end

        push!(result_emitters, SMLMData.Emitter2DFit(
            mx, my, Float64(cs.n), 0.0,
            sqrt(max(Σ_xx, 0.0)), sqrt(max(Σ_yy, 0.0)), Σ_xy,
            0.0, 0.0, 1, 1, 0, ki
        ))
    end

    return result_emitters, posterior_k
end

# ============================================================================
# PSM-corrected MAP-N: K from PSM, then standard Hungarian pipeline
# ============================================================================

"""
    _psm_cluster_count(psm; threshold=0.5) -> Int

Extract K from the PSM by thresholding at `threshold` and counting
connected components. A transient split shows C_ij ≈ 0.9 between the
split locs — still one cluster. This avoids the per-sample K histogram
bias where each transient split inflates K by +1.
"""
function _psm_cluster_count(psm::Matrix{Float64}; threshold::Float64 = 0.5)
    n = size(psm, 1)
    n == 0 && return 0

    # Union-Find via label propagation
    label = collect(1:n)

    function find_root(i)
        while label[i] != i
            label[i] = label[label[i]]  # path compression
            i = label[i]
        end
        return i
    end

    for i in 1:n
        for j in (i+1):n
            if psm[i, j] >= threshold
                ri, rj = find_root(i), find_root(j)
                if ri != rj
                    label[ri] = rj
                end
            end
        end
    end

    # Count distinct roots
    return length(unique(find_root(i) for i in 1:n))
end

"""
    estimate_mapn_psm(samples, locs, psm; threshold=0.5, n_refine=10)
        -> (Vector{Emitter2DFit}, Vector{Int})

PSM-corrected MAP-N: determines K from the PSM block structure
(threshold + connected components), then uses the standard Hungarian
matching pipeline from `estimate_mapn_collapsed` on samples with that K.

This fixes the linear K bias of histogram-mode MAP-N at large K,
while keeping the well-tested Hungarian + median position estimation.
"""
function estimate_mapn_psm(
    samples::Vector{Vector{Int16}},
    locs::Vector{<:SMLMData.AbstractEmitter},
    psm::Matrix{Float64};
    threshold::Float64 = 0.5,
    n_refine::Int = 10
)
    isempty(samples) && error("No assignment samples")

    # K from PSM
    k_psm = _psm_cluster_count(psm; threshold=threshold)
    if k_psm == 0
        return SMLMData.Emitter2DFit{Float64}[], Int[]
    end

    # Build K histogram (for diagnostics/return)
    ks = [length(unique(s)) for s in samples]
    k_max = maximum(ks)
    posterior_k = zeros(Int, k_max + 1)
    for k in ks
        posterior_k[k + 1] += 1
    end

    # Filter to samples with K = k_psm
    map_indices = findall(k -> k == k_psm, ks)
    if isempty(map_indices)
        # No samples with exact K — find closest K that has samples
        available_ks = sort!(unique(ks))
        _, idx = findmin(abs.(available_ks .- k_psm))
        k_psm = available_ks[idx]
        map_indices = findall(k -> k == k_psm, ks)
    end

    # From here: identical to estimate_mapn_collapsed pipeline
    map_n = k_psm

    # Extract positions from each matching sample
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

    # Build emitters with median positions and ClusterStats posterior covariances
    ref_sample = samples[map_indices[1]]
    result_emitters = SMLMData.Emitter2DFit{Float64}[]
    ref_labels = sort!(unique(ref_sample))

    for (id, positions) in enumerate(matched_positions)
        isempty(positions) && continue
        xs = [p[1] for p in positions]
        ys = [p[2] for p in positions]
        pos_x = median(xs)
        pos_y = median(ys)

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
            Float64(n_locs_cluster), 0.0,
            σ_x, σ_y, σ_xy,
            0.0, 0.0, 1, 1, 0, id
        ))
    end

    return result_emitters, posterior_k
end

# ============================================================================
# PSM-based partition estimation (Dahl's method + Greedy VI)
# ============================================================================

"""
    _association_psm_distance(assignments, psm) -> Float64

Squared Frobenius distance between the association matrix of `assignments`
and the PSM, using only the upper triangle:
  D = Σ_{i<j} (1[z_i == z_j] - C_ij)²
"""
function _association_psm_distance(assignments::Vector{Int16}, psm::Matrix{Float64})
    n = length(assignments)
    d = 0.0
    @inbounds for i in 1:n
        for j in (i+1):n
            coassigned = Float64(assignments[i] == assignments[j])
            d += (coassigned - psm[i, j])^2
        end
    end
    return d
end

"""
    _emitters_from_assignments(assignments, locs) -> Vector{Emitter2DFit}

Build emitters from a single assignment vector using ClusterStats posteriors.
Each cluster becomes one emitter with analytical position and covariance.
"""
function _emitters_from_assignments(
    assignments::AbstractVector{<:Integer},
    locs::Vector{<:SMLMData.AbstractEmitter}
)
    unique_labels = sort!(unique(assignments))
    emitters = SMLMData.Emitter2DFit{Float64}[]
    for (id, lab) in enumerate(unique_labels)
        cs = ClusterStats()
        for i in eachindex(assignments)
            if assignments[i] == lab
                cs = add_loc(cs, locs[i])
            end
        end
        cs.n == 0 && continue
        mx, my = posterior_mean(cs)
        Σ_xx, Σ_xy, Σ_yy = posterior_cov(cs)
        push!(emitters, SMLMData.Emitter2DFit(
            mx, my, Float64(cs.n), 0.0,
            sqrt(max(Σ_xx, 0.0)), sqrt(max(Σ_yy, 0.0)), Σ_xy,
            0.0, 0.0, 1, 1, 0, id
        ))
    end
    return emitters
end

"""
    _cluster_stability(assignments, psm) -> Vector{Float64}

Compute per-cluster stability score: mean pairwise co-occupancy within cluster.
Score ≈ 1.0 means cluster is always together; < 0.8 means ambiguous.
"""
function _cluster_stability(assignments::Vector{Int16}, psm::Matrix{Float64})
    unique_labels = sort!(unique(assignments))
    scores = Float64[]
    for lab in unique_labels
        members = findall(==(lab), assignments)
        nm = length(members)
        if nm <= 1
            push!(scores, 1.0)
            continue
        end
        s = 0.0
        np = 0
        @inbounds for a in 1:nm
            for b in (a+1):nm
                s += psm[members[a], members[b]]
                np += 1
            end
        end
        push!(scores, s / np)
    end
    return scores
end

"""
    estimate_dahl(samples, locs, psm) -> (emitters, posterior_k, stability)

Dahl's method: select the MCMC sample whose association matrix is closest
(squared Frobenius) to the PSM. Equivalent to minimizing posterior expected
Binder loss restricted to visited partitions.

Returns emitters with ClusterStats posteriors plus per-cluster stability scores.
"""
function estimate_dahl(
    samples::Vector{Vector{Int16}},
    locs::Vector{<:SMLMData.AbstractEmitter},
    psm::Matrix{Float64}
)
    isempty(samples) && error("No assignment samples")

    # Build K histogram
    ks = [length(unique(s)) for s in samples]
    k_max = maximum(ks)
    posterior_k = zeros(Int, k_max + 1)
    for k in ks
        posterior_k[k + 1] += 1
    end

    # Find sample closest to PSM
    best_idx = 1
    best_dist = Inf
    for (t, s) in enumerate(samples)
        d = _association_psm_distance(s, psm)
        if d < best_dist
            best_dist = d
            best_idx = t
        end
    end

    best_assignments = samples[best_idx]
    emitters = _emitters_from_assignments(best_assignments, locs)
    stability = _cluster_stability(best_assignments, psm)
    return emitters, posterior_k, stability, best_assignments
end

"""
    _xlogx(x) -> Float64

Compute x * log(x), with 0 * log(0) = 0 convention.
"""
@inline _xlogx(x::Int) = x > 0 ? x * log(x) : 0.0

"""
    _vi_from_contingency(ngh, ng, n, T) -> Float64

Compute VI(a, z) from pre-built contingency table.
VI = 2H(a,z) - H(a) - H(z), using raw counts (log-count form to avoid /N).
Since we sum over T samples and divide, the N factors cancel.
"""
function _vi_from_contingency(ngh::AbstractMatrix{<:Integer}, ng::AbstractVector{<:Integer},
                               nh::AbstractVector{<:Integer}, n::Int)
    inv_n = 1.0 / n
    H_a = 0.0
    @inbounds for g in eachindex(ng)
        H_a -= _xlogx(ng[g])
    end
    H_a = H_a * inv_n + log(n)

    H_z = 0.0
    @inbounds for h in eachindex(nh)
        H_z -= _xlogx(nh[h])
    end
    H_z = H_z * inv_n + log(n)

    H_az = 0.0
    @inbounds for idx in eachindex(ngh)
        H_az -= _xlogx(ngh[idx])
    end
    H_az = H_az * inv_n + log(n)

    return 2 * H_az - H_a - H_z
end

"""
    estimate_vi_greedy(samples, locs, psm; n_restarts=3, k_up=0) -> (emitters, posterior_k, stability)

Greedy search under Variation of Information loss (Rastelli & Friel 2018).

Uses cached contingency tables for O(1) per-sample delta computation.
Total cost per sweep: O(T × N × K_up) where T = samples, N = locs.

# Algorithm
1. Pre-compute T contingency tables between candidate partition and each sample
2. For each loc i (random order), try all clusters + new singleton
3. Compute ΔVI in O(T) by updating 2 contingency entries per sample
4. Accept the move with minimum expected VI
5. Repeat sweeps until no improvement; multiple restarts

Returns emitters with ClusterStats posteriors plus per-cluster stability scores.
"""
function estimate_vi_greedy(
    samples::Vector{Vector{Int16}},
    locs::Vector{<:SMLMData.AbstractEmitter},
    psm::Matrix{Float64};
    n_restarts::Int = 3,
    k_up::Int = 0
)
    isempty(samples) && error("No assignment samples")
    n = length(locs)
    T = length(samples)

    # Build K histogram
    ks = [length(unique(s)) for s in samples]
    k_max = maximum(ks)
    posterior_k = zeros(Int, k_max + 1)
    for k in ks
        posterior_k[k + 1] += 1
    end

    if k_up == 0
        k_up = min(n, k_max + 5)
    end

    # Remap all sample labels to contiguous 1:K_t
    # Also store the remapped label for each loc in each sample
    sample_labels = Matrix{Int}(undef, n, T)  # sample_labels[i, t] = cluster of loc i in sample t
    sample_ks = Vector{Int}(undef, T)
    for t in 1:T
        label_map = Dict{Int16, Int}()
        next_id = 0
        for i in 1:n
            lab = samples[t][i]
            if !haskey(label_map, lab)
                next_id += 1
                label_map[lab] = next_id
            end
            sample_labels[i, t] = label_map[lab]
        end
        sample_ks[t] = next_id
    end
    kz_max = maximum(sample_ks)

    # Precompute column sums nh[h, t] for each sample (fixed throughout)
    nh_all = zeros(Int, kz_max, T)
    for t in 1:T
        for i in 1:n
            nh_all[sample_labels[i, t], t] += 1
        end
    end

    # Get Dahl's partition as starting point
    dahl_idx = 1
    dahl_best = Inf
    for (t, s) in enumerate(samples)
        d = _association_psm_distance(s, psm)
        if d < dahl_best
            dahl_best = d
            dahl_idx = t
        end
    end

    best_partition = copy(samples[1])
    best_loss = Inf

    for restart in 1:n_restarts
        # Initialize partition a with contiguous labels 1:K_a
        if restart == 1
            a_source = samples[dahl_idx]
        else
            a_source = samples[rand(1:T)]
        end

        a = Vector{Int}(undef, n)
        label_map = Dict{Int16, Int}()
        next_id = 0
        for i in 1:n
            lab = a_source[i]
            if !haskey(label_map, lab)
                next_id += 1
                label_map[lab] = next_id
            end
            a[i] = label_map[lab]
        end
        ka = next_id

        # Build T contingency tables: ngh[g, h, t] and row sums ng[g]
        ngh = zeros(Int, k_up, kz_max, T)
        ng = zeros(Int, k_up)
        for t in 1:T
            for i in 1:n
                ngh[a[i], sample_labels[i, t], t] += 1
            end
        end
        for i in 1:n
            ng[a[i]] += 1
        end

        # Compute initial total VI (sum over samples, not mean)
        total_vi = 0.0
        for t in 1:T
            total_vi += _vi_from_contingency(
                @view(ngh[:, :, t]), ng, @view(nh_all[:, t]), n)
        end

        # Greedy sweeps
        for sweep in 1:50
            improved = false
            perm = randperm(n)

            for i in perm
                r = a[i]  # current cluster of loc i

                # Skip if r is a singleton and we're at k_up
                # Determine active clusters and one empty slot
                max_candidate = min(ka + 1, k_up)

                best_s = r
                best_delta = 0.0

                for s in 1:max_candidate
                    s == r && continue

                    # Compute Δ_total_vi for moving loc i from r to s
                    delta = 0.0
                    @inbounds for t in 1:T
                        v = sample_labels[i, t]  # loc i's cluster in sample t
                        nh_v = nh_all[v, t]

                        # Old contributions (will be removed)
                        old_nrv = ngh[r, v, t]
                        old_nsv = ngh[s, v, t]
                        old_nr = ng[r]
                        old_ns = ng[s]

                        # New contributions (after move)
                        new_nrv = old_nrv - 1
                        new_nsv = old_nsv + 1
                        new_nr = old_nr - 1
                        new_ns = old_ns + 1

                        # ΔVI = Δ(2H(a,z)) - Δ(H(a)) - Δ(H(z))
                        # H(z) doesn't change (z is fixed)
                        # ΔH(a) = -(new_nr*log(new_nr) + new_ns*log(new_ns)
                        #           - old_nr*log(old_nr) - old_ns*log(old_ns)) / N + 0
                        # Δ(2H(a,z)) = -2*(new_nrv*log(new_nrv) + new_nsv*log(new_nsv)
                        #                  - old_nrv*log(old_nrv) - old_nsv*log(old_nsv)) / N

                        inv_n = 1.0 / n

                        delta_H_a = -(_xlogx(new_nr) + _xlogx(new_ns) -
                                      _xlogx(old_nr) - _xlogx(old_ns)) * inv_n

                        delta_H_az = -(_xlogx(new_nrv) + _xlogx(new_nsv) -
                                       _xlogx(old_nrv) - _xlogx(old_nsv)) * inv_n

                        delta += 2 * delta_H_az - delta_H_a  # -ΔH(z) = 0
                    end

                    if delta < best_delta
                        best_delta = delta
                        best_s = s
                    end
                end

                # Apply best move
                if best_s != r
                    # Update contingency tables
                    @inbounds for t in 1:T
                        v = sample_labels[i, t]
                        ngh[r, v, t] -= 1
                        ngh[best_s, v, t] += 1
                    end
                    ng[r] -= 1
                    ng[best_s] += 1
                    a[i] = best_s
                    total_vi += best_delta

                    # Track max active cluster
                    if best_s > ka
                        ka = best_s
                    end

                    improved = true
                end
            end

            !improved && break
        end

        # Convert back to Int16 with compact labels
        a16 = Vector{Int16}(undef, n)
        compact_map = Dict{Int, Int16}()
        next_lab = Int16(1)
        for i in 1:n
            if !haskey(compact_map, a[i])
                compact_map[a[i]] = next_lab
                next_lab += Int16(1)
            end
            a16[i] = compact_map[a[i]]
        end

        if total_vi < best_loss
            best_loss = total_vi
            best_partition = a16
        end
    end

    emitters = _emitters_from_assignments(best_partition, locs)
    stability = _cluster_stability(best_partition, psm)
    return emitters, posterior_k, stability
end

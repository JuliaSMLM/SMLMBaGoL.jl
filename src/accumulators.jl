# Accumulator interface for collapsed Gibbs sampler
#
# Accumulators collect statistics from the MCMC chain without storing
# full samples. They are updated each iteration after burn-in and can
# be merged across partitions.

"""
    AbstractAccumulator

Base type for MCMC accumulators. Subtypes must implement:
- `update!(acc, state, locs, μ, shape, iter)` — add one iteration
- `result(acc)` — extract final result
- `merge!(acc_target, acc_source)` — combine results from two partitions
"""
abstract type AbstractAccumulator end

# ============================================================================
# EmitterCountHist — histogram of K per iteration
# ============================================================================

"""
    EmitterCountHist

Histogram of emitter count K across iterations.
Result: Vector{Int} where index k+1 = count of iterations with K=k.
"""
mutable struct EmitterCountHist <: AbstractAccumulator
    counts::Vector{Int}   # counts[k+1] = number of iterations with K=k
end

EmitterCountHist() = EmitterCountHist(Int[])

function accumulator_update!(acc::EmitterCountHist, state::CollapsedState,
                             locs::Vector{<:SMLMData.AbstractEmitter},
                             μ::Float64, shape::Float64, iter::Int)
    k = state.n_active
    # Grow histogram if needed
    while length(acc.counts) < k + 1
        push!(acc.counts, 0)
    end
    acc.counts[k + 1] += 1
end

accumulator_result(acc::EmitterCountHist) = acc.counts

function accumulator_merge!(target::EmitterCountHist, source::EmitterCountHist)
    while length(target.counts) < length(source.counts)
        push!(target.counts, 0)
    end
    for i in eachindex(source.counts)
        target.counts[i] += source.counts[i]
    end
end

# ============================================================================
# PosteriorImage — Rao-Blackwellized posterior image
# ============================================================================

"""
    PosteriorImage

Rao-Blackwellized posterior image accumulator. Each iteration adds a Gaussian
blob N(μ_post_j, Σ_post_j) per active cluster, rendering only within 4σ
of each blob center. This integrates over position uncertainty analytically,
producing much smoother images than point-delta histogramming.
"""
mutable struct PosteriorImage <: AbstractAccumulator
    image::Matrix{Float64}
    edges_x::Vector{Float64}
    edges_y::Vector{Float64}
    pixel_size::Float64
    n_sigma::Float64  # Render radius in sigma units
    initialized::Bool
end

function PosteriorImage(; pixel_size::Float64,
                         xlim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
                         ylim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
                         n_sigma::Float64 = 4.0)
    if xlim !== nothing && ylim !== nothing
        edges_x = collect(xlim[1]:pixel_size:(xlim[2] + pixel_size))
        edges_y = collect(ylim[1]:pixel_size:(ylim[2] + pixel_size))
        nx = length(edges_x) - 1
        ny = length(edges_y) - 1
        return PosteriorImage(zeros(nx, ny), edges_x, edges_y, pixel_size, n_sigma, true)
    end
    # Will be initialized on first update from data bounds
    return PosteriorImage(zeros(1, 1), Float64[0.0, 1.0], Float64[0.0, 1.0],
                          pixel_size, n_sigma, false)
end

function _ensure_initialized!(acc::PosteriorImage, locs::Vector{<:SMLMData.AbstractEmitter})
    acc.initialized && return
    xs = [loc.x for loc in locs]
    ys = [loc.y for loc in locs]
    pad = 3 * maximum(loc.σ_x for loc in locs)
    xmin, xmax = minimum(xs) - pad, maximum(xs) + pad
    ymin, ymax = minimum(ys) - pad, maximum(ys) + pad
    acc.edges_x = collect(xmin:acc.pixel_size:(xmax + acc.pixel_size))
    acc.edges_y = collect(ymin:acc.pixel_size:(ymax + acc.pixel_size))
    nx = length(acc.edges_x) - 1
    ny = length(acc.edges_y) - 1
    acc.image = zeros(nx, ny)
    acc.initialized = true
end

function accumulator_update!(acc::PosteriorImage, state::CollapsedState,
                             locs::Vector{<:SMLMData.AbstractEmitter},
                             μ::Float64, shape::Float64, iter::Int)
    _ensure_initialized!(acc, locs)

    nx = size(acc.image, 1)
    ny = size(acc.image, 2)
    xmin = acc.edges_x[1]
    ymin = acc.edges_y[1]
    ps = acc.pixel_size

    for (j, cs) in enumerate(state.clusters)
        state.active[j] || continue
        cs.n == 0 && continue

        mx, my = posterior_mean(cs)
        Σ_xx, Σ_xy, Σ_yy = posterior_cov(cs)

        # Effective sigma for bounding box
        σ_x = sqrt(Σ_xx)
        σ_y = sqrt(Σ_yy)
        radius_x = acc.n_sigma * σ_x
        radius_y = acc.n_sigma * σ_y

        # Pixel range to render
        ix_lo = max(1, floor(Int, (mx - radius_x - xmin) / ps) + 1)
        ix_hi = min(nx, floor(Int, (mx + radius_x - xmin) / ps) + 1)
        iy_lo = max(1, floor(Int, (my - radius_y - ymin) / ps) + 1)
        iy_hi = min(ny, floor(Int, (my + radius_y - ymin) / ps) + 1)

        # Precompute inverse covariance for 2D Gaussian
        det_Σ = Σ_xx * Σ_yy - Σ_xy^2
        if det_Σ <= 0
            continue
        end
        inv_det = 1.0 / det_Σ
        norm = 1.0 / (2π * sqrt(det_Σ))

        for iy in iy_lo:iy_hi
            py = ymin + (iy - 0.5) * ps
            dy = py - my
            for ix in ix_lo:ix_hi
                px = xmin + (ix - 0.5) * ps
                dx = px - mx
                # Quadratic form: d^T Σ^{-1} d
                qf = inv_det * (Σ_yy * dx^2 - 2 * Σ_xy * dx * dy + Σ_xx * dy^2)
                if qf < 2 * acc.n_sigma^2  # Within n_sigma radius
                    acc.image[ix, iy] += norm * exp(-0.5 * qf) * ps^2
                end
            end
        end
    end
end

function accumulator_result(acc::PosteriorImage)
    return (image=acc.image, edges_x=acc.edges_x, edges_y=acc.edges_y,
            pixel_size=acc.pixel_size)
end

function accumulator_merge!(target::PosteriorImage, source::PosteriorImage)
    if !source.initialized
        return
    end
    if !target.initialized
        target.image = copy(source.image)
        target.edges_x = copy(source.edges_x)
        target.edges_y = copy(source.edges_y)
        target.pixel_size = source.pixel_size
        target.initialized = true
        return
    end
    # Same grid: just add
    if target.edges_x == source.edges_x && target.edges_y == source.edges_y
        target.image .+= source.image
    else
        # Different grids: place source pixels into target grid
        sxmin = source.edges_x[1]
        symin = source.edges_y[1]
        txmin = target.edges_x[1]
        tymin = target.edges_y[1]
        ps = target.pixel_size
        tnx = size(target.image, 1)
        tny = size(target.image, 2)
        for iy in 1:size(source.image, 2)
            sy = symin + (iy - 0.5) * ps
            tiy = floor(Int, (sy - tymin) / ps) + 1
            (1 <= tiy <= tny) || continue
            for ix in 1:size(source.image, 1)
                sx = sxmin + (ix - 0.5) * ps
                tix = floor(Int, (sx - txmin) / ps) + 1
                (1 <= tix <= tnx) || continue
                target.image[tix, tiy] += source.image[ix, iy]
            end
        end
    end
end

# ============================================================================
# NNDistHist — nearest-neighbor distance histogram
# ============================================================================

"""
    NNDistHist

Histogram of nearest-neighbor distances between emitter posterior means,
computed per iteration. Useful for detecting clustering patterns.
Pre-allocates position buffers for zero-allocation updates.
"""
mutable struct NNDistHist <: AbstractAccumulator
    counts::Vector{Int}
    bin_edges::Vector{Float64}
    _pos_x::Vector{Float64}   # Workspace: x-coordinates of active clusters
    _pos_y::Vector{Float64}   # Workspace: y-coordinates of active clusters
end

function NNDistHist(; max_dist::Float64 = 0.1, n_bins::Int = 100)
    bin_size = max_dist / n_bins
    edges = collect(0.0:bin_size:max_dist)
    NNDistHist(zeros(Int, n_bins), edges, Float64[], Float64[])
end

function accumulator_update!(acc::NNDistHist, state::CollapsedState,
                             locs::Vector{<:SMLMData.AbstractEmitter},
                             μ::Float64, shape::Float64, iter::Int)
    # Collect active cluster positions into workspace buffers
    K = state.n_active
    # Grow buffers if needed (rare, once at most)
    if length(acc._pos_x) < K
        resize!(acc._pos_x, K)
        resize!(acc._pos_y, K)
    end

    n_pos = 0
    @inbounds for (j, cs) in enumerate(state.clusters)
        state.active[j] || continue
        cs.n == 0 && continue
        n_pos += 1
        mx, my = posterior_mean(cs)
        acc._pos_x[n_pos] = mx
        acc._pos_y[n_pos] = my
    end

    n_pos < 2 && return

    n_bins = length(acc.counts)
    bin_size = acc.bin_edges[2] - acc.bin_edges[1]

    # For each emitter, find nearest neighbor distance
    @inbounds for i in 1:n_pos
        min_d = Inf
        px_i = acc._pos_x[i]
        py_i = acc._pos_y[i]
        for j in 1:n_pos
            i == j && continue
            dx = px_i - acc._pos_x[j]
            dy = py_i - acc._pos_y[j]
            d = sqrt(dx^2 + dy^2)
            if d < min_d
                min_d = d
            end
        end
        # Bin the distance
        bin = floor(Int, min_d / bin_size) + 1
        if 1 <= bin <= n_bins
            acc.counts[bin] += 1
        end
    end
end

accumulator_result(acc::NNDistHist) = (counts=acc.counts, bin_edges=acc.bin_edges)

function accumulator_merge!(target::NNDistHist, source::NNDistHist)
    for i in eachindex(source.counts)
        if i <= length(target.counts)
            target.counts[i] += source.counts[i]
        end
    end
end

# ============================================================================
# CoAssignmentMatrix — posterior similarity matrix
# ============================================================================

"""
    CoAssignmentMatrix

Posterior similarity matrix (PSM) accumulator. For each pair of localizations
(i,j), accumulates the fraction of MCMC iterations where they are assigned
to the same cluster. The PSM is label-invariant and provides a principled
summary of partition uncertainty.

Use `consensus_partition` on the result to extract a point-estimate partition
via complete-linkage hierarchical clustering.
"""
mutable struct CoAssignmentMatrix <: AbstractAccumulator
    counts::Matrix{Int32}   # counts[i,j] for i<j: co-assignment count
    n_samples::Int32        # Total iterations accumulated
    initialized::Bool
end

CoAssignmentMatrix() = CoAssignmentMatrix(zeros(Int32, 0, 0), Int32(0), false)

function accumulator_update!(acc::CoAssignmentMatrix, state::CollapsedState,
                             locs::Vector{<:SMLMData.AbstractEmitter},
                             μ::Float64, shape::Float64, iter::Int)
    N = length(state.assignments)
    if !acc.initialized
        acc.counts = zeros(Int32, N, N)
        acc.initialized = true
    end

    assignments = state.assignments
    acc.n_samples += Int32(1)

    @inbounds for i in 1:N-1
        zi = assignments[i]
        for j in i+1:N
            if zi == assignments[j]
                acc.counts[i, j] += Int32(1)
            end
        end
    end
end

function accumulator_result(acc::CoAssignmentMatrix)
    N = size(acc.counts, 1)
    if N == 0 || acc.n_samples == 0
        return (psm=zeros(Float64, 0, 0), n_samples=0)
    end
    psm = zeros(Float64, N, N)
    ns = Float64(acc.n_samples)
    @inbounds for i in 1:N-1
        for j in i+1:N
            p = acc.counts[i, j] / ns
            psm[i, j] = p
            psm[j, i] = p
        end
    end
    for i in 1:N
        psm[i, i] = 1.0
    end
    return (psm=psm, n_samples=Int(acc.n_samples))
end

function accumulator_merge!(target::CoAssignmentMatrix, source::CoAssignmentMatrix)
    # PSMs from different partitions are over disjoint loc sets — no merge possible
end

# ============================================================================
# Consensus partition extraction from PSM
# ============================================================================

"""
    consensus_partition(psm; threshold=0.5) -> Vector{Int}

Extract a consensus partition from a posterior similarity matrix using
complete-linkage agglomerative clustering.

Iteratively merges the pair of clusters with highest minimum co-assignment
probability (complete linkage = most conservative merge). Stops when the
best merge has minimum co-assignment < `threshold`.

Set `threshold < 0.5` to bias toward merging (correct over-splitting).

Returns integer labels `1:K` for each localization.
"""
function consensus_partition(psm::Matrix{Float64}; threshold::Float64=0.5)
    N = size(psm, 1)
    N == 0 && return Int[]

    # Each loc starts in its own cluster
    labels = collect(1:N)

    # Complete-linkage: distance between clusters = min PSM over all cross-pairs
    # Merge pair with highest min-PSM, stop when best < threshold
    while true
        unique_labels = sort!(unique(labels))
        K = length(unique_labels)
        K <= 1 && break

        best_sim = -1.0
        best_a = 0
        best_b = 0

        for ai in 1:K-1
            la = unique_labels[ai]
            for bi in ai+1:K
                lb = unique_labels[bi]
                # Complete linkage: minimum PSM over all cross-pairs
                min_psm_val = 1.0
                for i in 1:N
                    labels[i] == la || continue
                    for j in 1:N
                        labels[j] == lb || continue
                        p = i < j ? psm[i, j] : psm[j, i]
                        if p < min_psm_val
                            min_psm_val = p
                        end
                    end
                end
                if min_psm_val > best_sim
                    best_sim = min_psm_val
                    best_a = la
                    best_b = lb
                end
            end
        end

        # Stop if best merge is below threshold
        best_sim < threshold && break

        # Merge: relabel best_b → best_a
        for i in 1:N
            if labels[i] == best_b
                labels[i] = best_a
            end
        end
    end

    # Compact labels to 1:K
    unique_labels = sort!(unique(labels))
    label_map = Dict(l => i for (i, l) in enumerate(unique_labels))
    return [label_map[l] for l in labels]
end

"""
    extract_emitters_consensus(psm, locs; threshold=0.5) -> Vector{Emitter2DFit}

Extract emitter positions from a PSM using consensus partition.
Builds ClusterStats for each consensus cluster and returns posterior
mean positions with posterior covariance uncertainties.
"""
function extract_emitters_consensus(psm::Matrix{Float64},
                                     locs::Vector{<:SMLMData.AbstractEmitter};
                                     threshold::Float64=0.5)
    labels = consensus_partition(psm; threshold)
    _emitters_from_labels(labels, locs)
end

"""
    _emitters_from_labels(labels, locs) -> Vector{Emitter2DFit}

Build emitters from a label vector and localizations using ClusterStats.
"""
function _emitters_from_labels(labels::Vector{Int},
                                locs::Vector{<:SMLMData.AbstractEmitter})
    K = maximum(labels; init=0)
    K == 0 && return SMLMData.Emitter2DFit[]

    emitters = SMLMData.Emitter2DFit[]
    for k in 1:K
        cs = ClusterStats()
        for i in eachindex(labels)
            if labels[i] == k
                cs = add_loc(cs, locs[i])
            end
        end
        cs.n == 0 && continue

        mx, my = posterior_mean(cs)
        Σ_xx, Σ_xy, Σ_yy = posterior_cov(cs)
        σ_x = sqrt(max(Σ_xx, 0.0))
        σ_y = sqrt(max(Σ_yy, 0.0))

        push!(emitters, SMLMData.Emitter2DFit(
            mx, my, 0.0, 0.0,
            σ_x, σ_y, Σ_xy,
            0.0, 0.0, 1, 1, 0, k
        ))
    end
    return emitters
end

# ============================================================================
# PartitionSamples — store thinned partition samples for Binder loss
# ============================================================================

"""
    PartitionSamples

Stores thinned partition samples from the MCMC chain. Used with
`binder_consensus` to find the visited partition that minimizes
expected Binder loss against the posterior similarity matrix.
"""
mutable struct PartitionSamples <: AbstractAccumulator
    samples::Vector{Vector{Int16}}   # Stored assignment vectors
    thin::Int                         # Store every thin-th sample
    _counter::Int                     # Internal counter
end

PartitionSamples(; thin::Int=10) = PartitionSamples(Vector{Int16}[], thin, 0)

function accumulator_update!(acc::PartitionSamples, state::CollapsedState,
                             locs::Vector{<:SMLMData.AbstractEmitter},
                             μ::Float64, shape::Float64, iter::Int)
    acc._counter += 1
    if acc._counter % acc.thin == 0
        push!(acc.samples, copy(state.assignments))
    end
end

accumulator_result(acc::PartitionSamples) = acc.samples

function accumulator_merge!(target::PartitionSamples, source::PartitionSamples)
    # Partition samples from different partitions can't be merged
end

# ============================================================================
# Binder loss consensus
# ============================================================================

"""
    binder_loss(assignments, psm; a=1.0, b=1.0) -> Float64

Compute Binder loss of a partition against a posterior similarity matrix.

    L(c) = Σ_{i<j} [a·𝟙(cᵢ=cⱼ)·(1-Pᵢⱼ) + b·𝟙(cᵢ≠cⱼ)·Pᵢⱼ]

- `a`: penalty for false co-assignment (merging locs that shouldn't be)
- `b`: penalty for false separation (splitting locs that should be together)
- Set `b > a` to penalize over-splitting (correct one-sided K bias)
"""
function binder_loss(assignments::Vector{Int16}, psm::Matrix{Float64};
                     a::Float64=1.0, b::Float64=1.0)
    N = length(assignments)
    loss = 0.0
    @inbounds for i in 1:N-1
        zi = assignments[i]
        for j in i+1:N
            p = psm[i, j]
            if zi == assignments[j]
                loss += a * (1.0 - p)
            else
                loss += b * p
            end
        end
    end
    return loss
end

"""
    binder_consensus(psm, samples; a=1.0, b=1.0) -> (labels, loss)

Find the visited partition that minimizes Binder loss against the PSM.

Returns compact integer labels `1:K` and the loss value.

# Arguments
- `psm`: Posterior similarity matrix from `CoAssignmentMatrix`
- `samples`: Vector of assignment vectors from `PartitionSamples`
- `a`: penalty for false co-assignment
- `b`: penalty for false separation (set > a to correct over-splitting)
"""
function binder_consensus(psm::Matrix{Float64}, samples::Vector{Vector{Int16}};
                          a::Float64=1.0, b::Float64=1.0)
    isempty(samples) && error("No partition samples to evaluate")

    best_loss = Inf
    best_idx = 0
    for (idx, s) in enumerate(samples)
        loss = binder_loss(s, psm; a, b)
        if loss < best_loss
            best_loss = loss
            best_idx = idx
        end
    end

    # Convert winning assignment to compact labels
    best = samples[best_idx]
    unique_labels = sort!(unique(best))
    label_map = Dict(l => i for (i, l) in enumerate(unique_labels))
    labels = [label_map[best[i]] for i in eachindex(best)]
    return labels, best_loss
end

"""
    extract_emitters_binder(psm, samples, locs; a=1.0, b=1.0) -> Vector{Emitter2DFit}

Extract emitter positions by finding the visited partition that minimizes
Binder loss against the PSM, then building ClusterStats for each cluster.

Set `b > a` to penalize over-splitting.
"""
function extract_emitters_binder(psm::Matrix{Float64},
                                  samples::Vector{Vector{Int16}},
                                  locs::Vector{<:SMLMData.AbstractEmitter};
                                  a::Float64=1.0, b::Float64=1.0)
    labels, _ = binder_consensus(psm, samples; a, b)
    _emitters_from_labels(labels, locs)
end

# ============================================================================
# Optimal Binder partition — greedy correlation clustering on PSM
# ============================================================================

"""
    binder_partition(psm; threshold=0.5) -> Vector{Int}

Find the partition minimizing Binder loss directly from the PSM,
without restricting to visited partitions.

Uses greedy agglomerative correlation clustering: start with singletons,
repeatedly merge the pair of clusters (A,B) with the largest positive merit

    merit(A,B) = Σ_{i∈A, j∈B} (Pᵢⱼ - threshold)

Stop when no merge has positive merit.

The threshold corresponds to the decision boundary: co-assign i,j when
the average cross-pair PSM exceeds `threshold`. Use `otsu_threshold(psm)`
to find the data-driven optimal threshold.

O(N³) worst case, O(N²) typical.
"""
function binder_partition(psm::Matrix{Float64}; threshold::Float64=0.5)
    N = size(psm, 1)
    N == 0 && return Int[]
    N == 1 && return Int[1]

    # Each loc starts as its own cluster
    cluster_members = [Int[i] for i in 1:N]
    active = trues(N)

    # Precompute pairwise merit matrix
    # merit[a,b] = Σ_{i∈a, j∈b} (P_ij - threshold)
    # For singletons: merit[a,b] = P_ab - threshold
    merit = zeros(N, N)
    @inbounds for a in 1:N-1
        for b in a+1:N
            m = psm[a, b] - threshold
            merit[a, b] = m
            merit[b, a] = m
        end
    end

    while true
        # Find best merge among active clusters
        best_merit = 0.0  # threshold: only merge if merit > 0
        best_a = 0
        best_b = 0

        @inbounds for a in 1:N
            active[a] || continue
            for b in a+1:N
                active[b] || continue
                if merit[a, b] > best_merit
                    best_merit = merit[a, b]
                    best_a = a
                    best_b = b
                end
            end
        end

        best_a == 0 && break  # no profitable merge

        # Merge b into a
        append!(cluster_members[best_a], cluster_members[best_b])
        empty!(cluster_members[best_b])
        active[best_b] = false

        # Update merits: merit[a,c] += merit[b,c] for all active c ≠ a
        @inbounds for c in 1:N
            (c == best_a || c == best_b || !active[c]) && continue
            merit[best_a, c] += merit[best_b, c]
            merit[c, best_a] = merit[best_a, c]
        end
    end

    # Build labels
    labels = zeros(Int, N)
    k = 0
    for c in 1:N
        active[c] || continue
        k += 1
        for i in cluster_members[c]
            labels[i] = k
        end
    end
    return labels
end

"""
    extract_emitters_psm(psm, samples, locs) -> Vector{Emitter2DFit}

Extract emitter positions from a PSM. Two-phase approach:
1. Find the best visited partition under symmetric Binder loss (b=1)
2. Greedily merge clusters in that partition where average cross-pair PSM > 0.5

This is principled (symmetric loss, no tuning), avoids the local-optima
problem of starting from singletons, and corrects the over-splitting in
visited partitions.
"""
function extract_emitters_psm(psm::Matrix{Float64},
                               samples::Vector{Vector{Int16}},
                               locs::Vector{<:SMLMData.AbstractEmitter})
    labels = refine_partition(psm, samples)
    _emitters_from_labels(labels, locs)
end

"""
    refine_partition(psm, samples) -> Vector{Int}

Two-phase partition extraction:
1. Find best visited partition under symmetric Binder loss
2. Greedily merge cluster pairs whose average cross-pair PSM > 0.5

No tuning parameters. The 0.5 threshold is the Bayes-optimal
decision boundary under symmetric pairwise loss.
"""
function refine_partition(psm::Matrix{Float64}, samples::Vector{Vector{Int16}})
    isempty(samples) && error("No partition samples")

    # Phase 1: Find best visited partition under symmetric Binder
    best_labels, _ = binder_consensus(psm, samples; a=1.0, b=1.0)
    N = length(best_labels)

    # Phase 2: Greedy merge of cluster pairs with average cross-PSM > 0.5
    labels = copy(best_labels)
    while true
        unique_labels = sort!(unique(labels))
        K = length(unique_labels)
        K <= 1 && break

        # Find best merge: largest positive merit
        best_merit = 0.0
        best_a = 0
        best_b = 0

        for ai in 1:K-1
            la = unique_labels[ai]
            for bi in ai+1:K
                lb = unique_labels[bi]
                # Merit = Σ (P_ij - 0.5) for all cross-pairs
                merit = 0.0
                @inbounds for i in 1:N
                    labels[i] == la || continue
                    for j in 1:N
                        labels[j] == lb || continue
                        p = i < j ? psm[i, j] : psm[j, i]
                        merit += p - 0.5
                    end
                end
                if merit > best_merit
                    best_merit = merit
                    best_a = la
                    best_b = lb
                end
            end
        end

        best_a == 0 && break  # no profitable merge

        # Merge
        for i in 1:N
            if labels[i] == best_b
                labels[i] = best_a
            end
        end
    end

    # Compact labels
    unique_labels = sort!(unique(labels))
    label_map = Dict(l => i for (i, l) in enumerate(unique_labels))
    return [label_map[l] for l in labels]
end

"""
    otsu_threshold(psm; n_bins=200) -> Float64

Find the optimal threshold for separating within-cluster from
between-cluster PSM values using Otsu's method.

The PSM value distribution is bimodal: a peak near 0 (between-cluster)
and a peak near the within-cluster co-assignment probability. Otsu's
method maximizes the inter-class variance to find the separation point.

This accounts for the sampler's one-sided split bias by adapting the
threshold to the actual PSM structure rather than assuming 0.5.
"""
function otsu_threshold(psm::Matrix{Float64}; n_bins::Int=200)
    N = size(psm, 1)
    N <= 1 && return 0.5

    # Collect off-diagonal PSM values
    vals = Float64[]
    sizehint!(vals, N * (N - 1) ÷ 2)
    @inbounds for i in 1:N-1
        for j in i+1:N
            push!(vals, psm[i, j])
        end
    end

    isempty(vals) && return 0.5

    # Build histogram
    bin_size = 1.0 / n_bins
    counts = zeros(Int, n_bins)
    @inbounds for v in vals
        bin = clamp(floor(Int, v / bin_size) + 1, 1, n_bins)
        counts[bin] += 1
    end

    total = length(vals)
    total_sum = sum(vals)

    # Otsu's method: maximize inter-class variance
    best_t = 0.5
    best_var = -1.0

    sum_bg = 0.0
    weight_bg = 0

    for i in 1:n_bins-1
        weight_bg += counts[i]
        weight_bg == 0 && continue
        weight_fg = total - weight_bg
        weight_fg == 0 && break

        bin_center = (i - 0.5) * bin_size
        sum_bg += counts[i] * bin_center

        mean_bg = sum_bg / weight_bg
        mean_fg = (total_sum - sum_bg) / weight_fg

        var_between = Float64(weight_bg) * Float64(weight_fg) * (mean_bg - mean_fg)^2
        if var_between > best_var
            best_var = var_between
            best_t = i * bin_size
        end
    end

    return best_t
end

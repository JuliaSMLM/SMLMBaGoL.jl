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
"""
mutable struct NNDistHist <: AbstractAccumulator
    counts::Vector{Int}
    bin_edges::Vector{Float64}
end

function NNDistHist(; max_dist::Float64 = 0.1, n_bins::Int = 100)
    bin_size = max_dist / n_bins
    edges = collect(0.0:bin_size:max_dist)
    NNDistHist(zeros(Int, n_bins), edges)
end

function accumulator_update!(acc::NNDistHist, state::CollapsedState,
                             locs::Vector{<:SMLMData.AbstractEmitter},
                             μ::Float64, shape::Float64, iter::Int)
    # Collect active cluster positions
    positions = Tuple{Float64, Float64}[]
    for (j, cs) in enumerate(state.clusters)
        state.active[j] || continue
        cs.n == 0 && continue
        push!(positions, posterior_mean(cs))
    end

    length(positions) < 2 && return

    n_bins = length(acc.counts)
    bin_size = acc.bin_edges[2] - acc.bin_edges[1]

    # For each emitter, find nearest neighbor distance
    for i in eachindex(positions)
        min_d = Inf
        pi = positions[i]
        for j in eachindex(positions)
            i == j && continue
            pj = positions[j]
            d = sqrt((pi[1] - pj[1])^2 + (pi[2] - pj[2])^2)
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

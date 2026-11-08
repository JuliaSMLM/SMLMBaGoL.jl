# Accumulator interface for collapsed Gibbs sampler
#
# Accumulators collect statistics from the MCMC chain without storing
# full samples. They are updated each iteration after burn-in and can
# be merged across partitions.

"""
    AbstractAccumulator

Base type for MCMC accumulators. Subtypes must implement:
- `accumulator_update!(acc, state, locs, μ, shape, iter)` — add one iteration
- `accumulator_result(acc)` — extract final result
- `accumulator_merge!(acc_target, acc_source)` — combine results from two partitions
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

# Pools two histograms bin by bin: right for two chains on the SAME localizations (more
# samples of one posterior over K), not for different partitions of a field, whose total
# K is a sum of their counts at each iteration (run_bagol records that itself).
function accumulator_merge!(target::EmitterCountHist, source::EmitterCountHist)
    while length(target.counts) < length(source.counts)
        push!(target.counts, 0)
    end
    for i in eachindex(source.counts)
        target.counts[i] += source.counts[i]
    end
end

# ============================================================================
# _CoreCountBuffer — field total K per iteration (run_bagol, internal)
# ============================================================================

# Internal (run_bagol): this partition's core-weighted emitter count (`_core_weighted_k`,
# defined with the posterior image below) at each iteration of the current sync block,
# stored in slot mod1(iter, block). Between syncs all partitions run the same iterations,
# so after each block run_bagol sums the slots across partitions into the field's total K
# at each recorded iteration (`_record_field_k!`), which gives `diagnostics.posterior_k`.
mutable struct _CoreCountBuffer <: AbstractAccumulator
    is_overlap::Vector{Bool}
    buf::Vector{Float64}
    _core::Vector{Int}
end

_CoreCountBuffer(is_overlap::AbstractVector{Bool}, block::Int) =
    _CoreCountBuffer(Vector{Bool}(is_overlap), zeros(max(block, 1)), Int[])

function accumulator_update!(acc::_CoreCountBuffer, state::CollapsedState,
                             locs::Vector{<:SMLMData.AbstractEmitter},
                             μ::Float64, shape::Float64, iter::Int)
    _check_overlap_length(acc.is_overlap, state)
    acc.buf[mod1(iter, length(acc.buf))] = _core_weighted_k(state, acc.is_overlap, acc._core)
end

accumulator_result(acc::_CoreCountBuffer) = acc.buf

"""
    _record_field_k!(counts, bufs, it_lo, it_hi, burn_in) -> counts

Internal: for each recorded iteration in `it_lo:it_hi` (after burn-in), sum the
partitions' core-weighted counts into the field total, round it to the nearest
integer K and add one to `counts[K+1]`.
"""
function _record_field_k!(counts::Vector{Int}, bufs::Vector{_CoreCountBuffer},
                          it_lo::Int, it_hi::Int, burn_in::Int)
    for it in max(it_lo, burn_in + 1):it_hi
        tot = 0.0
        for b in bufs
            tot += b.buf[mod1(it, length(b.buf))]
        end
        k = round(Int, tot)
        length(counts) < k + 1 && append!(counts, zeros(Int, k + 1 - length(counts)))
        counts[k + 1] += 1
    end
    return counts
end

"""Internal: the field total K now (core-weighted, rounded), for the convergence trace."""
_field_k(states, partitions, bufs::Vector{_CoreCountBuffer}) =
    round(Int, sum(_core_weighted_k(states[i], partitions[i].is_overlap, bufs[i]._core)
                   for i in eachindex(states); init = 0.0))

# ============================================================================
# PosteriorImage — Rao-Blackwellized posterior image
# ============================================================================

"""
    PosteriorImage(; pixel_size, xlim=nothing, ylim=nothing, n_sigma=4.0,
                     is_overlap=nothing, origin=nothing)

Rao-Blackwellized posterior image accumulator. Each iteration adds, per active
cluster, the posterior position density N(μ_post_j, Σ_post_j) integrated over
each pixel, so every cluster deposits exactly its weight (1 by default) per
iteration whatever its width relative to the pixel. The deposit covers
±`n_sigma` posterior standard deviations around the mean (always including the
pixel that holds the mean) and is renormalized to the weight inside that box.
The image is emitter mass per pixel summed over recorded iterations: divide by
the number of recorded iterations for the expected emitter count per pixel.

`is_overlap` (one flag per localization of the chain) marks localizations that
are overlap copies of a neighboring partition. When given, each cluster's
deposit is weighted by the fraction of its localizations that are core here,
so an emitter whose localizations two partitions share renders about once in the
merged image, not twice (1.04-1.07 on average in one simulated test, as overlap bands are partial).
`run_bagol` sets it; without it every cluster has weight 1.

Without `xlim`/`ylim` the grid is sized on the first update from the chain's
localizations; `origin = (x, y)` then snaps its pixel edges to the lattice
`origin + k·pixel_size`, so images of different partitions merge into a
full-field image on that lattice without resampling.

Pixel integration is exact for an axis-aligned posterior (σ_xy = 0) and, for a
correlation |ρ| ≤ 0.1 (fitted x-y correlations in real data are small; SMITE reports
no x-y covariance), accurate to 1e-10 per pixel
(the bivariate normal's tetrachoric series, each term separable in x and y). For
|ρ| > 0.1 each pixel's probability comes from the bivariate normal distribution
function at its corners (Genz's algorithm, about 1e-15), over the band of pixels
within ±`n_sigma` conditional standard deviations of the regression line inside
the box; this costs a few times more per deposit than the series.
"""
mutable struct PosteriorImage <: AbstractAccumulator
    image::Matrix{Float64}
    edges_x::Vector{Float64}
    edges_y::Vector{Float64}
    pixel_size::Float64
    n_sigma::Float64  # Render radius in sigma units
    initialized::Bool
    is_overlap::Union{Nothing, Vector{Bool}}   # halo flags per loc (run_bagol); nothing = weight 1
    origin::Union{Nothing, NTuple{2, Float64}} # lattice anchor for auto-sized grids
    _core::Vector{Int}                         # scratch: core-loc count per cluster slot
    _col::Vector{NTuple{2, Int}}               # scratch: per column (iy_lo, iy_hi) (correlated path)
    _edge::Vector{NTuple{3, Int}}              # scratch: per column edge (offset in _px, first, last row edge)
    _px::Vector{Float64}                       # scratch: column factors (series) or corner probabilities (correlated)
    _py::Vector{Float64}                       # scratch: row factors (series path)
    _cv::Vector{Float64}                       # scratch: one column's pixel values before they are added
end

function PosteriorImage(; pixel_size::Float64,
                         xlim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
                         ylim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
                         n_sigma::Float64 = 4.0,
                         is_overlap::Union{Nothing, AbstractVector{Bool}} = nothing,
                         origin::Union{Nothing, Tuple{<:Real, <:Real}} = nothing)
    n_sigma >= 0 || throw(ArgumentError("n_sigma must be non-negative (got $n_sigma)"))
    ov = is_overlap === nothing ? nothing : Vector{Bool}(is_overlap)
    org = origin === nothing ? nothing : (Float64(origin[1]), Float64(origin[2]))
    if xlim !== nothing && ylim !== nothing
        edges_x = collect(xlim[1]:pixel_size:(xlim[2] + pixel_size))
        edges_y = collect(ylim[1]:pixel_size:(ylim[2] + pixel_size))
        nx = length(edges_x) - 1
        ny = length(edges_y) - 1
        return PosteriorImage(zeros(nx, ny), edges_x, edges_y, pixel_size, n_sigma, true,
                              ov, org, Int[], NTuple{2, Int}[], NTuple{3, Int}[], Float64[], Float64[],
                              Float64[])
    end
    # Will be initialized on first update from data bounds
    return PosteriorImage(zeros(1, 1), Float64[0.0, 1.0], Float64[0.0, 1.0],
                          pixel_size, n_sigma, false, ov, org, Int[], NTuple{2, Int}[], NTuple{3, Int}[],
                          Float64[], Float64[], Float64[])
end

# Tile side (pixels, a power of 2) of the partition images run_bagol keeps
const _TSHIFT = 4
const _TILE = 1 << _TSHIFT

"""
    _TiledPosteriorImage(; pixel_size, is_overlap, origin, n_sigma=4.0)

Internal: one partition's posterior image in `run_bagol`. Same grid and deposits as an
auto-sized `PosteriorImage` (the partition's localizations, pixel edges on the full-field
lattice at `origin`), but stored as `_TILE`×`_TILE` tiles allocated where a deposit lands.
A dense image over each partition's bounding box can hold many times the area its deposits
cover, since a few partitions sprawl across the field. Merged into a dense `PosteriorImage`, it adds the same pixel values in
the same order as a dense partition image, so the merged image is unchanged.
"""
mutable struct _TiledPosteriorImage <: AbstractAccumulator
    edges_x::Vector{Float64}
    edges_y::Vector{Float64}
    nx::Int
    ny::Int
    pixel_size::Float64
    n_sigma::Float64
    initialized::Bool
    is_overlap::Union{Nothing, Vector{Bool}}
    origin::Union{Nothing, NTuple{2, Float64}}
    dir::Matrix{Int32}                         # tile number per (tile column, tile row); 0 = none
    tiles::Vector{Vector{Float64}}             # _TILE² pixels each, row index fastest
    _core::Vector{Int}
    _col::Vector{NTuple{2, Int}}
    _edge::Vector{NTuple{3, Int}}
    _px::Vector{Float64}
    _py::Vector{Float64}
    _cv::Vector{Float64}
end

function _TiledPosteriorImage(; pixel_size::Float64, n_sigma::Float64 = 4.0,
                              is_overlap::Union{Nothing, AbstractVector{Bool}} = nothing,
                              origin::Union{Nothing, Tuple{<:Real, <:Real}} = nothing)
    n_sigma >= 0 || throw(ArgumentError("n_sigma must be non-negative (got $n_sigma)"))
    ov = is_overlap === nothing ? nothing : Vector{Bool}(is_overlap)
    org = origin === nothing ? nothing : (Float64(origin[1]), Float64(origin[2]))
    return _TiledPosteriorImage(Float64[0.0, 1.0], Float64[0.0, 1.0], 0, 0, pixel_size, n_sigma,
                                false, ov, org, zeros(Int32, 0, 0), Vector{Float64}[], Int[],
                                NTuple{2, Int}[], NTuple{3, Int}[], Float64[], Float64[], Float64[])
end

const _ImageAcc = Union{PosteriorImage, _TiledPosteriorImage}

# Internal: grid geometry (nx, ny, x0, y0) and where a deposit's pixel values go.
_grid(acc::PosteriorImage) = (size(acc.image, 1), size(acc.image, 2), acc.edges_x[1], acc.edges_y[1])
_grid(acc::_TiledPosteriorImage) = (acc.nx, acc.ny, acc.edges_x[1], acc.edges_y[1])
_sink(acc::PosteriorImage) = acc.image
_sink(acc::_TiledPosteriorImage) = acc

# Internal: add cv[1:hi-lo+1] to pixels (ix, lo:hi).
@inline function _add_column!(img::Matrix{Float64}, ix::Int, lo::Int, hi::Int, cv::Vector{Float64})
    @inbounds for iy in lo:hi
        img[ix, iy] += cv[iy - lo + 1]
    end
    return nothing
end
function _add_column!(acc::_TiledPosteriorImage, ix::Int, lo::Int, hi::Int, cv::Vector{Float64})
    tx = ((ix - 1) >> _TSHIFT) + 1
    base0 = ((ix - 1) & (_TILE - 1)) * _TILE    # this column's offset within a tile
    iy = lo
    @inbounds while iy <= hi
        ty = ((iy - 1) >> _TSHIFT) + 1
        stop = min(hi, ty << _TSHIFT)              # last row of tile row ty
        k = acc.dir[tx, ty]
        if k == 0
            push!(acc.tiles, zeros(_TILE * _TILE))
            k = Int32(length(acc.tiles))
            acc.dir[tx, ty] = k
        end
        t = acc.tiles[k]
        base = base0 - ((ty - 1) << _TSHIFT)      # t[base + j] is pixel (ix, j)
        for j in iy:stop
            t[base + j] += cv[j - lo + 1]
        end
        iy = stop + 1
    end
    return nothing
end

# Internal: pixels the tiled image holds in memory.
_stored_pixels(acc::_TiledPosteriorImage) = length(acc.tiles) * _TILE^2

# Internal: pixel edges sized from the chain's localizations (snapped to `origin`'s lattice).
function _auto_edges(acc::_ImageAcc, locs::Vector{<:SMLMData.AbstractEmitter})
    xs = [loc.x for loc in locs]
    ys = [loc.y for loc in locs]
    ps = acc.pixel_size
    # A cluster of n ≥ 2 localizations has posterior σ ≤ σ_max/√2, so its ±4σ deposit box
    # (n_sigma = 4) fits inside this pad (with σ_xy ≠ 0 nearly so); only a single-localization
    # cluster at a corner of the partition can reach past it, losing at most ~0.3% of its
    # mass (dropped, not moved).
    pad = 3 * maximum(max(loc.σ_x, loc.σ_y) for loc in locs) + ps
    xmin, xmax = minimum(xs) - pad, maximum(xs) + pad
    ymin, ymax = minimum(ys) - pad, maximum(ys) + pad
    if acc.origin !== nothing
        ox, oy = acc.origin
        xmin = ox + floor((xmin - ox) / ps) * ps
        ymin = oy + floor((ymin - oy) / ps) * ps
    end
    return collect(xmin:ps:(xmax + ps)), collect(ymin:ps:(ymax + ps))
end

function _ensure_initialized!(acc::PosteriorImage, locs::Vector{<:SMLMData.AbstractEmitter})
    acc.initialized && return
    acc.edges_x, acc.edges_y = _auto_edges(acc, locs)
    acc.image = zeros(length(acc.edges_x) - 1, length(acc.edges_y) - 1)
    acc.initialized = true
end

function _ensure_initialized!(acc::_TiledPosteriorImage, locs::Vector{<:SMLMData.AbstractEmitter})
    acc.initialized && return
    acc.edges_x, acc.edges_y = _auto_edges(acc, locs)
    acc.nx = length(acc.edges_x) - 1
    acc.ny = length(acc.edges_y) - 1
    acc.dir = zeros(Int32, cld(acc.nx, _TILE), cld(acc.ny, _TILE))
    acc.initialized = true
end

# Internal: overlap flags must be one per localization of the chain (they are read
# under @inbounds below).
function _check_overlap_length(is_overlap::AbstractVector{Bool}, state::CollapsedState)
    length(is_overlap) == length(state.assignments) || throw(ArgumentError(
        "is_overlap has $(length(is_overlap)) flags but the chain has " *
        "$(length(state.assignments)) localizations; pass one flag per localization"))
    return nothing
end

"""
    _core_counts!(buf, state, is_overlap) -> buf

Internal: `buf[j]` = number of localizations in cluster slot `j` that are core in
this partition (not overlap copies of a neighbor).
"""
function _core_counts!(buf::Vector{Int}, state::CollapsedState, is_overlap::AbstractVector{Bool})
    _check_overlap_length(is_overlap, state)
    n = length(state.clusters)
    length(buf) == n || resize!(buf, n)
    fill!(buf, 0)
    @inbounds for i in eachindex(state.assignments)
        is_overlap[i] && continue
        c = state.assignments[i]
        c > 0 && (buf[c] += 1)
    end
    return buf
end

"""
    _core_weighted_k(state, is_overlap, buf) -> Float64

Internal: the partition's emitter count with each cluster weighted by its core fraction,
Σ_j n_core_j / n_j over active clusters (each cluster counts as the fraction of
its localizations that are core here). With no overlap localizations this is
exactly `state.n_active`. Summed over a run's partitions, every localization is
core in exactly one partition, so the sum is Σ over localizations of
1/(size of the cluster its own partition puts it in): an emitter whose
localizations sit in two partitions' copies counts about once, not twice.
"""
function _core_weighted_k(state::CollapsedState, is_overlap::AbstractVector{Bool}, buf::Vector{Int})
    _check_overlap_length(is_overlap, state)
    any(is_overlap) || return Float64(state.n_active)
    _core_counts!(buf, state, is_overlap)
    k = 0.0
    @inbounds for (j, cs) in enumerate(state.clusters)
        (state.active[j] && cs.n > 0) || continue
        k += buf[j] / cs.n
    end
    return k
end

# Standard normal CDF and density. Pixel masses are differences of _Φ at pixel
# edges (absolute error ~1e-16, ample for masses summed into an image).
@inline _Φ(z::Float64) = 0.5 * erfc(-z / sqrt(2.0))
@inline _φ(z::Float64) = exp(-0.5 * z * z) / sqrt(2π)

# Internal: number of tetrachoric-series terms beyond the first for correlation r = |ρ| ≤ 0.1:
# the smallest K with r^(K+1)/(K+1) ≤ 1e-10. (By Cramér's bound on Hermite functions the
# (K+1)-th term of a pixel's mass is at most ~0.75·r^(K+1)/(K+1).) r = 0 gives K = 0; r = 0.1
# gives K = 9. Terminates for any r < 1; the caller uses it only for r ≤ 0.1.
function _mehler_terms(r::Float64)
    K = 0
    p = r
    while p / (K + 1) > 1e-10
        K += 1
        p *= r
    end
    return K
end

# Internal: tetrachoric-series factors of the pixels lo:hi on the lattice e_i = o + (i−1)·ps
# for a normal of mean m and sd σ: buf[(i−1)(K+1) + k + 1] = Φ(b) − Φ(a) for k = 0 and
# He_{k−1}(a)φ(a) − He_{k−1}(b)φ(b) for k ≥ 1, with a, b the pixel's standardized edges.
function _mehler_factors!(buf::Vector{Float64}, o::Float64, ps::Float64, lo::Int, hi::Int,
                          m::Float64, σ::Float64, K::Int)
    n = hi - lo + 1
    K1 = K + 1
    resize!(buf, n * K1)
    za = (o + (lo - 1) * ps - m) / σ
    Fa = _Φ(za); φa = _φ(za)
    @inbounds for i in 1:n
        zb = (o + (lo - 1 + i) * ps - m) / σ
        Fb = _Φ(zb); φb = _φ(zb)
        off = (i - 1) * K1
        buf[off + 1] = Fb - Fa
        if K > 0
            # He_{k-1} at both edges by the recurrence He_{n+1}(z) = z He_n(z) − n He_{n−1}(z)
            ha0 = 1.0; ha1 = za; hb0 = 1.0; hb1 = zb
            for k in 1:K
                ta = φa == 0 ? 0.0 : ha0 * φa
                tb = φb == 0 ? 0.0 : hb0 * φb
                buf[off + k + 1] = ta - tb
                ha0, ha1 = ha1, za * ha1 - k * ha0
                hb0, hb1 = hb1, zb * hb1 - k * hb0
            end
        end
        za = zb; Fa = Fb; φa = φb
    end
    return buf
end

# Gauss–Legendre rules on [−1, 1] (positive nodes and their weights; the rules are symmetric)
# with 6, 12 and 20 points, for _bvnu.
const _GL6 = ((0.9324695142031522, 0.6612093864662647, 0.2386191860831970),
              (0.1713244923791705, 0.3607615730481384, 0.4679139345726904))
const _GL12 = ((0.9815606342467191, 0.9041172563704750, 0.7699026741943050,
                0.5873179542866171, 0.3678314989981802, 0.1252334085114692),
               (0.04717533638651177, 0.1069393259953183, 0.1600783285433464,
                0.2031674267230659, 0.2334925365383547, 0.2491470458134029))
const _GL20 = ((0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
                0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
                0.5108670019508271, 0.3737060887154196, 0.2277858511416451,
                0.07652652113349733),
               (0.01761400713915212, 0.04060142980038694, 0.06267204833410906,
                0.08327674157670475, 0.1019301198172404, 0.1181945319615184,
                0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
                0.1527533871307259))

"""
    _bvnu(h, k, r) -> Float64

Internal: P(X > h, Y > k) for standard normals X, Y with correlation r, to about 1e-15
absolute. Genz's algorithm (A. Genz, "Numerical computation of rectangular bivariate and
trivariate normal and t probabilities", Statistics and Computing 14:251-260, 2004; his
`bvnu`), after Drezner and Wesolowsky (1990): for |r| < 0.925 Gauss–Legendre quadrature
(6, 12 or 20 points by |r|) of P = Φ(−h)Φ(−k) + (1/2π) ∫₀^asin r exp(−(h² + k² − 2hk sin θ) /
(2cos² θ)) dθ; for |r| ≥ 0.925 the same integral rewritten in √(1 − r²) with its
singular part integrated analytically.
"""
function _bvnu(h::Float64, k::Float64, r::Float64)
    r == 0 && return _Φ(-h) * _Φ(-k)
    ar = abs(r)
    ar < 0.3 && return _bvnu_small(h, k, r, _GL6)
    ar < 0.75 && return _bvnu_small(h, k, r, _GL12)
    ar < 0.925 && return _bvnu_small(h, k, r, _GL20)
    X, W = _GL20
    kk = r < 0 ? -k : k
    hk = h * kk
    bvn = 0.0
    if ar < 1
        as = (1 - ar) * (1 + ar)
        a = sqrt(as)
        bs = (h - kk)^2
        c = (4 - hk) / 8
        d = (12 - hk) / 80
        asr = -(bs / as + hk) / 2
        if asr > -100
            bvn = a * exp(asr) * (1 - c * (bs - as) * (1 - d * bs) / 3 + c * d * as^2)
        end
        if hk > -100
            b = sqrt(bs)
            bvn -= exp(-hk / 2) * sqrt(2π) * _Φ(-b / a) * b * (1 - c * bs * (1 - d * bs) / 3)
        end
        a /= 2
        q = 0.0
        @inbounds for i in eachindex(X), t in (1 - X[i], 1 + X[i])
            xs = (a * t)^2
            asr = -(bs / xs + hk) / 2
            asr > -100 || continue
            sp = 1 + c * xs * (1 + 5 * d * xs)
            rs = sqrt(1 - xs)
            ep = exp(-(hk / 2) * xs / (1 + rs)^2) / rs
            q += W[i] * exp(asr) * (sp - ep)
        end
        bvn = (a * q - bvn) / (2π)
    end
    if r > 0
        bvn += _Φ(-max(h, kk))
    elseif h >= kk
        bvn = -bvn
    else
        L = h < 0 ? _Φ(kk) - _Φ(h) : _Φ(-h) - _Φ(-kk)
        bvn = L - bvn
    end
    return clamp(bvn, 0.0, 1.0)
end

function _bvnu_small(h::Float64, k::Float64, r::Float64, (X, W))
    hk = h * k
    hs = (h * h + k * k) / 2
    asr = asin(r) / 2
    q = 0.0
    @inbounds for i in eachindex(X), t in (1 - X[i], 1 + X[i])
        sn = sin(asr * t)
        q += W[i] * exp((sn * hk - hs) / (1 - sn * sn))
    end
    return clamp(q * asr / (2π) + _Φ(-h) * _Φ(-k), 0.0, 1.0)
end

"""
    _deposit_gaussian!(acc, w, mx, my, Σxx, Σxy, Σyy)

Internal: add weight `w` of N((mx, my), Σ) to the image (`acc.image`, or the tiles of a
`_TiledPosteriorImage`), integrated over each pixel. Columns span mx ± n_sigma·σx. |ρ| ≤ 0.1: the tetrachoric series below, rows
spanning my ± n_sigma·σy. |ρ| > 0.1: each pixel's probability from the bivariate
normal upper tail at its four corners (`_bvnu`); a column's rows span its range of
conditional means, my + (Σxy/Σxx)(x − mx) for x in the column, ± n_sigma conditional
sd √(det Σ / Σxx), within my ± n_sigma·σy. The deposit is renormalized to `w` over
the pixels computed, and only the in-image part is added (mass outside the image is
dropped, not moved).
"""
function _deposit_gaussian!(acc::_ImageAcc, w::Float64, mx::Float64, my::Float64,
                            Sxx::Float64, Sxy::Float64, Syy::Float64)
    det_Σ = Sxx * Syy - Sxy^2
    (Sxx > 0 && det_Σ > 0 && isfinite(det_Σ) && isfinite(mx) && isfinite(my)) || return
    sink = _sink(acc)
    cv = acc._cv
    nx, ny, x0, y0 = _grid(acc)
    ps = acc.pixel_size; ns = acc.n_sigma
    σx = sqrt(Sxx)
    s2 = det_Σ / Sxx

    jx_lo = floor(Int, (mx - ns * σx - x0) / ps) + 1
    jx_hi = floor(Int, (mx + ns * σx - x0) / ps) + 1
    # Axis-aligned or weakly correlated posterior (|ρ| ≤ 0.1: fitted x-y correlations in real
    # data are small, SMITE reports σ_xy = 0): the bivariate normal's tetrachoric (Mehler) series
    # φ₂(u, v; ρ) = φ(u)φ(v) Σ_k ρ^k/k! He_k(u) He_k(v) integrates over a pixel [a,b]×[c,d]
    # (standardized) to Σ_k ρ^k/k! C_k R_k, each term a column factor times a row factor:
    # C_0 = Φ(b) − Φ(a) and C_k = He_{k−1}(a)φ(a) − He_{k−1}(b)φ(b) (He = probabilists'
    # Hermite). Kept terms reach a truncation error below 1e-10 per pixel, so the cost is
    # (columns + rows)·(K+1) instead of columns·rows CDF evaluations; σ_xy = 0 is K = 0.
    σy = sqrt(Syy)
    ρ = Sxy / (σx * σy)
    if abs(ρ) <= 0.1
        K = _mehler_terms(abs(ρ))
        K1 = K + 1
        iy_lo = floor(Int, (my - ns * σy - y0) / ps) + 1
        iy_hi = floor(Int, (my + ns * σy - y0) / ps) + 1
        cx = _mehler_factors!(acc._px, x0, ps, jx_lo, jx_hi, mx, σx, K)
        cy = _mehler_factors!(acc._py, y0, ps, iy_lo, iy_hi, my, σy, K)
        ncol = jx_hi - jx_lo + 1; nrow = iy_hi - iy_lo + 1
        # fold ρ^k/k! into the column factors
        @inbounds for i in 1:ncol
            c = 1.0
            for k in 1:K
                c *= ρ / k
                cx[(i - 1) * K1 + k + 1] *= c
            end
        end
        Z = 0.0
        @inbounds for k in 1:K1
            sx = 0.0; sy = 0.0
            for i in 1:ncol; sx += cx[(i - 1) * K1 + k]; end
            for j in 1:nrow; sy += cy[(j - 1) * K1 + k]; end
            Z += sx * sy
        end
        Z > 0 || return
        scale = w / Z
        r_lo = max(iy_lo, 1); r_hi = min(iy_hi, ny)
        r_lo <= r_hi || return
        length(cv) < r_hi - r_lo + 1 && resize!(cv, r_hi - r_lo + 1)
        @inbounds for i in 1:ncol
            ix = jx_lo + i - 1
            1 <= ix <= nx || continue
            oi = (i - 1) * K1
            for iy in r_lo:r_hi
                oj = (iy - iy_lo) * K1
                v = 0.0
                for k in 1:K1
                    v += cx[oi + k] * cy[oj + k]
                end
                cv[iy - r_lo + 1] = scale * v
            end
            _add_column!(sink, ix, r_lo, r_hi, cv)
        end
        return
    end
    # |ρ| > 0.1: in standardized u = (x − mx)/σx, v = (y − my)/σy, pixel [ua, ub] × [vc, vd]
    # holds U(ua, vc) − U(ub, vc) − U(ua, vd) + U(ub, vd), U = _bvnu(·, ·, ρ); each column
    # edge's corner values serve the columns on both sides of it.
    ncol = jx_hi - jx_lo + 1
    rows = acc._col
    resize!(rows, ncol)
    sc = sqrt(s2) / σy                       # conditional sd of v given u, √(1 − ρ²)
    yb_lo = floor(Int, (my - ns * σy - y0) / ps) + 1
    yb_hi = floor(Int, (my + ns * σy - y0) / ps) + 1
    @inbounds for c in 1:ncol
        ua = (x0 + (jx_lo + c - 2) * ps - mx) / σx
        ub = (x0 + (jx_lo + c - 1) * ps - mx) / σx
        vlo = ρ * (ρ > 0 ? ua : ub) - ns * sc
        vhi = ρ * (ρ > 0 ? ub : ua) + ns * sc
        rows[c] = (max(floor(Int, (my + σy * vlo - y0) / ps) + 1, yb_lo),
                   min(floor(Int, (my + σy * vhi - y0) / ps) + 1, yb_hi))
    end
    edges = acc._edge
    resize!(edges, ncol + 1)
    U = acc._px
    empty!(U)
    @inbounds for e in 0:ncol
        tl = typemax(Int); th = typemin(Int)
        for c in (e, e + 1)
            1 <= c <= ncol || continue
            lo, hi = rows[c]
            lo <= hi || continue
            tl = min(tl, lo - 1); th = max(th, hi)
        end
        edges[e + 1] = (length(U), tl, th)   # U[o + t − tl + 1] is row edge t of this column edge
        h = (x0 + (jx_lo - 1 + e) * ps - mx) / σx
        for t in tl:th
            push!(U, _bvnu(h, (y0 + t * ps - my) / σy, ρ))
        end
    end
    Z = 0.0
    @inbounds for c in 1:ncol
        lo, hi = rows[c]
        oL, tL, _ = edges[c]; oR, tR, _ = edges[c + 1]
        for iy in lo:hi
            Z += max((U[oL + iy - tL] - U[oR + iy - tR]) - (U[oL + iy - tL + 1] - U[oR + iy - tR + 1]), 0.0)
        end
    end
    Z > 0 || return
    scale = w / Z
    @inbounds for c in 1:ncol
        ix = jx_lo + c - 1
        1 <= ix <= nx || continue
        lo, hi = rows[c]
        a = max(lo, 1); b = min(hi, ny)
        a <= b || continue
        length(cv) < b - a + 1 && resize!(cv, b - a + 1)
        oL, tL, _ = edges[c]; oR, tR, _ = edges[c + 1]
        for iy in a:b
            m = (U[oL + iy - tL] - U[oR + iy - tR]) - (U[oL + iy - tL + 1] - U[oR + iy - tR + 1])
            cv[iy - a + 1] = scale * max(m, 0.0)
        end
        _add_column!(sink, ix, a, b, cv)
    end
    return
end

# Internal: the x-y block (Σxx, Σxy, Σyy) of a cluster's posterior covariance, for the image
# (a 3D cluster's posterior_cov is ordered Σxx, Σyy, Σzz, Σxy, Σxz, Σyz).
_xy_cov(cs) = posterior_cov(cs)
function _xy_cov(cs::ClusterStats{3})
    c = posterior_cov(cs)
    return c[1], c[4], c[2]
end
_xy_cov(cs::MotionClusterStats) = _xy_cov(cs.base)

function accumulator_update!(acc::_ImageAcc, state::CollapsedState,
                             locs::Vector{<:SMLMData.AbstractEmitter},
                             μ::Float64, shape::Float64, iter::Int)
    _ensure_initialized!(acc, locs)
    ov = acc.is_overlap
    weighted = ov !== nothing && any(ov)
    weighted && _core_counts!(acc._core, state, ov)

    for (j, cs) in enumerate(state.clusters)
        (state.active[j] && cs.n > 0) || continue
        w = weighted ? acc._core[j] / cs.n : 1.0
        w > 0 || continue
        mx, my = posterior_mean(cs)
        Σ_xx, Σ_xy, Σ_yy = _xy_cov(cs)
        _deposit_gaussian!(acc, w, Float64(mx), Float64(my), Float64(Σ_xx), Float64(Σ_xy), Float64(Σ_yy))
    end
end

function accumulator_result(acc::PosteriorImage)
    return (image=acc.image, edges_x=acc.edges_x, edges_y=acc.edges_y,
            pixel_size=acc.pixel_size)
end

# The tiled image as a dense one on its own grid (tests; run_bagol merges tiles directly).
function accumulator_result(acc::_TiledPosteriorImage)
    img = zeros(acc.nx, acc.ny)
    _each_tile_pixel(acc) do ix, iy, v
        img[ix, iy] += v
    end
    return (image=img, edges_x=acc.edges_x, edges_y=acc.edges_y, pixel_size=acc.pixel_size)
end

# Internal: f(ix, iy, value) for every pixel of every allocated tile inside the grid.
function _each_tile_pixel(f, acc::_TiledPosteriorImage)
    @inbounds for ty in axes(acc.dir, 2), tx in axes(acc.dir, 1)
        k = acc.dir[tx, ty]
        k == 0 && continue
        t = acc.tiles[k]
        for c in 0:_TILE - 1
            ix = ((tx - 1) << _TSHIFT) + c + 1
            ix <= acc.nx || break
            for r in 0:_TILE - 1
                iy = ((ty - 1) << _TSHIFT) + r + 1
                iy <= acc.ny || break
                f(ix, iy, t[c * _TILE + r + 1])
            end
        end
    end
    return nothing
end

# Merge a partition's tiles into the full-field image. Both grids lie on one lattice, so
# source pixel (ix, iy) is target pixel (ix + dx, iy + dy), located by its center as in the
# dense merge; pixels outside the target are dropped. Each target pixel gets one addition per
# source, so merging sources in the same order gives the same sums as dense partition images.
function accumulator_merge!(target::PosteriorImage, source::_TiledPosteriorImage)
    source.initialized || return
    target.initialized ||
        throw(ArgumentError("a tiled partition image merges into a full-field image with a grid"))
    ps = target.pixel_size
    dx = floor(Int, (source.edges_x[1] + 0.5 * ps - target.edges_x[1]) / ps)
    dy = floor(Int, (source.edges_y[1] + 0.5 * ps - target.edges_y[1]) / ps)
    img = target.image
    tnx, tny = size(img)
    _each_tile_pixel(source) do ix, iy, v
        tix = ix + dx; tiy = iy + dy
        if 1 <= tix <= tnx && 1 <= tiy <= tny
            @inbounds img[tix, tiy] += v
        end
    end
    return nothing
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
# PartitionSamples — store thinned assignment samples for MAP-N
# ============================================================================

"""
    PartitionSamples

Stores thinned assignment vectors from the MCMC chain. Used for
MAP-N estimation: find the posterior mode of K, filter to matching
samples, and Hungarian-match to get consistent emitter positions.
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
# PSMAccumulator — Posterior Similarity Matrix for partition estimation
# ============================================================================

"""
    PSMAccumulator

Accumulates the Posterior Similarity Matrix (PSM) during the MCMC chain.
C[i,j] = fraction of post-burn-in iterations where locs i and j are
co-assigned to the same cluster.

Uses cluster-based iteration for efficiency: O(Σ n_k²) per iteration
rather than O(N²), since only locs in the same cluster contribute.

Result: (psm=Matrix{Float64}, n_samples=Int)
"""
mutable struct PSMAccumulator <: AbstractAccumulator
    counts::Matrix{UInt32}   # Upper-triangular co-assignment counts
    n_samples::Int           # Number of iterations accumulated
    n_locs::Int              # Set on first update
    _members::Vector{Int}    # Workspace: cluster member indices
end

PSMAccumulator() = PSMAccumulator(zeros(UInt32, 0, 0), 0, 0, Int[])

function accumulator_update!(acc::PSMAccumulator, state::CollapsedState,
                             locs::Vector{<:SMLMData.AbstractEmitter},
                             μ::Float64, shape::Float64, iter::Int)
    n = length(locs)

    # Initialize on first call
    if acc.n_locs == 0
        acc.n_locs = n
        acc.counts = zeros(UInt32, n, n)
        sizehint!(acc._members, n)
    end

    # Iterate active clusters, collect members, increment pairs
    @inbounds for (j, cs) in enumerate(state.clusters)
        state.active[j] || continue
        cs.n == 0 && continue

        # Collect member indices for this cluster
        empty!(acc._members)
        for i in 1:n
            if state.assignments[i] == j
                push!(acc._members, i)
            end
        end

        # Increment co-assignment counts for all pairs in this cluster
        nm = length(acc._members)
        for a in 1:nm
            ia = acc._members[a]
            acc.counts[ia, ia] += UInt32(1)  # diagonal
            for b in (a+1):nm
                ib = acc._members[b]
                acc.counts[ia, ib] += UInt32(1)
            end
        end
    end

    acc.n_samples += 1
end

function accumulator_result(acc::PSMAccumulator)
    n = acc.n_locs
    if n == 0 || acc.n_samples == 0
        return (psm=zeros(0, 0), n_samples=0)
    end
    # Build symmetric normalized PSM from upper triangle
    psm = zeros(Float64, n, n)
    inv_t = 1.0 / acc.n_samples
    @inbounds for i in 1:n
        psm[i, i] = 1.0
        for j in (i+1):n
            v = Float64(acc.counts[i, j]) * inv_t
            psm[i, j] = v
            psm[j, i] = v
        end
    end
    return (psm=psm, n_samples=acc.n_samples)
end

function accumulator_merge!(target::PSMAccumulator, source::PSMAccumulator)
    # PSM from different partitions can't be merged (different loc sets)
end

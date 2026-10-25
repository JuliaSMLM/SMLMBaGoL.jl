# Inhomogeneous Poisson process (IPP) spatial prior: a fixed pilot intensity shape g(θ)
#
# Emitters form a marked inhomogeneous Poisson process with intensity λ(θ) = ρ·g(θ).
# g is a FIXED dimensionless shape, built once from all localizations before
# partitioning, with ∫_box g = A_field (docs/math_reference.md §4.6). It is one
# read-only global object shared by every partition, so partition targets factorize.

"""
    IntensityMap

Read-only lookup of the pilot intensity shape `g(θ)` as `log g` on a regular grid (spacing
`bandwidth/4`, covering the normalization box plus 4·bandwidth), evaluated by bilinear
interpolation of `log g` (one lookup per `log g(θ̂)`, no `log` in the sampler's hot path).

`g(θ) = A_field · κ_h(θ) / N_all`, with `κ_h(θ) = Σ_j N(θ; d_j, h²I) / m_h(θ)` and `m_h(θ)`
the mass of the kernel at θ inside the box (edge correction). The final lookup function
`exp(bilinear(log g))` (floor and interpolation included) is normalized numerically
(tensor Gauss-Legendre quadrature per grid cell, clipped to the box) so that
`∫_box g = field_area(map)`.

Fields: `logg` (ny × nx), grid origin `x0, y0`, one `spacing`, the normalization
`box::UniformSpatialPrior`, `n_all` (localizations used to build the map), and kernel
`bandwidth` (μm). Use [`field_area`](@ref) for the box area and [`exposure`](@ref) for
a partition's localization share.
"""
struct IntensityMap
    logg::Matrix{Float64}
    x0::Float64
    y0::Float64
    spacing::Float64
    box::UniformSpatialPrior
    n_all::Int
    bandwidth::Float64
end

"""Default pilot-intensity kernel bandwidth in μm."""
const DEFAULT_INTENSITY_BANDWIDTH = 0.2

"""
    field_area(m::IntensityMap)

Area of the map's normalization box in μm².
"""
field_area(m::IntensityMap) = area(m.box)

"""
    exposure(m::IntensityMap, n_locs)
    exposure(box::UniformSpatialPrior, n_locs, n_all)

Partition exposure `field_area(m) * n_locs / m.n_all`, including overlap copies.
The box overload supplies the same field share for the flat spatial model.
"""
exposure(m::IntensityMap, n_locs) = exposure(m.box, n_locs, m.n_all)
exposure(box::UniformSpatialPrior, n_locs, n_all) = area(box) * n_locs / n_all

# Kernel truncation radius in bandwidths, grid spacing in bandwidths.
const _IPP_TRUNC = 4.0
const _IPP_GRID = 0.25

# At most 2^27 Float64 cells (1 GiB for the grid itself).
const _MAX_INTENSITY_CELLS = 2^27

function _intensity_grid(box::UniformSpatialPrior, h::Float64)
    (isfinite(h) && h > 0) ||
        throw(ArgumentError("bandwidth must be finite and positive (got $h)"))
    W, H = box.x_max - box.x_min, box.y_max - box.y_min
    (isfinite(W) && isfinite(H) && W > 0 && H > 0) ||
        throw(ArgumentError("intensity map box must have finite positive extents"))
    d, R = _IPP_GRID * h, _IPP_TRUNC * h
    # Check before integer conversion and allocation, even for extreme bandwidths.
    nx = ceil((W + 2R) / d) + 1
    ny = ceil((H + 2R) / d) + 1
    nx * ny <= _MAX_INTENSITY_CELLS || throw(ArgumentError(
        "bandwidth=$h μm on field [$box] ($W × $H μm) needs $(nx * ny) cells; " *
        "limit is $_MAX_INTENSITY_CELLS; increase bandwidth or reduce the field"))
    return box.x_min - R, box.y_min - R, d, Int(nx), Int(ny)
end

_norm_cdf(z) = 0.5 * (1 + erf(z / sqrt(2.0)))

"""
    build_intensity_map(locs, box::UniformSpatialPrior; bandwidth=0.2) -> IntensityMap

Build the pilot intensity shape from all localizations `locs`: Gaussian kernel density
(bandwidth `bandwidth`, μm) with the kernel mass inside `box` as edge correction,
scaled by `A_field/N_all` (`A_field = area(box)`, `N_all = length(locs)`), floored where the
kernels truncate to zero, then normalized so that the bilinear lookup integrates to
`A_field` over `box`. Cost O(N·(2·4h/(h/4))²), threaded over grid column strips.
"""
function build_intensity_map(locs::Vector{<:SMLMData.AbstractEmitter},
                             box::UniformSpatialPrior;
                             bandwidth::Real = DEFAULT_INTENSITY_BANDWIDTH)
    h = Float64(bandwidth)
    N = length(locs)
    N > 0 || throw(ArgumentError("build_intensity_map needs at least one localization"))
    A = area(box)
    A > 0 || throw(ArgumentError("intensity map box has zero area"))

    xlo, xhi, ylo, yhi = box.x_min, box.x_max, box.y_min, box.y_max
    x0, y0, d, nx, ny = _intensity_grid(box, h)
    R = _IPP_TRUNC * h

    xs0 = Float64[l.x for l in locs]
    order = sortperm(xs0)
    # single assignments: captured by the threaded loop
    xs = xs0[order]
    ys = Float64[locs[i].y for i in order]

    # Kernel density by splatting each localization onto the grid nodes within R. Column
    # strips are disjoint per task (no races, deterministic); each strip visits only the
    # localizations whose x-window reaches it.
    κ = zeros(ny, nx)
    inv2h2 = 1 / (2h^2)
    norm = 1 / (2π * h^2)
    nw = _IPP_TRUNC / _IPP_GRID           # half-window in nodes
    nstrip = max(1, min(nx, 4 * Threads.nthreads()))
    bounds = round.(Int, range(0, nx; length = nstrip + 1))
    Threads.@threads for s in 1:nstrip
        c1 = bounds[s] + 1; c2 = bounds[s + 1]
        c1 > c2 && continue
        j1 = searchsortedfirst(xs, x0 + (c1 - 1) * d - R)
        j2 = searchsortedlast(xs, x0 + (c2 - 1) * d + R)
        wx = Vector{Float64}(undef, 2 * ceil(Int, nw) + 1)
        wy = similar(wx)
        @inbounds for j in j1:j2
            fx = (xs[j] - x0) / d + 1.0
            fy = (ys[j] - y0) / d + 1.0
            ixa = max(c1, ceil(Int, fx - nw)); ixb = min(c2, floor(Int, fx + nw))
            iya = max(1, ceil(Int, fy - nw)); iyb = min(ny, floor(Int, fy + nw))
            (ixa > ixb || iya > iyb) && continue
            for ix in ixa:ixb
                ddx = (ix - fx) * d
                wx[ix - ixa + 1] = exp(-ddx^2 * inv2h2)
            end
            for iy in iya:iyb
                ddy = (iy - fy) * d
                wy[iy - iya + 1] = exp(-ddy^2 * inv2h2)
            end
            for ix in ixa:ixb
                a = norm * wx[ix - ixa + 1]
                for iy in iya:iyb
                    κ[iy, ix] += a * wy[iy - iya + 1]
                end
            end
        end
    end

    # Edge correction (kernel mass inside the box) and scale A_field/N_all.
    mx = [max(_norm_cdf((xhi - (x0 + (i - 1) * d)) / h) -
              _norm_cdf((xlo - (x0 + (i - 1) * d)) / h), 1e-12) for i in 1:nx]
    my = [max(_norm_cdf((yhi - (y0 + (i - 1) * d)) / h) -
              _norm_cdf((ylo - (y0 + (i - 1) * d)) / h), 1e-12) for i in 1:ny]
    # Floor: one isolated kernel at the truncation radius, so g stays positive and
    # continuous where the truncated kernels vanish.
    g_floor = (A / N) * norm * exp(-_IPP_TRUNC^2 / 2)
    logg = κ                               # in place: one grid-sized array in memory
    @inbounds for ix in 1:nx, iy in 1:ny
        logg[iy, ix] = log(max((A / N) * κ[iy, ix] / (mx[ix] * my[iy]), g_floor))
    end

    m = IntensityMap(logg, x0, y0, d, box, N, h)
    return _normalize_intensity(m)
end

"""
    constant_intensity_map(box::UniformSpatialPrior; n_all=1, bandwidth=0.2) -> IntensityMap

The map g ≡ 1 over `box` (∫ g = area(box)): `:ipp` then reduces to `:flat` with a
Poisson K prior.
"""
function constant_intensity_map(box::UniformSpatialPrior; n_all::Int = 1,
                                bandwidth::Real = DEFAULT_INTENSITY_BANDWIDTH)
    h = Float64(bandwidth)
    n_all > 0 || throw(ArgumentError("n_all must be positive (got $n_all)"))
    x0, y0, d, nx, ny = _intensity_grid(box, h)
    return IntensityMap(zeros(ny, nx), x0, y0, d, box, n_all, h)
end

# 4-point Gauss-Legendre rule on [-1, 1].
const _GL4_X = (-0.8611363115940526, -0.3399810435848563, 0.3399810435848563,
    0.8611363115940526)
const _GL4_W = (0.3478548451374538, 0.6521451548625461, 0.6521451548625461,
    0.3478548451374538)

"""
    intensity_integral(m::IntensityMap, xlo, xhi, ylo, yhi) -> Float64

Integral of the lookup `g = exp(bilinear(log g))` over a rectangle (clipped to the grid):
tensor 4-point Gauss-Legendre quadrature on each grid cell overlapped, over the part of the
cell inside the rectangle (relative error far below 1e-8 for cells of a `h/4` grid).
Threaded over column strips; the result is deterministic.
"""
function intensity_integral(m::IntensityMap, xlo::Real, xhi::Real, ylo::Real, yhi::Real)
    ixa = max(1, floor(Int, (xlo - m.x0) / m.spacing) + 1)
    ixb = min(size(m.logg, 2) - 1, floor(Int, (xhi - m.x0) / m.spacing) + 1)
    iya = max(1, floor(Int, (ylo - m.y0) / m.spacing) + 1)
    iyb = min(size(m.logg, 1) - 1, floor(Int, (yhi - m.y0) / m.spacing) + 1)
    ixb < ixa && return 0.0
    ncol = ixb - ixa + 1
    nstrip = max(1, min(ncol, 4 * Threads.nthreads()))
    bounds = round.(Int, range(0, ncol; length = nstrip + 1))
    partial = zeros(nstrip)
    Threads.@threads for s in 1:nstrip
        acc = 0.0
        @inbounds for ix in (ixa + bounds[s]):(ixa + bounds[s + 1] - 1)
            cxa = max(xlo, m.x0 + (ix - 1) * m.spacing)
            cxb = min(xhi, m.x0 + ix * m.spacing)
            cxb > cxa || continue
            hx = (cxb - cxa) / 2; mx = (cxb + cxa) / 2
            for iy in iya:iyb
                cya = max(ylo, m.y0 + (iy - 1) * m.spacing)
                cyb = min(yhi, m.y0 + iy * m.spacing)
                cyb > cya || continue
                hy = (cyb - cya) / 2; my = (cyb + cya) / 2
                cell = 0.0
                for a in 1:4, b in 1:4
                    cell += _GL4_W[a] * _GL4_W[b] * exp(_bilinear_log(
                        m, mx + hx * _GL4_X[a], my + hy * _GL4_X[b]))
                end
                acc += hx * hy * cell
            end
        end
        partial[s] = acc
    end
    return sum(partial)
end

# Shift log g so that the lookup integrates to field_area over the box.
function _normalize_intensity(m::IntensityMap)
    I = intensity_integral(m, m.box.x_min, m.box.x_max, m.box.y_min, m.box.y_max)
    (isfinite(I) && I > 0) || error("intensity map has non-positive integral over its box")
    m.logg .+= log(field_area(m) / I)       # in place (the map is not yet shared)
    return m
end

@inline function _bilinear_log(m::IntensityMap, x::Float64, y::Float64)
    fx = clamp((x - m.x0) / m.spacing + 1.0, 1.0, Float64(size(m.logg, 2)))
    fy = clamp((y - m.y0) / m.spacing + 1.0, 1.0, Float64(size(m.logg, 1)))
    ix = max(min(floor(Int, fx), size(m.logg, 2) - 1), 1)
    iy = max(min(floor(Int, fy), size(m.logg, 1) - 1), 1)
    sx = fx - ix
    sy = fy - iy
    lg = m.logg
    @inbounds begin
        v00 = lg[iy, ix]; v10 = lg[iy, ix + 1]
        v01 = lg[iy + 1, ix]; v11 = lg[iy + 1, ix + 1]
    end
    return (1 - sx) * (1 - sy) * v00 + sx * (1 - sy) * v10 +
           (1 - sx) * sy * v01 + sx * sy * v11
end

"""
    log_intensity(m::IntensityMap, x, y) -> Float64

`log g(x, y)` by bilinear interpolation of the stored `log g` (clamped to the grid). O(1).
"""
@inline log_intensity(m::IntensityMap, x::Real, y::Real) = _bilinear_log(m, Float64(x),
    Float64(y))

"""
    IntensitySpatial <: AbstractSpatialModel

Inhomogeneous Poisson process spatial prior. Partition exposure `exp(log_area)`
(`A_i' = A_field·n_i/N_all` when partitioned) and ONE shared read-only
[`IntensityMap`](@ref).
The per-emitter factor is `ρ·g(θ̂)` at the cluster's posterior mean (saddle point):

    spatial_ml(cs, sp) = log_marginal_likelihood(cs, log_area) + log g(θ̂)

paired with the Poisson(ρA_i') K prior (as `FlatSpatial`), so the flat `-log A` cancels and
the per-cluster factor is `ρ g(θ̂)·ML`. With `g ≡ 1` it is exactly `FlatSpatial`.

Retained convention: ρ is the density of emitters with at least one localization and the
count marks are not zero-truncated (the flat + Poisson convention; the omitted term is
`-K log(1 - p₀)`, `p₀ = (α/(α+μ))^α`). docs/math_reference.md §4.6.
"""
struct IntensitySpatial <: AbstractSpatialModel
    log_area::Float64
    map::IntensityMap
end

"""
    IntensitySpatial(m::IntensityMap, n_locs)

Use a shared map with the exposure of `n_locs` localizations from its field.
"""
IntensitySpatial(m::IntensityMap, n_locs::Integer) =
    IntensitySpatial(log(exposure(m, n_locs)), m)

# g uses x, y; 3D features share the same xy exposure.
@inline function spatial_ml(cs::ClusterStats, sp::IntensitySpatial)
    lml = log_marginal_likelihood(cs, sp.log_area)
    (cs.n == 0 || !isfinite(lml)) && return lml
    μ = cs.Λ \ cs.η
    return lml + log_intensity(sp.map, μ[1], μ[2])
end

@inline function spatial_pred(cs::ClusterStats, lp::LocPrecision, sp::IntensitySpatial)
    if cs.n == 0
        return -sp.log_area + log_intensity(sp.map, lp.pos[1], lp.pos[2])
    end
    return spatial_ml(add_loc(cs, lp), sp) - spatial_ml(cs, sp)
end

spatial_area(sp::IntensitySpatial) = exp(sp.log_area)
spatial_log_area(sp::IntensitySpatial) = sp.log_area

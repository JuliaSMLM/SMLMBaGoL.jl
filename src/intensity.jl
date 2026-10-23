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
`∫_box g = field_area`.

Fields: `logg` (ny × nx), grid origin `x0, y0` and spacing `dx, dy`, the normalization box
`(xlo, xhi, ylo, yhi)`, `field_area` (area of the box), `n_all` (localizations the map was
built from; a partition of `n_i` localizations has exposure `field_area·n_i/n_all`) and the
kernel `bandwidth` (µm).
"""
struct IntensityMap
    logg::Matrix{Float64}
    x0::Float64
    y0::Float64
    dx::Float64
    dy::Float64
    nx::Int
    ny::Int
    xlo::Float64
    xhi::Float64
    ylo::Float64
    yhi::Float64
    field_area::Float64
    n_all::Int
    bandwidth::Float64
end

# Kernel truncation radius in bandwidths, grid spacing in bandwidths.
const _IPP_TRUNC = 4.0
const _IPP_GRID = 0.25

_norm_cdf(z) = 0.5 * (1 + erf(z / sqrt(2.0)))

"""
    build_intensity_map(locs, box::UniformSpatialPrior; bandwidth=0.2) -> IntensityMap

Build the pilot intensity shape from all localizations `locs`: Gaussian kernel density
(bandwidth `bandwidth`, µm) with the kernel mass inside `box` as edge correction,
scaled by `A_field/N_all` (`A_field = area(box)`, `N_all = length(locs)`), floored where the
kernels truncate to zero, then normalized so that the bilinear lookup integrates to
`A_field` over `box`. Cost O(N·(2·4h/(h/4))²), threaded over grid column strips.
"""
function build_intensity_map(locs::Vector{<:SMLMData.AbstractEmitter}, box::UniformSpatialPrior;
                             bandwidth::Real = 0.2)
    h = Float64(bandwidth)
    h > 0 || throw(ArgumentError("intensity_bandwidth must be positive (got $bandwidth)"))
    N = length(locs)
    N > 0 || throw(ArgumentError("build_intensity_map needs at least one localization"))
    A = area(box)
    A > 0 || throw(ArgumentError("intensity map box has zero area"))

    xlo, xhi, ylo, yhi = box.x_min, box.x_max, box.y_min, box.y_max
    d = _IPP_GRID * h
    R = _IPP_TRUNC * h
    x0 = xlo - R; y0 = ylo - R
    nx = ceil(Int, (xhi - xlo + 2R) / d) + 1
    ny = ceil(Int, (yhi - ylo + 2R) / d) + 1

    xs0 = Float64[l.x for l in locs]
    order = sortperm(xs0)
    xs = xs0[order]                        # single assignments: captured by the threaded loop
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
    mx = [max(_norm_cdf((xhi - (x0 + (i - 1) * d)) / h) - _norm_cdf((xlo - (x0 + (i - 1) * d)) / h), 1e-12) for i in 1:nx]
    my = [max(_norm_cdf((yhi - (y0 + (i - 1) * d)) / h) - _norm_cdf((ylo - (y0 + (i - 1) * d)) / h), 1e-12) for i in 1:ny]
    # Floor: one isolated kernel at the truncation radius, so g stays positive and
    # continuous where the truncated kernels vanish.
    g_floor = (A / N) * norm * exp(-_IPP_TRUNC^2 / 2)
    logg = κ                               # in place: one grid-sized array in memory
    @inbounds for ix in 1:nx, iy in 1:ny
        logg[iy, ix] = log(max((A / N) * κ[iy, ix] / (mx[ix] * my[iy]), g_floor))
    end

    m = IntensityMap(logg, x0, y0, d, d, nx, ny, xlo, xhi, ylo, yhi, A, N, h)
    return _normalize_intensity(m)
end

"""
    constant_intensity_map(box::UniformSpatialPrior; n_all=1, bandwidth=0.2) -> IntensityMap

The map g ≡ 1 over `box` (∫ g = area(box)): `:ipp` then reduces to `:flat` + Poisson K prior.
"""
function constant_intensity_map(box::UniformSpatialPrior; n_all::Int = 1, bandwidth::Real = 0.2)
    h = Float64(bandwidth)
    d = _IPP_GRID * h
    R = _IPP_TRUNC * h
    x0 = box.x_min - R; y0 = box.y_min - R
    nx = ceil(Int, (box.x_max - box.x_min + 2R) / d) + 1
    ny = ceil(Int, (box.y_max - box.y_min + 2R) / d) + 1
    return IntensityMap(zeros(ny, nx), x0, y0, d, d, nx, ny,
                        box.x_min, box.x_max, box.y_min, box.y_max, area(box), n_all, h)
end

# 4-point Gauss-Legendre rule on [-1, 1].
const _GL4_X = (-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526)
const _GL4_W = (0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538)

"""
    intensity_integral(m::IntensityMap, xlo, xhi, ylo, yhi) -> Float64

Integral of the lookup `g = exp(bilinear(log g))` over a rectangle (clipped to the grid):
tensor 4-point Gauss-Legendre quadrature on each grid cell overlapped, over the part of the
cell inside the rectangle (relative error far below 1e-8 for cells of a `h/4` grid). Threaded
over column strips; the result is deterministic.
"""
function intensity_integral(m::IntensityMap, xlo::Real, xhi::Real, ylo::Real, yhi::Real)
    ixa = max(1, floor(Int, (xlo - m.x0) / m.dx) + 1)
    ixb = min(m.nx - 1, floor(Int, (xhi - m.x0) / m.dx) + 1)
    iya = max(1, floor(Int, (ylo - m.y0) / m.dy) + 1)
    iyb = min(m.ny - 1, floor(Int, (yhi - m.y0) / m.dy) + 1)
    ixb < ixa && return 0.0
    ncol = ixb - ixa + 1
    nstrip = max(1, min(ncol, 4 * Threads.nthreads()))
    bounds = round.(Int, range(0, ncol; length = nstrip + 1))
    partial = zeros(nstrip)
    Threads.@threads for s in 1:nstrip
        acc = 0.0
        @inbounds for ix in (ixa + bounds[s]):(ixa + bounds[s + 1] - 1)
            cxa = max(xlo, m.x0 + (ix - 1) * m.dx); cxb = min(xhi, m.x0 + ix * m.dx)
            cxb > cxa || continue
            hx = (cxb - cxa) / 2; mx = (cxb + cxa) / 2
            for iy in iya:iyb
                cya = max(ylo, m.y0 + (iy - 1) * m.dy); cyb = min(yhi, m.y0 + iy * m.dy)
                cyb > cya || continue
                hy = (cyb - cya) / 2; my = (cyb + cya) / 2
                cell = 0.0
                for a in 1:4, b in 1:4
                    cell += _GL4_W[a] * _GL4_W[b] * exp(_bilinear_log(m, mx + hx * _GL4_X[a], my + hy * _GL4_X[b]))
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
    I = intensity_integral(m, m.xlo, m.xhi, m.ylo, m.yhi)
    (isfinite(I) && I > 0) || error("intensity map has non-positive integral over its box")
    m.logg .+= log(m.field_area / I)       # in place (the map is not yet shared)
    return m
end

@inline function _bilinear_log(m::IntensityMap, x::Float64, y::Float64)
    fx = clamp((x - m.x0) / m.dx + 1.0, 1.0, Float64(m.nx))
    fy = clamp((y - m.y0) / m.dy + 1.0, 1.0, Float64(m.ny))
    ix = max(min(floor(Int, fx), m.nx - 1), 1)
    iy = max(min(floor(Int, fy), m.ny - 1), 1)
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
@inline log_intensity(m::IntensityMap, x::Real, y::Real) = _bilinear_log(m, Float64(x), Float64(y))

"""
    IntensitySpatial <: AbstractSpatialModel

Inhomogeneous Poisson process spatial prior. Partition exposure `exp(log_area)`
(`A_i' = A_field·n_i/N_all` when partitioned) and ONE shared read-only [`IntensityMap`](@ref).
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

# Empty cluster contributes nothing (guard before the solve). g is a map of the first two
# coordinates (x, y), as the flat area is (3D features use the same xy exposure).
@inline function spatial_ml(cs::ClusterStats{D}, sp::IntensitySpatial) where D
    n = Int(cs.n)
    n == 0 && return 0.0
    det_Λ = det(cs.Λ)
    det_Λ <= 0 && return -Inf
    μ = cs.Λ \ cs.η                       # one solve: the flat ML term and θ̂ share it
    # = log_marginal_likelihood(cs, sp.log_area) (cluster_stats.jl) + log g(θ̂)
    lml = (1 - n) * (D / 2) * log(2π) -
          0.5 * cs.log_det_sum -
          0.5 * (cs.quad - dot(cs.η, μ)) -
          0.5 * log(det_Λ) -
          sp.log_area
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

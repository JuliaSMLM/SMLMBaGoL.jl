using Test
using SMLMBaGoL
using SMLMData
using Random

# PosteriorImage deposits: every cluster puts exactly its weight into the image per
# iteration, whatever its posterior width relative to the pixel and wherever it sits in
# the pixel (the density is integrated over each pixel, not sampled at pixel centers),
# and in a partition with overlap copies each cluster is weighted by its core fraction.

const B = SMLMBaGoL

# One cluster: n localizations at (x, y) with σ = σ_loc; posterior σ = σ_loc/√n.
function one_cluster_state(x, y, σ_loc, n; σ_xy = 0.0)
    locs = [SMLMData.Emitter2DFit{Float64}(x, y, 1000.0, 10.0, σ_loc, σ_loc, 0.0, 0.0;
                                            σ_xy = σ_xy, frame = i) for i in 1:n]
    log_area = log(1.0)
    state = B._reconstruct_state(fill(Int16(1), n), locs, log_area, B.FlatSpatial(log_area))
    return state, locs
end

@testset "Posterior image: one emitter deposits mass 1 at any width and sub-pixel offset" begin
    n = 4
    for ps in (0.001, 0.002, 0.005), σ_px in (0.1, 0.3, 0.7, 2.0, 5.0),
        (fx, fy) in ((0.0, 0.0), (0.25, 0.0), (0.5, 0.5), (0.13, 0.71))
        σ_post = σ_px * ps
        x = 0.1 + fx * ps; y = 0.1 + fy * ps
        state, locs = one_cluster_state(x, y, σ_post * sqrt(n), n)
        acc = PosteriorImage(pixel_size = ps, xlim = (0.0, 0.2), ylim = (0.0, 0.2))
        B.accumulator_update!(acc, state, locs, 10.0, 2.0, 1)
        @test sum(acc.image) ≈ 1.0 atol = 1e-9
    end
    # Correlated posterior (σ_xy ≠ 0): still exactly mass 1.
    for ρ in (-0.8, 0.5)
        σ = 0.004
        state, locs = one_cluster_state(0.1003, 0.0997, σ, 3; σ_xy = ρ * σ^2)
        acc = PosteriorImage(pixel_size = 0.002, xlim = (0.0, 0.2), ylim = (0.0, 0.2))
        B.accumulator_update!(acc, state, locs, 10.0, 2.0, 1)
        @test sum(acc.image) ≈ 1.0 atol = 1e-9
    end
    # Axis-aligned pixel values are the exact pixel integrals (product of normal CDF
    # differences): a 0.3-pixel emitter on a pixel edge puts Φ(1/0.3) − 1/2 in each of the
    # two columns beside the edge and the rest, 1 − Φ(1/0.3), in the next columns out.
    ps = 0.001
    state, locs = one_cluster_state(0.1, 0.1005, 0.3 * ps * 2, 4)
    acc = PosteriorImage(pixel_size = ps, xlim = (0.0, 0.2), ylim = (0.0, 0.2))
    B.accumulator_update!(acc, state, locs, 10.0, 2.0, 1)
    cols = sort(vec(sum(acc.image, dims = 2)), rev = true)
    @test cols[1] ≈ 0.4995709396798759 atol = 1e-9   # Φ(10/3) − 1/2
    @test cols[2] ≈ 0.4995709396798759 atol = 1e-9
    @test cols[3] ≈ 0.000429060320124 atol = 1e-9    # 1 − Φ(10/3) (Φ(20/3) ≈ 1)
end

@testset "Posterior image: overlap copies weight a cluster by its core fraction" begin
    state, locs = one_cluster_state(0.1, 0.1, 0.004, 4)
    for (ov, w) in (([false, false, false, false], 1.0), ([true, false, false, false], 0.75),
                    ([true, true, false, false], 0.5), ([true, true, true, true], 0.0))
        acc = PosteriorImage(pixel_size = 0.001, xlim = (0.0, 0.2), ylim = (0.0, 0.2),
                             is_overlap = ov)
        B.accumulator_update!(acc, state, locs, 10.0, 2.0, 1)
        @test sum(acc.image) ≈ w atol = 1e-9
        @test B._core_weighted_k(state, ov, Int[]) ≈ w
    end
    # Auto-sized grids snap to the origin lattice, so partition images merge in place.
    acc = PosteriorImage(pixel_size = 0.001, origin = (0.0123, -0.0456))
    B.accumulator_update!(acc, state, locs, 10.0, 2.0, 1)
    k = (acc.edges_x[1] - 0.0123) / 0.001
    @test k ≈ round(k) atol = 1e-6
    k = (acc.edges_y[1] + 0.0456) / 0.001
    @test k ≈ round(k) atol = 1e-6
end

@testset "Posterior image: weakly correlated pixels equal an independent pixel integral" begin
    # Independent reference: exact normal CDF in y given x, Simpson's rule in x (400 panels per
    # pixel). The deposit's series for |ρ| ≤ 0.1 must match it to 1e-9 per pixel; σ_xy = 0 too.
    function ref_pixel(xl, xr, yl, yr, mx, my, Sxx, Sxy, Syy)
        β = Sxy / Sxx; s = sqrt(Syy - Sxy^2 / Sxx); σx = sqrt(Sxx)
        f(x) = exp(-0.5 * ((x - mx) / σx)^2) / (sqrt(2π) * σx) *
               (B._Φ((yr - my - β * (x - mx)) / s) - B._Φ((yl - my - β * (x - mx)) / s))
        n = 400; h = (xr - xl) / n
        return h / 3 * (f(xl) + f(xr) + 4 * sum(f(xl + (2i - 1) * h) for i in 1:n ÷ 2) +
                        2 * sum(f(xl + 2i * h) for i in 1:(n ÷ 2 - 1)))
    end
    ps = 0.001
    for (σx, σy, ρ) in ((0.0012, 0.0009, 0.0), (0.0012, 0.0009, 0.002), (0.0007, 0.0016, -0.05),
                        (0.0015, 0.0015, 0.1), (0.0006, 0.0006, 0.09),
                        (0.0002, 0.00025, 0.08), (0.004, 0.003, -0.1))   # narrow and wide
        # ±8σ box: the renormalization to mass 1 then changes pixels by < 1e-14
        acc = PosteriorImage(pixel_size = ps, xlim = (0.0, 0.2), ylim = (0.0, 0.2), n_sigma = 8.0)
        mx, my = 0.10031, 0.09977
        Sxx, Syy, Sxy = σx^2, σy^2, ρ * σx * σy
        B._deposit_gaussian!(acc, 1.0, mx, my, Sxx, Sxy, Syy)
        @test sum(acc.image) ≈ 1.0 atol = 1e-12
        ix0 = floor(Int, (mx - acc.edges_x[1]) / ps) + 1; iy0 = floor(Int, (my - acc.edges_y[1]) / ps) + 1
        worst = 0.0
        for ix in ix0 - 3:ix0 + 3, iy in iy0 - 3:iy0 + 3
            xl = acc.edges_x[1] + (ix - 1) * ps; yl = acc.edges_y[1] + (iy - 1) * ps
            r = ref_pixel(xl, xl + ps, yl, yl + ps, mx, my, Sxx, Sxy, Syy)
            worst = max(worst, abs(acc.image[ix, iy] - r))
        end
        @test worst < 1e-9
    end
end

@testset "Posterior image: strongly correlated pixels hold their exact probabilities" begin
    # Checks independent of the deposit's method. (1) Mean on a pixel corner and σ ≪ pixel:
    # the four pixels at the corner hold the quadrant probabilities 1/4 ± asin(ρ)/(2π).
    ps = 0.001
    for σ_px in (0.01, 0.001), ρ in (0.3, 0.9, 0.9999, -0.95)
        σ = σ_px * ps
        acc = PosteriorImage(pixel_size = ps, xlim = (-0.01, 0.01), ylim = (-0.01, 0.01))
        B._deposit_gaussian!(acc, 1.0, 0.0, 0.0, σ^2, ρ * σ^2, σ^2)
        i0 = findfirst(e -> abs(e) < ps / 2, acc.edges_x)   # pixels i0, j0 lie above the edge at 0
        j0 = findfirst(e -> abs(e) < ps / 2, acc.edges_y)
        same = 0.25 + asin(ρ) / (2π); opposite = 0.25 - asin(ρ) / (2π)
        @test acc.image[i0, j0] ≈ same atol = 1e-12
        @test acc.image[i0 - 1, j0 - 1] ≈ same atol = 1e-12
        @test acc.image[i0, j0 - 1] ≈ opposite atol = 1e-12
        @test acc.image[i0 - 1, j0] ≈ opposite atol = 1e-12
    end
    # (2) Summed over rows (columns), a pixel-integrated bivariate normal is the 1D pixel
    # integral of its x (y) marginal, for any ρ (±8σ box: truncation < 1e-14).
    for (mx, my, σx, σy, ρ) in ((0.00031, -0.00017, 0.0003, 0.0003, 0.95), (0.00031, -0.00017, 0.0012, 0.0009, 0.5),
                                (0.00031, -0.00017, 0.003, 0.002, -0.7), (0.00031, -0.00017, 0.0004, 0.0015, 0.3),
                                (0.0, 0.0, 1e-5, 0.005, 0.99))
        acc = PosteriorImage(pixel_size = ps, xlim = (-0.1, 0.1), ylim = (-0.1, 0.1), n_sigma = 8.0)
        B._deposit_gaussian!(acc, 1.0, mx, my, σx^2, ρ * σx * σy, σy^2)
        ex, ey = acc.edges_x, acc.edges_y
        colx = vec(sum(acc.image, dims = 2)); rowy = vec(sum(acc.image, dims = 1))
        @test maximum(abs(colx[i] - (B._Φ((ex[i + 1] - mx) / σx) - B._Φ((ex[i] - mx) / σx))) for i in eachindex(colx)) < 1e-12
        @test maximum(abs(rowy[j] - (B._Φ((ey[j + 1] - my) / σy) - B._Φ((ey[j] - my) / σy))) for j in eachindex(rowy)) < 1e-12
    end
end

@testset "Bivariate normal upper tail (_bvnu)" begin
    rs = (-0.9999, -0.95, -0.8, -0.5, -0.2, 0.15, 0.29, 0.31, 0.6, 0.74, 0.76, 0.9, 0.924, 0.926, 0.99, 0.999999)
    for r in rs
        @test B._bvnu(0.0, 0.0, r) ≈ 0.25 + asin(r) / (2π) atol = 1e-15
        for (h, k) in ((-2.3, 0.2), (0.7, -1.7), (3.1, 2.6), (-0.4, -0.4))
            @test B._bvnu(h, k, r) ≈ B._bvnu(k, h, r) atol = 1e-15            # exchange
            @test B._bvnu(h, k, r) + B._bvnu(h, -k, -r) ≈ B._Φ(-h) atol = 1e-14  # P(X > h) split on Y
        end
    end
    # General points against ∫_h^∞ φ(x) Φ((r x − k)/√(1 − r²)) dx (Simpson, 200,000 panels)
    function ref(h, k, r)
        s = sqrt((1 - r) * (1 + r)); lo = max(h, -12.0); hi = 12.0; n = 200_000; d = (hi - lo) / n
        f(x) = exp(-x^2 / 2) / sqrt(2π) * B._Φ((r * x - k) / s)
        return d / 3 * (f(lo) + f(hi) + 4 * sum(f(lo + (2i - 1) * d) for i in 1:n ÷ 2) +
                        2 * sum(f(lo + 2i * d) for i in 1:(n ÷ 2 - 1)))
    end
    for r in (-0.97, 0.5, 0.9999), (h, k) in ((-2.3, 0.2), (0.7, -1.7), (0.0, 2.6))
        @test B._bvnu(h, k, r) ≈ ref(h, k, r) atol = 1e-12
    end
end

@testset "Posterior image: overlap flags must match the chain" begin
    state, locs = one_cluster_state(0.1, 0.1, 0.004, 4)
    acc = PosteriorImage(pixel_size = 0.001, xlim = (0.0, 0.2), ylim = (0.0, 0.2), is_overlap = [true])
    @test_throws ArgumentError B.accumulator_update!(acc, state, locs, 10.0, 2.0, 1)
    @test_throws ArgumentError B._core_weighted_k(state, [true, false], Int[])
end

@testset "Posterior image: 3D chain clusters render their x-y covariance" begin
    # σ_z ≫ σ_x, σ_y: the image must use the x-y block of a 3D cluster's covariance, never
    # σ_z (a 3D posterior_cov is ordered Σxx, Σyy, Σzz, Σxy, Σxz, Σyz).
    n = 4
    locs = [SMLMData.Emitter3DFit{Float64}(0.1, 0.1, 0.0, 1000.0, 10.0, 0.008, 0.004, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.0, i, 1, 0, i) for i in 1:n]
    cs = B.empty_cluster(locs)
    for l in locs
        cs = B.add_loc(cs, l)
    end
    Sxx, Sxy, Syy = B._xy_cov(cs)
    @test Sxx ≈ 0.008^2 / n rtol = 1e-9
    @test Syy ≈ 0.004^2 / n rtol = 1e-9
    @test abs(Sxy) < 1e-15
end

@testset "Render resampling conserves each emitter's mass at any render pixel" begin
    # A 0.3 nm emitter imaged at 1 nm and shown at 2 nm render pixels: the render panel's
    # total and peak must not depend on where the emitter sits within a pixel.
    peaks = Float64[]
    for f in 0.0:0.125:0.875
        state, locs = one_cluster_state(0.1 + f * 0.001, 0.1 + 0.3 * f * 0.001, 0.0003 * 2, 4)
        acc = PosteriorImage(pixel_size = 0.001, xlim = (0.0, 0.2), ylim = (0.0, 0.2))
        B.accumulator_update!(acc, state, locs, 10.0, 2.0, 1)
        post = (image = acc.image, edges_x = acc.edges_x, edges_y = acc.edges_y, pixel_size = 0.001)
        out = B._resample_mass(post, 0.0, 0.2, 0.0, 0.2, 100, 100)
        @test sum(out) ≈ 1.0 atol = 1e-9
        push!(peaks, maximum(out))
    end
    @test minimum(peaks) >= 0.25 - 1e-9            # at worst split over 4 render pixels (main: 0.002)
    # Partial overlap of a render window keeps exactly the mass inside it.
    state, locs = one_cluster_state(0.1, 0.1, 0.004, 4)
    acc = PosteriorImage(pixel_size = 0.001, xlim = (0.0, 0.2), ylim = (0.0, 0.2))
    B.accumulator_update!(acc, state, locs, 10.0, 2.0, 1)
    post = (image = acc.image, edges_x = acc.edges_x, edges_y = acc.edges_y, pixel_size = 0.001)
    @test sum(B._resample_mass(post, 0.1, 0.3, 0.0, 0.2, 77, 131)) ≈ 0.5 atol = 1e-4  # ±4σ box tails
end

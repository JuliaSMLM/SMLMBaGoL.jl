using Test
using SMLMBaGoL
using SMLMData
using Random
import SMLMSim
const B = SMLMBaGoL

# run_bagol's partition images are stored as tiles where deposits land, not densely over each
# partition's bounding box (a few partitions sprawl across the field, so dense partition images can
# need many times the field image's memory). The tiles must hold exactly the dense image's pixel values and merge
# into the full-field image exactly as a dense partition image does.

loc(x, y, σ) = SMLMData.Emitter2DFit{Float64}(x, y, 1.0, 0.0, σ, σ, 0.0, 0.0)

@testset "Tiled partition image holds the dense image's pixels exactly" begin
    rng = Xoshiro(5)
    locs = [loc(0.2 + 0.6 * rand(rng), 0.3 + 0.5 * rand(rng), 0.01) for _ in 1:40]
    origin = (0.0123, -0.0071)
    for ps in (0.001, 0.003)
        dense = PosteriorImage(pixel_size = ps, origin = origin)
        tiled = B._TiledPosteriorImage(pixel_size = ps, origin = origin)
        B._ensure_initialized!(dense, locs)
        B._ensure_initialized!(tiled, locs)
        for _ in 1:300
            mx = 0.2 + 0.6 * rand(rng); my = 0.3 + 0.5 * rand(rng)
            σx = ps * (0.05 + 6 * rand(rng)); σy = ps * (0.05 + 6 * rand(rng))
            ρ = rand(rng, (0.0, 0.05, -0.08, 0.5, -0.97))      # series and corner paths
            w = rand(rng)
            for a in (dense, tiled)
                B._deposit_gaussian!(a, w, mx, my, σx^2, ρ * σx * σy, σy^2)
            end
        end
        r = B.accumulator_result(tiled)
        @test r.edges_x == dense.edges_x && r.edges_y == dense.edges_y
        @test r.image == dense.image                                  # bitwise
        # merged into a full-field image, a tiled and a dense partition image give equal sums
        full_d = PosteriorImage(pixel_size = ps, xlim = (origin[1], 1.0), ylim = (origin[2], 1.0))
        full_t = PosteriorImage(pixel_size = ps, xlim = (origin[1], 1.0), ylim = (origin[2], 1.0))
        for f in (full_d, full_t)
            f.image .= 0.25                                           # an earlier partition
        end
        B.accumulator_merge!(full_d, dense)
        B.accumulator_merge!(full_t, tiled)
        @test full_t.image == full_d.image
        @test sum(full_t.image) ≈ 0.25 * length(full_t.image) + sum(dense.image)
    end
end

@testset "A sprawling partition stores only where it deposits" begin
    # two localizations 20 μm apart at 1 nm pixels: a 4e8-pixel bounding box
    locs = [loc(0.0, 0.0, 0.005), loc(20.0, 20.0, 0.005)]
    tiled = B._TiledPosteriorImage(pixel_size = 0.001, origin = (-1.0, -1.0))
    B._ensure_initialized!(tiled, locs)
    @test tiled.nx * tiled.ny > 4e8
    for (x, y) in ((0.0, 0.0), (20.0, 20.0))
        B._deposit_gaussian!(tiled, 1.0, x, y, 0.004^2, 0.0, 0.004^2)
    end
    @test B._stored_pixels(tiled) < 2e4
    tot = 0.0
    B._each_tile_pixel((ix, iy, v) -> (tot += v), tiled)
    @test tot ≈ 2.0
end

@testset "run_bagol: partition image memory follows the deposits, not the bounding boxes" begin
    # A diagonal line of emitters 50 nm apart, 7 μm long: one DBSCAN component cut by METIS into
    # pieces whose bounding boxes are ~5 μm² each (dense at 1 nm: ~43 MB each), while the
    # deposits cover a thin band. The merged image is restricted to a 1 μm window.
    Random.seed!(3)
    cam = SMLMData.IdealCamera(1:80, 1:80, 0.1)
    params = SMLMSim.StaticSMLMConfig(; density = 1.0, nframes = 1000, framerate = 50.0,
        ndatasets = 1, σ_psf = 0.13, minphotons = 50)
    sites = [SMLMData.Emitter2DFit{Float64}(0.5 + 0.035k, 0.5 + 0.035k, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
             track_id = k + 1) for k in 0:199]
    smld, _ = SMLMSim.simulate(params; starting_conditions = sites,
        molecule = SMLMSim.GenericFluor(photons = 1e4, k_off = 50.0, k_on = 0.5), camera = cam)
    kw = (n_iterations = 120, burn_in = 60, verbose = false,
          posterior_xlim = (0.5, 1.5), posterior_ylim = (0.5, 1.5))
    go(ps) = (Random.seed!(1); run_bagol(smld; posterior_pixel_size = ps, kw...))
    go(0.001); go(0.0)                                                # compile
    res = nothing
    on = @allocated (res = go(0.001))
    off = @allocated go(0.0)
    d = res[2]
    @test d.n_partitions > 1
    @test (on - off) / 2^20 < 40          # dense partition images: > 100 MB here
    @test d.posterior_image.image isa Matrix{Float64}
    @test size(d.posterior_image.image) == (1001, 1001)
end

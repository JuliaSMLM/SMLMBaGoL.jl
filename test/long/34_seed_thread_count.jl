using Test

# The same seed gives the same run_bagol result in a fresh process with 1 thread and with
# 4 threads, on the first call and the second (before, partition chains drew from task
# streams that depend on the thread count and on tasks spawned during the first call), and
# the :ipp intensity map and a seeded :ipp run are the same too (before, the map's
# normalization summed a thread-count-dependent number of strips).

const SCRIPT = raw"""
using SMLMBaGoL, SMLMData, Random
import SMLMSim
Random.seed!(7)
cam = SMLMData.IdealCamera(1:16, 1:16, 0.1)
params = SMLMSim.StaticSMLMConfig(; density = 1.0, nframes = 1000, framerate = 50.0,
    ndatasets = 1, σ_psf = 0.13, minphotons = 50)
sites = [SMLMData.Emitter2DFit{Float64}(0.5 + 0.05i, 0.5 + 0.05j, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
         track_id = 10i + j + 1) for i in 0:9 for j in 0:9]
smld, _ = SMLMSim.simulate(params; starting_conditions = sites,
    molecule = SMLMSim.GenericFluor(photons = 1e4, k_off = 50.0, k_on = 0.5), camera = cam)
kw = (n_iterations = 200, burn_in = 100, max_partition_size = 300,
      posterior_pixel_size = 0.002, verbose = false)
for call in 1:2
    Random.seed!(1)
    r, d = run_bagol(smld; kw...)
    h = hash(([(e.x, e.y) for e in r.emitters], d.posterior_k, d.posterior_image.image))
    println("CALL ", call, " ", d.n_partitions, " ", h)
end
box = SMLMBaGoL.UniformSpatialPrior(smld.emitters)
m = SMLMBaGoL.build_intensity_map(smld.emitters, box; bandwidth = 0.05)
println("IPPMAP ", size(m.logg, 2), " ", hash(m.logg))
Random.seed!(1)
r, d = run_bagol(smld; kw..., spatial_model = :ipp, intensity_bandwidth = 0.05)
println("IPPRUN ", hash(([(e.x, e.y) for e in r.emitters], d.posterior_k, d.posterior_image.image)))
"""

function calls(nthreads)
    cmd = `$(Base.julia_cmd()) --startup-file=no -t $nthreads --project=$(Base.active_project()) -e $SCRIPT`
    out = read(cmd, String)
    return [l for l in split(out, '\n') if startswith(l, "CALL ") || startswith(l, "IPP")]
end

@testset "run_bagol: same seed, same result with 1 and 4 threads, first and second call" begin
    one, four = calls(1), calls(4)
    pick(ls, tag) = [l for l in ls if startswith(l, tag)]
    c1, c4 = pick(one, "CALL "), pick(four, "CALL ")
    @test length(c1) == 2 && length(c4) == 2
    @test parse(Int, split(c1[1])[3]) > 1                     # several partition chains
    @test allequal([last(split(l)) for l in vcat(c1, c4)])
    # :ipp: the intensity map (more grid columns than strips at either thread count) and a run
    m1, m4 = pick(one, "IPPMAP"), pick(four, "IPPMAP")
    @test length(m1) == 1 && parse(Int, split(m1[1])[2]) > 64
    @test m1 == m4
    @test length(pick(one, "IPPRUN")) == 1 && pick(one, "IPPRUN") == pick(four, "IPPRUN")
end

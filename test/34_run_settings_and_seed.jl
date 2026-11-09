using Test
using SMLMBaGoL
using SMLMData
using Random
import SMLMSim

# run_bagol checks its chain lengths before any work (before, a run with too few recorded
# iterations sampled to the end and then threw from a task; sync_interval = 0 threw
# DivideError), and Random.seed! before the call fixes the result (before, every partition
# chain drew from a task stream that depends on how many tasks the caller had spawned since
# the seed, so the result changed with the thread count and with any earlier spawn).

function sim_field(starting; seed, npix = 16)
    Random.seed!(seed)
    cam = SMLMData.IdealCamera(1:npix, 1:npix, 0.1)
    params = SMLMSim.StaticSMLMConfig(; density = 1.0, nframes = 1000, framerate = 50.0,
        ndatasets = 1, σ_psf = 0.13, minphotons = 50)
    smld, _ = SMLMSim.simulate(params; starting_conditions = starting,
        molecule = SMLMSim.GenericFluor(photons = 1e4, k_off = 50.0, k_on = 0.5), camera = cam)
    return smld
end
site(x, y, i) = SMLMData.Emitter2DFit{Float64}(x, y, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0; track_id = i)

@testset "run_bagol checks chain lengths before any work" begin
    smld = sim_field([site(0.6 + 0.06 * k, 0.6, k) for k in 0:2]; seed = 11)
    go(; kw...) = run_bagol(smld; posterior_pixel_size = 0.0, verbose = false, kw...)
    @test_throws ArgumentError go(; n_iterations = 300, burn_in = 300)
    @test_throws ArgumentError go(; n_iterations = 300, burn_in = 296)   # 4 recorded, no sample
    @test_throws ArgumentError go(; n_iterations = 300, burn_in = 150, sync_interval = 0)
    @test_throws ArgumentError run_bagol(smld, BaGoLConfig(; n_iterations = 300, burn_in = 300,
                                                           posterior_pixel_size = 0.0, verbose = false))
    # the shortest legal run: five recorded iterations give one assignment sample
    Random.seed!(3)
    r, d = go(; n_iterations = 105, burn_in = 100)
    @test sum(d.posterior_k) == 5
    @test d.n_emitters == length(r.emitters) > 0
end

# run_collapsed_chain divides the iteration count by both intervals; zero threw DivideError.
@testset "run_collapsed_chain checks its intervals before any work" begin
    locs = sim_field([site(0.6 + 0.06 * k, 0.6, k) for k in 0:2]; seed = 11).emitters
    chain(; kw...) = run_collapsed_chain(locs; n_iterations = 20, burn_in = 10, kw...)
    @test_throws ArgumentError chain(; hierarchical_interval = 0)
    @test_throws ArgumentError chain(; callback = (args...) -> nothing, callback_interval = 0)
    # without a callback the callback interval is unused
    @test chain(; callback_interval = 0) isa SMLMBaGoL.CollapsedChainResult
    calls = Ref(0)
    chain(; callback = (args...) -> (calls[] += 1), callback_interval = 5)
    @test calls[] == 4
end

@testset "Random.seed! fixes run_bagol's result, whatever tasks were spawned before" begin
    # METIS-split field (several partitions, several chain tasks)
    sites = [site(0.5 + 0.05 * i, 0.5 + 0.05 * j, 15i + j + 1) for i in 0:9 for j in 0:9]
    smld = sim_field(sites; seed = 7)
    kw = (n_iterations = 200, burn_in = 100, max_partition_size = 300,
          posterior_pixel_size = 0.002, verbose = false)
    Random.seed!(1)
    r1, d1 = run_bagol(smld; kw...)
    Random.seed!(1)
    fetch(Threads.@spawn nothing)          # as compiling, logging or user code may do
    r2, d2 = run_bagol(smld; kw...)
    @test d1.n_partitions > 1
    @test [(e.x, e.y) for e in r1.emitters] == [(e.x, e.y) for e in r2.emitters]
    @test d1.posterior_k == d2.posterior_k
    @test d1.posterior_image.image == d2.posterior_image.image
    # the caller's own stream after the run is fixed by the seed too
    Random.seed!(1); run_bagol(smld; kw...); a = rand()
    Random.seed!(1); fetch(Threads.@spawn nothing); run_bagol(smld; kw...); b = rand()
    @test a == b
end

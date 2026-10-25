using Test
using SMLMBaGoL
using SMLMData
using Random
import SMLMSim

@testset "run_bagol(:auto) forwards run settings to the τ finder" begin
    # Small field of 6 two-loc pairs; the finder keeps its own chain length, so the
    # run is cheap only because the field is tiny.
    cam = SMLMData.IdealCamera(1:64, 1:64, 0.1)
    Random.seed!(11)
    em = SMLMData.Emitter2DFit{Float64}[]
    for k in 1:6, j in 1:2
        push!(em, SMLMData.Emitter2DFit(0.5 + 0.4k, 1.0,
                1000.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 1, length(em) + 1, 0, 1))
    end
    truth = SMLMData.BasicSMLD(em, cam, 1, 1, Dict{String,Any}())
    smld = SMLMSim.apply_noise(truth, 0.01 * sqrt(1000.0))
    _, diag = run_bagol(smld; se_adjust = :auto, keep_se_finder = true,
                        max_partition_size = 7, overlap = 0.02, spatial_model = :flat,
                        bridge_ratio = 0.2, min_split_size = 2,
                        n_iterations = 100, burn_in = 20, verbose = false)
    kw = diag.se_finder.bagol_kwargs
    @test kw.max_partition_size == 7
    @test kw.overlap == 0.02
    @test kw.spatial_model === :flat
    @test kw.bridge_ratio == 0.2
    @test kw.min_split_size == 2
    @test kw.n_bd_substeps == 5
    @test_throws ArgumentError run_bagol(smld, BaGoLConfig(bridge_ratio = -1.0,
        se_adjust = 0.0, verbose = false))
    @test_throws ArgumentError run_bagol(smld, BaGoLConfig(min_split_size = 0,
        se_adjust = 0.0, verbose = false))
    @test !haskey(kw, :neighbor_graph) && !haskey(kw, :n_iterations)
    # Direct callers: explicitly given kwargs are used as given.
    f = estimate_se_adjust(smld; max_partition_size = 9, n_boot = 5, n_iterations = 100,
                           burn_in = 20, max_steps = 1)
    @test f.bagol_kwargs.max_partition_size == 9
    @test f.bagol_kwargs isa NamedTuple
    # The finder does not model drift: :auto with motion=:linear must refuse.
    @test_throws ArgumentError run_bagol(smld; se_adjust = :auto, motion = :linear,
                                         n_iterations = 100, burn_in = 20, verbose = false)
end

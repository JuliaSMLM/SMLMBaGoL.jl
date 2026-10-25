using Test
using SMLMBaGoL
using SMLMData

@testset "Bridge Refinement" begin
    σ = 0.005
    locs = SMLMData.Emitter2DFit[]
    id = 0

    # Two dense groups connected by a sparse transitive DBSCAN bridge.
    for x0 in (0.1, 0.2)
        for dx in (-0.002, -0.001, 0.0, 0.001, 0.002, 0.003)
            id += 1
            push!(locs, SMLMData.Emitter2DFit(
                x0 + dx, 0.1, 1000.0, 10.0,
                σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, id))
        end
    end
    for x in (0.128, 0.156, 0.184)
        id += 1
        push!(locs, SMLMData.Emitter2DFit(
            x, 0.1, 1000.0, 10.0,
            σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, id))
    end

    plain, _ = partition_locs(locs; partition_sigma=3.0, min_size=0,
                              max_size=100, bridge_ratio=0.0)
    refined, _ = partition_locs(locs; partition_sigma=3.0, min_size=0,
                                max_size=100, bridge_ratio=0.6,
                                min_split_size=3)

    @test length(plain) == 1
    @test length(refined) == 2
    @test sum(length(p.locs) for p in refined) == length(locs)
    @test all(length(p.locs) >= 3 for p in refined)
end

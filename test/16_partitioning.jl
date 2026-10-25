using Test
using SMLMBaGoL
using SMLMData
using Random

@testset "Partitioning" begin
    Random.seed!(456)

    sim1 = simulate_localizations([(0.1, 0.1)];
        fixed_sigma=0.005, mean_count=20.0, count_model=:fixed)
    sim2 = simulate_localizations([(0.5, 0.5)];
        fixed_sigma=0.005, mean_count=15.0, count_model=:fixed)
    locs = vcat(sim1.smld.emitters, sim2.smld.emitters)

    partitions, skipped = partition_locs(locs; partition_sigma=4.0, min_size=5,
        max_size=100)

    @test length(partitions) == 2
    @test isempty(skipped)

    sizes = sort([length(p.locs) for p in partitions])
    @test sizes == [15, 20]

    for p in partitions
        @test length(p.original_indices) == length(p.locs)
        for (i, idx) in enumerate(p.original_indices)
            @test p.locs[i] === locs[idx]
        end
    end

    # Public min_size counts the point itself; precision_neighbors excludes
    # self internally, so exactly-min-size dense clusters should survive.
    σ = 0.005
    tiny = [
        SMLMData.Emitter2DFit(0.1 + 0.001 * i, 0.1, 1000.0, 10.0,
                              σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i)
        for i in 1:3
    ]
    tiny_parts, _ = partition_locs(tiny; partition_sigma=3.0, min_size=3, max_size=100)
    @test length(tiny_parts) == 1
    @test length(tiny_parts[1].locs) == 3
end

using Test
using SMLMBaGoL
using SMLMData
using Random

@testset "Oversized Cluster Splitting" begin
    Random.seed!(789)

    locs = SMLMData.Emitter2DFit[]
    σ = 0.005

    for i in 1:50
        x = 0.1 + randn() * 0.02
        y = 0.1 + randn() * 0.02
        push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1,
            0, i))
    end

    partitions, skipped = partition_locs(locs; partition_sigma=6.0, min_size=3, max_size=20)

    @test isempty(skipped)
    @test length(partitions) >= 2

    for p in partitions
        # Core locs (non-overlap) should be near max_size.
        # With overlap wider than the cluster, cores may slightly exceed max_size
        # because recursion stops when bisection can't reduce core further.
        n_core = count(.!p.is_overlap)
        @test n_core <= 30  # allow some slack for overlap edge cases
    end

    # With overlap, locs may appear in multiple partitions.
    # But unique original indices must cover all 50.
    all_indices = reduce(vcat, [p.original_indices for p in partitions])
    @test length(unique(all_indices)) == 50

    partitions_skip, skipped_skip = partition_locs(locs; partition_sigma=6.0,
        min_size=3, max_size=20, skip_size=20)
    @test length(skipped_skip) >= 1
end

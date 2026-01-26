using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics

@testset "SMLMBaGoL.jl Tests" begin
    @testset "Basic Types" begin
        emitter = Emitter(0.1, 0.2)
        @test emitter.x == 0.1
        @test emitter.y == 0.2
        @test isempty(emitter.allocated)
    end

    @testset "Simple Integration" begin
        Random.seed!(123)

        # Create simple test data: 2 well-separated emitters
        locs = SMLMData.Emitter2DFit[]
        σ = 0.005

        # Emitter 1 at (0.1, 0.1)
        for i in 1:5
            x = 0.1 + randn() * σ
            y = 0.1 + randn() * σ
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i))
        end

        # Emitter 2 at (0.2, 0.2)
        for i in 1:5
            x = 0.2 + randn() * σ
            y = 0.2 + randn() * σ
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i+5))
        end

        # Create camera and SMLD for unified API
        camera = SMLMData.IdealCamera(64, 64, 0.1)
        smld = SMLMData.BasicSMLD(locs, camera, 100, 1)

        # Run with minimal iterations for testing - returns (BasicSMLD, BaGoLDiagnostics)
        result_smld, diagnostics = run_bagol(smld;
            partition_threshold = 0,  # Disable partitioning for simple test
            n_iterations = 500,
            burn_in = 100,
            verbose = false
        )

        @test result_smld isa SMLMData.BasicSMLD
        @test diagnostics isa BaGoLDiagnostics
        @test diagnostics.n_emitters >= 0
        @test length(result_smld.emitters) == diagnostics.n_emitters

        # Emitters are Emitter2DFit with uncertainties as fields
        if diagnostics.n_emitters > 0
            @test result_smld.emitters[1] isa SMLMData.Emitter2DFit
        end

        # Test chain access via run_bagol_chain for advanced users
        chain = run_bagol_chain(locs;
            n_iterations = 500,
            burn_in = 100,
            verbose = false
        )
        @test length(chain.samples) > 0

        emitters, posterior_k = estimate_mapn(chain)
        @test length(emitters) >= 0
    end

    @testset "Spatial Utilities" begin
        # Test get_coords and get_sigma with Emitter2DFit
        loc = SMLMData.Emitter2DFit(0.1, 0.2, 1000.0, 10.0, 0.005, 0.006, 0.0, 0.0, 0.0, 1, 1, 0, 1)
        coords = SMLMBaGoL.get_coords(loc)
        @test coords[1] ≈ 0.1
        @test coords[2] ≈ 0.2

        sigma = SMLMBaGoL.get_sigma(loc)
        @test sigma[1] ≈ 0.005
        @test sigma[2] ≈ 0.006

        # Test mean_sigma (geometric mean)
        ms = SMLMBaGoL.mean_sigma(loc)
        @test ms ≈ sqrt(0.005 * 0.006)

        # Test precision_weighted_distance
        loc2 = SMLMData.Emitter2DFit(0.2, 0.2, 1000.0, 10.0, 0.005, 0.005, 0.0, 0.0, 0.0, 1, 1, 0, 2)
        d_eff = SMLMBaGoL.precision_weighted_distance(loc, loc2)
        @test d_eff > 0
    end

    @testset "Partitioning" begin
        Random.seed!(456)

        # Create two well-separated clusters
        locs = SMLMData.Emitter2DFit[]
        σ = 0.005

        # Cluster 1 centered at (0.1, 0.1)
        for i in 1:20
            x = 0.1 + randn() * σ
            y = 0.1 + randn() * σ
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i))
        end

        # Cluster 2 centered at (0.5, 0.5) - well separated
        for i in 1:15
            x = 0.5 + randn() * σ
            y = 0.5 + randn() * σ
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i+20))
        end

        # Test partition_locs
        partitions, skipped = partition_locs(locs; nsigma=4.0, min_size=5, max_size=100)

        @test length(partitions) == 2
        @test isempty(skipped)

        # Check that partitions contain correct number of locs
        sizes = sort([length(p.locs) for p in partitions])
        @test sizes == [15, 20]

        # Verify original_indices map back correctly
        for p in partitions
            @test length(p.original_indices) == length(p.locs)
            for (i, idx) in enumerate(p.original_indices)
                @test p.locs[i] === locs[idx]
            end
        end
    end

    @testset "Oversized Cluster Splitting" begin
        Random.seed!(789)

        # Create one large cluster that exceeds max_size
        locs = SMLMData.Emitter2DFit[]
        σ = 0.005

        for i in 1:50
            x = 0.1 + randn() * 0.02  # Spread out a bit
            y = 0.1 + randn() * 0.02
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i))
        end

        # Test with max_size=20, should split
        partitions, skipped = partition_locs(locs; nsigma=6.0, min_size=3, max_size=20, oversized=:split)

        @test isempty(skipped)
        @test length(partitions) >= 2  # Should be split into at least 2

        # All partitions should be under max_size
        for p in partitions
            @test length(p.locs) <= 20
        end

        # Total locs should be preserved
        total_locs = sum(length(p.locs) for p in partitions)
        @test total_locs == 50

        # Test skip mode
        partitions_skip, skipped_skip = partition_locs(locs; nsigma=6.0, min_size=3, max_size=20, oversized=:skip)

        @test length(skipped_skip) >= 1  # The large cluster should be skipped
    end

    @testset "Partitioned BaGoL via SMLD" begin
        Random.seed!(321)

        # Create two distinct clusters with a few emitters each
        locs = SMLMData.Emitter2DFit[]
        σ = 0.005

        # Cluster 1: 2 emitters at (0.1, 0.1) and (0.12, 0.1)
        for i in 1:5
            push!(locs, SMLMData.Emitter2DFit(0.1 + randn()*σ, 0.1 + randn()*σ, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i))
        end
        for i in 1:5
            push!(locs, SMLMData.Emitter2DFit(0.12 + randn()*σ, 0.1 + randn()*σ, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i+5))
        end

        # Cluster 2: 1 emitter at (0.5, 0.5)
        for i in 1:5
            push!(locs, SMLMData.Emitter2DFit(0.5 + randn()*σ, 0.5 + randn()*σ, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i+10))
        end

        # Wrap in SMLD for dispatch
        camera = SMLMData.IdealCamera(64, 64, 0.1)  # n_pixels_x, n_pixels_y, pixel_size
        smld = SMLMData.BasicSMLD(locs, camera, 100, 1)

        # Run via unified interface - triggers partitioning for n_locs > threshold
        result_smld, diagnostics = run_bagol(smld;
            partition_threshold=5,  # Force partitioning
            nsigma=4.0,
            min_partition_size=3,
            max_partition_size=100,
            sync_interval=100,
            n_iterations=500,
            burn_in=100,
            verbose=false
        )

        @test result_smld isa SMLMData.BasicSMLD
        @test diagnostics isa BaGoLDiagnostics
        @test diagnostics.n_emitters >= 0
        @test diagnostics.n_partitions >= 1  # Partitioning was triggered
    end
end

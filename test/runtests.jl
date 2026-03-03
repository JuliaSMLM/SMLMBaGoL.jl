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

    # ================================================================
    # ClusterStats unit tests
    # ================================================================
    @testset "ClusterStats" begin
        σ = 0.005
        locs = [
            SMLMData.Emitter2DFit(0.1, 0.1, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, 1),
            SMLMData.Emitter2DFit(0.102, 0.098, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, 2),
            SMLMData.Emitter2DFit(0.099, 0.101, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, 3),
        ]

        # Empty cluster
        cs0 = ClusterStats()
        @test cs0.n == 0

        # Add locs one by one
        cs1 = SMLMBaGoL.add_loc(cs0, locs[1])
        @test cs1.n == 1
        cs2 = SMLMBaGoL.add_loc(cs1, locs[2])
        @test cs2.n == 2
        cs3 = SMLMBaGoL.add_loc(cs2, locs[3])
        @test cs3.n == 3

        # Posterior mean should be near precision-weighted centroid
        mx, my = SMLMBaGoL.posterior_mean(cs3)
        @test abs(mx - mean([l.x for l in locs])) < 0.01
        @test abs(my - mean([l.y for l in locs])) < 0.01

        # add/remove are exact inverses
        cs_back = SMLMBaGoL.remove_loc(cs3, locs[3])
        @test cs_back.n == 2
        @test cs_back.Λ_xx ≈ cs2.Λ_xx atol=1e-12
        @test cs_back.η_x ≈ cs2.η_x atol=1e-12
        @test cs_back.quad ≈ cs2.quad atol=1e-12

        # Remove all → back to empty
        cs_empty = SMLMBaGoL.remove_loc(SMLMBaGoL.remove_loc(cs_back, locs[2]), locs[1])
        @test cs_empty.n == 0
        @test abs(cs_empty.Λ_xx) < 1e-10

        # build_cluster_stats matches sequential add
        cs_bulk = SMLMBaGoL.build_cluster_stats(locs, 1:3)
        @test cs_bulk.n == cs3.n
        @test cs_bulk.Λ_xx ≈ cs3.Λ_xx atol=1e-12
        @test cs_bulk.η_x ≈ cs3.η_x atol=1e-12

        # Posterior covariance should be positive definite
        Σ_xx, Σ_xy, Σ_yy = SMLMBaGoL.posterior_cov(cs3)
        @test Σ_xx > 0
        @test Σ_yy > 0
        @test Σ_xx * Σ_yy - Σ_xy^2 > 0

        # Marginal likelihood is finite for non-empty cluster
        log_area = log(0.01)  # 0.1 × 0.1 μm²
        lml = SMLMBaGoL.log_marginal_likelihood(cs3, log_area)
        @test isfinite(lml)

        # Empty cluster has zero marginal likelihood
        @test SMLMBaGoL.log_marginal_likelihood(cs0, log_area) == 0.0

        # Predictive probability
        new_loc = SMLMData.Emitter2DFit(0.101, 0.1, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, 4)
        lp = SMLMBaGoL.log_predictive(cs3, new_loc, log_area)
        @test isfinite(lp)

        # Predictive for empty cluster = -log_area (uniform)
        lp_empty = SMLMBaGoL.log_predictive(cs0, new_loc, log_area)
        @test lp_empty ≈ -log_area
    end

    # ================================================================
    # Collapsed sampler integration test
    # ================================================================
    @testset "Collapsed Sampler - 2 Emitters" begin
        Random.seed!(42)

        locs = SMLMData.Emitter2DFit[]
        σ = 0.005

        # Emitter 1 at (0.1, 0.1) — 5 locs
        for i in 1:5
            x = 0.1 + randn() * σ
            y = 0.1 + randn() * σ
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i))
        end

        # Emitter 2 at (0.2, 0.2) — 5 locs
        for i in 1:5
            x = 0.2 + randn() * σ
            y = 0.2 + randn() * σ
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i+5))
        end

        # Run collapsed chain directly
        count_hist = EmitterCountHist()
        result = run_collapsed_chain(locs;
            n_iterations=2000,
            burn_in=500,
            accumulators=AbstractAccumulator[count_hist],
            verbose=false
        )

        @test result isa CollapsedChainResult
        @test result.state.n_active >= 1
        @test result.n_iterations == 2000

        # Count histogram should exist
        counts = SMLMBaGoL.accumulator_result(count_hist)
        @test length(counts) > 0
        @test sum(counts) > 0  # Should have recorded samples

        # Extract emitters
        emitters = SMLMBaGoL.extract_emitters(result.state, locs)
        @test length(emitters) >= 1
        for e in emitters
            @test e isa SMLMData.Emitter2DFit
            @test e.σ_x >= 0
            @test e.σ_y >= 0
        end
    end

    # ================================================================
    # run_bagol with collapsed sampler (default)
    # ================================================================
    @testset "run_bagol Collapsed Integration" begin
        Random.seed!(123)

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

        camera = SMLMData.IdealCamera(64, 64, 0.1)
        smld = SMLMData.BasicSMLD(locs, camera, 100, 1)

        # Default sampler (collapsed)
        result_smld, diagnostics = run_bagol(smld;
            n_iterations=1000,
            burn_in=200,
            verbose=false
        )

        @test result_smld isa SMLMData.BasicSMLD
        @test diagnostics isa BaGoLDiagnostics
        @test diagnostics.n_emitters >= 0
        @test length(result_smld.emitters) == diagnostics.n_emitters

        if diagnostics.n_emitters > 0
            @test result_smld.emitters[1] isa SMLMData.Emitter2DFit
        end
    end

    # ================================================================
    # Legacy RJMCMC sampler still works
    # ================================================================
    @testset "run_bagol RJMCMC Legacy" begin
        Random.seed!(123)

        locs = SMLMData.Emitter2DFit[]
        σ = 0.005

        for i in 1:5
            x = 0.1 + randn() * σ
            y = 0.1 + randn() * σ
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i))
        end
        for i in 1:5
            x = 0.2 + randn() * σ
            y = 0.2 + randn() * σ
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i+5))
        end

        camera = SMLMData.IdealCamera(64, 64, 0.1)
        smld = SMLMData.BasicSMLD(locs, camera, 100, 1)

        result_smld, diagnostics = run_bagol(smld;
            sampler=:rjmcmc,
            n_iterations=500,
            burn_in=100,
            verbose=false
        )

        @test result_smld isa SMLMData.BasicSMLD
        @test diagnostics isa BaGoLDiagnostics

        # run_bagol_chain still works
        chain = run_bagol_chain(locs;
            n_iterations=500,
            burn_in=100,
            verbose=false
        )
        @test length(chain.samples) > 0

        emitters, posterior_k = estimate_mapn(chain)
        @test length(emitters) >= 0
    end

    # ================================================================
    # Accumulators
    # ================================================================
    @testset "Accumulators" begin
        # EmitterCountHist merge
        h1 = EmitterCountHist()
        h1.counts = [10, 20, 5]
        h2 = EmitterCountHist()
        h2.counts = [5, 15, 10, 3]

        SMLMBaGoL.accumulator_merge!(h1, h2)
        @test h1.counts == [15, 35, 15, 3]

        # NNDistHist basic
        nn = NNDistHist(max_dist=0.1, n_bins=50)
        @test length(nn.counts) == 50
    end

    @testset "Spatial Utilities" begin
        loc = SMLMData.Emitter2DFit(0.1, 0.2, 1000.0, 10.0, 0.005, 0.006, 0.0, 0.0, 0.0, 1, 1, 0, 1)
        coords = SMLMBaGoL.get_coords(loc)
        @test coords[1] ≈ 0.1
        @test coords[2] ≈ 0.2

        sigma = SMLMBaGoL.get_sigma(loc)
        @test sigma[1] ≈ 0.005
        @test sigma[2] ≈ 0.006

        ms = SMLMBaGoL.mean_sigma(loc)
        @test ms ≈ sqrt(0.005 * 0.006)

        loc2 = SMLMData.Emitter2DFit(0.2, 0.2, 1000.0, 10.0, 0.005, 0.005, 0.0, 0.0, 0.0, 1, 1, 0, 2)
        d_eff = SMLMBaGoL.precision_weighted_distance(loc, loc2)
        @test d_eff > 0
    end

    @testset "Partitioning" begin
        Random.seed!(456)

        locs = SMLMData.Emitter2DFit[]
        σ = 0.005

        for i in 1:20
            x = 0.1 + randn() * σ
            y = 0.1 + randn() * σ
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i))
        end

        for i in 1:15
            x = 0.5 + randn() * σ
            y = 0.5 + randn() * σ
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i+20))
        end

        partitions, skipped = partition_locs(locs; nsigma=4.0, min_size=5, max_size=100)

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
    end

    @testset "Oversized Cluster Splitting" begin
        Random.seed!(789)

        locs = SMLMData.Emitter2DFit[]
        σ = 0.005

        for i in 1:50
            x = 0.1 + randn() * 0.02
            y = 0.1 + randn() * 0.02
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i))
        end

        partitions, skipped = partition_locs(locs; nsigma=6.0, min_size=3, max_size=20)

        @test isempty(skipped)
        @test length(partitions) >= 2

        for p in partitions
            @test length(p.locs) <= 20
        end

        total_locs = sum(length(p.locs) for p in partitions)
        @test total_locs == 50

        partitions_skip, skipped_skip = partition_locs(locs; nsigma=6.0, min_size=3, max_size=20, skip_size=20)
        @test length(skipped_skip) >= 1
    end

    @testset "Posterior Image (RJMCMC)" begin
        Random.seed!(555)

        locs = SMLMData.Emitter2DFit[]
        σ = 0.005
        for i in 1:10
            x = 0.1 + randn() * σ
            y = 0.1 + randn() * σ
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i))
        end

        chain = run_bagol_chain(locs; n_iterations=500, burn_in=100, verbose=false)

        post = posterior_image(chain; pixel_size=0.005)
        @test post.image isa Matrix{Int}
        @test post.pixel_size == 0.005
        @test length(post.edges_x) == size(post.image, 1) + 1
        @test sum(post.image) > 0
    end

    @testset "Posterior Image (Collapsed)" begin
        Random.seed!(555)

        locs = SMLMData.Emitter2DFit[]
        σ = 0.005
        for i in 1:10
            x = 0.1 + randn() * σ
            y = 0.1 + randn() * σ
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i))
        end

        camera = SMLMData.IdealCamera(64, 64, 0.1)
        smld = SMLMData.BasicSMLD(locs, camera, 100, 1)

        result_smld, diagnostics = run_bagol(smld;
            n_iterations=1000, burn_in=200, verbose=false,
            posterior_pixel_size=0.005)
        @test diagnostics.posterior_image !== nothing
        @test diagnostics.posterior_image.pixel_size == 0.005
        @test sum(diagnostics.posterior_image.image) > 0

        # Default: no posterior image
        _, diag_default = run_bagol(smld;
            n_iterations=500, burn_in=100, verbose=false)
        @test diag_default.posterior_image === nothing
    end

    @testset "Partitioned BaGoL Collapsed" begin
        Random.seed!(321)

        locs = SMLMData.Emitter2DFit[]
        σ = 0.005

        # Cluster 1
        for i in 1:5
            push!(locs, SMLMData.Emitter2DFit(0.1 + randn()*σ, 0.1 + randn()*σ, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i))
        end
        for i in 1:5
            push!(locs, SMLMData.Emitter2DFit(0.12 + randn()*σ, 0.1 + randn()*σ, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i+5))
        end

        # Cluster 2
        for i in 1:5
            push!(locs, SMLMData.Emitter2DFit(0.5 + randn()*σ, 0.5 + randn()*σ, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i+10))
        end

        camera = SMLMData.IdealCamera(64, 64, 0.1)
        smld = SMLMData.BasicSMLD(locs, camera, 100, 1)

        result_smld, diagnostics = run_bagol(smld;
            nsigma=4.0,
            min_partition_size=3,
            max_partition_size=100,
            sync_interval=100,
            n_iterations=1000,
            burn_in=200,
            verbose=false
        )

        @test result_smld isa SMLMData.BasicSMLD
        @test diagnostics isa BaGoLDiagnostics
        @test diagnostics.n_emitters >= 0
        @test diagnostics.n_partitions >= 1
    end

    # ================================================================
    # Archive
    # ================================================================
    @testset "Archive Write/Read" begin
        Random.seed!(999)

        locs = SMLMData.Emitter2DFit[]
        σ = 0.005
        for i in 1:10
            x = 0.1 + randn() * σ
            y = 0.1 + randn() * σ
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i))
        end

        archive_dir = mktempdir()

        camera = SMLMData.IdealCamera(64, 64, 0.1)
        smld = SMLMData.BasicSMLD(locs, camera, 100, 1)

        # Run with archive
        result_smld, diagnostics = run_bagol(smld;
            n_iterations=500,
            burn_in=100,
            archive_path=archive_dir,
            verbose=false
        )

        @test result_smld isa SMLMData.BasicSMLD

        # Read archive back
        archive = BaGoLArchive(archive_dir)
        @test archive.n_partitions >= 1
        @test archive.sample_counts[1] > 0

        assignments, μs, shapes = SMLMBaGoL.read_partition(archive, 1)
        @test size(assignments, 2) == archive.sample_counts[1]
        @test length(μs) == archive.sample_counts[1]
        @test all(isfinite, μs)
        @test all(isfinite, shapes)
    end
end

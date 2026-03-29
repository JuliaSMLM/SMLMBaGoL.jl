using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions

@testset "SMLMBaGoL.jl Tests" begin
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

        sim = simulate_localizations([(0.1, 0.1), (0.2, 0.2)];
            fixed_sigma=0.005, mean_count=5.0, count_model=:fixed)
        locs = sim.smld.emitters

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
    # run_bagol integration
    # ================================================================
    @testset "run_bagol Integration" begin
        Random.seed!(123)

        sim = simulate_localizations([(0.1, 0.1), (0.2, 0.2)];
            fixed_sigma=0.005, mean_count=5.0, count_model=:fixed,
            field_size=6.4, pixel_size=0.1)
        smld = sim.smld

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

    @testset "Collapsed MAP-N" begin
        Random.seed!(42)

        sim = simulate_localizations([(0.1, 0.1), (0.2, 0.2)];
            fixed_sigma=0.005, mean_count=10.0, count_model=:fixed)
        locs = sim.smld.emitters

        # Run with PartitionSamples accumulator
        ps_acc = PartitionSamples(thin=5)
        result = run_collapsed_chain(locs;
            n_iterations=5000, burn_in=1000,
            accumulators=AbstractAccumulator[ps_acc],
            verbose=false)

        samples = SMLMBaGoL.accumulator_result(ps_acc)
        @test length(samples) > 0

        # MAP-N estimation from collapsed samples
        emitters, posterior_k = estimate_mapn_collapsed(samples, locs)
        @test length(emitters) >= 1
        @test length(posterior_k) > 0
        @test sum(posterior_k) == length(samples)

        # With well-separated emitters, MAP-N should find 2
        @test length(emitters) == 2

        # Positions should be near true emitter locations
        positions = sort([(e.x, e.y) for e in emitters])
        @test abs(positions[1][1] - 0.1) < 0.01
        @test abs(positions[2][1] - 0.2) < 0.01

        # Uncertainties should be positive
        for e in emitters
            @test e.σ_x >= 0
            @test e.σ_y >= 0
        end
    end

    @testset "overlap_hungarian" begin
        # Identity match: same assignments
        ref = Int16[1, 1, 2, 2, 3, 3]
        sample = Int16[1, 1, 2, 2, 3, 3]
        ref_labels = Int16[1, 2, 3]
        sample_labels = Int16[1, 2, 3]
        assignment, total, coverlaps = SMLMBaGoL.overlap_hungarian(ref, sample, ref_labels, sample_labels)
        @test total == 6
        @test coverlaps == [2, 2, 2]
        # Each ref cluster should map to itself
        @test assignment == [1, 2, 3]

        # Permuted labels: ref 1↔sample 3, ref 2↔sample 1, ref 3↔sample 2
        sample_perm = Int16[3, 3, 1, 1, 2, 2]
        sample_labels_perm = Int16[1, 2, 3]
        assignment_p, total_p, coverlaps_p = SMLMBaGoL.overlap_hungarian(ref, sample_perm, ref_labels, sample_labels_perm)
        @test total_p == 6
        @test coverlaps_p == [2, 2, 2]
        # ref cluster 1 (indices 1,2 which have sample label 3) → sample cluster 3
        @test assignment_p[1] == 3
        @test assignment_p[2] == 1
        @test assignment_p[3] == 2

        # K=1 short-circuit
        ref1 = Int16[1, 1, 1]
        sample1 = Int16[1, 1, 1]
        a1, t1, c1 = SMLMBaGoL.overlap_hungarian(ref1, sample1, Int16[1], Int16[1])
        @test a1 == [1]
        @test t1 == 3
        @test c1 == [3]
    end

    @testset "Overlap-Based MAP-N" begin
        Random.seed!(42)

        sim = simulate_localizations([(0.1, 0.1), (0.2, 0.2)];
            fixed_sigma=0.005, mean_count=10.0, count_model=:fixed)
        locs = sim.smld.emitters

        # Run chain with both PartitionSamples and PSMAccumulator
        ps_acc = PartitionSamples(thin=5)
        psm_acc = PSMAccumulator()
        result = run_collapsed_chain(locs;
            n_iterations=5000, burn_in=1000,
            accumulators=AbstractAccumulator[ps_acc, psm_acc],
            verbose=false)

        samples = SMLMBaGoL.accumulator_result(ps_acc)
        psm = SMLMBaGoL.accumulator_result(psm_acc).psm
        @test length(samples) > 0

        # Get Dahl assignments
        _, _, _, dahl_assignments = estimate_dahl(samples, locs, psm)

        # Overlap-based MAP-N
        emitters_overlap, posterior_k = estimate_mapn_overlap(samples, locs, dahl_assignments)

        # Should find 2 emitters
        @test length(emitters_overlap) == 2

        # Positions should be near true emitter locations
        positions = sort([(e.x, e.y) for e in emitters_overlap])
        @test abs(positions[1][1] - 0.1) < 0.01
        @test abs(positions[2][1] - 0.2) < 0.01

        # Uncertainties should be positive
        for e in emitters_overlap
            @test e.σ_x > 0
            @test e.σ_y > 0
        end

        # Compare with estimate_mapn_collapsed — positions should agree closely
        k_dahl = length(unique(dahl_assignments))
        emitters_old, _ = estimate_mapn_collapsed(samples, locs; k_override=k_dahl)
        @test length(emitters_old) == length(emitters_overlap)

        if length(emitters_old) == length(emitters_overlap)
            pos_old = sort([(e.x, e.y) for e in emitters_old])
            pos_new = sort([(e.x, e.y) for e in emitters_overlap])
            for i in eachindex(pos_old)
                @test abs(pos_old[i][1] - pos_new[i][1]) < 0.005
                @test abs(pos_old[i][2] - pos_new[i][2]) < 0.005
            end
        end
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

        sim1 = simulate_localizations([(0.1, 0.1)];
            fixed_sigma=0.005, mean_count=20.0, count_model=:fixed)
        sim2 = simulate_localizations([(0.5, 0.5)];
            fixed_sigma=0.005, mean_count=15.0, count_model=:fixed)
        locs = vcat(sim1.smld.emitters, sim2.smld.emitters)

        partitions, skipped = partition_locs(locs; partition_sigma=4.0, min_size=5, max_size=100)

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

        partitions, skipped = partition_locs(locs; partition_sigma=6.0, min_size=3, max_size=20)

        @test isempty(skipped)
        @test length(partitions) >= 2

        for p in partitions
            @test length(p.locs) <= 20
        end

        total_locs = sum(length(p.locs) for p in partitions)
        @test total_locs == 50

        partitions_skip, skipped_skip = partition_locs(locs; partition_sigma=6.0, min_size=3, max_size=20, skip_size=20)
        @test length(skipped_skip) >= 1
    end

    @testset "Posterior Image (Collapsed)" begin
        Random.seed!(555)

        sim = simulate_localizations([(0.1, 0.1)];
            fixed_sigma=0.005, mean_count=10.0, count_model=:fixed,
            field_size=6.4, pixel_size=0.1)
        smld = sim.smld

        result_smld, diagnostics = run_bagol(smld;
            n_iterations=1000, burn_in=200, verbose=false,
            posterior_pixel_size=0.005)
        @test diagnostics.posterior_image !== nothing
        @test diagnostics.posterior_image.pixel_size == 0.005
        @test sum(diagnostics.posterior_image.image) > 0

        # Disable posterior image explicitly
        _, diag_default = run_bagol(smld;
            n_iterations=500, burn_in=100, verbose=false,
            posterior_pixel_size=0.0)
        @test diag_default.posterior_image === nothing
    end

    @testset "Partitioned BaGoL" begin
        Random.seed!(321)

        sim = simulate_localizations([(0.1, 0.1), (0.12, 0.1), (0.5, 0.5)];
            fixed_sigma=0.005, mean_count=5.0, count_model=:fixed,
            field_size=6.4, pixel_size=0.1)
        smld = sim.smld

        result_smld, diagnostics = run_bagol(smld;
            partition_sigma=4.0,
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

        sim = simulate_localizations([(0.1, 0.1)];
            fixed_sigma=0.005, mean_count=10.0, count_model=:fixed,
            field_size=6.4, pixel_size=0.1)
        smld = sim.smld

        archive_dir = mktempdir()

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

    # ================================================================
    # Count-Model Optimality with Hierarchical Learning
    # ================================================================
    # Spatially separated emitters with hierarchical μ/shape learning.
    # Hier must learn (μ, α) from cluster counts during the chain.
    # Spatial separation resolves K; hier calibrates the count model.
    # Pass: observed MAP-K accuracy ≥ 80% of oracle (known μ, α).
    @testset "Count-Model Optimality (Hierarchical)" begin
        Random.seed!(2024)

        μ_true = 20.0
        α_true = 5.0
        σ = 0.005  # Localization precision (μm)
        separation = 0.05  # 10σ between emitters — well resolved

        # Oracle MAP-K accuracy with known (μ, α) — pure count model, no K prior
        function oracle_accuracy(K_true; n_mc=20000)
            p = α_true / (α_true + μ_true)
            dist_N = NegativeBinomial(K_true * α_true, p)
            n_correct = 0
            n_valid = 0
            K_max = max(3 * K_true, 15)
            for _ in 1:n_mc
                N = rand(dist_N)
                N < K_true && continue
                n_valid += 1
                best_K = 1
                best_lp = -Inf
                for k in 1:K_max
                    lp = SMLMBaGoL._log_count_posterior(k, N, α_true, μ_true)
                    if lp > best_lp
                        best_lp = lp
                        best_K = k
                    end
                end
                n_correct += (best_K == K_true)
            end
            return n_correct / n_valid
        end

        for K_true in [1, 2, 4]
            @testset "K=$K_true separated, hier on" begin
                p_theory = oracle_accuracy(K_true)

                n_trials = 25
                n_correct = 0

                for _ in 1:n_trials
                    # Generate well-separated emitters on a line
                    locs = SMLMData.Emitter2DFit[]
                    p = α_true / (α_true + μ_true)
                    id = 0
                    for k in 1:K_true
                        cx = 0.1 + (k - 1) * separation
                        cy = 0.1
                        n_k = max(rand(NegativeBinomial(α_true, p)), 1)
                        for _ in 1:n_k
                            id += 1
                            x = cx + randn() * σ
                            y = cy + randn() * σ
                            push!(locs, SMLMData.Emitter2DFit(
                                x, y, 1000.0, 10.0, σ, σ,
                                0.0, 0.0, 0.0, 1, 1, 0, id))
                        end
                    end

                    ps_acc = PartitionSamples(thin=5)
                    result = run_collapsed_chain(locs;
                        n_iterations=4000, burn_in=800,
                        shape=2.0, learn_distribution=true,
                        accumulators=AbstractAccumulator[ps_acc],
                        verbose=false)

                    samples = SMLMBaGoL.accumulator_result(ps_acc)
                    emitters, _ = estimate_mapn_collapsed(samples, locs)
                    n_correct += (length(emitters) == K_true)
                end

                observed = n_correct / n_trials
                threshold = 0.80 * p_theory
                @test observed >= threshold
                println("  K=$K_true: $(n_correct)/$(n_trials) = $(round(observed*100, digits=1))% " *
                        "(theory=$(round(p_theory*100, digits=1))%, threshold=$(round(threshold*100, digits=1))%)")
            end
        end
    end

    # ================================================================
    # Diagnostics module tests
    # ================================================================
    @testset "Diagnostics" begin

        @testset "Enumeration" begin
            # Bell number B₃ = 5
            parts_3 = enumerate_canonical_partitions(3, 3)
            @test length(parts_3) == 5

            # S(4,1) + S(4,2) = 1 + 7 = 8
            parts_4_2 = enumerate_canonical_partitions(4, 2)
            @test length(parts_4_2) == 8

            # All partitions should be canonical (labels in first-appearance order)
            for z in parts_3
                @test z == canonicalize(z)
            end

            # Canonicalize tests
            @test canonicalize([3, 3, 1, 1]) == [1, 1, 2, 2]
            @test canonicalize([2, 1, 2, 1]) == [1, 2, 1, 2]
            @test canonicalize([1, 1, 1]) == [1, 1, 1]
            @test canonicalize([5, 5, 3, 3, 5]) == [1, 1, 2, 2, 1]
        end

        @testset "Partition Metrics" begin
            # VI of identical partitions = 0
            z = [1, 1, 2, 2, 3]
            vi = variation_of_information(z, z)
            @test vi.vi_total ≈ 0.0
            @test vi.overseg ≈ 0.0
            @test vi.underseg ≈ 0.0

            # VI is positive for different partitions
            z1 = [1, 1, 2, 2]
            z2 = [1, 2, 1, 2]
            vi = variation_of_information(z1, z2)
            @test vi.vi_total > 0

            # VI total is symmetric
            vi_rev = variation_of_information(z2, z1)
            @test vi.vi_total ≈ vi_rev.vi_total

            # Overseg/underseg swap when arguments swap
            @test vi.overseg ≈ vi_rev.underseg
            @test vi.underseg ≈ vi_rev.overseg

            # EPL with all-identical samples = 0
            candidate = [1, 1, 2, 2]
            samples = [candidate, candidate, candidate]
            epl = expected_posterior_loss(candidate, samples)
            @test epl ≈ 0.0
        end

        @testset "Chain Diagnostics" begin
            # ESS of iid samples should be ≈ n
            Random.seed!(42)
            iid = randn(1000)
            ess_iid = effective_sample_size(iid)
            @test ess_iid > 500  # Should be close to 1000

            # ESS of constant trace = effective 1
            constant = fill(5.0, 100)
            ess_const = effective_sample_size(constant)
            @test ess_const ≤ 100  # Degenerate — var is 0

            # Autocorrelation at lag 0 = 1.0
            acf = autocorrelation(iid; max_lag=10)
            @test acf[1] ≈ 1.0

            # ACF of iid should be small at lag > 0
            @test all(abs.(acf[2:end]) .< 0.1)

            # R-hat of identical chains ≈ 1.0
            chain1 = randn(500)
            rhat_val = gelman_rubin([chain1, copy(chain1)])
            @test rhat_val ≈ 1.0 atol=0.01

            # R-hat of very different chains > 1
            chain_a = randn(500)
            chain_b = randn(500) .+ 10.0
            rhat_diff = gelman_rubin([chain_a, chain_b])
            @test rhat_diff > 1.5
        end

        @testset "Count Model" begin
            # Posterior should sum to 1
            ks, probs = count_model_posterior(20, 10.0, 2.0)
            @test sum(probs) ≈ 1.0 atol=1e-10

            # MAP should be near N/μ
            map_idx = argmax(probs)
            @test ks[map_idx] == 2  # 20 locs / 10 mean = 2

            # Recovery rate for trivial case (K=1, lots of data) should be high
            rate = qpaint_recovery_rate(1, 5.0, 2.0; n_mc=10_000)
            @test rate > 0.5

            # Confusion matrix rows should sum to 1
            C = count_model_confusion_matrix(4, 10.0, 2.0; n_mc=10_000)
            for k in 1:4
                @test sum(C[k, :]) ≈ 1.0 atol=0.01
            end

            # Theoretical bound should be in [0, 1]
            bound = theoretical_accuracy_bound(2, 10.0, 2.0)
            @test 0.0 < bound < 1.0
        end

        @testset "Target Density" begin
            σ = 0.005
            locs = [
                SMLMData.Emitter2DFit(0.1, 0.1, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, 1),
                SMLMData.Emitter2DFit(0.12, 0.1, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, 2),
                SMLMData.Emitter2DFit(0.1, 0.12, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, 3),
                SMLMData.Emitter2DFit(0.12, 0.12, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, 4),
            ]

            z_all_one = [1, 1, 1, 1]
            z_two = [1, 1, 2, 2]

            # DecoupledTarget gives finite values
            td = DecoupledTarget()
            lt1 = evaluate_target(td, z_all_one, locs; μ=10.0, shape=2.0)
            lt2 = evaluate_target(td, z_two, locs; μ=10.0, shape=2.0)
            @test isfinite(lt1)
            @test isfinite(lt2)

            # MFMTarget gives finite values
            td_mfm = MFMTarget()
            lt1_mfm = evaluate_target(td_mfm, z_all_one, locs; μ=10.0, shape=2.0)
            lt2_mfm = evaluate_target(td_mfm, z_two, locs; μ=10.0, shape=2.0)
            @test isfinite(lt1_mfm)
            @test isfinite(lt2_mfm)

            # For K=1, MFM partition prior = log(1) = 0, so targets are equal
            @test lt1_mfm ≈ lt1
            # For K=2, MFM partition prior ≠ 0, so targets differ
            @test lt2_mfm != lt2

            # UniformPriorTarget gives finite values
            td_unif = UniformPriorTarget(log(0.01))
            lt1_unif = evaluate_target(td_unif, z_all_one, locs; μ=10.0, shape=2.0)
            @test isfinite(lt1_unif)

            # Relabeling should not change target density
            z_a = [1, 2, 1, 2]
            z_b = [2, 1, 2, 1]  # Same partition, different labels
            lt_a = evaluate_target(td, canonicalize(z_a), locs; μ=10.0, shape=2.0)
            lt_b = evaluate_target(td, canonicalize(z_b), locs; μ=10.0, shape=2.0)
            @test lt_a ≈ lt_b atol=1e-10

            # Exact posterior should sum to 1
            parts = enumerate_canonical_partitions(4, 3)
            probs, log_targets = exact_posterior(parts, locs, td; μ=10.0, shape=2.0)
            @test sum(probs) ≈ 1.0 atol=1e-10
            @test all(isfinite, log_targets)
        end

    end
end

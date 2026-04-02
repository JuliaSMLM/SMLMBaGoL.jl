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

        @testset "Finite State" begin
            # Bell number B₃ = 5
            parts_3 = enumerate_canonical_partitions(3, 3)
            @test length(parts_3) == 5

            # S(4,1) + S(4,2) = 1 + 7 = 8
            parts_4_2 = enumerate_canonical_partitions(4, 2)
            @test length(parts_4_2) == 8

            # Canonical partitions
            for z in parts_3
                @test z == canonicalize(z)
            end
            @test canonicalize([3, 3, 1, 1]) == [1, 1, 2, 2]
            @test canonicalize([2, 1, 2, 1]) == [1, 2, 1, 2]
            @test canonicalize([1, 1, 1]) == [1, 1, 1]
            @test canonicalize([5, 5, 3, 3, 5]) == [1, 1, 2, 2, 1]
        end

        @testset "Mixing" begin
            Random.seed!(42)
            iid = randn(1000)
            ess_iid = effective_sample_size(iid)
            @test ess_iid > 500

            constant = fill(5.0, 100)
            ess_const = effective_sample_size(constant)
            @test ess_const ≤ 100

            acf = autocorrelation(iid; max_lag=10)
            @test acf[1] ≈ 1.0
            @test all(abs.(acf[2:end]) .< 0.1)

            # Split R-hat of identical chains ≈ 1.0
            chain1 = randn(500)
            rhat_val = split_gelman_rubin([chain1, copy(chain1)])
            @test rhat_val ≈ 1.0 atol=0.05

            # Split R-hat of very different chains > 1
            chain_a = randn(500)
            chain_b = randn(500) .+ 10.0
            rhat_diff = split_gelman_rubin([chain_a, chain_b])
            @test rhat_diff > 1.5
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

            td = DecoupledTarget()
            lt1 = evaluate_target(td, z_all_one, locs; μ=10.0, shape=2.0)
            lt2 = evaluate_target(td, z_two, locs; μ=10.0, shape=2.0)
            @test isfinite(lt1)
            @test isfinite(lt2)

            # Relabeling should not change target density
            z_a = [1, 2, 1, 2]
            z_b = [2, 1, 2, 1]
            lt_a = evaluate_target(td, canonicalize(z_a), locs; μ=10.0, shape=2.0)
            lt_b = evaluate_target(td, canonicalize(z_b), locs; μ=10.0, shape=2.0)
            @test lt_a ≈ lt_b atol=1e-10

            # Exact posterior should sum to 1
            parts = enumerate_canonical_partitions(4, 3)
            probs, log_targets = exact_posterior(parts, locs, td; μ=10.0, shape=2.0)
            @test sum(probs) ≈ 1.0 atol=1e-10
            @test all(isfinite, log_targets)

            probs_none, log_targets_none = exact_posterior(
                parts, locs, td; μ=10.0, shape=2.0, label_multiplicity=:none)
            @test sum(probs_none) ≈ 1.0 atol=1e-10
            for (i, z) in enumerate(parts)
                K = maximum(z)
                @test log_targets[i] ≈ log_targets_none[i] + SMLMBaGoL.logfactorial(K) atol=1e-10
            end

            # The collapsed DM form must match the direct NegBin-assignment form
            # exactly for every labeled partition.
            td_dm = DMFlatTarget()
            td_direct = DirectNegBinFlatTarget()
            for z in parts
                lt_dm = evaluate_target(td_dm, z, locs; μ=10.0, shape=2.0, ρ=2.0)
                lt_direct = evaluate_target(td_direct, z, locs; μ=10.0, shape=2.0, ρ=2.0)
                @test lt_dm ≈ lt_direct atol=1e-10
            end

            probs_dm, _ = exact_posterior(parts, locs, td_dm; μ=10.0, shape=2.0, ρ=2.0)
            probs_direct, _ = exact_posterior(parts, locs, td_direct; μ=10.0, shape=2.0, ρ=2.0)
            @test all(isapprox.(probs_dm, probs_direct; atol=1e-12))
        end

    end
end

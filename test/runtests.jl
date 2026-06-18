using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions

@testset "SMLMBaGoL.jl Tests" begin
    # ================================================================
    # Empty / tiny input — must not crash (graceful empty result)
    # Regression: run_bagol returned abstract-eltype Vector{Emitter2DFit},
    # so a downstream reduction over an empty result hit zero(Type{Any}).
    # ================================================================
    @testset "Empty and tiny input" begin
        cam = SMLMData.IdealCamera(1:64, 1:64, 0.1)
        mksmld(n) = SMLMData.BasicSMLD(
            [SMLMData.Emitter2DFit(0.5 + 0.02i, 0.5, 1000.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 1, 1, 0, i) for i in 1:n],
            cam, 1, 1, Dict{String,Any}())

        # 0 localizations: graceful empty result, not a crash
        r0, d0 = run_bagol(mksmld(0); n_iterations=100, burn_in=20, verbose=false)
        @test length(r0.emitters) == 0
        @test d0.n_emitters == 0
        # Concrete eltype so downstream reductions don't fall to zero(Type{Any})
        @test isconcretetype(eltype(r0.emitters))
        @test eltype(r0.emitters) == SMLMData.Emitter2DFit{Float64}
        @test sum([e.x for e in r0.emitters]) == 0.0   # the exact crashing pattern

        # 1 and 2 localizations: run without crashing
        for n in (1, 2)
            r, d = run_bagol(mksmld(n); n_iterations=100, burn_in=20, verbose=false)
            @test isconcretetype(eltype(r.emitters))
            @test d.n_emitters >= 0
        end
    end

    # ================================================================
    # se_adjust — independent-error σ correction (quadrature, per-axis, guarded)
    # ================================================================
    @testset "se_adjust uncertainty correction" begin
        cam = SMLMData.IdealCamera(1:64, 1:64, 0.1)
        mk(x, y, sx, sy) = SMLMData.Emitter2DFit(x, y, 1000.0, 0.0, sx, sy, 0.0, 0.0, 0.0, 1, 1, 0, 1)

        # _resolve_tau: scalar / per-axis tuple / per-loc vector / per-loc per-axis
        τx, τy = SMLMBaGoL._resolve_tau(0.01, 3)
        @test τx == fill(0.01, 3) && τy == fill(0.01, 3)
        τx, τy = SMLMBaGoL._resolve_tau((0.0047, 0.0076), 4)
        @test τx == fill(0.0047, 4) && τy == fill(0.0076, 4)
        τx, τy = SMLMBaGoL._resolve_tau([0.001, 0.002], 2)
        @test τx == [0.001, 0.002] && τy == [0.001, 0.002]
        τx, τy = SMLMBaGoL._resolve_tau(([0.001, 0.002], [0.003, 0.004]), 2)
        @test τx == [0.001, 0.002] && τy == [0.003, 0.004]
        @test_throws ArgumentError SMLMBaGoL._resolve_tau([0.1, 0.2, 0.3], 2)  # wrong length
        @test_throws ArgumentError SMLMBaGoL._resolve_tau(-0.01, 2)            # negative τ

        # _inflate_sigma: exact quadrature, per-axis; σ_xy + position untouched
        e = mk(1.0, 2.0, 0.003, 0.004)
        e2 = SMLMBaGoL._inflate_sigma(e, 0.004, 0.003)
        @test e2 isa SMLMData.Emitter2DFit{Float64}
        @test e2.σ_x ≈ sqrt(0.003^2 + 0.004^2)
        @test e2.σ_y ≈ sqrt(0.004^2 + 0.003^2)
        @test e2.σ_xy == e.σ_xy
        @test e2.x == e.x && e2.y == e.y

        # apply-once guard via metadata stamp (2nd return = applied (τx,τy) or nothing)
        locs = [mk(1.0, 1.0, 0.005, 0.005)]
        stamped = Dict{String,Any}("sigma_corrected" => true, "tau_x_nm" => 3.0)
        out, tau, _ = SMLMBaGoL._maybe_apply_se_adjust(locs, stamped, 0.004, false)
        @test tau === nothing && out === locs                  # skipped (guard)
        out, tau, _ = SMLMBaGoL._maybe_apply_se_adjust(locs, stamped, 0.004, true)
        @test tau == (0.004, 0.004)                            # force override
        @test out[1].σ_x ≈ sqrt(0.005^2 + 0.004^2)
        out, tau, _ = SMLMBaGoL._maybe_apply_se_adjust(locs, Dict{String,Any}(), 0.004, false)
        @test tau == (0.004, 0.004)                            # no stamp → applies
        out, tau, _ = SMLMBaGoL._maybe_apply_se_adjust(locs, Dict{String,Any}(), 0.0, false)
        @test tau === nothing && out === locs                  # zero τ → no-op

        # apply_se_adjust (public SMLD->SMLD): inflates σ, preserves count; guard + no-op
        smld_raw = SMLMData.BasicSMLD([mk(1.0,1.0,0.005,0.006), mk(1.1,1.0,0.005,0.006)], cam, 1, 1, Dict{String,Any}())
        smld_inf = apply_se_adjust(smld_raw, (0.004, 0.003))
        @test length(smld_inf.emitters) == 2
        @test smld_inf.emitters[1].σ_x ≈ sqrt(0.005^2 + 0.004^2)
        @test smld_inf.emitters[1].σ_y ≈ sqrt(0.006^2 + 0.003^2)
        @test apply_se_adjust(smld_raw, 0.0) === smld_raw       # no-op returns same object
        smld_stamp = SMLMData.BasicSMLD(smld_raw.emitters, cam, 1, 1, Dict{String,Any}("sigma_corrected"=>true))
        @test apply_se_adjust(smld_stamp, 0.004) === smld_stamp # guard: no-op on corrected

        # end-to-end: run_bagol with se_adjust runs, returns concrete SMLD + records τ in Info
        elist = [SMLMData.Emitter2DFit(0.5 + 0.02i, 0.5, 1000.0, 0.0, 0.005, 0.005, 0.0, 0.0, 0.0, 1, 1, 0, i) for i in 1:6]
        smld = SMLMData.BasicSMLD(elist, cam, 1, 1, Dict{String,Any}())
        r, d = run_bagol(smld; se_adjust=(0.004, 0.005), n_iterations=200, burn_in=40, verbose=false)
        @test r isa SMLMData.BasicSMLD
        @test isconcretetype(eltype(r.emitters))
        @test d.se_adjust == (0.004, 0.005)                    # applied τ recorded in diagnostics (Info)
        r0, d0 = run_bagol(smld; n_iterations=200, burn_in=40, verbose=false)
        @test d0.se_adjust === nothing                         # default se_adjust=0 → not recorded
        # already-corrected SMLD: se_adjust skipped (warns); τ not recorded
        smld_c = SMLMData.BasicSMLD(elist, cam, 1, 1, Dict{String,Any}("sigma_corrected" => true))
        rc, dc = run_bagol(smld_c; se_adjust=0.004, n_iterations=200, burn_in=40, verbose=false)
        @test rc isa SMLMData.BasicSMLD
        @test dc.se_adjust === nothing
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
        @test cs_back.Λ[1,1] ≈ cs2.Λ[1,1] atol=1e-12
        @test cs_back.η[1] ≈ cs2.η[1] atol=1e-12
        @test cs_back.quad ≈ cs2.quad atol=1e-12

        # Remove all → back to empty
        cs_empty = SMLMBaGoL.remove_loc(SMLMBaGoL.remove_loc(cs_back, locs[2]), locs[1])
        @test cs_empty.n == 0
        @test abs(cs_empty.Λ[1,1]) < 1e-10

        # build_cluster_stats matches sequential add
        cs_bulk = SMLMBaGoL.build_cluster_stats(locs, 1:3)
        @test cs_bulk.n == cs3.n
        @test cs_bulk.Λ[1,1] ≈ cs3.Λ[1,1] atol=1e-12
        @test cs_bulk.η[1] ≈ cs3.η[1] atol=1e-12

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
    # 3D feature (Emitter3DFit) — dimension-parametric core + round-trip
    # ================================================================
    @testset "3D feature (Emitter3DFit)" begin
        mk3(x, y, z, i) = SMLMData.Emitter3DFit(x, y, z, 1000.0, 0.0, 0.01, 0.01, 0.02,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 0, i)
        # 3D conjugate core: posterior mean recovers the precision-weighted centroid
        Random.seed!(11)
        ctr = (1.0, 2.0, 0.5)
        locs = [mk3(ctr[1]+0.01randn(), ctr[2]+0.01randn(), ctr[3]+0.02randn(), i) for i in 1:8]
        cs = foldl((c, loc) -> SMLMBaGoL.add_loc(c, SMLMBaGoL._loc_precision(loc)), locs;
                   init = zero(SMLMBaGoL.ClusterStats{3,9}))
        @test cs.n == 8
        m = SMLMBaGoL.posterior_mean(cs)
        @test length(m) == 3
        @test isapprox(m[1], sum(l.x for l in locs)/8; atol=1e-9)
        @test isapprox(m[3], sum(l.z for l in locs)/8; atol=1e-9)
        @test isfinite(SMLMBaGoL.log_marginal_likelihood(cs, log(1.0)))
        # add/remove exact inverse at D=3
        lp = SMLMBaGoL._loc_precision(locs[1])
        cs_back = SMLMBaGoL.remove_loc(SMLMBaGoL.add_loc(cs, lp), lp)
        @test isapprox(cs_back.Λ, cs.Λ; atol=1e-9)
        @test isbitstype(SMLMBaGoL.ClusterStats{3,9})

        # 3D round-trip: two emitters at the SAME (x,y), separated only in z.
        # (A 2D feature could never distinguish these.)
        Random.seed!(11)
        σxy, σz = 0.01, 0.02
        locs3 = SMLMData.Emitter3DFit{Float64}[]
        for cz in (0.2, 0.6), _ in 1:12
            push!(locs3, mk3(1.0+σxy*randn(), 1.0+σxy*randn(), cz+σz*randn(), length(locs3)+1))
        end
        r = run_collapsed_chain(locs3; n_iterations=8000, burn_in=2000,
                                spatial_model=:flat, learn_distribution=false, shape=10.0)
        em = SMLMBaGoL.extract_emitters(r.state, locs3)
        @test eltype(em) == SMLMData.Emitter3DFit{Float64}
        @test length(em) == 2
        zs = sort([e.z for e in em])
        @test isapprox(zs[1], 0.2; atol=0.03)
        @test isapprox(zs[2], 0.6; atol=0.03)
    end

    # ================================================================
    # Multi-cue (position + spectral) — block-diagonal composite
    # ================================================================
    @testset "Multi-cue (position + spectral)" begin
        Random.seed!(3)
        σxy, σλ = 0.01, 5.0
        positions = SMLMData.Emitter2DFit{Float64}[]
        values = Float64[]
        for λ_true in (580.0, 620.0), _ in 1:12
            push!(positions, SMLMData.Emitter2DFit(1.0+σxy*randn(), 1.0+σxy*randn(),
                  1000.0, 0.0, σxy, σxy, 0.0, 0.0, 0.0, 1, 1, 0, length(positions)+1))
            push!(values, λ_true + σλ*randn())
        end
        σ_values = fill(σλ, length(values))
        fs = FeatureSet((SMLMBaGoL.FlatSpatial(log(0.01)), SMLMBaGoL.FlatSpatial(log(200.0))))

        # Block-diagonal marginal = sum of per-feature marginals
        mp = build_multicue_precisions(positions, values, σ_values)
        mcs = foldl((c, p) -> SMLMBaGoL.add_loc(c, p), mp; init = SMLMBaGoL.empty_cluster(mp))
        @test mcs.n == 24
        @test isbitstype(typeof(mcs))
        @test SMLMBaGoL.spatial_ml(mcs, fs) ≈
              SMLMBaGoL.spatial_ml(mcs.blocks[1], fs.priors[1]) +
              SMLMBaGoL.spatial_ml(mcs.blocks[2], fs.priors[2])

        # Joint grouping: the spectral cue separates spatially-identical emitters
        state = run_multicue_chain(positions, values, σ_values, fs;
                                   n_iterations=8000, burn_in=2000, μ=12.0, shape=10.0)
        em = extract_multicue(state)
        @test length(em) == 2
        λs = sort([e.value for e in em])
        @test isapprox(λs[1], 580.0; atol=8.0)
        @test isapprox(λs[2], 620.0; atol=8.0)
        @test all(isapprox(e.x, 1.0; atol=0.02) for e in em)  # positions identical
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

            # Locmix targets should also normalize and the DM decomposition should
            # match the direct NegBin assignment form, without a Poisson K prior.
            td_locmix = DMLocmixTarget()
            td_direct_locmix = DirectNegBinLocmixTarget()
            for z in parts
                lt_dm = evaluate_target(td_locmix, z, locs; μ=10.0, shape=2.0, ρ=2.0)
                lt_direct = evaluate_target(td_direct_locmix, z, locs; μ=10.0, shape=2.0, ρ=2.0)
                @test isfinite(lt_dm)
                @test lt_dm ≈ lt_direct atol=1e-10
            end

            probs_locmix, _ = exact_posterior(parts, locs, td_locmix; μ=10.0, shape=2.0, ρ=2.0)
            probs_direct_locmix, _ = exact_posterior(parts, locs, td_direct_locmix; μ=10.0, shape=2.0, ρ=2.0)
            @test sum(probs_locmix) ≈ 1.0 atol=1e-10
            @test all(isapprox.(probs_locmix, probs_direct_locmix; atol=1e-12))

            @test SMLMBaGoL._diagnostic_sampler_kwargs(td_dm).spatial_model == :flat
            @test SMLMBaGoL._diagnostic_sampler_kwargs(td_locmix).spatial_model == :locmix
            @test SMLMBaGoL._diagnostic_sampler_kwargs(DecoupledTarget()).allocation_model == :decoupled
            @test SMLMBaGoL._diagnostic_sampler_kwargs(DecoupledLocmixTarget()).allocation_model == :decoupled

            # FazelFlatTarget: count × K^-N × ∏ ML_flat, NO K prior
            td_fazel = FazelFlatTarget()
            for z in parts
                lt_fazel = evaluate_target(td_fazel, z, locs; μ=10.0, shape=2.0, ρ=2.0)
                @test isfinite(lt_fazel)
                K = maximum(z); N = length(z)
                # Verify the log target differs from DMFlatTarget by exactly
                # (-N log K) - log_dm_term - log_poisson_k_prior, which the
                # helper algebra proves indirectly. Here we just check
                # ρ-invariance: no Poisson(ρA) dependence on ρ.
                lt_fazel_ρ5 = evaluate_target(td_fazel, z, locs; μ=10.0, shape=2.0, ρ=5.0)
                @test lt_fazel ≈ lt_fazel_ρ5 atol=1e-10
            end
            probs_fazel, _ = exact_posterior(parts, locs, td_fazel; μ=10.0, shape=2.0, ρ=2.0)
            @test sum(probs_fazel) ≈ 1.0 atol=1e-10
            kw_fazel = SMLMBaGoL._diagnostic_sampler_kwargs(td_fazel)
            @test kw_fazel.spatial_model == :flat
            @test kw_fazel.allocation_model == :categorical
            @test kw_fazel.k_prior == :none
        end

    end

    @testset "Locmix vs Flat Target Routing" begin
        # P0 fix verification: the sampler must route through spatial_ml /
        # spatial_pred dispatch on state.spatial, not hardcoded flat calls;
        # and the Poisson K prior + ρ hier update must be gated off under
        # LocmixSpatial per docs/math_reference.md.

        @testset "_uses_poisson_k_prior dispatch" begin
            # Flat uses the K prior + ρ update
            @test SMLMBaGoL._uses_poisson_k_prior(FlatSpatial(log(1.0)))
            # Locmix does not
            sim = simulate_localizations([(0.0, 0.0), (0.05, 0.0)];
                count_model=:fixed, mean_count=5.0,
                fixed_sigma=0.005)
            loc_precs = SMLMBaGoL.precompute_loc_precisions(sim.smld.emitters)
            grid = SMLMBaGoL.build_locmix_grid(loc_precs)
            @test !SMLMBaGoL._uses_poisson_k_prior(LocmixSpatial(grid))
        end

        @testset "_total_spatial_lml dispatches on spatial model" begin
            sim = simulate_localizations([(0.0, 0.0)];
                count_model=:fixed, mean_count=8.0,
                fixed_sigma=0.005)
            locs = sim.smld.emitters

            # Flat state
            state_flat = SMLMBaGoL.initialize_collapsed_state(locs,
                FlatSpatial(log(SMLMBaGoL.area(UniformSpatialPrior(locs)))))
            lml_flat = SMLMBaGoL._total_spatial_lml(state_flat)

            # Locmix state
            state_lm = SMLMBaGoL.initialize_collapsed_state(locs, LocmixSpatial(locs))
            lml_lm = SMLMBaGoL._total_spatial_lml(state_lm)

            # Flat and locmix produce different values for same allocation —
            # this fails if both paths are secretly using the flat formula.
            @test isfinite(lml_flat) && isfinite(lml_lm)
            @test lml_flat != lml_lm
        end

        @testset "Flat path still runs and converges" begin
            sim = simulate_localizations([(0.0, 0.0), (0.05, 0.0)];
                count_model=:fixed, mean_count=10.0,
                fixed_sigma=0.005)
            result = run_collapsed_chain(sim.smld.emitters;
                spatial_model=:flat, allocation_model=:dm,
                n_iterations=1000, burn_in=500,
                learn_distribution=false, shape=2.0,
                μ_prior_shape=2.0, μ_prior_scale=2.5,
                verbose=false)
            @test result.state.n_active >= 1
        end

        @testset "Locmix path runs and converges" begin
            sim = simulate_localizations([(0.0, 0.0), (0.05, 0.0)];
                count_model=:fixed, mean_count=10.0,
                fixed_sigma=0.005)
            result = run_collapsed_chain(sim.smld.emitters;
                spatial_model=:locmix, allocation_model=:dm,
                n_iterations=1000, burn_in=500,
                learn_distribution=false, shape=2.0,
                μ_prior_shape=2.0, μ_prior_scale=2.5,
                verbose=false)
            @test result.state.n_active >= 1
            @test isa(result.state.spatial, LocmixSpatial)
        end
    end

    @testset "Fazel Target Configuration" begin
        # Verify the public API for the Fazel-equivalent target:
        # allocation_model=:categorical + k_prior=:none.

        @testset "CategoricalAllocation K^-N partition prior" begin
            sim = simulate_localizations([(0.0, 0.0), (0.05, 0.0)];
                count_model=:fixed, mean_count=6.0, fixed_sigma=0.005)
            locs = sim.smld.emitters
            N = length(locs)
            # Build states at K=1 and K=2 manually to verify partition_prior
            sp = FlatSpatial(log(SMLMBaGoL.area(UniformSpatialPrior(locs))))
            am = CategoricalAllocation()
            state_k1 = SMLMBaGoL.initialize_from_assignments(fill(1, N), locs;
                sp=sp, am=am)
            state_k2 = SMLMBaGoL.initialize_from_assignments(
                [i <= N÷2 ? 1 : 2 for i in 1:N], locs; sp=sp, am=am)
            pp_k1 = SMLMBaGoL.partition_prior(am, state_k1, N, 2.0)
            pp_k2 = SMLMBaGoL.partition_prior(am, state_k2, N, 2.0)
            # partition_prior = -N·log(K) — K=1 gives 0, K=2 gives -N·log(2)
            @test pp_k1 ≈ 0.0 atol=1e-10
            @test pp_k2 ≈ -N * log(2.0) atol=1e-10
        end

        @testset "k_prior=:none disables Poisson K prior on flat" begin
            sim = simulate_localizations([(0.0, 0.0)];
                count_model=:fixed, mean_count=5.0, fixed_sigma=0.005)
            locs = sim.smld.emitters
            result_default = run_collapsed_chain(locs;
                spatial_model=:flat, allocation_model=:dm,
                n_iterations=200, burn_in=100, learn_distribution=false,
                shape=2.0, μ_prior_shape=2.0, μ_prior_scale=2.5, verbose=false)
            @test result_default.state.use_poisson_k_prior == true

            result_none = run_collapsed_chain(locs;
                spatial_model=:flat, allocation_model=:dm, k_prior=:none,
                n_iterations=200, burn_in=100, learn_distribution=false,
                shape=2.0, μ_prior_shape=2.0, μ_prior_scale=2.5, verbose=false)
            @test result_none.state.use_poisson_k_prior == false
        end

        @testset "k_prior=:poisson rejected outside :flat" begin
            sim = simulate_localizations([(0.0, 0.0)];
                count_model=:fixed, mean_count=5.0, fixed_sigma=0.005)
            locs = sim.smld.emitters
            result_default = run_collapsed_chain(locs;
                spatial_model=:locmix, allocation_model=:dm,
                n_iterations=200, burn_in=100, learn_distribution=false,
                shape=2.0, μ_prior_shape=2.0, μ_prior_scale=2.5, verbose=false)
            @test result_default.state.use_poisson_k_prior == false

            # k_prior=:poisson is the flat-area-cancelled prior; mixing it
            # with :locmix would silently combine inconsistent target pieces.
            @test_throws ArgumentError run_collapsed_chain(locs;
                spatial_model=:locmix, k_prior=:poisson,
                n_iterations=10, burn_in=0, verbose=false)

            # Same check on the run_bagol public path. Validation must
            # happen BEFORE the @spawn partition loop or the ArgumentError
            # gets wrapped in CompositeException.
            @test_throws ArgumentError run_bagol(sim.smld;
                spatial_model=:locmix, k_prior=:poisson,
                n_iterations=10, burn_in=0,
                partition_sigma=Inf, posterior_pixel_size=0.0,
                verbose=false)
        end

        @testset "Fazel exact configuration runs" begin
            sim = simulate_localizations([(0.0, 0.0), (0.05, 0.0)];
                count_model=:fixed, mean_count=6.0, fixed_sigma=0.005)
            locs = sim.smld.emitters
            # Exact Fazel config: flat + categorical + no K prior
            result = run_collapsed_chain(locs;
                spatial_model=:flat, allocation_model=:categorical, k_prior=:none,
                n_iterations=500, burn_in=200, learn_distribution=false,
                shape=2.0, μ_prior_shape=2.0, μ_prior_scale=3.0, verbose=false)
            @test isa(result.state.allocation, CategoricalAllocation)
            @test isa(result.state.spatial, FlatSpatial)
            @test result.state.use_poisson_k_prior == false
        end

        @testset "Invalid kwargs error" begin
            sim = simulate_localizations([(0.0, 0.0)];
                count_model=:fixed, mean_count=3.0, fixed_sigma=0.005)
            locs = sim.smld.emitters
            @test_throws ArgumentError run_collapsed_chain(locs;
                allocation_model=:bogus, n_iterations=10, burn_in=0, verbose=false)
            @test_throws ArgumentError run_collapsed_chain(locs;
                k_prior=:bogus, n_iterations=10, burn_in=0, verbose=false)
        end

        @testset "Archive _reconstruct_state populates use_poisson_k_prior" begin
            # Regression for the new positional field on CollapsedState —
            # previously _reconstruct_state used the old field list and
            # threw MethodError after the use_poisson_k_prior field landed.
            sim = simulate_localizations([(0.0, 0.0), (0.05, 0.0)];
                count_model=:fixed, mean_count=4.0, fixed_sigma=0.005)
            locs = sim.smld.emitters
            assignments = Vector{Int16}(repeat(1:2, outer=cld(length(locs), 2))[1:length(locs)])
            log_area = log(SMLMBaGoL.area(UniformSpatialPrior(locs)))
            state = SMLMBaGoL._reconstruct_state(assignments, locs, log_area)
            @test state.n_active >= 1
            @test isa(state.spatial, FlatSpatial)
            @test state.use_poisson_k_prior == true   # legacy flat default
        end
    end
end

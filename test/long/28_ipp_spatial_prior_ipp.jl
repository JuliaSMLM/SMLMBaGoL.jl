using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions
import LinearAlgebra
import SMLMSim

# ================================================================
# Inhomogeneous Poisson process (:ipp) spatial prior — the default
# ================================================================
@testset "IPP spatial prior (:ipp)" begin
    S = SMLMBaGoL
    # A small field of three well-separated groups of localizations.
    function ipp_groups(; seed = 5)
        rng = Random.MersenneTwister(seed)
        em = SMLMData.Emitter2DFit{Float64}[]
        for (gx, gy, n) in ((0.0, 0.0, 5), (1.2, 0.3, 4), (0.4, 1.4, 5))
            for j in 1:n
                push!(em, SMLMData.Emitter2DFit{Float64}(gx + 0.01 * randn(rng), gy + 0.01 * randn(rng),
                    1000.0, 10.0, 0.01, 0.01, 0.0, 0.0; frame = length(em) + 1))
            end
        end
        em
    end
    locs_g = ipp_groups()
    box_g = S.UniformSpatialPrior(locs_g)
    map_g = S.build_intensity_map(locs_g, box_g; bandwidth = 0.2)   # non-constant

    @testset "P6 default is :locmix, :ipp is an option" begin
        @test BaGoLConfig().spatial_model === :locmix
        @test BaGoLConfig(spatial_model = :ipp).spatial_model === :ipp
        @test BaGoLConfig().intensity_bandwidth == 0.2
        @test BaGoLConfig(intensity_bandwidth = 0.3).intensity_bandwidth == 0.3
        r = run_collapsed_chain(locs_g; spatial_model = :ipp, n_iterations = 20, burn_in = 5, verbose = false)
        @test r.state.spatial isa IntensitySpatial
        @test r.state.use_poisson_k_prior
        pf = tempname()
        smld = SMLMData.BasicSMLD(locs_g, SMLMData.IdealCamera(1:16, 1:16, 0.1), 1, 1)
        run_bagol(smld; spatial_model = :ipp, n_iterations = 40, burn_in = 10, sync_interval = 20, verbose = false,
                  posterior_pixel_size = 0.0, progress_file = pf)
        @test occursin("ipp pilot intensity map (bandwidth 0.2", read(pf, String))
        # config path forwards the bandwidth
        pf2 = tempname()
        run_bagol(smld, BaGoLConfig(spatial_model = :ipp, n_iterations = 40, burn_in = 10, sync_interval = 20, verbose = false,
                                    posterior_pixel_size = 0.0, progress_file = pf2,
                                    intensity_bandwidth = 0.3, se_adjust = 0.0))
        @test occursin("ipp pilot intensity map (bandwidth 0.3", read(pf2, String))
    end

    @testset "P2 normalization, floor, edge correction" begin
        # exact quadrature of the final lookup over the box equals A_field
        m = map_g
        @test m.field_area ≈ S.area(box_g)
        @test m.n_all == length(locs_g)
        @test all(isfinite, m.logg)
        # independent 8-point Gauss-Legendre (Golub-Welsch nodes) per (clipped) cell, evaluated
        # through log_intensity
        gl_x, gl_w = let n = 8
            ev = LinearAlgebra.eigen(LinearAlgebra.SymTridiagonal(zeros(n), [k / sqrt(4k^2 - 1) for k in 1:n-1]))
            ev.values, 2 .* ev.vectors[1, :] .^ 2
        end
        function gl_integral(m)
            tot = 0.0
            for ix in 1:m.nx-1, iy in 1:m.ny-1
                xa = max(m.xlo, m.x0 + (ix - 1) * m.dx); xb = min(m.xhi, m.x0 + ix * m.dx)
                ya = max(m.ylo, m.y0 + (iy - 1) * m.dy); yb = min(m.yhi, m.y0 + iy * m.dy)
                (xb > xa && yb > ya) || continue
                for (s, ws) in zip(gl_x, gl_w), (t, wt) in zip(gl_x, gl_w)
                    tot += ws * wt * (xb - xa) * (yb - ya) / 4 *
                           exp(S.log_intensity(m, (xa + xb) / 2 + s * (xb - xa) / 2, (ya + yb) / 2 + t * (yb - ya) / 2))
                end
            end
            tot
        end
        @test gl_integral(m) ≈ m.field_area rtol = 1e-6
        @test S.intensity_integral(m, m.xlo, m.xhi, m.ylo, m.yhi) ≈ m.field_area rtol = 1e-6
        # g is larger inside the groups than far from every localization
        @test exp(S.log_intensity(m, 0.0, 0.0)) > 5 * exp(S.log_intensity(m, 0.9, 1.0))
        # a homogeneous SMLMSim field: mean g over the localizations is ≈ 1, edges included
        Random.seed!(1)
        cam = SMLMData.IdealCamera(1:11, 1:11, 0.1)
        params = SMLMSim.StaticSMLMConfig(; density = 250, nframes = 300, framerate = 50.0,
            ndatasets = 1, σ_psf = 0.13, minphotons = 30)
        smld_h, _ = SMLMSim.simulate(params; pattern = SMLMSim.Nmer2D(n = 1, d = 0.0),
            molecule = SMLMSim.GenericFluor(photons = 12000.0, k_off = 50.0, k_on = 0.5), camera = cam)
        lh = smld_h.emitters
        xs = extrema(e.x for e in lh); ys = extrema(e.y for e in lh)
        bh = S.UniformSpatialPrior(xs..., ys...)
        mh = S.build_intensity_map(lh, bh; bandwidth = 0.2)
        gvals = [exp(S.log_intensity(mh, e.x, e.y)) for e in lh]
        @test abs(mean(gvals) - 1) < 0.05
        @test S.intensity_integral(mh, mh.xlo, mh.xhi, mh.ylo, mh.yhi) ≈ mh.field_area rtol = 1e-6
        # edge correction: the band within h of the box edge is not depleted
        edge = [g for (g, e) in zip(gvals, lh) if min(e.x - bh.x_min, bh.x_max - e.x, e.y - bh.y_min, bh.y_max - e.y) < 0.1]
        inner = [g for (g, e) in zip(gvals, lh) if min(e.x - bh.x_min, bh.x_max - e.x, e.y - bh.y_min, bh.y_max - e.y) > 0.3]
        @test length(edge) > 20 && length(inner) > 20
        @test abs(mean(edge) / mean(inner) - 1) < 0.15
        # invalid bandwidth
        @test_throws ArgumentError S.build_intensity_map(locs_g, box_g; bandwidth = 0.0)
    end

    @testset "P1 g ≡ 1 reduces to :flat + Poisson K prior" begin
        cmap = S.constant_intensity_map(box_g; n_all = length(locs_g))
        la = log(S.area(box_g))
        sp_i = IntensitySpatial(la, cmap)
        sp_f = FlatSpatial(la)
        @test S.spatial_area(sp_i) == S.spatial_area(sp_f)
        @test S._uses_poisson_k_prior(sp_i) == S._uses_poisson_k_prior(sp_f) == true
        lps = S.precompute_loc_precisions(locs_g)
        z = [1, 1, 2, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7]
        st_i = S.initialize_from_assignments(z, locs_g; sp = sp_i, am = DMAllocation(nothing), use_poisson_k_prior = true)
        st_f = S.initialize_from_assignments(z, locs_g; sp = sp_f, am = DMAllocation(nothing), use_poisson_k_prior = true)
        @test S._total_spatial_lml(st_i) ≈ S._total_spatial_lml(st_f) atol = 1e-12
        for j in 1:st_i.n_active, k in (1, 9, 14)
            @test S.spatial_pred(st_i.clusters[j], lps[k], sp_i) ≈ S.spatial_pred(st_f.clusters[j], lps[k], sp_f) atol = 1e-12
        end
        # K-move ratios: identical accept/reject path and states under the same RNG stream
        for (rng_seed) in (1, 2)
            a = S.initialize_from_assignments(z, locs_g; sp = sp_i, am = DMAllocation(nothing), use_poisson_k_prior = true)
            b = S.initialize_from_assignments(z, locs_g; sp = sp_f, am = DMAllocation(nothing), use_poisson_k_prior = true)
            Random.seed!(rng_seed)
            acc_a = Bool[]
            for _ in 1:150
                push!(acc_a, S.propose_split_merge!(a, locs_g, 8.0, 2.0, 3.0)[1])
                push!(acc_a, S.propose_birth_death!(a, locs_g, 8.0, 2.0, 3.0)[1])
            end
            Random.seed!(rng_seed)
            acc_b = Bool[]
            for _ in 1:150
                push!(acc_b, S.propose_split_merge!(b, locs_g, 8.0, 2.0, 3.0)[1])
                push!(acc_b, S.propose_birth_death!(b, locs_g, 8.0, 2.0, 3.0)[1])
            end
            @test acc_a == acc_b
            @test a.assignments == b.assignments
        end
        # whole standalone chain: :ipp with the constant map equals :flat, learned ρ included
        kw = (n_iterations = 400, burn_in = 100, hierarchical_interval = 50, verbose = false)
        Random.seed!(7); rf = run_collapsed_chain(locs_g; spatial_model = :flat, kw...)
        Random.seed!(7); ri = run_collapsed_chain(locs_g; spatial_model = :ipp, intensity_map = cmap, kw...)
        @test ri.state.assignments == rf.state.assignments
        @test ri.ρ ≈ rf.ρ
    end

    @testset "empty-cluster guard, predictive, exposure" begin
        sp = IntensitySpatial(log(2.0), map_g)
        lps = S.precompute_loc_precisions(locs_g)
        ecs = S.empty_cluster(lps)
        @test S.spatial_ml(ecs, sp) == 0.0
        @test S.spatial_pred(ecs, lps[3], sp) ≈ -log(2.0) + S.log_intensity(map_g, lps[3].pos[1], lps[3].pos[2])
        # saddle point: ML_flat + log g at the posterior mean
        cs = S.add_loc(S.add_loc(ecs, lps[1]), lps[2])
        μ̂ = S.posterior_mean(cs)
        @test S.spatial_ml(cs, sp) ≈ S.log_marginal_likelihood(cs, log(2.0)) + S.log_intensity(map_g, μ̂...)
        # a supplied map fixes the exposure at field_area·n/n_all
        r = run_collapsed_chain(locs_g[1:5]; spatial_model = :ipp, n_iterations = 10, burn_in = 2, intensity_map = map_g, verbose = false)
        @test S.spatial_area(r.state.spatial) ≈ map_g.field_area * 5 / map_g.n_all
        @test_throws ArgumentError run_collapsed_chain(locs_g; spatial_model = :flat, intensity_map = map_g,
                                                       n_iterations = 5, burn_in = 1, verbose = false)
        # data-scaled ρ start when rho is not given, explicit rho respected
        r0 = run_collapsed_chain(locs_g; spatial_model = :ipp, n_iterations = 3, burn_in = 0, hierarchical_interval = 100,
                                 verbose = false, μ_prior_shape = 1.0, μ_prior_scale = 10.0)
        @test r0.ρ ≈ length(locs_g) / (10.0 * S.area(box_g))
        r1 = run_collapsed_chain(locs_g; spatial_model = :ipp, n_iterations = 3, burn_in = 0, rho = 4.0, verbose = false)
        @test r1.ρ == 4.0
    end

    @testset "standalone positive-area policy: singleton, co-located, collinear" begin
        mkl(x, y, i) = SMLMData.Emitter2DFit{Float64}(x, y, 1000.0, 10.0, 0.01, 0.01, 0.0, 0.0; frame = i)
        for locs in ([mkl(0.1, 0.2, 1)], [mkl(0.1, 0.2, i) for i in 1:6], [mkl(0.02i, 0.0, i) for i in 1:8])
            ri = run_collapsed_chain(locs; spatial_model = :ipp, n_iterations = 100, burn_in = 20, hierarchical_interval = 20, verbose = false)
            rf = run_collapsed_chain(locs; spatial_model = :flat, n_iterations = 100, burn_in = 20,
                                     hierarchical_interval = 20, verbose = false)
            # same padded, floored exposure as standalone flat, positive, learned rho finite
            @test S.spatial_area(ri.state.spatial) > 0
            @test S.spatial_area(ri.state.spatial) == S.spatial_area(rf.state.spatial)
            @test isfinite(ri.ρ) && ri.ρ > 0
        end
    end

    @testset "P3 partition consistency (target factorizes across cuts)" begin
        N = length(locs_g)
        xs = extrema(l.x for l in locs_g); ys = extrema(l.y for l in locs_g)
        A_field = (xs[2] - xs[1]) * (ys[2] - ys[1])
        fmap = S.build_intensity_map(locs_g, S.UniformSpatialPrior(xs..., ys...); bandwidth = 0.2)
        grp = vcat(fill(1, 5), fill(2, 4), fill(3, 5))
        @test length(grp) == N
        μ, shape, ρ = 8.0, 2.0, 3.0
        # log of the product of the partition targets of a cut (parts = vectors of group ids)
        function cut_target(parts, zfull)
            tot = 0.0
            for g in parts
                idx = findall(in(g), grp)
                pl = locs_g[idx]
                sp = IntensitySpatial(log(A_field * length(idx) / N), fmap)
                tot += S.log_target(IPPTarget(fmap), S.canonicalize(zfull[idx]), S.precompute_loc_precisions(pl), sp, μ, shape, ρ)
            end
            tot
        end
        rng = Random.MersenneTwister(3)
        diffs = Float64[]
        for _ in 1:12
            # random assignment confined to the groups (labels unique per group)
            zfull = [grp[i] * 10 + rand(rng, 1:3) for i in 1:N]
            push!(diffs, cut_target([[1, 2], [3]], zfull) - cut_target([[1], [2, 3]], zfull))
        end
        @test maximum(diffs) - minimum(diffs) < 1e-9     # z-independent constant between cuts
        # sanity: the target does depend on z within a cut
        z1 = [grp[i] * 10 + 1 for i in 1:N]; z2 = [grp[i] * 10 + (i % 3) for i in 1:N]
        @test abs(cut_target([[1, 2], [3]], z1) - cut_target([[1, 2], [3]], z2)) > 1e-3
    end

    @testset "P5 rho conditional: Gamma(a + ΣK, b + ΣA')" begin
        parts, _ = S.partition_locs(locs_g; max_size = 8, overlap = 0.05)
        @test length(parts) > 1
        N = length(locs_g)
        xs = extrema(l.x for l in locs_g); ys = extrema(l.y for l in locs_g)
        A_field = (xs[2] - xs[1]) * (ys[2] - ys[1])
        fmap = S.build_intensity_map(locs_g, S.UniformSpatialPrior(xs..., ys...); bandwidth = 0.2)
        sts = [S.initialize_collapsed_state(p.locs,
                   IntensitySpatial(log(S._partition_area_share(A_field, length(p.locs), N)), fmap),
                   DMAllocation(nothing); use_poisson_k_prior = true) for p in parts]
        # several clusters per partition so ΣK is not just the partition count
        sts = [S.initialize_from_assignments(mod1.(1:length(p.locs), 3), p.locs; sp = s.spatial,
                   am = DMAllocation(nothing), use_poisson_k_prior = true) for (s, p) in zip(sts, parts)]
        @test sum(s.n_active for s in sts) > length(sts)
        areas = [S.spatial_area(s.spatial) for s in sts]
        cfg = (ρ_prior_shape = 2.0, ρ_prior_rate = 1.0)
        shp, rate = S._rho_posterior_params(sts, areas, cfg)
        @test shp == 2.0 + sum(s.n_active for s in sts)
        @test rate ≈ 1.0 + A_field * sum(length(p.locs) for p in parts) / N
        @test rate ≈ 1.0 + sum(S.spatial_area(s.spatial) for s in sts)
    end

    @testset "IPPTarget conventions" begin
        # the cancelled canonical target equals DMFlatTarget(:factorial) at g ≡ 1
        sub = locs_g[1:6]
        bsub = S.UniformSpatialPrior(sub)
        cm = S.constant_intensity_map(bsub; n_all = 6)
        parts6 = enumerate_canonical_partitions(6, 6)
        p_ipp, lt_ipp = exact_posterior(parts6, sub, IPPTarget(cm); μ = 6.0, shape = 2.0, ρ = 3.0)
        p_flat, lt_flat = exact_posterior(parts6, sub, DMFlatTarget(); μ = 6.0, shape = 2.0, ρ = 3.0)
        @test p_ipp ≈ p_flat atol = 1e-10
        # with the non-constant map the exact posterior differs from flat
        p_g, _ = exact_posterior(parts6, sub, IPPTarget(S.build_intensity_map(sub, bsub; bandwidth = 0.05)); μ = 6.0, shape = 2.0, ρ = 3.0)
        @test maximum(abs.(p_g .- p_flat)) > 1e-6
    end

    @testset "kernel invariance with a non-constant map (IPPTarget)" begin
        Random.seed!(11)
        sub = [SMLMData.Emitter2DFit{Float64}(x, y, 1000.0, 10.0, 0.01, 0.01, 0.0, 0.0; frame = i)
               for (i, (x, y)) in enumerate([(0.0, 0.0), (0.02, 0.01), (0.03, -0.01), (0.55, 0.3)])]
        bsub = S.UniformSpatialPrior(sub)
        m4 = S.build_intensity_map(sub, bsub; bandwidth = 0.1)
        res = run_kernel_invariance_test(sub, IPPTarget(m4); μ = 3.0, shape = 2.0, ρ = 3.0, K_max = 4,
            n_steps_per_state = 6000, n_mcmc_iterations = 100_000, mcmc_burn_in = 2_000)
        @test res.label_multiplicity === :none
        # The Wald p-value is not calibrated at this size (it also rejects :flat in 9 of 12 seeds
        # while the TV stays ~4e-4), so bound the upper 95% confidence limit of the one-step invariance TV instead: pi'P - pi'.
        @test res.wald.tv_upper_95 < 0.005
        @test res.k_test.verdict == :pass        # long-run K marginal matches the exact posterior
    end

    @testset "run_bagol :ipp: degenerate field, k_prior, motion, finder forwarding" begin
        cam = SMLMData.IdealCamera(1:10, 1:10, 0.1)
        # collinear (zero y-extent) field: warns, falls back to padded per-partition areas
        line = [SMLMData.Emitter2DFit{Float64}(0.004 * i, 0.0, 1000.0, 10.0, 0.004, 0.004,
                    0.0, 0.0; frame = i) for i in 1:120]
        r, d = @test_logs (:warn, r"zero spatial extent") run_bagol(SMLMData.BasicSMLD(line, cam, 1, 1); spatial_model = :ipp,
            max_partition_size = 40, overlap = 0.05, n_iterations = 100, burn_in = 50, sync_interval = 50,
            verbose = false, posterior_pixel_size = 0.0)
        @test isfinite(d.final_ρ) && d.final_ρ > 0
        # k_prior=:poisson is valid with :ipp; k_prior=:none runs (per-partition padded areas)
        smld = SMLMData.BasicSMLD(locs_g, cam, 1, 1)
        _, dp = run_bagol(smld; spatial_model = :ipp, k_prior = :poisson, n_iterations = 60, burn_in = 20, sync_interval = 20, verbose = false, posterior_pixel_size = 0.0)
        @test isfinite(dp.final_ρ)
        _, dn = run_bagol(smld; spatial_model = :ipp, k_prior = :none, n_iterations = 60, burn_in = 20, sync_interval = 20, verbose = false, posterior_pixel_size = 0.0)
        @test dn isa BaGoLDiagnostics
        # fixed rho respected
        _, df = run_bagol(smld; spatial_model = :ipp, learn_rho = false, rho = 5.0, n_iterations = 60, burn_in = 20, sync_interval = 20, verbose = false, posterior_pixel_size = 0.0)
        @test df.final_ρ == 5.0
        # motion keeps its own prior; logged once
        pm = tempname()
        em = [SMLMData.Emitter2DFit{Float64}(0.0 + 0.002 * randn(), 0.0 + 0.002 * randn(), 1000.0, 10.0, 0.005, 0.005, 0.0, 0.0; frame = 10 * i) for i in 1:12]
        run_bagol(SMLMData.BasicSMLD(em, cam, 1, 1); spatial_model = :ipp, motion = :linear, se_adjust = 0.0, n_iterations = 60, burn_in = 20,
                  sync_interval = 20, verbose = false, posterior_pixel_size = 0.0, progress_file = pm)
        @test count("motion model's own spatial prior", read(pm, String)) == 1
        @test !occursin("ipp pilot intensity map", read(pm, String))
        # the τ finder's E-steps run with the run's intensity_bandwidth
        cam2 = SMLMData.IdealCamera(1:64, 1:64, 0.1)
        rng = Random.MersenneTwister(11)
        em2 = SMLMData.Emitter2DFit{Float64}[]
        for k in 1:6, j in 1:2
            push!(em2, SMLMData.Emitter2DFit(0.5 + 0.4k + 0.01randn(rng), 1.0 + 0.01randn(rng),
                    1000.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 1, length(em2) + 1, 0, 1))
        end
        smld2 = SMLMData.BasicSMLD(em2, cam2, 1, 1, Dict{String,Any}())
        _, dg = run_bagol(smld2; spatial_model = :ipp, se_adjust = :auto, keep_se_finder = true, intensity_bandwidth = 0.3,
                          max_partition_size = 7, overlap = 0.02, n_iterations = 100, burn_in = 20, verbose = false)
        @test dg.se_finder.bagol_kwargs.intensity_bandwidth == 0.3
    end

    @testset "3D features run under :ipp (g is a map of x, y)" begin
        mk3(x, y, z, i) = SMLMData.Emitter3DFit(x, y, z, 1000.0, 0.0, 0.01, 0.01, 0.02,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 0, i)
        Random.seed!(11)
        locs3 = SMLMData.Emitter3DFit{Float64}[]
        for cz in (0.2, 0.6), _ in 1:12
            push!(locs3, mk3(1.0 + 0.01randn(), 1.0 + 0.01randn(), cz + 0.02randn(), length(locs3) + 1))
        end
        r = run_collapsed_chain(locs3; spatial_model = :ipp, n_iterations = 8000, burn_in = 2000,
                                learn_distribution = false, shape = 10.0)
        @test r.state.spatial isa IntensitySpatial
        @test length(S.extract_emitters(r.state, locs3)) == 2
    end

    @testset "archive replay is model-independent under :ipp" begin
        dir = mktempdir()
        smld = SMLMData.BasicSMLD(locs_g, SMLMData.IdealCamera(1:16, 1:16, 0.1), 1, 1)
        run_bagol(smld; spatial_model = :ipp, n_iterations = 100, burn_in = 20, sync_interval = 20, archive_path = dir,
                  verbose = false, posterior_pixel_size = 0.0)
        ar = BaGoLArchive(dir)
        @test ar.n_partitions >= 1
        counts = S.compute_from_archive(EmitterCountHist, ar, 1, S.partition_locs(locs_g)[1][1].locs)
        @test sum(counts) > 0
    end

    # a custom accumulator that evaluates the spatial model
    @eval struct SpatialScoreAcc <: AbstractAccumulator
        scores::Vector{Float64}
    end
    @eval SpatialScoreAcc(; kw...) = SpatialScoreAcc(Float64[])
    @eval SMLMBaGoL.needs_spatial_model(::Type{SpatialScoreAcc}) = true
    @eval SMLMBaGoL.accumulator_update!(a::SpatialScoreAcc, state, locs, μ, shape, it) =
        push!(a.scores, sum(spatial_ml(cs, state.spatial) for (k, cs) in enumerate(state.clusters) if state.active[k]))
    @eval SMLMBaGoL.accumulator_result(a::SpatialScoreAcc) = a.scores
    # a custom accumulator that has not declared its replay needs
    @eval struct UndeclaredAcc <: AbstractAccumulator end
    @eval UndeclaredAcc(; kw...) = UndeclaredAcc()
    @eval SMLMBaGoL.accumulator_update!(a::UndeclaredAcc, state, locs, μ, shape, it) = nothing
    @eval SMLMBaGoL.accumulator_result(a::UndeclaredAcc) = nothing

    @testset "archive records the spatial model; replay of spatial-model accumulators" begin
        plocs = S.partition_locs(locs_g)[1][1].locs
        smld = SMLMData.BasicSMLD(locs_g, SMLMData.IdealCamera(1:16, 1:16, 0.1), 1, 1)
        @test needs_spatial_model(EmitterCountHist) === false
        @test needs_spatial_model(SpatialScoreAcc) === true
        for T in (EmitterCountHist, PosteriorImage, NNDistHist, S.PartitionSamples, S.PSMAccumulator)
            @test needs_spatial_model(T) === false
        end
        @test_throws ArgumentError needs_spatial_model(UndeclaredAcc)
        @test needs_spatial_model(ChainDiagnosticAccumulator) === false
        dir_i = mktempdir()
        run_bagol(smld; spatial_model = :ipp, n_iterations = 100, burn_in = 20, sync_interval = 20,
                  archive_path = dir_i, verbose = false, posterior_pixel_size = 0.0)
        ar_i = BaGoLArchive(dir_i)
        @test ar_i.spatial_model === :ipp
        @test_throws ArgumentError S.compute_from_archive(SpatialScoreAcc, ar_i, 1, plocs)
        la = log(S.area(S.UniformSpatialPrior(plocs)))
        sc = S.compute_from_archive(SpatialScoreAcc, ar_i, 1, plocs; spatial = S.IntensitySpatial(la, map_g))
        @test !isempty(sc) && all(isfinite, sc)
        @test sum(S.compute_from_archive(EmitterCountHist, ar_i, 1, plocs)) > 0
        kt = S.compute_from_archive(ChainDiagnosticAccumulator, ar_i, 1, plocs)
        @test length(kt.k_trace) == ar_i.sample_counts[1] && all(>(0), kt.k_trace)
        dir_l = mktempdir()
        run_bagol(smld; n_iterations = 100, burn_in = 20, sync_interval = 20,
                  archive_path = dir_l, verbose = false, posterior_pixel_size = 0.0)
        ar_l = BaGoLArchive(dir_l)
        @test ar_l.spatial_model === :locmix
        @test_throws ArgumentError S.compute_from_archive(SpatialScoreAcc, ar_l, 1, plocs)
        @test_throws ArgumentError S.compute_from_archive(UndeclaredAcc, ar_l, 1, plocs)
        @test_throws ArgumentError S.compute_from_archive(UndeclaredAcc, ar_i, 1, plocs)
        dir_f = mktempdir()
        run_bagol(smld; spatial_model = :flat, n_iterations = 100, burn_in = 20, sync_interval = 20,
                  archive_path = dir_f, verbose = false, posterior_pixel_size = 0.0)
        ar_f = BaGoLArchive(dir_f)
        @test ar_f.spatial_model === :flat
        # flat archives too: replay does not rebuild the chain's field-share exposure
        @test_throws ArgumentError S.compute_from_archive(SpatialScoreAcc, ar_f, 1, plocs)
        @test !isempty(S.compute_from_archive(SpatialScoreAcc, ar_f, 1, plocs;
                                              spatial = S.FlatSpatial(la)))
        @test sum(S.compute_from_archive(EmitterCountHist, ar_f, 1, plocs)) > 0

        # motion archives cannot be replayed with spatial accumulators
        em = [SMLMData.Emitter2DFit{Float64}(0.002 * randn(), 0.002 * randn(), 1000.0, 10.0, 0.005, 0.005, 0.0, 0.0; frame = 10 * i) for i in 1:12]
        dir_m = mktempdir()
        run_bagol(SMLMData.BasicSMLD(em, SMLMData.IdealCamera(1:16, 1:16, 0.1), 1, 1); motion = :linear, se_adjust = 0.0,
                  n_iterations = 60, burn_in = 20, sync_interval = 20, archive_path = dir_m,
                  verbose = false, posterior_pixel_size = 0.0)
        ar_m = BaGoLArchive(dir_m)
        @test ar_m.spatial_model === :motion
        mlocs = S.partition_locs(em)[1][1].locs
        @test_throws ArgumentError S.compute_from_archive(SpatialScoreAcc, ar_m, 1, mlocs)
        msp = S.motion_spatial(mlocs, 0.002, log(S.area(S.UniformSpatialPrior(mlocs))), 10.0, 110.0)
        @test_throws ArgumentError S.compute_from_archive(SpatialScoreAcc, ar_m, 1, mlocs; spatial = msp)
        @test_throws ArgumentError S.compute_from_archive(SpatialScoreAcc, ar_f, 1, plocs; spatial = msp)
        @test sum(S.compute_from_archive(EmitterCountHist, ar_m, 1, mlocs)) > 0
    end

    @testset "motion=:linear rejects spatial_model=:ipp with k_prior=:poisson" begin
        @test_throws ArgumentError run_collapsed_chain(locs_g; spatial_model = :ipp, motion = :linear,
            k_prior = :poisson, n_iterations = 5, burn_in = 1, verbose = false)
        smld = SMLMData.BasicSMLD(locs_g, SMLMData.IdealCamera(1:16, 1:16, 0.1), 1, 1)
        @test_throws ArgumentError run_bagol(smld; spatial_model = :ipp, motion = :linear, k_prior = :poisson,
            se_adjust = 0.0, n_iterations = 20, burn_in = 5, sync_interval = 10, verbose = false)
    end
end

using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics
import SMLMSim

# ================================================================
# Inhomogeneous Poisson process (:ipp) spatial prior — optional model
# ================================================================
@testset "IPP spatial prior (:ipp)" begin
    S = SMLMBaGoL
    # A small field of three well-separated groups of localizations.
    function ipp_groups(; seed = 5)
        Random.seed!(seed)
        em = SMLMData.Emitter2DFit{Float64}[]
        for (gx, gy, n) in ((0.0, 0.0, 5), (1.2, 0.3, 4), (0.4, 1.4, 5))
            for j in 1:n
                push!(em, SMLMData.Emitter2DFit{Float64}(gx, gy,
                    1000.0, 10.0, 0.01, 0.01, 0.0, 0.0; frame = length(em) + 1))
            end
        end
        cam = SMLMData.IdealCamera(1:16, 1:16, 0.1)
        truth = SMLMData.BasicSMLD(em, cam, 1, 1)
        SMLMSim.apply_noise(truth, 0.01 * sqrt(1000.0)).emitters
    end
    locs_g = ipp_groups()
    box_g = S.UniformSpatialPrior(locs_g)
    map_g = S.build_intensity_map(locs_g, box_g; bandwidth = 0.2)   # non-constant

    @testset "P6 default is :locmix, :ipp is an option" begin
        @test BaGoLConfig().spatial_model === :locmix
        @test BaGoLConfig(spatial_model = :ipp).spatial_model === :ipp
        @test BaGoLConfig().intensity_bandwidth == 0.2
        @test BaGoLConfig(intensity_bandwidth = 0.3).intensity_bandwidth == 0.3
        r = run_collapsed_chain(locs_g; spatial_model = :ipp, n_iterations = 20,
            burn_in = 5, verbose = false)
        @test r.state.spatial isa IntensitySpatial
        @test r.state.use_poisson_k_prior
        smld = SMLMData.BasicSMLD(locs_g, SMLMData.IdealCamera(1:16, 1:16, 0.1), 1, 1)
        @test_throws ArgumentError run_bagol(smld; spatial_model = :ipp,
            intensity_bandwidth = 0.0, n_iterations = 40, burn_in = 10,
            verbose = false, posterior_pixel_size = 0.0)
        @test_throws ArgumentError run_bagol(smld, BaGoLConfig(spatial_model = :ipp,
            intensity_bandwidth = 0.0, se_adjust = 0.0, n_iterations = 40, burn_in = 10,
            verbose = false, posterior_pixel_size = 0.0))

    end

    @testset "P2 homogeneous SMLMSim field" begin
        # a homogeneous SMLMSim field: mean g over the localizations is ≈ 1, edges included
        Random.seed!(1)
        cam = SMLMData.IdealCamera(1:11, 1:11, 0.1)
        params = SMLMSim.StaticSMLMConfig(; density = 250, nframes = 300, framerate = 50.0,
            ndatasets = 1, σ_psf = 0.13, minphotons = 30)
        smld_h, _ = SMLMSim.simulate(params; pattern = SMLMSim.Nmer2D(n = 1, d = 0.0),
            molecule = SMLMSim.GenericFluor(photons = 12000.0, k_off = 50.0, k_on =
                0.5), camera = cam)
        lh = smld_h.emitters
        bh = field_box(lh)
        mh = S.build_intensity_map(lh, bh; bandwidth = 0.2)
        gvals = [exp(S.log_intensity(mh, e.x, e.y)) for e in lh]
        @test abs(mean(gvals) - 1) < 0.05
        @test S.intensity_integral(mh, bh.x_min, bh.x_max, bh.y_min,
            bh.y_max) ≈ field_area(mh) rtol = 1e-6
        # edge correction: the band within h of the box edge is not depleted
        edge = [g for (g, e) in zip(gvals, lh) if min(e.x - bh.x_min, bh.x_max - e.x,
            e.y - bh.y_min, bh.y_max - e.y) < 0.1]
        inner = [g for (g, e) in zip(gvals, lh) if min(e.x - bh.x_min, bh.x_max - e.x,
            e.y - bh.y_min, bh.y_max - e.y) > 0.3]
        @test length(edge) > 20 && length(inner) > 20
        @test abs(mean(edge) / mean(inner) - 1) < 0.15
    end

    @testset "P1 g ≡ 1 reduces to :flat + Poisson K prior" begin
        cmap = S.constant_intensity_map(box_g; n_all = length(locs_g))
        la = log(S.area(box_g))
        sp_i = IntensitySpatial(la, cmap)
        sp_f = FlatSpatial(la)
        z = [1, 1, 2, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7]
        # K-move ratios: identical accept/reject path and states under the same RNG stream
        for (rng_seed) in (1, 2)
            a = S.initialize_from_assignments(z, locs_g; sp = sp_i,
                am = DMAllocation(nothing), use_poisson_k_prior = true)
            b = S.initialize_from_assignments(z, locs_g; sp = sp_f,
                am = DMAllocation(nothing), use_poisson_k_prior = true)
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
        # whole standalone chain: :ipp with the constant map equals :flat, learned ρ
        # included
        kw = (n_iterations = 400, burn_in = 100, hierarchical_interval = 50, verbose =
            false)
        Random.seed!(7); rf = run_collapsed_chain(locs_g; spatial_model = :flat, kw...)
        Random.seed!(7); ri = run_collapsed_chain(locs_g; spatial_model = :ipp,
            intensity_map = cmap, kw...)
        @test ri.state.assignments == rf.state.assignments
        @test ri.ρ ≈ rf.ρ
    end

    @testset "standalone positive-area policy: singleton, co-located, collinear" begin
        mkl(x, y, i) = SMLMData.Emitter2DFit{Float64}(x, y, 1000.0, 10.0, 0.01, 0.01,
            0.0, 0.0; frame = i)
        for locs in ([mkl(0.1, 0.2, 1)], [mkl(0.1, 0.2, i) for i in 1:6], [mkl(0.02i,
            0.0, i) for i in 1:8])
            ri = run_collapsed_chain(locs; spatial_model = :ipp, n_iterations = 100,
                burn_in = 20, hierarchical_interval = 20, verbose = false)
            rf = run_collapsed_chain(locs; spatial_model = :flat, n_iterations = 100,
                burn_in = 20, hierarchical_interval = 20, verbose = false)
            # same padded, floored exposure as standalone flat, positive, learned rho finite
            @test S.spatial_area(ri.state.spatial) > 0
            @test S.spatial_area(ri.state.spatial) == S.spatial_area(rf.state.spatial)
            @test isfinite(ri.ρ) && ri.ρ > 0
        end
    end

    @testset "kernel invariance with a non-constant map (IPPTarget)" begin
        Random.seed!(11)
        sub = [SMLMData.Emitter2DFit{Float64}(x, y, 1000.0, 10.0, 0.01, 0.01, 0.0, 0.0;
            frame = i)
               for (i, (x, y)) in enumerate([(0.0, 0.0), (0.02, 0.01), (0.03, -0.01),
                   (0.55, 0.3)])]
        bsub = S.UniformSpatialPrior(sub)
        m4 = S.build_intensity_map(sub, bsub; bandwidth = 0.1)
        res = run_kernel_invariance_test(sub, IPPTarget(m4);
            μ = 3.0, shape = 2.0, ρ = 3.0, K_max = 4,
            n_steps_per_state = 6000, n_mcmc_iterations = 100_000, mcmc_burn_in = 2_000)
        @test res.label_multiplicity === :factorial
        # The Wald p-value also rejects :flat in 9 of 12 seeds while TV stays ~4e-4.
        # Bound the upper 95% confidence limit of one-step TV instead: pi'P - pi'.
        @test res.wald.tv_upper_95 < 0.005
        # long-run K marginal matches the exact posterior
        @test res.k_test.verdict == :pass
    end

    @testset "run_bagol :ipp: degenerate field, k_prior, motion, finder forwarding" begin
        cam = SMLMData.IdealCamera(1:10, 1:10, 0.1)
        # collinear (zero y-extent) field: warns, falls back to padded per-partition areas
        line = [SMLMData.Emitter2DFit{Float64}(0.004 * i, 0.0, 1000.0, 10.0, 0.004, 0.004,
                    0.0, 0.0; frame = i) for i in 1:120]
        r, d = @test_logs (:warn,
            r"zero spatial extent") run_bagol(SMLMData.BasicSMLD(line, cam, 1, 1);
            spatial_model = :ipp,
            max_partition_size = 40, overlap = 0.05, n_iterations = 100,
            burn_in = 50, sync_interval = 50,
            verbose = false, posterior_pixel_size = 0.0)
        @test isfinite(d.final_ρ) && d.final_ρ > 0
        # :poisson is valid with :ipp; :none uses padded per-partition areas.
        smld = SMLMData.BasicSMLD(locs_g, cam, 1, 1)
        _, dp = run_bagol(smld; spatial_model = :ipp, k_prior = :poisson,
            n_iterations = 60, burn_in = 20, sync_interval = 20,
            verbose = false, posterior_pixel_size = 0.0)
        @test isfinite(dp.final_ρ)
        _, dn = run_bagol(smld; spatial_model = :ipp, k_prior = :none,
            n_iterations = 60, burn_in = 20, sync_interval = 20,
            verbose = false, posterior_pixel_size = 0.0)
        @test dn isa BaGoLDiagnostics
        # fixed rho respected
        _, df = run_bagol(smld; spatial_model = :ipp, learn_rho = false, rho = 5.0,
            n_iterations = 60, burn_in = 20, sync_interval = 20,
            verbose = false, posterior_pixel_size = 0.0)
        @test df.final_ρ == 5.0
        # Motion keeps its own spatial prior and permits an explicit Poisson K prior.
        pm = tempname()
        em = [SMLMData.Emitter2DFit{Float64}(0.0, 0.0, 1000.0, 10.0, 0.005, 0.005, 0.0,
            0.0; frame = 10 * i) for i in 1:12]
        em = SMLMSim.apply_noise(SMLMData.BasicSMLD(em, cam, 1, 1),
                                 0.005 * sqrt(1000.0)).emitters
        run_bagol(SMLMData.BasicSMLD(em, cam, 1, 1); spatial_model = :ipp,
            motion = :linear, k_prior = :poisson, se_adjust = 0.0,
            n_iterations = 60, burn_in = 20,
            sync_interval = 20, verbose = false, posterior_pixel_size = 0.0,
            progress_file = pm)
        @test count("motion model's own spatial prior", read(pm, String)) == 1
        @test !occursin("ipp pilot intensity map", read(pm, String))
        motion_kw = (motion = :linear, k_prior = :poisson, n_iterations = 10,
                     burn_in = 2, hierarchical_interval = 100, verbose = false)
        Random.seed!(19)
        rm_ipp = run_collapsed_chain(em; spatial_model = :ipp, motion_kw...)
        Random.seed!(19)
        rm_flat = run_collapsed_chain(em; spatial_model = :flat, motion_kw...)
        @test rm_ipp.state.assignments == rm_flat.state.assignments
        @test rm_ipp.ρ == rm_flat.ρ == 2.0   # motion retains its prior-mean start
        # the τ finder's E-steps run with the run's intensity_bandwidth
        cam2 = SMLMData.IdealCamera(1:64, 1:64, 0.1)
        Random.seed!(11)
        em2 = SMLMData.Emitter2DFit{Float64}[]
        for k in 1:6, j in 1:2
            push!(em2, SMLMData.Emitter2DFit(0.5 + 0.4k, 1.0,
                    1000.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 1, length(em2) + 1, 0, 1))
        end
        truth2 = SMLMData.BasicSMLD(em2, cam2, 1, 1, Dict{String,Any}())
        smld2 = SMLMSim.apply_noise(truth2, 0.01 * sqrt(1000.0))
        _, dg = run_bagol(smld2; spatial_model = :ipp, se_adjust = :auto,
            keep_se_finder = true, intensity_bandwidth = 0.3,
            max_partition_size = 7, overlap = 0.02, n_iterations = 100,
            burn_in = 20, verbose = false)
        @test dg.se_finder.bagol_kwargs.intensity_bandwidth == 0.3
    end

    @testset "3D features run under :ipp (g is a map of x, y)" begin
        mk3(x, y, z, i) = SMLMData.Emitter3DFit(x, y, z, 1000.0, 0.0, 0.01, 0.01, 0.02,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 0, i)
        Random.seed!(11)
        locs3 = SMLMData.Emitter3DFit{Float64}[]
        for cz in (0.2, 0.6), _ in 1:12
            push!(locs3, mk3(1.0, 1.0, cz, length(locs3) + 1))
        end
        cam3 = SMLMData.IdealCamera(1:16, 1:16, 0.1)
        truth3 = SMLMData.BasicSMLD(locs3, cam3, 1, 1)
        locs3 = SMLMSim.apply_noise(truth3, sqrt(1000.0) .* [0.01, 0.01, 0.02]).emitters
        r = run_collapsed_chain(locs3; spatial_model = :ipp, n_iterations = 8000,
            burn_in = 2000, learn_distribution = false, shape = 10.0, verbose = false)
        @test r.state.spatial isa IntensitySpatial
        @test length(S.extract_emitters(r.state, locs3)) == 2
    end

    @testset "run_bagol records its sampling model for replay" begin
        cam = SMLMData.IdealCamera(1:16, 1:16, 0.1)
        smld = SMLMData.BasicSMLD(locs_g, cam, 14, 1)
        plocs = S.partition_locs(locs_g)[1][1].locs
        for (model, motion) in ((:ipp, :none), (:flat, :none),
                                (:locmix, :none), (:locmix, :linear))
            mktempdir() do dir
                run_bagol(smld; spatial_model = model, motion, se_adjust = 0.0,
                    n_iterations = 60, burn_in = 20, sync_interval = 20,
                    archive_path = dir, verbose = false, posterior_pixel_size = 0.0)
                ar = BaGoLArchive(dir)
                @test ar.spatial_model === (motion === :linear ? :motion : model)
                counts = S.compute_from_archive(EmitterCountHist, ar, 1, plocs)
                @test sum(counts) == ar.sample_counts[1] > 0
            end
        end
    end

end

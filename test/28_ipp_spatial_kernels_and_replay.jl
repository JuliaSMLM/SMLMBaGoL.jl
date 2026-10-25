using Test
using SMLMBaGoL
using SMLMData
using Random
import LinearAlgebra

# Each test file has its own module; extension types belong at top level.
struct SpatialScoreAcc <: AbstractAccumulator
    scores::Vector{Float64}
end
SpatialScoreAcc(; kw...) = SpatialScoreAcc(Float64[])
function SMLMBaGoL.accumulator_update!(a::SpatialScoreAcc, state, locs, μ, shape, it)
    push!(a.scores, sum(spatial_ml(cs, state.spatial)
        for (k, cs) in enumerate(state.clusters) if state.active[k]))
end
SMLMBaGoL.accumulator_result(a::SpatialScoreAcc) = a.scores

struct UndeclaredAcc <: AbstractAccumulator end
SMLMBaGoL.accumulator_update!(::UndeclaredAcc, state, locs, μ, shape, it) = nothing
SMLMBaGoL.accumulator_result(::UndeclaredAcc) = nothing

@testset "IPP spatial kernels and archive replay" begin
    S = SMLMBaGoL
    # Fixed coordinates exercise the kernels without simulating localization error.
    xy = [(0.00, 0.00), (0.01, 0.01), (-0.01, 0.00), (0.00, -0.01), (0.01, -0.01),
          (1.20, 0.30), (1.21, 0.31), (1.19, 0.30), (1.20, 0.29),
          (0.40, 1.40), (0.41, 1.41), (0.39, 1.40), (0.40, 1.39), (0.41, 1.39)]
    locs_g = [Emitter2DFit{Float64}(x, y, 1000.0, 10.0, 0.01, 0.01, 0.0, 0.0;
                                  frame = i) for (i, (x, y)) in enumerate(xy)]
    box_g = UniformSpatialPrior(locs_g)
    map_g = build_intensity_map(locs_g, box_g)

    @testset "P2 normalization, floor, edge correction" begin
        # exact quadrature of the final lookup over the box equals A_field
        m = map_g
        @test field_area(m) ≈ S.area(box_g)
        @test m.n_all == length(locs_g)
        @test all(isfinite, m.logg)
        # Independent 8-point Gauss-Legendre (Golub-Welsch nodes) per clipped cell,
        # evaluated through log_intensity.
        gl_x, gl_w = let n = 8
            ev = LinearAlgebra.eigen(LinearAlgebra.SymTridiagonal(zeros(n),
                [k / sqrt(4k^2 - 1) for k in 1:n-1]))
            ev.values, 2 .* ev.vectors[1, :] .^ 2
        end
        function gl_integral(m)
            tot = 0.0
            for ix in 1:size(m.logg, 2)-1, iy in 1:size(m.logg, 1)-1
                xa = max(m.box.x_min, m.x0 + (ix - 1) * m.spacing)
                xb = min(m.box.x_max, m.x0 + ix * m.spacing)
                ya = max(m.box.y_min, m.y0 + (iy - 1) * m.spacing)
                yb = min(m.box.y_max, m.y0 + iy * m.spacing)
                (xb > xa && yb > ya) || continue
                for (s, ws) in zip(gl_x, gl_w), (t, wt) in zip(gl_x, gl_w)
                    tot += ws * wt * (xb - xa) * (yb - ya) / 4 *
                           exp(S.log_intensity(m, (xa + xb) / 2 + s * (xb - xa) / 2,
                               (ya + yb) / 2 + t * (yb - ya) / 2))
                end
            end
            tot
        end
        @test gl_integral(m) ≈ field_area(m) rtol = 1e-6
        @test S.intensity_integral(m, m.box.x_min, m.box.x_max, m.box.y_min,
            m.box.y_max) ≈ field_area(m) rtol = 1e-6
        # g is larger inside the groups than far from every localization
        @test exp(S.log_intensity(m, 0.0, 0.0)) > 5 * exp(S.log_intensity(m, 0.9, 1.0))
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
        st_i = S.initialize_from_assignments(z, locs_g; sp = sp_i,
            am = DMAllocation(nothing), use_poisson_k_prior = true)
        st_f = S.initialize_from_assignments(z, locs_g; sp = sp_f,
            am = DMAllocation(nothing), use_poisson_k_prior = true)
        @test S._total_spatial_lml(st_i) ≈ S._total_spatial_lml(st_f) atol = 1e-12
        for j in 1:st_i.n_active, k in (1, 9, 14)
            @test S.spatial_pred(st_i.clusters[j], lps[k],
                sp_i) ≈ S.spatial_pred(st_f.clusters[j], lps[k], sp_f) atol = 1e-12
        end
    end

    @testset "empty-cluster guard, predictive, exposure" begin
        sp = IntensitySpatial(log(2.0), map_g)
        lps = S.precompute_loc_precisions(locs_g)
        ecs = S.empty_cluster(lps)
        @test S.spatial_ml(ecs, sp) == 0.0
        @test S.spatial_pred(ecs, lps[3], sp) ≈ -log(2.0) + S.log_intensity(map_g,
            lps[3].pos[1], lps[3].pos[2])
        # saddle point: ML_flat + log g at the posterior mean
        cs = S.add_loc(S.add_loc(ecs, lps[1]), lps[2])
        μ̂ = S.posterior_mean(cs)
        @test S.spatial_ml(cs, sp) ≈ S.log_marginal_likelihood(cs, log(2.0)) +
            S.log_intensity(map_g, μ̂...)
        # a supplied map fixes the exposure at field_area·n/n_all
        r = run_collapsed_chain(locs_g[1:5]; spatial_model = :ipp, n_iterations = 10,
            burn_in = 2, intensity_map = map_g, verbose = false)
        @test S.spatial_area(r.state.spatial) ≈ exposure(map_g, 5)
        @test_throws ArgumentError run_collapsed_chain(locs_g; spatial_model = :flat,
            intensity_map = map_g, n_iterations = 5, burn_in = 1, verbose = false)
        # data-scaled ρ start when rho is not given, explicit rho respected
        r0 = run_collapsed_chain(locs_g; spatial_model = :ipp, n_iterations = 3,
            burn_in = 0, hierarchical_interval = 100, verbose = false,
            μ_prior_shape = 1.0, μ_prior_scale = 10.0)
        @test r0.ρ ≈ length(locs_g) / (10.0 * S.area(box_g))
        r1 = run_collapsed_chain(locs_g; spatial_model = :ipp, n_iterations = 3,
            burn_in = 0, rho = 4.0, verbose = false)
        @test r1.ρ == 4.0
    end

    @testset "P3 partition consistency (target factorizes across cuts)" begin
        N = length(locs_g)
        box = field_box(locs_g)
        fmap = build_intensity_map(locs_g, box)
        grp = vcat(fill(1, 5), fill(2, 4), fill(3, 5))
        @test length(grp) == N
        μ, shape, ρ = 8.0, 2.0, 3.0
        # Log product of partition targets for a cut (parts are vectors of group ids).
        function cut_target(parts, zfull)
            tot = 0.0
            for g in parts
                idx = findall(in(g), grp)
                pl = locs_g[idx]
                sp = IntensitySpatial(fmap, length(idx))
                z = S.canonicalize(zfull[idx])
                tot += S.log_target(IPPTarget(fmap), z, S.precompute_loc_precisions(pl),
                                    sp, μ, shape, ρ) + S.logfactorial(maximum(z))
            end
            tot
        end
        rng = Random.MersenneTwister(3)
        diffs = Float64[]
        for _ in 1:12
            # random assignment confined to the groups (labels unique per group)
            zfull = [grp[i] * 10 + rand(rng, 1:3) for i in 1:N]
            push!(diffs, cut_target([[1, 2], [3]], zfull) - cut_target([[1], [2, 3]],
                zfull))
        end
        # z-independent constant between cuts
        @test maximum(diffs) - minimum(diffs) < 1e-9
        # sanity: the target does depend on z within a cut
        z1 = [grp[i] * 10 + 1 for i in 1:N]; z2 = [grp[i] * 10 + (i % 3) for i in 1:N]
        @test abs(cut_target([[1, 2], [3]], z1) - cut_target([[1, 2], [3]], z2)) > 1e-3
    end

    @testset "P5 rho conditional: Gamma(a + ΣK, b + ΣA')" begin
        parts, _ = S.partition_locs(locs_g; max_size = 8, overlap = 0.05)
        @test length(parts) > 1
        box = field_box(locs_g)
        fmap = build_intensity_map(locs_g, box)
        sts = [S.initialize_collapsed_state(p.locs,
                   IntensitySpatial(fmap, length(p.locs)),
                   DMAllocation(nothing); use_poisson_k_prior = true) for p in parts]
        # several clusters per partition so ΣK is not just the partition count
        sts = [S.initialize_from_assignments(mod1.(1:length(p.locs), 3), p.locs;
                   sp = s.spatial, am = DMAllocation(nothing), use_poisson_k_prior = true)
               for (s, p) in zip(sts, parts)]
        @test sum(s.n_active for s in sts) > length(sts)
        areas = [S.spatial_area(s.spatial) for s in sts]
        cfg = (ρ_prior_shape = 2.0, ρ_prior_rate = 1.0)
        shp, rate = S._rho_posterior_params(sts, areas, cfg)
        @test shp == 2.0 + sum(s.n_active for s in sts)
        @test rate ≈ 1.0 + sum(exposure(fmap, length(p.locs)) for p in parts)
        @test rate ≈ 1.0 + sum(S.spatial_area(s.spatial) for s in sts)
    end

    @testset "IPPTarget conventions" begin
        # Labeled and canonical targets both equal DMFlatTarget at g ≡ 1
        sub = locs_g[1:6]
        bsub = S.UniformSpatialPrior(sub)
        cm = S.constant_intensity_map(bsub; n_all = 6)
        parts6 = enumerate_canonical_partitions(6, 6)
        p_ipp, lt_ipp = exact_posterior(parts6, sub, IPPTarget(cm);
            μ = 6.0, shape = 2.0, ρ = 3.0)
        p_flat, lt_flat = exact_posterior(parts6, sub, DMFlatTarget();
            μ = 6.0, shape = 2.0, ρ = 3.0)
        @test p_ipp ≈ p_flat atol = 1e-10
        @test lt_ipp ≈ lt_flat atol = 1e-10
        z = first(parts6)
        lps = S.precompute_loc_precisions(sub)
        @test log_target(IPPTarget(cm), z, lps, log(field_area(cm)), 6.0, 2.0, 3.0) ≈
              evaluate_target(IPPTarget(cm), z, sub; μ = 6.0, shape = 2.0, ρ = 3.0)
        lps = S.precompute_loc_precisions(sub)
        for z in parts6
            sp = IntensitySpatial(cm, length(sub))
            @test log_target(IPPTarget(cm), z, lps, sp, 6.0, 2.0, 3.0) ≈
                  log_target(DMFlatTarget(), z, lps, FlatSpatial(sp.log_area), 6.0, 2.0,
                      3.0)
        end
        # with the non-constant map the exact posterior differs from flat
        p_g, _ = exact_posterior(parts6, sub, IPPTarget(S.build_intensity_map(sub, bsub;
            bandwidth = 0.05)); μ = 6.0, shape = 2.0, ρ = 3.0)
        @test maximum(abs.(p_g .- p_flat)) > 1e-6
    end

    @testset "field box and map allocation guard" begin
        box = field_box(locs_g)
        @test (box.x_min, box.x_max, box.y_min, box.y_max) == (-0.01, 1.21, -0.01, 1.41)
        @test field_box(locs_g[1:0]) === nothing
        @test field_box(locs_g[1:1]) === nothing
        @test field_box(locs_g[[1, 3]]) === nothing
        # Check full-camera grid sizing without allocating the map.
        camera_box = UniformSpatialPrior(0.0, 205.0, 0.0, 205.0)
        _, _, _, nx, ny = S._intensity_grid(camera_box,
            S.DEFAULT_INTENSITY_BANDWIDTH)
        @test nx == ny
        @test nx * ny > 16_000_000
        @test_throws r"bandwidth=.*field" S._intensity_grid(
            UniformSpatialPrior(0.0, 50.0, 0.0, 50.0), 0.002)
        @test_throws r"bandwidth=.*field" build_intensity_map(locs_g, box;
            bandwidth = 0.002e-3)
        @test_throws ArgumentError S.constant_intensity_map(box; bandwidth = Inf)
        @test_throws ArgumentError S.constant_intensity_map(box; n_all = 0)
        @test S._initial_rho(nothing, false, 14, 10.0, 2.0, 2.0, 1.0) == 2.0
        @test S._initial_rho(nothing, true, 14, 10.0, 2.0, 2.0, 1.0) == 0.7
        @test S._initial_rho(4.0, true, 14, 10.0, 2.0, 2.0, 1.0) == 4.0
        @test S._data_scaled_rho_start(FeatureSet((FlatSpatial(log(2.0)),)))
    end

    @testset "archive codes and replay sentinel" begin
        replay_error = "pass spatial= to replay a spatial-model accumulator"
        mktempdir() do dir
            ar = BaGoLArchive(dir)
            @test ar.n_partitions == 0
            @test ar.spatial_model === :unrecorded
            @test isempty(ar.partition_sizes) && isempty(ar.sample_counts)
        end
        sp = IntensitySpatial(map_g, length(locs_g))
        state = S.initialize_from_assignments(mod1.(1:length(locs_g), 3), locs_g; sp)
        for model in (:unrecorded, :flat, :locmix, :ipp, :motion)
            @test S._archive_spatial_symbol(S._archive_spatial_code(model)) === model
            mktempdir() do dir
                writer = BaGoLArchive(dir, 1, [length(locs_g)]; spatial_model = model)
                S.write_sample!(writer, 1, state, 8.0, 2.0)
                close(writer)
                ar = BaGoLArchive(dir)
                @test ar.spatial_model === model
                @test S.compute_from_archive(UndeclaredAcc, ar, 1, locs_g) === nothing
                @test sum(S.compute_from_archive(EmitterCountHist, ar, 1, locs_g)) == 1
                kt = S.compute_from_archive(ChainDiagnosticAccumulator, ar, 1, locs_g)
                @test kt.k_trace == [3]
                if model === :unrecorded
                    @test all(isfinite, S.compute_from_archive(SpatialScoreAcc, ar, 1,
                        locs_g))
                else
                    @test_throws replay_error S.compute_from_archive(
                        SpatialScoreAcc, ar, 1, locs_g)
                end
                scores = S.compute_from_archive(SpatialScoreAcc, ar, 1, locs_g;
                    spatial = sp)
                expected = sum(spatial_ml(cs, sp) for cs in state.clusters if cs.n > 0)
                @test scores ≈ [expected]
            end
        end
        @test_throws ArgumentError S._archive_spatial_code(:unknown)
        @test_throws ErrorException S._archive_spatial_symbol(99)
        mktempdir() do dir
            writer = BaGoLArchive(dir, 2, [1, 1]; spatial_model = :ipp)
            close(writer)
            open(joinpath(dir, "partition_00002.bin"), "r+") do io
                seek(io, 24)
                write(io, S._archive_spatial_code(:flat))
            end
            @test_throws ErrorException BaGoLArchive(dir)
        end
        sentinel = S.ReplaySpatial()
        lps = S.precompute_loc_precisions(locs_g)
        cs = S.empty_cluster(lps)
        for use in (() -> spatial_ml(cs, sentinel),
                    () -> spatial_pred(cs, lps[1], sentinel),
                    () -> S.spatial_area(sentinel), () -> S.spatial_log_area(sentinel))
            @test_throws replay_error use()
        end
    end
end

using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions
import LinearAlgebra
import SMLMSim

@testset "Partitioned flat rho = emitter density over analyzed area" begin
    # Uniform field of single emitters from SMLMSim (Nmer2D n=1), σ ≈ 11 nm. The
    # pooled flat ρ update must learn the density of emitters that produced ≥ 1
    # localization over the bounding box of the localizations, independent of how
    # METIS chunks the field (padded/floored chunk areas used to give ρ ≈ 1/6 true).
    function rho_field(density, npix, seed)
        Random.seed!(seed)
        cam = SMLMData.IdealCamera(1:npix, 1:npix, 0.1)
        params = SMLMSim.StaticSMLMConfig(; density=density, nframes=300, framerate=50.0,
            ndatasets=1, σ_psf=0.13, minphotons=30)
        smld, _ = SMLMSim.simulate(params; pattern=SMLMSim.Nmer2D(n=1, d=0.0),
            molecule=SMLMSim.GenericFluor(photons=12000.0, k_off=50.0, k_on=0.5),
            camera=cam)
        xs = [e.x for e in smld.emitters]; ys = [e.y for e in smld.emitters]
        A = (maximum(xs) - minimum(xs)) * (maximum(ys) - minimum(ys))
        smld, length(unique(e.track_id for e in smld.emitters)) / A
    end
    learned_rho(smld, mps) = run_bagol(smld; spatial_model=:flat, learn_rho=true,
        max_partition_size=mps, ρ_prior_rate=0.01, n_iterations=4000, burn_in=2000,
        sync_interval=100, verbose=false)[2]

    smld, ρ_true = rho_field(250, 11, 1)   # ~1 µm², ~210 emitters/µm² visible
    d300 = learned_rho(smld, 300)
    d100 = learned_rho(smld, 100)
    @test d300.n_partitions > 1
    @test d100.n_partitions > d300.n_partitions
    @test abs(d300.final_ρ / ρ_true - 1) < 0.30            # P1
    @test abs(d100.final_ρ / ρ_true - 1) < 0.30
    @test abs(d300.final_ρ / d100.final_ρ - 1) < 0.20      # P2

    # Deterministic: area shares sum to A_field·(N with copies)/N_all, and a partition's
    # FlatSpatial area (used by its Poisson K prior via spatial_area) equals its share.
    let locs = smld.emitters
        parts, _ = SMLMBaGoL.partition_locs(locs; max_size=300)
        xs = extrema(e.x for e in locs); ys = extrema(e.y for e in locs)
        A_field = (xs[2] - xs[1]) * (ys[2] - ys[1])
        shares = [SMLMBaGoL._partition_area_share(A_field, length(p.locs), length(locs)) for p in parts]
        @test sum(shares) ≈ A_field * sum(length(p.locs) for p in parts) / length(locs)
        for (p, A) in zip(parts, shares)
            sp = SMLMBaGoL.FlatSpatial(log(A))
            @test SMLMBaGoL.spatial_area(sp) ≈ A
        end
    end

    # Degenerate (collinear, zero y-extent) field forced into several METIS partitions
    # WITH overlap copies: the update must use every active cluster and every partition
    # area, never the overlap-masked subset.
    let locs = [SMLMData.Emitter2DFit{Float64}(0.004 * i, 0.0, 1000.0, 10.0, 0.004, 0.004,
                    0.0, 0.0; frame=i) for i in 1:120]
        parts, _ = SMLMBaGoL.partition_locs(locs; partition_sigma=3.0, min_size=1,
                                            max_size=40, overlap=0.05)
        @test length(parts) > 1
        @test sum(count(p.is_overlap) for p in parts) > 0
        sts = [SMLMBaGoL.initialize_collapsed_state(p.locs,
                   SMLMBaGoL.FlatSpatial(log(SMLMBaGoL.area(SMLMBaGoL.UniformSpatialPrior(p.locs)))),
                   SMLMBaGoL.DMAllocation(nothing); use_poisson_k_prior=true) for p in parts]
        masks = SMLMBaGoL._compute_cluster_overlap_masks(sts, parts)
        # the fixture discriminates: some active cluster touches an overlap copy
        @test any(any(sts[i].active .& .!masks[i]) for i in eachindex(sts))
        areas = [SMLMBaGoL.spatial_area(s.spatial) for s in sts]
        cfg = (ρ_prior_shape=2.0, ρ_prior_rate=1.0)
        shp, rate = SMLMBaGoL._rho_posterior_params(sts, areas, cfg)
        @test shp == 2.0 + sum(s.n_active for s in sts)
        @test rate ≈ 1.0 + sum(areas)
        r, d = run_bagol(SMLMData.BasicSMLD(locs, SMLMData.IdealCamera(1:10, 1:10, 0.1), 1, 1);
            spatial_model=:flat, learn_rho=true, max_partition_size=40, overlap=0.05,
            n_iterations=200, burn_in=100, sync_interval=50, verbose=false)
        @test isfinite(d.final_ρ) && d.final_ρ > 0
    end

    smld_s, ρ_true_s = rho_field(3, 40, 2)                 # sparse, ~3 emitters/µm²
    d_s = learned_rho(smld_s, 300)
    @test abs(d_s.final_ρ / ρ_true_s - 1) < 0.30           # P3
    @info "flat ρ test" dense=(ρ_true, d300.final_ρ, d100.final_ρ) sparse=(ρ_true_s, d_s.final_ρ)
end

using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions
import LinearAlgebra
import SMLMSim

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

    @testset "estimate_se_adjust τ finder" begin
        # M-step kernel: at the true τ the within-group scaled neighbor distances
        # z = d/√(σa²+σb²+2τ²) are Rayleigh(1). Synthesize d = z·√(s2+2τ²) with
        # z ~ Rayleigh(1) and check the KS-minimizing τ (the M-step) recovers the
        # planted τ. Pure kernel — no BaGoL, no sim.
        rng = Random.MersenneTwister(20260620)
        npair = 4000
        s2_nm2 = fill(2 * 13.0^2, npair)               # σa²+σb² (nm²), 13 nm σ pairs
        grid_nm = collect(0.0:0.1:14.0)
        for tau_nm in (0.0, 6.0, 10.0)
            z = sqrt.(-2 .* log.(rand(rng, npair)))    # Rayleigh(1)
            d_nm = z .* sqrt.(s2_nm2 .+ 2 * tau_nm^2)
            tau_hat = SMLMBaGoL._mstep(d_nm, s2_nm2, grid_nm, eachindex(d_nm))
            @test abs(tau_hat - tau_nm) ≤ 1.0
        end

        # _ks_rayleigh1 ≥ 0 everywhere and minimized at the planted τ over the grid.
        z = sqrt.(-2 .* log.(rand(rng, 3000)))
        d6 = z .* sqrt.(s2_nm2[1:3000] .+ 2 * 6.0^2)
        ks = [SMLMBaGoL._ks_rayleigh1(d6, s2_nm2[1:3000], τ, eachindex(d6)) for τ in grid_nm]
        @test all(≥(0), ks)
        @test abs(grid_nm[argmin(ks)] - 6.0) ≤ 1.0

        # Right-biased M-step (production kernel): recovers the planted τ and is
        # ≥ the plain argmin M-step (largest τ within ks_noise of the KS min).
        tau_plain = SMLMBaGoL._mstep(d6, s2_nm2[1:3000], grid_nm, eachindex(d6))
        tau_rb = SMLMBaGoL._mstep_rb(d6, s2_nm2[1:3000], grid_nm, eachindex(d6))
        @test abs(tau_rb - 6.0) ≤ 1.5
        @test tau_rb ≥ tau_plain

        # Guards fail fast (before any BaGoL run).
        cam = SMLMData.IdealCamera(1:64, 1:64, 0.1)
        elist = [SMLMData.Emitter2DFit(0.5, 0.5, 1000.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 1, 1, 0, 1)]
        smld_corr = SMLMData.BasicSMLD(elist, cam, 1, 1, Dict{String,Any}("sigma_corrected" => true))
        @test_throws ArgumentError estimate_se_adjust(smld_corr)                  # already σ-corrected
        smld_raw = SMLMData.BasicSMLD(elist, cam, 1, 1, Dict{String,Any}())
        @test_throws ArgumentError estimate_se_adjust(smld_raw; grouping=:nope)   # unknown grouping

        # :dahl grouping (now the default; wired via _dahl_out): group original loc
        # indices by global Dahl label, skipping overlap-dup label 0.
        dgroups = SMLMBaGoL._dahl_label_groups([1, 1, 2, 0, 2])
        @test Set(Set.(dgroups)) == Set([Set([1, 2]), Set([3, 5])])

        # Spatial-block bootstrap resampler returns valid in-range indices.
        pos = [(0.1i, 0.1j) for i in 1:5 for j in 1:5]
        bi = SMLMBaGoL._block_indices(pos, 1.0, Random.MersenneTwister(1))
        @test !isempty(bi) && all(1 .≤ bi .≤ length(pos))
    end

    @testset "run_bagol(:auto) forwards run settings to the τ finder" begin
        # Small field of 6 two-loc pairs; the finder keeps its own chain length, so the
        # run is cheap only because the field is tiny.
        cam = SMLMData.IdealCamera(1:64, 1:64, 0.1)
        rng = Random.MersenneTwister(11)
        em = SMLMData.Emitter2DFit{Float64}[]
        for k in 1:6, j in 1:2
            push!(em, SMLMData.Emitter2DFit(0.5 + 0.4k + 0.01randn(rng), 1.0 + 0.01randn(rng),
                    1000.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 1, length(em) + 1, 0, 1))
        end
        smld = SMLMData.BasicSMLD(em, cam, 1, 1, Dict{String,Any}())
        _, diag = run_bagol(smld; se_adjust = :auto, keep_se_finder = true,
                            max_partition_size = 7, overlap = 0.02, spatial_model = :flat,
                            n_iterations = 100, burn_in = 20, verbose = false)
        kw = diag.se_finder.bagol_kwargs
        @test kw.max_partition_size == 7
        @test kw.overlap == 0.02
        @test kw.spatial_model === :flat
        @test !haskey(kw, :neighbor_graph) && !haskey(kw, :n_iterations)
        # Direct callers: explicitly given kwargs are used as given.
        f = estimate_se_adjust(smld; max_partition_size = 9, n_boot = 5, n_iterations = 100,
                               burn_in = 20, max_steps = 1)
        @test f.bagol_kwargs.max_partition_size == 9
        @test f.bagol_kwargs isa NamedTuple
        # The finder does not model drift: :auto with motion=:linear must refuse.
        @test_throws ArgumentError run_bagol(smld; se_adjust = :auto, motion = :linear,
                                             n_iterations = 100, burn_in = 20, verbose = false)
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
    # Linear motion model (per-emitter drift; opt-in motion=:linear)
    # ================================================================
    @testset "Linear motion model" begin
        SB = SMLMBaGoL
        mk2(x,y,sx,sy; sxy=0.0, f=1, k=0, id=1) =
            SMLMData.Emitter2DFit{Float64}(x,y,1e3,0.0,sx,sy,sxy,0.0,0.0,f,1,k,id)
        mk3(x,y,z,sx,sy,sz) = SMLMData.Emitter3DFit{Float64}(x,y,z,1e3,0.0,sx,sy,sz,0.0,0.0)

        # kernel: motion marginal vs brute-force numerical reference (2D)
        locs2 = [mk2(0.0,0.0,0.003,0.0035), mk2(0.004,-0.001,0.0025,0.003;sxy=0.0005), mk2(-0.002,0.003,0.004,0.0028)]
        δ2 = [-0.4,0.1,0.45]; σv = 0.003; logA = 0.0
        cs2 = SB.MotionClusterStats{2,4}()
        for (l,δ) in zip(locs2,δ2); cs2 = SB.add_loc(cs2, SB.MotionLocPrecision(SB._loc_precision(l), δ)); end
        cf2 = SB.spatial_ml(cs2, SB.motion_spatial(locs2, σv, logA, 0.0, 1.0))
        _static_shift(vx,vy) = begin
            csb = SB.ClusterStats{2,4}()
            for (l,δ) in zip(locs2,δ2)
                csb = SB.add_loc(csb, SB._loc_precision(mk2(l.x-δ*vx, l.y-δ*vy, l.σ_x, l.σ_y; sxy=l.σ_xy)))
            end
            SB.log_marginal_likelihood(csb, logA)
        end
        ng=121; Lg=6σv; vs=range(-Lg,Lg,length=ng); dv=Float64(vs.step)
        lN(vx,vy) = -log(2π) - 2log(σv) - 0.5*(vx^2+vy^2)/σv^2
        vals = Float64[_static_shift(vx,vy)+lN(vx,vy) for vx in vs for vy in vs]; mmax=maximum(vals)
        ref2 = mmax + log(sum(exp.(vals .- mmax))) + 2log(dv)
        @test isapprox(cf2, ref2; atol=1e-3)

        # static reduction σv→0 (2D and 3D) and add/remove inverse
        csb2 = SB.ClusterStats{2,4}(); for l in locs2; csb2 = SB.add_loc(csb2, SB._loc_precision(l)); end
        @test isapprox(SB.spatial_ml(cs2, SB.motion_spatial(locs2, 1e-6, logA, 0.0, 1.0)),
                       SB.log_marginal_likelihood(csb2, logA); atol=1e-5)
        locs3 = [mk3(0.0,0.0,0.0,0.003,0.003,0.004), mk3(0.001,0.0,0.002,0.003,0.0035,0.004), mk3(0.0,0.001,-0.001,0.0028,0.003,0.0045)]
        δ3 = [-0.4,0.0,0.4]
        cs3 = SB.MotionClusterStats{3,9}(); for (l,δ) in zip(locs3,δ3); cs3 = SB.add_loc(cs3, SB.MotionLocPrecision(SB._loc_precision(l), δ)); end
        csb3 = SB.ClusterStats{3,9}(); for l in locs3; csb3 = SB.add_loc(csb3, SB._loc_precision(l)); end
        @test isapprox(SB.spatial_ml(cs3, SB.motion_spatial(locs3, 1e-6, 0.5, 0.0, 1.0)),
                       SB.log_marginal_likelihood(csb3, 0.5); atol=1e-5)
        lp1 = SB.MotionLocPrecision(SB._loc_precision(locs2[1]), δ2[1])
        spm = SB.motion_spatial(locs2, σv, logA, 0.0, 1.0)
        @test isapprox(SB.spatial_ml(SB.remove_loc(SB.add_loc(cs2, lp1), lp1), spm), SB.spatial_ml(cs2, spm); atol=1e-9)

        # run_bagol motion=:linear: grouping + velocity output; recovers a 30nm-drifting mark
        function ruler_smld(; v2=(0.0,0.03), nf=200, Mb=30, σ=0.004, seed=7)
            rng = Random.MersenneTwister(seed); t0=(1+nf)/2; span=nf-1
            ls = SMLMData.Emitter2DFit{Float64}[]; id=1
            for (k,(x0,y0,vv)) in enumerate([(0.30,0.30,(0.0,0.0)),(0.32,0.30,v2)]), _ in 1:Mb
                f=rand(rng,1:nf); δ=(f-t0)/span
                push!(ls, mk2(x0+vv[1]*δ+randn(rng)*σ, y0+vv[2]*δ+randn(rng)*σ, σ, σ; f=f, k=k, id=id)); id+=1
            end
            SMLMData.BasicSMLD(ls, SMLMData.IdealCamera(1:100,1:100,0.1), nf, 1)
        end
        smld = ruler_smld()
        rm, dm = run_bagol(smld; motion=:linear, motion_sigma=0.015, n_iterations=2500, burn_in=1200, verbose=false)
        @test dm.motion !== nothing
        @test size(dm.motion.velocities, 2) == 2
        @test size(dm.motion.velocities, 1) == dm.n_emitters
        @test length(dm.motion.axis_mean) == 2 && length(dm.motion.axis_std) == 2
        @test maximum(abs.(dm.motion.velocities[:, 2])) > 0.010   # the drifting mark's y-velocity recovered
        # per-emitter n + velocity variance exposed (for downstream gating/weighting)
        @test size(dm.motion.velocity_var) == size(dm.motion.velocities) && all(dm.motion.velocity_var .>= 0)
        @test length(dm.motion.n) == dm.n_emitters
        # static run: motion field is nothing (no regression)
        _, ds = run_bagol(smld; n_iterations=2500, burn_in=1200, verbose=false)
        @test ds.motion === nothing

        # adversarial: two STATIC marks 20nm apart, active in DISJOINT frame windows, must NOT
        # be merged into one "moving" emitter under the (tight, default) velocity prior.
        function disjoint_smld(; nf=200, Mb=30, σ=0.004, seed=5)
            rng = Random.MersenneTwister(seed); ls = SMLMData.Emitter2DFit{Float64}[]; id=1
            for _ in 1:Mb; f=rand(rng,1:(nf÷2));    push!(ls, mk2(0.30+randn(rng)*σ,0.30+randn(rng)*σ,σ,σ; f=f, id=id)); id+=1; end
            for _ in 1:Mb; f=rand(rng,(nf÷2+1):nf); push!(ls, mk2(0.32+randn(rng)*σ,0.30+randn(rng)*σ,σ,σ; f=f, id=id)); id+=1; end
            SMLMData.BasicSMLD(ls, SMLMData.IdealCamera(1:100,1:100,0.1), nf, 1)
        end
        _, da = run_bagol(disjoint_smld(); motion=:linear, motion_sigma=0.002, n_iterations=2500, burn_in=1200, verbose=false)
        @test da.n_emitters >= 2
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

    @testset "Fixed rho control" begin
        Random.seed!(322)

        sim = simulate_localizations([(0.1, 0.1), (0.14, 0.1)];
            fixed_sigma=0.005, mean_count=4.0, count_model=:fixed,
            field_size=1.0, pixel_size=0.1)
        fixed_rho = 7.5

        _, diagnostics = run_bagol(sim.smld;
            spatial_model=:flat,
            k_prior=:poisson,
            learn_distribution=false,
            learn_rho=false,
            rho=fixed_rho,
            partition_sigma=Inf,
            sync_interval=20,
            n_iterations=120,
            burn_in=40,
            posterior_pixel_size=0.0,
            verbose=false)

        @test diagnostics.final_ρ == fixed_rho
        @test !isempty(diagnostics.convergence_trace.rho)
        @test all(==(fixed_rho), diagnostics.convergence_trace.rho)
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
    # Pass: observed MAP-K accuracy ≥ 80% of oracle (known μ, α), scored on the Dahl
    # consensus MAP-N (production estimator), which is robust to the locmix K-marginal's
    # diffuse upward bias that the peak-K estimator conflates with recovery error.
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
                    psm_acc = PSMAccumulator()
                    result = run_collapsed_chain(locs;
                        n_iterations=4000, burn_in=800,
                        shape=2.0, learn_distribution=true,
                        accumulators=AbstractAccumulator[ps_acc, psm_acc],
                        verbose=false)

                    # Score the PRODUCTION estimator (Dahl consensus), NOT peak-K.
                    # peak-K (estimate_mapn_collapsed = histogram-mode of per-iteration K) reads
                    # the locmix marginal's diffuse upward K-bias directly and regresses ~15-20pt
                    # once Fix A removes the compensating over-merge; Dahl reads a PSM consensus,
                    # is robust (mean-K stays at truth), and is what production/render ship.
                    # See dev/KNOWLEDGE_BASE.md (peak-K vs Dahl under Fix A).
                    samples = SMLMBaGoL.accumulator_result(ps_acc)
                    psm = SMLMBaGoL.accumulator_result(psm_acc).psm
                    emitters, _, _, _ = estimate_dahl(samples, locs, psm)
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

            # Regression: the flat + Poisson(ρA) K target must be AREA-INVARIANT.
            # log_prior_k_poisson's A^K cancels the flat ML's per-cluster A^{-K}, so the
            # normalized P(K) is independent of region area A (ρ held fixed). Before the
            # v0.4 fix the K prior lacked the +K·logA term and P(K) drifted with A.
            let lps_area = [SMLMBaGoL._loc_precision(l) for l in locs]
                pk_at = A -> begin
                    sp = SMLMBaGoL.FlatSpatial(log(A))
                    lts = [SMLMBaGoL.log_target(td_dm, z, lps_area, sp, 10.0, 2.0, 2.0) for z in parts]
                    m = maximum(lts); w = exp.(lts .- m); w ./= sum(w)
                    p = zeros(length(locs))
                    for (z, wi) in zip(parts, w); p[length(unique(z))] += wi; end
                    p
                end
                @test maximum(abs.(pk_at(1.0) .- pk_at(10.0))) < 1e-10
                @test maximum(abs.(pk_at(1.0) .- pk_at(0.05))) < 1e-10
            end

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
end

using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions
import LinearAlgebra
import SMLMSim

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

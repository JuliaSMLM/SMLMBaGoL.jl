using Test
using SMLMBaGoL
using SMLMData
using Random

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

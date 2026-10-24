using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions
import LinearAlgebra
import SMLMSim

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

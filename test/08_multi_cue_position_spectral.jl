using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions
import LinearAlgebra
import SMLMSim

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

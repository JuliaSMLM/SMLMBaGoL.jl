using Test
using SMLMBaGoL
using SMLMData
using Random

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

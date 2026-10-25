using Test
using SMLMBaGoL
using SMLMData
using Random

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

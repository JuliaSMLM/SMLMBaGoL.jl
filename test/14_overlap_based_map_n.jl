using Test
using SMLMBaGoL
using SMLMData
using Random

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

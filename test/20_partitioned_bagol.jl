using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions
import LinearAlgebra
import SMLMSim

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

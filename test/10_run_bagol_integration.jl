using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions
import LinearAlgebra
import SMLMSim

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

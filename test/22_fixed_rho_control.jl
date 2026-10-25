using Test
using SMLMBaGoL
using SMLMData
using Random

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

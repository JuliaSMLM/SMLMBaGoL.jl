using Test
using SMLMBaGoL
using SMLMData
using Random

@testset "Posterior Image (Collapsed)" begin
    Random.seed!(555)

    sim = simulate_localizations([(0.1, 0.1)];
        fixed_sigma=0.005, mean_count=10.0, count_model=:fixed,
        field_size=6.4, pixel_size=0.1)
    smld = sim.smld

    result_smld, diagnostics = run_bagol(smld;
        n_iterations=1000, burn_in=200, verbose=false,
        posterior_pixel_size=0.005)
    @test diagnostics.posterior_image !== nothing
    @test diagnostics.posterior_image.pixel_size == 0.005
    @test sum(diagnostics.posterior_image.image) > 0

    # Disable posterior image explicitly
    _, diag_default = run_bagol(smld;
        n_iterations=500, burn_in=100, verbose=false,
        posterior_pixel_size=0.0)
    @test diag_default.posterior_image === nothing
end

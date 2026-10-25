using Test
using SMLMBaGoL
using SMLMData

@testset "Locmix vs Flat Target Routing" begin
    # P0 fix verification: the sampler must route through spatial_ml /
    # spatial_pred dispatch on state.spatial, not hardcoded flat calls;
    # and the Poisson K prior + ρ hier update must be gated off under
    # LocmixSpatial per docs/math_reference.md.

    @testset "_uses_poisson_k_prior dispatch" begin
        # Flat uses the K prior + ρ update
        @test SMLMBaGoL._uses_poisson_k_prior(FlatSpatial(log(1.0)))
        # Locmix does not
        sim = simulate_localizations([(0.0, 0.0), (0.05, 0.0)];
            count_model=:fixed, mean_count=5.0,
            fixed_sigma=0.005)
        loc_precs = SMLMBaGoL.precompute_loc_precisions(sim.smld.emitters)
        grid = SMLMBaGoL.build_locmix_grid(loc_precs)
        @test !SMLMBaGoL._uses_poisson_k_prior(LocmixSpatial(grid))
    end

    @testset "_total_spatial_lml dispatches on spatial model" begin
        sim = simulate_localizations([(0.0, 0.0)];
            count_model=:fixed, mean_count=8.0,
            fixed_sigma=0.005)
        locs = sim.smld.emitters

        # Flat state
        state_flat = SMLMBaGoL.initialize_collapsed_state(locs,
            FlatSpatial(log(SMLMBaGoL.area(UniformSpatialPrior(locs)))))
        lml_flat = SMLMBaGoL._total_spatial_lml(state_flat)

        # Locmix state
        state_lm = SMLMBaGoL.initialize_collapsed_state(locs, LocmixSpatial(locs))
        lml_lm = SMLMBaGoL._total_spatial_lml(state_lm)

        # Flat and locmix produce different values for same allocation —
        # this fails if both paths are secretly using the flat formula.
        @test isfinite(lml_flat) && isfinite(lml_lm)
        @test lml_flat != lml_lm
    end

    @testset "Flat path still runs and converges" begin
        sim = simulate_localizations([(0.0, 0.0), (0.05, 0.0)];
            count_model=:fixed, mean_count=10.0,
            fixed_sigma=0.005)
        result = run_collapsed_chain(sim.smld.emitters;
            spatial_model=:flat, allocation_model=:dm,
            n_iterations=1000, burn_in=500,
            learn_distribution=false, shape=2.0,
            μ_prior_shape=2.0, μ_prior_scale=2.5,
            verbose=false)
        @test result.state.n_active >= 1
    end

    @testset "Locmix path runs and converges" begin
        sim = simulate_localizations([(0.0, 0.0), (0.05, 0.0)];
            count_model=:fixed, mean_count=10.0,
            fixed_sigma=0.005)
        result = run_collapsed_chain(sim.smld.emitters;
            spatial_model=:locmix, allocation_model=:dm,
            n_iterations=1000, burn_in=500,
            learn_distribution=false, shape=2.0,
            μ_prior_shape=2.0, μ_prior_scale=2.5,
            verbose=false)
        @test result.state.n_active >= 1
        @test isa(result.state.spatial, LocmixSpatial)
    end
end

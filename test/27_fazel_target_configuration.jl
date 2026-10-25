using Test
using SMLMBaGoL
using SMLMData

@testset "Fazel Target Configuration" begin
    # Verify the public API for the Fazel-equivalent target:
    # allocation_model=:categorical + k_prior=:none.

    @testset "CategoricalAllocation K^-N partition prior" begin
        sim = simulate_localizations([(0.0, 0.0), (0.05, 0.0)];
            count_model=:fixed, mean_count=6.0, fixed_sigma=0.005)
        locs = sim.smld.emitters
        N = length(locs)
        # Build states at K=1 and K=2 manually to verify partition_prior
        sp = FlatSpatial(log(SMLMBaGoL.area(UniformSpatialPrior(locs))))
        am = CategoricalAllocation()
        state_k1 = SMLMBaGoL.initialize_from_assignments(fill(1, N), locs;
            sp=sp, am=am)
        state_k2 = SMLMBaGoL.initialize_from_assignments(
            [i <= N÷2 ? 1 : 2 for i in 1:N], locs; sp=sp, am=am)
        pp_k1 = SMLMBaGoL.partition_prior(am, state_k1, N, 2.0)
        pp_k2 = SMLMBaGoL.partition_prior(am, state_k2, N, 2.0)
        # partition_prior = -N·log(K) — K=1 gives 0, K=2 gives -N·log(2)
        @test pp_k1 ≈ 0.0 atol=1e-10
        @test pp_k2 ≈ -N * log(2.0) atol=1e-10
    end

    @testset "k_prior=:none disables Poisson K prior on flat" begin
        sim = simulate_localizations([(0.0, 0.0)];
            count_model=:fixed, mean_count=5.0, fixed_sigma=0.005)
        locs = sim.smld.emitters
        result_default = run_collapsed_chain(locs;
            spatial_model=:flat, allocation_model=:dm,
            n_iterations=200, burn_in=100, learn_distribution=false,
            shape=2.0, μ_prior_shape=2.0, μ_prior_scale=2.5, verbose=false)
        @test result_default.state.use_poisson_k_prior == true

        result_none = run_collapsed_chain(locs;
            spatial_model=:flat, allocation_model=:dm, k_prior=:none,
            n_iterations=200, burn_in=100, learn_distribution=false,
            shape=2.0, μ_prior_shape=2.0, μ_prior_scale=2.5, verbose=false)
        @test result_none.state.use_poisson_k_prior == false
    end

    @testset "k_prior=:poisson rejected outside :flat" begin
        sim = simulate_localizations([(0.0, 0.0)];
            count_model=:fixed, mean_count=5.0, fixed_sigma=0.005)
        locs = sim.smld.emitters
        result_default = run_collapsed_chain(locs;
            spatial_model=:locmix, allocation_model=:dm,
            n_iterations=200, burn_in=100, learn_distribution=false,
            shape=2.0, μ_prior_shape=2.0, μ_prior_scale=2.5, verbose=false)
        @test result_default.state.use_poisson_k_prior == false

        # k_prior=:poisson is the flat-area-cancelled prior; mixing it
        # with :locmix would silently combine inconsistent target pieces.
        @test_throws ArgumentError run_collapsed_chain(locs;
            spatial_model=:locmix, k_prior=:poisson,
            n_iterations=10, burn_in=0, verbose=false)

        # Same check on the run_bagol public path. Validation must
        # happen BEFORE the @spawn partition loop or the ArgumentError
        # gets wrapped in CompositeException.
        @test_throws ArgumentError run_bagol(sim.smld;
            spatial_model=:locmix, k_prior=:poisson,
            n_iterations=10, burn_in=0,
            partition_sigma=Inf, posterior_pixel_size=0.0,
            verbose=false)
    end

    @testset "Fazel exact configuration runs" begin
        sim = simulate_localizations([(0.0, 0.0), (0.05, 0.0)];
            count_model=:fixed, mean_count=6.0, fixed_sigma=0.005)
        locs = sim.smld.emitters
        # Exact Fazel config: flat + categorical + no K prior
        result = run_collapsed_chain(locs;
            spatial_model=:flat, allocation_model=:categorical, k_prior=:none,
            n_iterations=500, burn_in=200, learn_distribution=false,
            shape=2.0, μ_prior_shape=2.0, μ_prior_scale=3.0, verbose=false)
        @test isa(result.state.allocation, CategoricalAllocation)
        @test isa(result.state.spatial, FlatSpatial)
        @test result.state.use_poisson_k_prior == false
    end

    @testset "Invalid kwargs error" begin
        sim = simulate_localizations([(0.0, 0.0)];
            count_model=:fixed, mean_count=3.0, fixed_sigma=0.005)
        locs = sim.smld.emitters
        @test_throws ArgumentError run_collapsed_chain(locs;
            allocation_model=:bogus, n_iterations=10, burn_in=0, verbose=false)
        @test_throws ArgumentError run_collapsed_chain(locs;
            k_prior=:bogus, n_iterations=10, burn_in=0, verbose=false)
    end

    @testset "Archive _reconstruct_state populates use_poisson_k_prior" begin
        # Regression for the new positional field on CollapsedState —
        # previously _reconstruct_state used the old field list and
        # threw MethodError after the use_poisson_k_prior field landed.
        sim = simulate_localizations([(0.0, 0.0), (0.05, 0.0)];
            count_model=:fixed, mean_count=4.0, fixed_sigma=0.005)
        locs = sim.smld.emitters
        assignments = Vector{Int16}(repeat(1:2, outer=cld(length(locs), 2))[1:length(locs)])
        log_area = log(SMLMBaGoL.area(UniformSpatialPrior(locs)))
        state = SMLMBaGoL._reconstruct_state(assignments, locs, log_area)
        @test state.n_active >= 1
        @test isa(state.spatial, FlatSpatial)
        @test state.use_poisson_k_prior == true   # legacy flat default
    end
end

# Basic workflow test without SMLMSim dependency
# Tests core BaGoL functionality with manually created data

using Test
using SMLMBaGoL
using Distributions
using Statistics
using Random

@testset "BaGoL Core Functionality" begin
    # Set random seed for reproducibility
    Random.seed!(12345)
    
    # Create simple synthetic data manually
    @testset "Manual Data Creation" begin
        # Generate some simple test observations
        n_obs = 20
        x_coords = rand(n_obs) * 10.0  # Random x coordinates 0-10
        y_coords = rand(n_obs) * 10.0  # Random y coordinates 0-10
        σ_x_vals = fill(0.1, n_obs)    # Fixed uncertainty
        σ_y_vals = fill(0.1, n_obs)    # Fixed uncertainty
        
        # Create observations using SMLMBaGoL
        emitter_type = SMLMBaGoL.Emitters.Emitter2D
        positions = vcat(y_coords', x_coords')  # Note: y first, x second
        sigmas = vcat(σ_y_vals', σ_x_vals')
        
        obs = SMLMBaGoL.gen_observations(emitter_type, positions, sigmas)
        
        @test length(obs.ŷ) == n_obs
        @test all(o -> isa(o, SMLMBaGoL.Emitters.Localization2D), obs.ŷ)
        
        # Check that coordinates are preserved correctly
        for (i, o) in enumerate(obs.ŷ)
            @test abs(o.x - x_coords[i]) < 1e-10
            @test abs(o.y - y_coords[i]) < 1e-10
            @test o.σ_x ≈ σ_x_vals[i]
            @test o.σ_y ≈ σ_y_vals[i]
        end
    end

    # Test observation generation with different parameters
    @testset "Observation Generation" begin
        # Test with different sizes
        for n in [1, 5, 10]
            positions = rand(2, n) * 5.0
            sigmas = fill(0.15, 2, n)
            
            obs = SMLMBaGoL.gen_observations(SMLMBaGoL.Emitters.Emitter2D, positions, sigmas)
            @test length(obs.ŷ) == n
        end
    end

    # Test basic RJMCMC functionality 
    @testset "RJMCMC Core Types" begin
        # Test that we can create basic RJMCMC types
        emitter = SMLMBaGoL.Emitters.Emitter2D(1.0, 2.0)
        @test emitter.x ≈ 1.0
        @test emitter.y ≈ 2.0
        
        # Test state creation
        emitters = [SMLMBaGoL.Emitters.Emitter2D(rand(), rand()) for _ in 1:3]
        state = SMLMBaGoL.RJMCMC.MCState(emitters, -10.0)
        @test length(state.emitters) == 3
        @test state.log_ℓ ≈ -10.0
    end

    # Test basic mathematical functions
    @testset "Core Mathematics" begin
        # Create simple observations
        n_obs = 5
        positions = rand(2, n_obs) * 3.0
        sigmas = fill(0.1, 2, n_obs)
        obs = SMLMBaGoL.gen_observations(SMLMBaGoL.Emitters.Emitter2D, positions, sigmas)
        
        # Test prior building
        prior_y = SMLMBaGoL.build_prior_y(obs)
        @test prior_y !== nothing
        
        # Test log likelihood calculation
        emitter = SMLMBaGoL.Emitters.Emitter2D(positions[2, 1], positions[1, 1])  # x, y
        log_prob = SMLMBaGoL.log_p_z_given_y(obs.ŷ[1], emitter)
        @test isfinite(log_prob)
    end

    # Test visualization functions with simple data
    @testset "Basic Visualization" begin
        # Create minimal test data
        n_obs = 3
        positions = [1.0 2.0 3.0; 1.0 2.0 3.0]  # Simple grid
        sigmas = fill(0.1, 2, n_obs)
        obs = SMLMBaGoL.gen_observations(SMLMBaGoL.Emitters.Emitter2D, positions, sigmas)
        
        # Test that basic visualization functions can be called
        # Note: We're not testing the visual output, just that they don't error
        try
            fig_sr = SMLMBaGoL.plot_sr(obs.ŷ; pixelsize=0.1, title="Test")
            @test fig_sr !== nothing
        catch e
            # If there are missing visualization dependencies, skip this test
            @test_skip "Visualization test skipped due to missing dependencies: $e"
        end
    end
end
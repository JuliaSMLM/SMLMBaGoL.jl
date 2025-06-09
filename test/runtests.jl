using Test
using SMLMBaGoL
using SMLMData
using Distributions

@testset "SMLMBaGoL Tests" begin
    
    # Include analysis tests
    include("test_analysis.jl")
    
    @testset "Basic API" begin
        # Create synthetic data
        observations = [
            Emitter2DFit(
                1.0 + 0.1*randn(), 1.0 + 0.1*randn(),  # x, y with noise
                1000.0, 100.0,                          # photons, bg
                0.1, 0.1,                               # σ_x, σ_y
                10.0, 10.0,                             # σ_photons, σ_bg
                i, 1, 1, i                              # frame, dataset, track_id, id
            ) for i in 1:20
        ]
        
        # Test basic bagol call
        result = bagol(observations; mcmc_steps=500, burnin=100, verbose=false)
        
        @test isa(result, BaGoLResult)
        @test length(result.chains) > 0
        @test !isempty(result.mapn_emitters)
        @test size(result.posterior, 1) > 0
        @test size(result.posterior, 2) > 0
        @test isa(result.updated_prior, HierarchicalPrior)
    end
    
    @testset "Generic Emitter Support" begin
        # Test with basic Emitter2D (no frame info)
        emitters = [Emitter2D(randn(), randn(), 1000.0) for _ in 1:10]
        
        result = bagol(emitters; mcmc_steps=200, burnin=50, verbose=false)
        @test isa(result, BaGoLResult)
    end
    
    @testset "Duck Typing" begin
        # Test with any object that has .emitters field
        observations = [Emitter2DFit(randn(), randn(), 1000.0, 100.0, 0.1, 0.1, 10.0, 10.0, i, 1, 1, i) 
                       for i in 1:10]
        
        smld = (; emitters = observations)  # Simple named tuple
        
        result = bagol(smld; mcmc_steps=200, burnin=50, verbose=false)
        @test isa(result, BaGoLResult)
    end
end

println("✅ All tests passed! SMLMBaGoL is ready to use.")
using Test
using SMLMData
using Distributions

# Include new implementation
include("../src/new/SMLMBaGoL.jl")
using .SMLMBaGoL

@testset "New BaGoL Design Tests" begin
    
    @testset "Basic Types" begin
        # Test HierarchicalPrior construction
        prior = HierarchicalPrior{Float64}()
        @test prior.α == 1.0
        @test prior.β == 1.0
        
        # Test MoveProbs construction
        probs = MoveProbs{Float64}()
        @test sum([probs.move, probs.add, probs.remove, 
                  probs.split, probs.merge, probs.reallocate]) ≈ 1.0
    end
    
    @testset "Simple RJMCMC Test" begin
        # Create synthetic data
        n_true = 3
        n_obs = 30
        
        # Generate observations around true emitters
        true_positions = [(1.0, 1.0), (2.0, 1.0), (1.5, 2.0)]
        observations = Emitter2DFit{Float64}[]
        
        for (x_true, y_true) in true_positions
            for i in 1:10
                x = x_true + 0.1 * randn()
                y = y_true + 0.1 * randn()
                push!(observations, Emitter2DFit(
                    x, y,           # x, y
                    1000.0, 100.0,  # photons, bg
                    0.1, 0.1,       # σ_x, σ_y
                    10.0, 10.0,     # σ_photons, σ_bg
                    i, 1, 1, i      # frame, dataset, track_id, id
                ))
            end
        end
        
        # Run BaGoL
        result = bagol(
            observations;
            mcmc_steps = 1000,
            burnin = 200,
            posterior_resolution = 0.05,
            verbose = false
        )
        
        @test isa(result, BaGoLResult)
        @test length(result.chains) > 0
        @test !isempty(result.mapn_emitters)
        @test size(result.posterior, 1) > 0
        @test size(result.posterior, 2) > 0
    end
    
    @testset "Hierarchical Prior Updates" begin
        prior = HierarchicalPrior{Float64}()
        
        # Create mock chain with known statistics
        observations = [Emitter2DFit(0.0, 0.0, 1000.0, 100.0, 0.1, 0.1, 10.0, 10.0, 1, 1, 1, 1) 
                       for _ in 1:10]
        
        states = [
            [Emitter2D(0.0, 0.0, 1000.0)],
            [Emitter2D(0.0, 0.0, 1000.0)],
        ]
        
        allocations = [
            fill(1, 10),  # All observations to emitter 1
            fill(1, 10),  # All observations to emitter 1
        ]
        
        chain = BaGoLChain(states, [0.0, 0.0], allocations, observations)
        
        # Update prior
        updated = SMLMBaGoL.update_hierarchical_prior([chain], prior)
        
        @test isa(updated, HierarchicalPrior)
        @test updated.α > 0
        @test updated.β > 0
    end
    
    @testset "Clustering" begin
        # Test DBSCAN clustering
        observations = [
            Emitter2DFit(0.0, 0.0, 1000.0, 100.0, 0.1, 0.1, 10.0, 10.0, 1, 1, 1, 1),
            Emitter2DFit(0.1, 0.1, 1000.0, 100.0, 0.1, 0.1, 10.0, 10.0, 2, 1, 1, 2),
            Emitter2DFit(5.0, 5.0, 1000.0, 100.0, 0.1, 0.1, 10.0, 10.0, 3, 1, 1, 3),
        ]
        
        subregions = SMLMBaGoL.cluster_subregions(observations, 1.0)
        
        @test length(subregions) == 2  # Two clusters
        @test length(subregions[1]) == 2 || length(subregions[2]) == 2
    end
    
    @testset "Direct SMLD Support" begin
        # Test that we can pass SMLD objects directly
        emitters = [Emitter2DFit(randn(), randn(), 1000.0, 100.0, 0.1, 0.1, 10.0, 10.0, i, 1, 1, i) 
                   for i in 1:10]
        
        # Create minimal SMLD structure
        smld = (; emitters = emitters)
        
        result = bagol(smld; mcmc_steps=100, burnin=50, verbose=false)
        @test isa(result, BaGoLResult)
    end
    
    @testset "Generic Emitter Support" begin
        # Test with basic Emitter2D (no frame info)
        emitters = [Emitter2D(randn(), randn(), 1000.0) for _ in 1:10]
        
        # This should work with any AbstractEmitter subtype
        result = bagol(emitters; mcmc_steps=100, burnin=50, verbose=false)
        @test isa(result, BaGoLResult)
    end
end

println("✅ All tests passed!")
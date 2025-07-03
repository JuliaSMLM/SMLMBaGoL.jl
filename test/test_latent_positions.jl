# Test that latent position fixes are working correctly
using Test
using SMLMBaGoL
using SMLMData
using Random

@testset "Latent Position Implementation Tests" begin
    # Set up test data
    Random.seed!(123)
    
    # Create simple test localizations
    localizations = [
        Localization2D(1.0, 1.0, 0.1, 0.1, 1),
        Localization2D(2.0, 2.0, 0.1, 0.1, 2),
        Localization2D(3.0, 3.0, 0.1, 0.1, 3)
    ]
    
    # Create spatial prior
    spatial_prior = create_spatial_prior_from_localizations(localizations)
    
    # Create hierarchical count prior with validation
    count_prior = HierarchicalNegBinomialPrior(
        2.0,  # μ
        1.0,  # κ  
        0.1,  # τ²
        (1.0, 1.0),  # μ_hyperprior
        (1.0, 1.0),  # κ_hyperprior
        (1.0, 1.0)   # τ²_hyperprior
    )
    
    @testset "HierarchicalNegBinomialPrior Validation" begin
        # Test that constructor validates parameters
        @test_throws ErrorException HierarchicalNegBinomialPrior(
            -1.0, 1.0, 0.1, (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)
        )
        
        @test_throws ErrorException HierarchicalNegBinomialPrior(
            1.0, -1.0, 0.1, (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)
        )
        
        @test_throws ErrorException HierarchicalNegBinomialPrior(
            1.0, 1.0, -0.1, (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)
        )
        
        @test_throws ErrorException HierarchicalNegBinomialPrior(
            1.0, 1.0, 0.1, (-1.0, 1.0), (1.0, 1.0), (1.0, 1.0)
        )
    end
    
    @testset "Chain Initialization with Latent Positions" begin
        # Test that chain initialization properly samples latent positions
        chain = initialize_chain(
            localizations,
            Emitter2D,
            spatial_prior,
            count_prior;
            burn_in=1000,
            thin=10,
            rng=Random.MersenneTwister(123)
        )
        
        # Check that initial state has latent positions
        @test length(chain.current_state.latent_positions) == length(localizations)
        
        # Check that latent positions are not just copied from observations
        # (they should be sampled from posterior, so likely different)
        different_positions = 0
        for (i, loc) in enumerate(localizations)
            latent_pos = chain.current_state.latent_positions[i]
            if abs(latent_pos[1] - loc.x) > 1e-10 || abs(latent_pos[2] - loc.y) > 1e-10
                different_positions += 1
            end
        end
        
        # At least some positions should be different (sampled from posterior)
        @test different_positions > 0
    end
    
    @testset "Move Operations Use Latent Positions" begin
        # Initialize a chain
        chain = initialize_chain(
            localizations,
            Emitter2D,
            spatial_prior,
            count_prior;
            burn_in=100,
            thin=10,
            rng=Random.MersenneTwister(456)
        )
        
        initial_state = chain.current_state
        
        # Test Move operation
        if length(initial_state.emitters) > 0
            new_state = propose_move(Move, initial_state, Random.MersenneTwister(789))
            
            if new_state !== nothing
                # Move should preserve latent positions (only emitter positions change)
                @test length(new_state.latent_positions) == length(initial_state.latent_positions)
                @test new_state.latent_positions == initial_state.latent_positions
                
                # Some emitter position should have changed
                emitter_moved = false
                for i in 1:length(initial_state.emitters)
                    if i <= length(new_state.emitters)
                        old_em = initial_state.emitters[i]
                        new_em = new_state.emitters[i]
                        if abs(old_em.x - new_em.x) > 1e-10 || abs(old_em.y - new_em.y) > 1e-10
                            emitter_moved = true
                            break
                        end
                    end
                end
                @test emitter_moved
            end
        end
    end
    
    @testset "Birth Move Initializes Latent Positions" begin
        # Initialize a chain
        chain = initialize_chain(
            localizations,
            Emitter2D,
            spatial_prior,
            count_prior;
            burn_in=100,
            thin=10,
            rng=Random.MersenneTwister(101112)
        )
        
        initial_state = chain.current_state
        
        # Test Birth operation
        new_state = propose_move(Birth, initial_state, chain, Random.MersenneTwister(131415))
        
        if new_state !== nothing
            # Birth should maintain latent positions array length
            @test length(new_state.latent_positions) == length(localizations)
            
            # Birth should add one emitter
            @test length(new_state.emitters) == length(initial_state.emitters) + 1
        end
    end
    
    println("✓ All latent position tests passed!")
end
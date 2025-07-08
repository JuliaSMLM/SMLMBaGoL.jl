using Test
using SMLMBaGoL
using Random
using SMLMData

# Import necessary types from SMLMBaGoL
using SMLMBaGoL: BaGoLState, UniformSpatialPrior, HierarchicalNegBinomialPrior, 
                 remove_empty_emitters, propose_move, Allocate

@testset "Emitter Utilities Tests" begin
    Random.seed!(42)
    
    # Create test data
    function create_test_state(n_emitters::Int, n_locs::Int, allocation_pattern::Vector{Int})
        emitters = [Emitter2D(Float64(i), Float64(i), 1000.0) for i in 1:n_emitters]
        localizations = [Localization2D(rand(), rand(), 0.1, 0.1, 1) for _ in 1:n_locs]
        latent_positions = [(rand(), rand()) for _ in 1:n_locs]
        
        spatial_prior = UniformSpatialPrior(0.0, 10.0, 0.0, 10.0)
        count_prior = HierarchicalNegBinomialPrior(
            Float64(n_locs / n_emitters),  # μ: mean locs per emitter
            1.0,   # κ: overdispersion
            0.001, # τ²: additional variance
            (1.0, 1.0),  # μ hyperprior
            (1.0, 1.0),  # κ hyperprior
            (1.0, 1.0)   # τ² hyperprior
        )
        
        state = BaGoLState(
            emitters,
            localizations,
            allocation_pattern,
            latent_positions,
            spatial_prior,
            count_prior,
            0.001,
            0.0  # dummy likelihood
        )
        return state
    end
    
    @testset "remove_empty_emitters" begin
        @testset "No empty emitters" begin
            # All emitters have allocations
            state = create_test_state(3, 6, [1, 1, 2, 2, 3, 3])
            cleaned = remove_empty_emitters(state)
            
            @test length(cleaned.emitters) == 3
            @test cleaned.allocations == state.allocations
            @test cleaned.emitters == state.emitters
        end
        
        @testset "Single empty emitter" begin
            # Emitter 2 has no allocations
            state = create_test_state(3, 4, [1, 1, 3, 3])
            cleaned = remove_empty_emitters(state)
            
            @test length(cleaned.emitters) == 2
            @test cleaned.allocations == [1, 1, 2, 2]  # 3 becomes 2
            @test cleaned.emitters[1] == state.emitters[1]
            @test cleaned.emitters[2] == state.emitters[3]  # Old emitter 3 is now 2
        end
        
        @testset "Multiple empty emitters" begin
            # Emitters 2 and 4 have no allocations
            state = create_test_state(5, 6, [1, 1, 3, 3, 5, 5])
            cleaned = remove_empty_emitters(state)
            
            @test length(cleaned.emitters) == 3
            @test cleaned.allocations == [1, 1, 2, 2, 3, 3]  # 3->2, 5->3
            @test cleaned.emitters[1] == state.emitters[1]
            @test cleaned.emitters[2] == state.emitters[3]
            @test cleaned.emitters[3] == state.emitters[5]
        end
        
        @testset "All emitters empty" begin
            # Edge case: no allocations to any emitter
            state = create_test_state(3, 3, [0, 0, 0])
            cleaned = remove_empty_emitters(state)
            
            @test length(cleaned.emitters) == 0
            @test all(alloc == 0 for alloc in cleaned.allocations)
        end
        
        @testset "Preserves other state components" begin
            state = create_test_state(3, 4, [1, 1, 3, 3])
            cleaned = remove_empty_emitters(state)
            
            # Check that other components are preserved
            @test cleaned.localizations === state.localizations
            @test cleaned.latent_positions === state.latent_positions
            @test cleaned.spatial_prior === state.spatial_prior
            @test cleaned.count_prior === state.count_prior
            @test cleaned.τ² == state.τ²
        end
        
        @testset "Handles out-of-range allocations" begin
            # Some allocations might be 0 or > n_emitters
            allocations = [0, 1, 1, 4, 5]  # 4 and 5 are out of range for 3 emitters
            state = create_test_state(3, 5, allocations)
            cleaned = remove_empty_emitters(state)
            
            @test length(cleaned.emitters) == 1  # Only emitter 1 has valid allocations
            @test cleaned.allocations[1] == 0  # Stays 0
            @test cleaned.allocations[2] == 1  # Maps to new index 1
            @test cleaned.allocations[3] == 1  # Maps to new index 1
            @test cleaned.allocations[4] == 4  # Stays out of range
            @test cleaned.allocations[5] == 5  # Stays out of range
        end
    end
    
    @testset "Integration with allocate move" begin
        # Create a state where reallocation might leave empty emitters
        emitters = [Emitter2D(1.0, 1.0, 1000.0), 
                   Emitter2D(5.0, 5.0, 1000.0),
                   Emitter2D(9.0, 9.0, 1000.0)]
        
        # All localizations near emitter 1
        localizations = [Localization2D(1.0 + 0.1*randn(), 1.0 + 0.1*randn(), 0.1, 0.1, 1) 
                        for _ in 1:10]
        
        latent_positions = [(loc.x, loc.y) for loc in localizations]
        allocations = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]  # Initially spread
        
        spatial_prior = UniformSpatialPrior(0.0, 10.0, 0.0, 10.0)
        count_prior = HierarchicalNegBinomialPrior(
            10.0/3.0,  # μ: mean locs per emitter (10 locs / 3 emitters)
            1.0,   # κ: overdispersion
            0.001, # τ²: additional variance
            (1.0, 1.0),  # μ hyperprior
            (1.0, 1.0),  # κ hyperprior
            (1.0, 1.0)   # τ² hyperprior
        )
        
        state = BaGoLState(
            emitters,
            localizations,
            allocations,
            latent_positions,
            spatial_prior,
            count_prior,
            0.001,
            0.0
        )
        
        # Run allocate move
        new_state = propose_move(Allocate, state)
        
        # Check that no emitters are empty
        allocation_counts = zeros(Int, length(new_state.emitters))
        for alloc in new_state.allocations
            if 1 ≤ alloc ≤ length(new_state.emitters)
                allocation_counts[alloc] += 1
            end
        end
        
        @test all(count > 0 for count in allocation_counts)
        @test length(new_state.emitters) ≤ length(state.emitters)
    end
end
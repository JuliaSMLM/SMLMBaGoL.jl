# Test analysis functions
using Test
using SMLMBaGoL
using SMLMData
using Random

@testset "Analysis Functions" begin
    
    @testset "Chain Statistics" begin
        # Create a simple test chain
        Random.seed!(42)
        
        # Generate simple test data
        n_obs = 20
        observations = [Emitter2D(Float64(rand()*10), Float64(rand()*10), Float64(100 + 50*rand())) for _ in 1:n_obs]
        
        # Create mock chain data
        n_samples = 100
        states = Vector{Vector{Emitter2D{Float64}}}()
        log_probs = Vector{Float64}()
        allocations = Vector{Vector{Int}}()
        
        # Simulate a simple chain with varying numbers of emitters
        for i in 1:n_samples
            # Vary emitter count slightly around 5
            n_emitters = max(1, 5 + rand(-2:2))
            
            # Create random emitters
            state = [Emitter2D(Float64(rand()*10), Float64(rand()*10), Float64(100 + 50*rand())) 
                    for _ in 1:n_emitters]
            push!(states, state)
            
            # Mock log probability (should generally increase)
            log_prob = -1000.0 - 100*rand() + i*0.1  # Slight upward trend
            push!(log_probs, log_prob)
            
            # Create random allocations
            alloc = [rand(1:n_emitters) for _ in 1:n_obs]
            push!(allocations, alloc)
        end
        
        # Create test chain
        chain = BaGoLChain(states, log_probs, allocations, observations)
        
        # Test statistics function
        stats = statistics(chain)
        
        @test stats isa ChainStatistics{Float64}
        @test stats.chain_length == n_samples
        @test stats.mean_emitters > 0
        @test stats.std_emitters >= 0
        @test 0 <= stats.overall_acceptance <= 1
        @test stats.effective_sample_size > 0
        @test stats.autocorr_time >= 0
        @test isfinite(stats.geweke_z_score)
        @test isfinite(stats.mean_log_prob)
        @test stats.std_log_prob >= 0
        @test isfinite(stats.log_prob_trend)
        @test stats.acceptance_rate_stability >= 0
        @test 0 <= stats.state_transition_rate <= 1
        
        # Test summary function doesn't error
        @test_nowarn summary(stats)
    end
    
    @testset "Acceptance Rate Estimation" begin
        # Test with known state changes
        observations = [Emitter2D(1.0, 1.0, 100.0), Emitter2D(2.0, 2.0, 100.0)]
        
        # Create states with known patterns
        state1 = [Emitter2D(1.0, 1.0, 100.0)]  # 1 emitter
        state2 = [Emitter2D(1.0, 1.0, 100.0), Emitter2D(2.0, 2.0, 100.0)]  # Add emitter
        state3 = [Emitter2D(1.1, 1.1, 100.0), Emitter2D(2.0, 2.0, 100.0)]  # Move emitter
        state4 = [Emitter2D(1.1, 1.1, 100.0)]  # Remove emitter
        
        states = [state1, state2, state3, state4]
        log_probs = [-100.0, -95.0, -94.0, -96.0]
        allocations = [[1, 1], [1, 2], [1, 2], [1, 1]]
        
        chain = BaGoLChain(states, log_probs, allocations, observations)
        rates = SMLMBaGoL.estimate_acceptance_rates(chain)
        
        @test rates isa SMLMBaGoL.AcceptanceRates{Float64}
        @test rates.add > 0  # Should detect the add move
        @test rates.remove > 0  # Should detect the remove move
        @test rates.move > 0  # Should detect the position move
        @test 0 <= rates.overall <= 1
    end
    
    @testset "Convergence Diagnostics" begin
        # Test effective sample size
        x1 = randn(100)  # Uncorrelated
        ess1 = SMLMBaGoL.effective_sample_size(x1)
        @test ess1 > 50  # Should be reasonably high for uncorrelated data
        
        # Test with highly correlated data
        x2 = cumsum(randn(100) * 0.1)  # Highly correlated
        ess2 = SMLMBaGoL.effective_sample_size(x2)
        @test ess2 < ess1  # Should be lower than uncorrelated
        
        # Test autocorrelation time
        τ1 = SMLMBaGoL.autocorrelation_time(x1)
        τ2 = SMLMBaGoL.autocorrelation_time(x2)
        @test τ1 >= 0
        @test τ2 >= 0
        @test τ2 > τ1  # Correlated data should have longer autocorr time
        
        # Test Geweke diagnostic
        z1 = SMLMBaGoL.geweke_diagnostic(x1)
        @test isfinite(z1)
        @test abs(z1) < 5  # Should be reasonable for random data
        
        # Test with trending data
        x3 = collect(1:100) .+ randn(100) * 0.1
        z3 = SMLMBaGoL.geweke_diagnostic(x3)
        @test isfinite(z3)
        @test abs(z3) > abs(z1)  # Should detect the trend
    end
    
    @testset "State Comparison Functions" begin
        # Test state difference detection
        s1 = [Emitter2D(1.0, 1.0, 100.0), Emitter2D(2.0, 2.0, 100.0)]
        s2 = [Emitter2D(1.0, 1.0, 100.0), Emitter2D(2.0, 2.0, 100.0)]  # Same
        s3 = [Emitter2D(1.1, 1.0, 100.0), Emitter2D(2.0, 2.0, 100.0)]  # Different
        s4 = [Emitter2D(1.0, 1.0, 100.0)]  # Different length
        
        @test !SMLMBaGoL.states_different(s1, s2)
        @test SMLMBaGoL.states_different(s1, s3)
        @test SMLMBaGoL.states_different(s1, s4)
        
        # Test state similarity
        @test SMLMBaGoL.states_similar(s1, s2)
        @test !SMLMBaGoL.states_similar(s1, s4)
        @test SMLMBaGoL.states_similar(s1, s3, 0.2)  # With larger tolerance
        
        # Test allocation differences
        a1 = [1, 2, 1]
        a2 = [1, 2, 1]
        a3 = [1, 2, 2]
        
        @test !SMLMBaGoL.allocations_different(a1, a2)
        @test SMLMBaGoL.allocations_different(a1, a3)
        
        # Test split/merge pattern detection
        prev = [Emitter2D(1.0, 1.0, 100.0)]
        curr_split = [Emitter2D(1.0, 1.0, 50.0), Emitter2D(1.05, 1.05, 50.0)]  # Close together
        curr_add = [Emitter2D(1.0, 1.0, 100.0), Emitter2D(5.0, 5.0, 100.0)]  # Far apart
        
        @test SMLMBaGoL.detect_split_pattern(prev, curr_split)
        @test !SMLMBaGoL.detect_split_pattern(prev, curr_add)
        
        @test SMLMBaGoL.detect_merge_pattern(curr_split, prev)
        @test SMLMBaGoL.detect_merge_pattern(curr_add, prev)
    end
    
    @testset "Multi-Chain Analysis" begin
        # Create multiple test chains
        Random.seed!(123)
        observations = [Emitter2D(Float64(rand()*10), Float64(rand()*10), Float64(100)) for _ in 1:10]
        
        chains = BaGoLChain{Float64, Emitter2D{Float64}}[]
        
        for chain_idx in 1:3
            states = Vector{Vector{Emitter2D{Float64}}}()
            log_probs = Vector{Float64}()
            allocations = Vector{Vector{Int}}()
            
            for i in 1:50
                n_emitters = max(1, 3 + rand(-1:1))
                state = [Emitter2D(Float64(rand()*10), Float64(rand()*10), Float64(100)) 
                        for _ in 1:n_emitters]
                push!(states, state)
                push!(log_probs, -500.0 - 50*rand() + i*0.05)
                push!(allocations, [rand(1:n_emitters) for _ in 1:10])
            end
            
            push!(chains, BaGoLChain(states, log_probs, allocations, observations))
        end
        
        # Create mock result
        posterior = zeros(Float64, 20, 20)
        mapn_emitters = [Emitter2D(5.0, 5.0, 100.0)]
        log_evidence = -1000.0
        updated_prior = HierarchicalPrior{Float64}()
        
        result = BaGoLResult(chains, posterior, mapn_emitters, log_evidence, updated_prior)
        
        # Test multi-chain statistics
        chain_stats = statistics(result)
        @test length(chain_stats) == 3
        @test all(s -> s isa ChainStatistics{Float64}, chain_stats)
        
        # Test multi-chain summary
        @test_nowarn summary(chain_stats)
        @test_nowarn convergence_summary(result)
    end
    
    @testset "Edge Cases" begin
        # Test empty chain
        empty_observations = Emitter2D{Float64}[]
        empty_chain = BaGoLChain(Vector{Vector{Emitter2D{Float64}}}(), 
                                Float64[], Vector{Vector{Int}}(), empty_observations)
        @test_throws Exception statistics(empty_chain)
        
        # Test single sample chain
        observations = [Emitter2D(1.0, 1.0, 100.0)]
        single_state = [Emitter2D(1.0, 1.0, 100.0)]
        single_chain = BaGoLChain([single_state], [-100.0], [[1]], observations)
        
        stats = statistics(single_chain)
        @test stats.chain_length == 1
        @test stats.effective_sample_size == 1.0
        @test stats.autocorr_time == 0.0
        
        # Test with very short time series for diagnostics
        short_x = [1.0, 2.0, 3.0]
        @test SMLMBaGoL.autocorrelation_time(short_x) >= 0
        @test isfinite(SMLMBaGoL.geweke_diagnostic(short_x))
        @test SMLMBaGoL.effective_sample_size(short_x) > 0
    end
end
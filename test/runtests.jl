using Test
using SMLMBaGoL
using SMLMData
using Random

@testset "SMLMBaGoL.jl Tests" begin
    @testset "Basic Types" begin
        emitter = Emitter(0.1, 0.2)
        @test emitter.x == 0.1
        @test emitter.y == 0.2
        @test isempty(emitter.allocated)
    end

    @testset "Simple Integration" begin
        Random.seed!(123)

        # Create simple test data: 2 well-separated emitters
        locs = SMLMData.Emitter2DFit[]
        σ = 0.005

        # Emitter 1 at (0.1, 0.1)
        for i in 1:5
            x = 0.1 + randn() * σ
            y = 0.1 + randn() * σ
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 1, 1, 0, i))
        end

        # Emitter 2 at (0.2, 0.2)
        for i in 1:5
            x = 0.2 + randn() * σ
            y = 0.2 + randn() * σ
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 1, 1, 0, i+5))
        end

        # Run with minimal iterations for testing
        chain = run_bagol(locs;
            τ = 0.005,
            n_iterations = 500,
            burn_in = 100,
            verbose = false
        )

        @test length(chain.samples) > 0

        result = estimate_mapn(chain)
        # Smoke test: just verify it returns a valid result
        @test result.n_emitters >= 0
        @test length(result.emitters) == result.n_emitters
        @test length(result.uncertainties) == result.n_emitters
    end
end

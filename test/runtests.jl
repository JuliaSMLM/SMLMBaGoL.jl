using Test

# Run all test files
@testset "SMLMBaGoL.jl Tests" begin
    include("test_latent_positions.jl")
    include("test_hierarchical_k_math.jl")
end
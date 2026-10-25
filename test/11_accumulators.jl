using Test
using SMLMBaGoL
using SMLMData

# ================================================================
# Accumulators
# ================================================================
@testset "Accumulators" begin
    # EmitterCountHist merge
    h1 = EmitterCountHist()
    h1.counts = [10, 20, 5]
    h2 = EmitterCountHist()
    h2.counts = [5, 15, 10, 3]

    SMLMBaGoL.accumulator_merge!(h1, h2)
    @test h1.counts == [15, 35, 15, 3]

    # NNDistHist basic
    nn = NNDistHist(max_dist=0.1, n_bins=50)
    @test length(nn.counts) == 50
end

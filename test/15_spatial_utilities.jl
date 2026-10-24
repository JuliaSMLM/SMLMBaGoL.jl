using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions
import LinearAlgebra
import SMLMSim

@testset "Spatial Utilities" begin
    loc = SMLMData.Emitter2DFit(0.1, 0.2, 1000.0, 10.0, 0.005, 0.006, 0.0, 0.0, 0.0, 1, 1, 0, 1)
    coords = SMLMBaGoL.get_coords(loc)
    @test coords[1] ≈ 0.1
    @test coords[2] ≈ 0.2

    sigma = SMLMBaGoL.get_sigma(loc)
    @test sigma[1] ≈ 0.005
    @test sigma[2] ≈ 0.006

    ms = SMLMBaGoL.mean_sigma(loc)
    @test ms ≈ sqrt(0.005 * 0.006)

    loc2 = SMLMData.Emitter2DFit(0.2, 0.2, 1000.0, 10.0, 0.005, 0.005, 0.0, 0.0, 0.0, 1, 1, 0, 2)
    d_eff = SMLMBaGoL.precision_weighted_distance(loc, loc2)
    @test d_eff > 0
end

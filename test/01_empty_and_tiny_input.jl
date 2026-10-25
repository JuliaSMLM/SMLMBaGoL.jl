using Test
using SMLMBaGoL
using SMLMData

# ================================================================
# Empty / tiny input — must not crash (graceful empty result)
# Regression: run_bagol returned abstract-eltype Vector{Emitter2DFit},
# so a downstream reduction over an empty result hit zero(Type{Any}).
# ================================================================
@testset "Empty and tiny input" begin
    cam = SMLMData.IdealCamera(1:64, 1:64, 0.1)
    mksmld(n) = SMLMData.BasicSMLD(
        [SMLMData.Emitter2DFit(0.5 + 0.02i, 0.5, 1000.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0,
            1, 1, 0, i) for i in 1:n],
        cam, 1, 1, Dict{String,Any}())

    # 0 localizations: graceful empty result, not a crash
    r0, d0 = run_bagol(mksmld(0); n_iterations=100, burn_in=20, verbose=false)
    @test length(r0.emitters) == 0
    @test d0.n_emitters == 0
    # Concrete eltype so downstream reductions don't fall to zero(Type{Any})
    @test isconcretetype(eltype(r0.emitters))
    @test eltype(r0.emitters) == SMLMData.Emitter2DFit{Float64}
    @test sum([e.x for e in r0.emitters]) == 0.0   # the exact crashing pattern

    # 1 and 2 localizations: run without crashing
    for n in (1, 2)
        r, d = run_bagol(mksmld(n); n_iterations=100, burn_in=20, verbose=false)
        @test isconcretetype(eltype(r.emitters))
        @test d.n_emitters >= 0
    end
end

using Test
using SMLMBaGoL
using SMLMData

# run_bagol's pipeline is 2D: any localization carrying z is refused with an
# ArgumentError instead of being grouped on x, y and returned without z.

# A fitter's own 3D type (the field layout of GaussMLE's Emitter3DFitGaussMLE): a 3D fit
# that is not SMLMData.Emitter3DFit.
mutable struct Fit3DLike{T} <: SMLMData.AbstractEmitter
    x::T; y::T; z::T
    photons::T; bg::T
    σ_x::T; σ_y::T; σ_z::T; σ_xy::T; σ_xz::T; σ_yz::T; σ_photons::T; σ_bg::T
    pvalue::T
    frame::Int; dataset::Int; track_id::Int; id::Int
end

@testset "3D localizations are refused" begin
    cam = SMLMData.IdealCamera(1:32, 1:32, 0.1)
    # Two molecules at the same x, y, 0.6 μm apart in z, ten localizations each.
    xy = [(1.0 + 0.002 * (i % 3), 1.0 + 0.002 * (i % 5)) for i in 1:20]
    zs = [i <= 10 ? 0.0 : 0.6 for i in 1:20]
    l3 = [SMLMData.Emitter3DFit{Float64}(xy[i]..., zs[i], 1000.0, 10.0, 0.01, 0.01, 0.03,
              0.0, 0.0, 0.0, 0.0, 0.0, i, 1, 0, i) for i in 1:20]
    lg = [Fit3DLike{Float64}(xy[i]..., zs[i], 1000.0, 10.0, 0.01, 0.01, 0.03,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.5, i, 1, 0, i) for i in 1:20]
    for locs in (l3, lg)
        smld = SMLMData.BasicSMLD(locs, cam, 20, 1)
        @test_throws ArgumentError run_bagol(smld; n_iterations = 50, burn_in = 10, verbose = false)
        @test_throws ArgumentError run_bagol(locs; camera = cam, n_iterations = 50, burn_in = 10,
                                             verbose = false)
        @test_throws ArgumentError estimate_se_adjust(smld)
        err = try
            run_bagol(smld; n_iterations = 50, burn_in = 10, verbose = false); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("z coordinate", err.msg)
    end
    # The chain entry point refuses a 3D type it would run as 2D (SMLMData's Emitter3DFit has
    # its own 3D chain path); the guard sits where every path builds per-localization precision.
    @test_throws ArgumentError run_collapsed_chain(lg; n_iterations = 20, burn_in = 5)
    # A mixed vector holding one 3D localization is refused too.
    mixed = SMLMData.AbstractEmitter[SMLMData.Emitter2DFit{Float64}(1.0, 1.0, 1000.0, 10.0,
                                         0.01, 0.01, 0.0, 0.0), l3[1]]
    @test_throws ArgumentError SMLMBaGoL._require_2d(mixed, "run_bagol")
    # 2D input passes the check.
    l2 = [SMLMData.Emitter2DFit{Float64}(xy[i]..., 1000.0, 10.0, 0.01, 0.01, 0.0, 0.0;
              frame = i) for i in 1:20]
    @test SMLMBaGoL._require_2d(l2, "run_bagol") === nothing
end

using Test
using SMLMBaGoL
using SMLMData

# ================================================================
# se_adjust — independent-error σ correction (quadrature, per-axis, guarded)
# ================================================================
@testset "se_adjust uncertainty correction" begin
    cam = SMLMData.IdealCamera(1:64, 1:64, 0.1)
    mk(x, y, sx, sy) = SMLMData.Emitter2DFit(x, y, 1000.0, 0.0, sx, sy, 0.0, 0.0, 0.0,
        1, 1, 0, 1)

    # _resolve_tau: scalar / per-axis tuple / per-loc vector / per-loc per-axis
    τx, τy = SMLMBaGoL._resolve_tau(0.01, 3)
    @test τx == fill(0.01, 3) && τy == fill(0.01, 3)
    τx, τy = SMLMBaGoL._resolve_tau((0.0047, 0.0076), 4)
    @test τx == fill(0.0047, 4) && τy == fill(0.0076, 4)
    τx, τy = SMLMBaGoL._resolve_tau([0.001, 0.002], 2)
    @test τx == [0.001, 0.002] && τy == [0.001, 0.002]
    τx, τy = SMLMBaGoL._resolve_tau(([0.001, 0.002], [0.003, 0.004]), 2)
    @test τx == [0.001, 0.002] && τy == [0.003, 0.004]
    @test_throws ArgumentError SMLMBaGoL._resolve_tau([0.1, 0.2, 0.3], 2)  # wrong length
    @test_throws ArgumentError SMLMBaGoL._resolve_tau(-0.01, 2)            # negative τ

    # _inflate_sigma: exact quadrature, per-axis; σ_xy + position untouched
    e = mk(1.0, 2.0, 0.003, 0.004)
    e2 = SMLMBaGoL._inflate_sigma(e, 0.004, 0.003)
    @test e2 isa SMLMData.Emitter2DFit{Float64}
    @test e2.σ_x ≈ sqrt(0.003^2 + 0.004^2)
    @test e2.σ_y ≈ sqrt(0.004^2 + 0.003^2)
    @test e2.σ_xy == e.σ_xy
    @test e2.x == e.x && e2.y == e.y

    # apply-once guard via metadata stamp (2nd return = applied (τx,τy) or nothing)
    locs = [mk(1.0, 1.0, 0.005, 0.005)]
    stamped = Dict{String,Any}("sigma_corrected" => true, "tau_x_nm" => 3.0)
    out, tau, _ = SMLMBaGoL._maybe_apply_se_adjust(locs, stamped, 0.004, false)
    @test tau === nothing && out === locs                  # skipped (guard)
    out, tau, _ = SMLMBaGoL._maybe_apply_se_adjust(locs, stamped, 0.004, true)
    @test tau == (0.004, 0.004)                            # force override
    @test out[1].σ_x ≈ sqrt(0.005^2 + 0.004^2)
    out, tau, _ = SMLMBaGoL._maybe_apply_se_adjust(locs, Dict{String,Any}(), 0.004, false)
    @test tau == (0.004, 0.004)                            # no stamp → applies
    out, tau, _ = SMLMBaGoL._maybe_apply_se_adjust(locs, Dict{String,Any}(), 0.0, false)
    @test tau === nothing && out === locs                  # zero τ → no-op

    # apply_se_adjust (public SMLD->SMLD): inflates σ, preserves count; guard + no-op
    smld_raw = SMLMData.BasicSMLD([mk(1.0,1.0,0.005,0.006), mk(1.1,1.0,0.005,0.006)],
        cam, 1, 1, Dict{String,Any}())
    smld_inf = apply_se_adjust(smld_raw, (0.004, 0.003))
    @test length(smld_inf.emitters) == 2
    @test smld_inf.emitters[1].σ_x ≈ sqrt(0.005^2 + 0.004^2)
    @test smld_inf.emitters[1].σ_y ≈ sqrt(0.006^2 + 0.003^2)
    @test apply_se_adjust(smld_raw, 0.0) === smld_raw       # no-op returns same object
    smld_stamp = SMLMData.BasicSMLD(smld_raw.emitters, cam, 1, 1, Dict{String,
        Any}("sigma_corrected"=>true))
    @test apply_se_adjust(smld_stamp, 0.004) === smld_stamp # guard: no-op on corrected

    # end-to-end: run_bagol with se_adjust runs, returns concrete SMLD + records τ in Info
    elist = [SMLMData.Emitter2DFit(0.5 + 0.02i, 0.5, 1000.0, 0.0, 0.005, 0.005, 0.0,
        0.0, 0.0, 1, 1, 0, i) for i in 1:6]
    smld = SMLMData.BasicSMLD(elist, cam, 1, 1, Dict{String,Any}())
    r, d = run_bagol(smld; se_adjust=(0.004, 0.005), n_iterations=200, burn_in=40,
        verbose=false)
    @test r isa SMLMData.BasicSMLD
    @test isconcretetype(eltype(r.emitters))
    # applied τ recorded in diagnostics (Info)
    @test d.se_adjust == (0.004, 0.005)
    r0, d0 = run_bagol(smld; n_iterations=200, burn_in=40, verbose=false)
    # default se_adjust=0 → not recorded
    @test d0.se_adjust === nothing
    # already-corrected SMLD: se_adjust skipped (warns); τ not recorded
    smld_c = SMLMData.BasicSMLD(elist, cam, 1, 1, Dict{String,
        Any}("sigma_corrected" => true))
    rc, dc = run_bagol(smld_c; se_adjust=0.004, n_iterations=200, burn_in=40, verbose=false)
    @test rc isa SMLMData.BasicSMLD
    @test dc.se_adjust === nothing
end

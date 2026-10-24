using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions
import LinearAlgebra
import SMLMSim

@testset "estimate_se_adjust τ finder" begin
    # M-step kernel: at the true τ the within-group scaled neighbor distances
    # z = d/√(σa²+σb²+2τ²) are Rayleigh(1). Synthesize d = z·√(s2+2τ²) with
    # z ~ Rayleigh(1) and check the KS-minimizing τ (the M-step) recovers the
    # planted τ. Pure kernel — no BaGoL, no sim.
    rng = Random.MersenneTwister(20260620)
    npair = 4000
    s2_nm2 = fill(2 * 13.0^2, npair)               # σa²+σb² (nm²), 13 nm σ pairs
    grid_nm = collect(0.0:0.1:14.0)
    for tau_nm in (0.0, 6.0, 10.0)
        z = sqrt.(-2 .* log.(rand(rng, npair)))    # Rayleigh(1)
        d_nm = z .* sqrt.(s2_nm2 .+ 2 * tau_nm^2)
        tau_hat = SMLMBaGoL._mstep(d_nm, s2_nm2, grid_nm, eachindex(d_nm))
        @test abs(tau_hat - tau_nm) ≤ 1.0
    end

    # _ks_rayleigh1 ≥ 0 everywhere and minimized at the planted τ over the grid.
    z = sqrt.(-2 .* log.(rand(rng, 3000)))
    d6 = z .* sqrt.(s2_nm2[1:3000] .+ 2 * 6.0^2)
    ks = [SMLMBaGoL._ks_rayleigh1(d6, s2_nm2[1:3000], τ, eachindex(d6)) for τ in grid_nm]
    @test all(≥(0), ks)
    @test abs(grid_nm[argmin(ks)] - 6.0) ≤ 1.0

    # Right-biased M-step (production kernel): recovers the planted τ and is
    # ≥ the plain argmin M-step (largest τ within ks_noise of the KS min).
    tau_plain = SMLMBaGoL._mstep(d6, s2_nm2[1:3000], grid_nm, eachindex(d6))
    tau_rb = SMLMBaGoL._mstep_rb(d6, s2_nm2[1:3000], grid_nm, eachindex(d6))
    @test abs(tau_rb - 6.0) ≤ 1.5
    @test tau_rb ≥ tau_plain

    # Guards fail fast (before any BaGoL run).
    cam = SMLMData.IdealCamera(1:64, 1:64, 0.1)
    elist = [SMLMData.Emitter2DFit(0.5, 0.5, 1000.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 1, 1, 0, 1)]
    smld_corr = SMLMData.BasicSMLD(elist, cam, 1, 1, Dict{String,Any}("sigma_corrected" => true))
    @test_throws ArgumentError estimate_se_adjust(smld_corr)                  # already σ-corrected
    smld_raw = SMLMData.BasicSMLD(elist, cam, 1, 1, Dict{String,Any}())
    @test_throws ArgumentError estimate_se_adjust(smld_raw; grouping=:nope)   # unknown grouping

    # :dahl grouping (now the default; wired via _dahl_out): group original loc
    # indices by global Dahl label, skipping overlap-dup label 0.
    dgroups = SMLMBaGoL._dahl_label_groups([1, 1, 2, 0, 2])
    @test Set(Set.(dgroups)) == Set([Set([1, 2]), Set([3, 5])])

    # Spatial-block bootstrap resampler returns valid in-range indices.
    pos = [(0.1i, 0.1j) for i in 1:5 for j in 1:5]
    bi = SMLMBaGoL._block_indices(pos, 1.0, Random.MersenneTwister(1))
    @test !isempty(bi) && all(1 .≤ bi .≤ length(pos))
end

using Test
using SMLMBaGoL
using SMLMData
using Random

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

# Reference kernels and serial resampling from before the memory changes.
function reference_ks(d, s2, tau_nm, idxs)
    isempty(idxs) && return NaN
    t2 = tau_nm^2
    z = sort!([d[k] / sqrt(s2[k] + 2t2) for k in idxs])
    n = length(z)
    maximum(abs(i / n - (1 - exp(-z[i]^2 / 2))) for i in 1:n)
end

reference_mstep(d, s2, grid_nm, idxs) =
    grid_nm[argmin([reference_ks(d, s2, τ, idxs) for τ in grid_nm])]

function reference_mstep_rb(d, s2, grid_nm, idxs; ks_noise = 0.01, interpolate = false)
    ks = [reference_ks(d, s2, τ, idxs) for τ in grid_nm]
    kmin = minimum(ks)
    thr = kmin + ks_noise
    j = argmax([ks[i] ≤ thr ? grid_nm[i] : -Inf for i in eachindex(grid_nm)])
    (!interpolate || j == lastindex(grid_nm) || !(ks[j + 1] > thr)) && return grid_nm[j]
    grid_nm[j] + (thr - ks[j]) / (ks[j + 1] - ks[j]) * (grid_nm[j + 1] - grid_nm[j])
end

function reference_block_indices(pos, block_um, rng)
    isempty(pos) && return Int[]
    x0 = minimum(p[1] for p in pos); y0 = minimum(p[2] for p in pos)
    tile(p) = (floor(Int, (p[1] - x0) / block_um), floor(Int, (p[2] - y0) / block_um))
    tiles = Dict{Tuple{Int, Int}, Vector{Int}}()
    for (i, p) in enumerate(pos)
        push!(get!(tiles, tile(p), Int[]), i)
    end
    kv = collect(keys(tiles)); out = Int[]
    for _ in 1:length(kv)
        append!(out, tiles[kv[rand(rng, 1:length(kv))]])
    end
    out
end

@testset "finder M-step scratch preserves exact values" begin
    for T in (Float64, Float32), seed in (1, 29)
        rng = MersenneTwister(seed)
        s2 = T.(100 .+ 300 .* rand(rng, 128))
        d = T.(sqrt.(-2 .* log.(rand(rng, 128))) .* sqrt.(s2 .+ 2 * 8^2))
        d_before = copy(d); s2_before = copy(s2)
        grid = collect(T, 0.0:0.1:40.0)
        for ix in (eachindex(d), randperm(rng, length(d))[1:40],
                   rand(rng, eachindex(d), 256), [3], Int[])
            @test isequal(SMLMBaGoL._ks_rayleigh1_curve(d, s2, grid, ix),
                          [reference_ks(d, s2, τ, ix) for τ in grid])
            @test SMLMBaGoL._mstep(d, s2, grid, ix) === reference_mstep(d, s2, grid, ix)
            for ks_noise in (0.0, 0.01, 1.0), interpolate in (false, true)
                @test SMLMBaGoL._mstep_rb(d, s2, grid, ix; ks_noise, interpolate) ===
                      reference_mstep_rb(d, s2, grid, ix; ks_noise, interpolate)
            end
        end
        @test d == d_before && s2 == s2_before
    end

    d = s2 = Float32[1, 2, 3]
    grid = Real[0f0, 1.0, 2.0]
    for ix in (eachindex(d), [3, 1, 3, 2])
        @test SMLMBaGoL._ks_rayleigh1_curve(d, s2, grid, ix) ==
              [reference_ks(d, s2, τ, ix) for τ in grid]
        @test SMLMBaGoL._mstep(d, s2, grid, ix) === reference_mstep(d, s2, grid, ix)
        for interpolate in (false, true)
            @test SMLMBaGoL._mstep_rb(d, s2, grid, ix; interpolate) ===
                  reference_mstep_rb(d, s2, grid, ix; interpolate)
        end
    end

    # Grid refinement may grow the KS curve, but must reuse the pair-sized buffer.
    rng = MersenneTwister(500)
    d = 20 .* rand(rng, 4096); s2 = 100 .+ 300 .* rand(rng, length(d))
    grid = collect(0.0:0.1:40.0); coarse = grid[1:5]; ix = eachindex(d)
    SMLMBaGoL._ks_rayleigh1_curve(d, s2, coarse, ix)
    SMLMBaGoL._ks_rayleigh1_curve(d, s2, grid, ix)
    coarse_bytes = @allocated SMLMBaGoL._ks_rayleigh1_curve(d, s2, coarse, ix)
    fine_bytes = @allocated SMLMBaGoL._ks_rayleigh1_curve(d, s2, grid, ix)
    @test fine_bytes < 4coarse_bytes
end

@testset "finder bootstrap preserves serial resamples and CI" begin
    rng = MersenneTwister(201)
    s2 = 100 .+ 300 .* rand(rng, 160)
    d = sqrt.(-2 .* log.(rand(rng, length(s2)))) .* sqrt.(s2 .+ 2 * 8^2)
    pos = [(4rand(rng) - 2, 3rand(rng) - 1) for _ in eachindex(d)]
    grid = collect(0.0:0.2:20.0)
    n_boot = 73                                  # crosses a BitVector word boundary
    for seed in (1, 72), block_um in (0.4, 1.0, 10.0)
        old_rng = MersenneTwister(seed); new_rng = MersenneTwister(seed)
        bis = [reference_block_indices(pos, block_um, old_rng) for _ in 1:n_boot]
        tiles = SMLMBaGoL._block_tiles(pos, block_um)
        picks = [[rand(new_rng, 1:length(tiles)) for _ in eachindex(tiles)] for _ in 1:n_boot]
        @test [SMLMBaGoL._expand_block_indices(tiles, p) for p in picks] == bis
        @test rand(old_rng) === rand(new_rng)
        @test SMLMBaGoL._block_indices(pos, block_um, MersenneTwister(seed)) == first(bis)
        expected = [reference_mstep_rb(d, s2, grid, bi; interpolate = true) for bi in bis]
        boots = SMLMBaGoL._bootstrap_msteps(d, s2, pos, grid;
                    n_boot, block_um, seed, ks_noise = 0.01)
        @test length(boots) == n_boot
        @test boots == expected
        @test SMLMBaGoL.quantile(boots, [0.025, 0.975]) ==
              SMLMBaGoL.quantile(expected, [0.025, 0.975])
    end
    empty_pos = NTuple{2, Float64}[]
    @test isempty(SMLMBaGoL._block_indices(empty_pos, 1.0, MersenneTwister(1)))
    @test isempty(SMLMBaGoL._bootstrap_msteps(Float64[], Float64[], empty_pos, grid;
                  n_boot, block_um = 1.0, seed = 1, ks_noise = 0.01))
    @test isempty(SMLMBaGoL._bootstrap_msteps(d, s2, pos, grid;
                  n_boot = 0, block_um = 1.0, seed = 1, ks_noise = 0.01))
    @test SMLMBaGoL._bootstrap_msteps(d, s2, nothing, grid;
              n_boot = 0, block_um = 0.0, seed = 1, ks_noise = 0.01) isa Vector{Float64}
end

@testset "finder neighbor cache budget and labels" begin
    rng = MersenneTwister(340)
    cam = SMLMData.IdealCamera(1:64, 1:64, 0.1)
    em = [SMLMData.Emitter2DFit(4rand(rng), 4rand(rng), 1000.0, 0.0,
              0.008 + 0.012rand(rng), 0.008 + 0.012rand(rng),
              0.0, 0.0, 0.0, 1, i, 0, 1) for i in 1:4096]
    smld = SMLMData.BasicSMLD(em, cam, 1, 1, Dict{String, Any}())
    est = Ref(NaN)
    g = SMLMBaGoL._build_finder_neighbor_graph(smld, 0.030, 3.0;
            graph_max_gb = Inf, graph_est_gb = est)
    @test g !== nothing
    actual_gb = 32 * (length(g.neighbors) ÷ 2) / 1e9
    @test abs(est[] / actual_gb - 1) < 0.10
    @test SMLMBaGoL._build_finder_neighbor_graph(smld, 0.030, 3.0;
              graph_max_gb = est[] / 2) === nothing
    @test SMLMBaGoL._build_finder_neighbor_graph(smld, 0.030, 3.0;
              graph_max_gb = est[]) !== nothing
    @test SMLMBaGoL._build_finder_neighbor_graph(smld, 0.030, 3.0;
              graph_max_gb = 2est[]) !== nothing
    @test SMLMBaGoL._build_finder_neighbor_graph(smld, 0.030, 3.0;
              graph_max_gb = 0.0) === nothing
    repeat_est = Ref(NaN)
    SMLMBaGoL._build_finder_neighbor_graph(smld, 0.030, 3.0;
        graph_max_gb = est[] / 2, graph_est_gb = repeat_est)
    @test repeat_est[] === est[]
    @test SMLMBaGoL._build_finder_neighbor_graph(nothing, 0.030, 3.0;
              graph_max_gb = 0.0, graph_est_gb = repeat_est) === nothing
    @test isnan(repeat_est[])

    # A small subset has several components at each inflated precision.
    small = SMLMData.BasicSMLD(em[1:160], cam, 1, 1, Dict{String, Any}())
    sg = SMLMBaGoL._build_finder_neighbor_graph(small, 0.030, 3.0; graph_max_gb = Inf)
    for tau in (0.0, 0.008, 0.030)
        infl, _, _ = SMLMBaGoL._maybe_apply_se_adjust(small.emitters, small.metadata, (tau, tau), true)
        fresh = SMLMBaGoL.precision_dbscan(infl, 3.0, 0)
        @test SMLMBaGoL.precision_dbscan(infl, 3.0, 0; neighbor_graph = sg) == fresh
        @test length(unique(fresh)) > 1
    end

    empty_smld = SMLMData.BasicSMLD(empty(em), cam, 1, 1, Dict{String, Any}())
    for (data, sigma) in ((empty_smld, 3.0), (smld, Inf))
        skipped_est = Ref(0.0)
        @test SMLMBaGoL._build_finder_neighbor_graph(data, 0.030, sigma;
                  graph_est_gb = skipped_est) === nothing
        @test isnan(skipped_est[])
    end
    for budget in (-1.0, NaN)
        @test_throws ArgumentError SMLMBaGoL._build_finder_neighbor_graph(smld, 0.030, 3.0;
                                      graph_max_gb = budget)
    end
end

@testset "finder cache budget smoke run" begin
    cam = SMLMData.IdealCamera(1:64, 1:64, 0.1)
    em = [SMLMData.Emitter2DFit(0.5 + 0.4k + 0.002j, 1.0,
              1000.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 1, 2k + j, 0, 1)
          for k in 1:4 for j in 1:2]
    smld = SMLMData.BasicSMLD(em, cam, 1, 1, Dict{String, Any}())
    for graph_max_gb in (0.0, Inf)
        Random.seed!(11)
        f = estimate_se_adjust(smld; graph_max_gb, max_steps = 1,
                n_iterations = 100, burn_in = 20, n_boot = 73, grid_um = 0.0:0.002:0.014,
                return_diagnostics = true)
        @test f.graph_cached == (graph_max_gb == Inf)
        @test f.graph_est_gb isa Float64
        @test graph_max_gb == 0.0 ? isnan(f.graph_est_gb) : isfinite(f.graph_est_gb)
        @test !haskey(f.bagol_kwargs, :graph_max_gb)
        @test !haskey(f.bagol_kwargs, :neighbor_graph)
        @test f.n_bagol == 1
        @test isfinite(f.tau_hat_um) && isfinite(f.ci_lo_um) && isfinite(f.ci_hi_um)
    end

    for (data, sigma) in ((smld, Inf),
                         (SMLMData.BasicSMLD(empty(em), cam, 1, 1, Dict{String, Any}()), 3.0))
        f = estimate_se_adjust(data; partition_sigma = sigma, max_steps = 0, n_boot = 0)
        @test !f.graph_cached && isnan(f.graph_est_gb)
    end
end

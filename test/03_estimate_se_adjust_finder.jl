using Test
using SMLMBaGoL
using SMLMData
using Random
import SMLMSim

# The plain (argmin) and right-biased M-steps on one grid, composed from the package's
# KS curve and pick rule.
finder_mstep_plain(d, s2, grid, ix) =
    grid[argmin(SMLMBaGoL._ks_rayleigh1_curve(d, s2, grid, ix))]
finder_mstep_rb(d, s2, grid, ix; ks_noise = 0.01, interpolate = false) =
    SMLMBaGoL._mstep_rb_pick(SMLMBaGoL._ks_rayleigh1_curve(d, s2, grid, ix), grid;
                             ks_noise, interpolate)

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
        tau_hat = finder_mstep_plain(d_nm, s2_nm2, grid_nm, eachindex(d_nm))
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
    tau_plain = finder_mstep_plain(d6, s2_nm2[1:3000], grid_nm, eachindex(d6))
    tau_rb = finder_mstep_rb(d6, s2_nm2[1:3000], grid_nm, eachindex(d6))
    @test abs(tau_rb - 6.0) ≤ 1.5
    @test tau_rb ≥ tau_plain

    # Guards fail fast (before any BaGoL run).
    cam = SMLMData.IdealCamera(1:64, 1:64, 0.1)
    elist = [SMLMData.Emitter2DFit(0.5, 0.5, 1000.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 1, 1, 0, 1)]
    smld_corr = SMLMData.BasicSMLD(elist, cam, 1, 1, Dict{String,Any}("sigma_corrected" => true))
    @test_throws ArgumentError estimate_se_adjust(smld_corr)                  # already σ-corrected
    smld_raw = SMLMData.BasicSMLD(elist, cam, 1, 1, Dict{String,Any}())
    @test_throws ArgumentError estimate_se_adjust(smld_raw; grouping=:nope)   # unknown grouping
    @test_throws ArgumentError estimate_se_adjust(smld_raw; grid_um = Float64[])
    @test_throws ArgumentError estimate_se_adjust(smld_raw; grid_um = [0.0])
    @test_throws ArgumentError estimate_se_adjust(smld_raw; tau_max_um = 0.012)

    # :dahl grouping (now the default; wired via _dahl_out): group original loc
    # indices by global Dahl label, skipping overlap-dup label 0.
    dgroups = SMLMBaGoL._dahl_label_groups([1, 1, 2, 0, 2])
    @test Set(Set.(dgroups)) == Set([Set([1, 2]), Set([3, 5])])

    # Spatial-block bootstrap resampler returns valid in-range indices.
    pos = [(0.1i, 0.1j) for i in 1:5 for j in 1:5]
    bi = SMLMBaGoL._block_indices(pos, 1.0, Random.MersenneTwister(1))
    @test !isempty(bi) && all(1 .≤ bi .≤ length(pos))
end

@testset "estimate_se_adjust above the default grid ceiling" begin
    # SMLMSim supplies both independent noise passes. The second pass overwrites
    # the reported σ with 5 nm, leaving the first pass's planted 16 nm unreported.
    Random.seed!(20261028)
    photons = 1000.0
    cam = SMLMData.IdealCamera(1:128, 1:128, 0.1)
    em = SMLMData.Emitter2DFit{Float64}[]
    for x in 1:10, y in 1:10, blink in 1:20
        push!(em, SMLMData.Emitter2DFit(Float64(x), Float64(y), photons, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, blink, 1, 0, length(em) + 1))
    end
    truth = SMLMData.BasicSMLD(em, cam, 20, 1, Dict{String,Any}())
    unreported = SMLMSim.apply_noise(truth, 0.016 * sqrt(photons))
    smld = SMLMSim.apply_noise(unreported, 0.005 * sqrt(photons))
    @test all(e -> e.σ_x ≈ 0.005 && e.σ_y ≈ 0.005, smld.emitters)

    # Keep the default grid and cap, but start above the planted 16 nm root.
    timed = @timed estimate_se_adjust(smld; g_start_um = 0.03,
        n_iterations = 300, burn_in = 150, n_boot = 4)
    found = timed.value
    @info "τ finder above-default-grid regression runtime" seconds = round(timed.time; digits = 3)
    @test found.converged
    @test !found.censored
    @test found.start_overmerged
    @test abs(1000 * found.tau_hat_um - 16.0) ≤ 1.5
    @test found.grid_top_um > 0.014
    @test found.tau_max_um == 0.05
    @test 0 ≤ found.n_boot_censored ≤ 4

    # The default 8 nm start is below the planted root. One E-step is enough
    # to diagnose the failed start premise, without searching for a spurious root.
    below = @test_logs (:warn, r"finder start is not over-merged") match_mode = :any begin
        estimate_se_adjust(smld; max_steps = 1, n_iterations = 300, burn_in = 150, n_boot = 0)
    end
    @test below.start_overmerged == false
    @test below.n_bagol == 1
end

# Reference kernels and serial resampling from before the memory changes (the right-biased
# pick ignores nonfinite KS points; the extending M-step grows its grid while the top
# qualifies).
function reference_ks(d, s2, tau_nm, idxs)
    isempty(idxs) && return NaN
    t2 = tau_nm^2
    z = sort!([d[k] / sqrt(s2[k] + 2t2) for k in idxs])
    n = length(z)
    maximum(abs(i / n - (1 - exp(-z[i]^2 / 2))) for i in 1:n)
end

reference_mstep(d, s2, grid_nm, idxs) =
    grid_nm[argmin([reference_ks(d, s2, τ, idxs) for τ in grid_nm])]

function reference_pick(ks, grid_nm; ks_noise = 0.01, interpolate = false)
    finite = findall(isfinite, ks)
    isempty(finite) && return NaN
    thr = minimum(ks[i] for i in finite) + ks_noise
    p = argmax([ks[i] ≤ thr ? grid_nm[i] : -Inf for i in finite])
    j = finite[p]
    (!interpolate || p == lastindex(finite)) && return grid_nm[j]
    k = finite[p + 1]
    !(ks[k] > thr) && return grid_nm[j]
    grid_nm[j] + (thr - ks[j]) / (ks[k] - ks[j]) * (grid_nm[k] - grid_nm[j])
end

reference_mstep_rb(d, s2, grid_nm, idxs; ks_noise = 0.01, interpolate = false) =
    reference_pick([reference_ks(d, s2, τ, idxs) for τ in grid_nm], grid_nm;
                   ks_noise, interpolate)

function reference_mstep_rb_ext(d, s2, grid_nm, idxs; ks_noise = 0.01, cap_nm)
    grid = collect(Float64, grid_nm)
    ks = [reference_ks(d, s2, τ, idxs) for τ in grid]
    qualifies() = isfinite(ks[end]) &&
        ks[end] ≤ minimum(k for k in ks if isfinite(k)) + ks_noise
    while qualifies() && grid[end] < cap_nm
        top = grid[end]; step = top - grid[end - 1]
        target = min(cap_nm, max(2top, top + step))
        new_points = collect((top + step):step:target)
        if target == cap_nm
            snap = !isempty(new_points) &&
                isapprox(new_points[end], cap_nm; rtol = 8eps(Float64))
            if snap
                new_points[end] = cap_nm
            else
                push!(new_points, Float64(cap_nm))
            end
        end
        append!(grid, new_points)
        append!(ks, [reference_ks(d, s2, τ, idxs) for τ in new_points])
    end
    (; m = reference_pick(ks, grid; ks_noise, interpolate = true), grid,
       censored = grid[end] == cap_nm && qualifies())
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
            @test finder_mstep_plain(d, s2, grid, ix) === reference_mstep(d, s2, grid, ix)
            for ks_noise in (0.0, 0.01, 1.0), interpolate in (false, true)
                @test finder_mstep_rb(d, s2, grid, ix; ks_noise, interpolate) ===
                      reference_mstep_rb(d, s2, grid, ix; ks_noise, interpolate)
            end
        end
        for (g, cap) in ((grid, 40.0), (grid, 60.0), (collect(T, 0.0:0.1:4.0), 40.0),
                         (collect(T, 0.0:0.1:4.0), 6.25))
            for ix in (eachindex(d), randperm(rng, length(d))[1:40], [3], Int[])
                got = SMLMBaGoL._mstep_rb_ext(d, s2, g, ix; cap_nm = cap)
                want = reference_mstep_rb_ext(d, s2, g, ix; cap_nm = cap)
                @test got.m === want.m && got.grid == want.grid
                @test got.censored == want.censored
            end
        end
        @test d == d_before && s2 == s2_before
    end

    d = s2 = Float32[1, 2, 3]
    grid = Real[0f0, 1.0, 2.0]
    for ix in (eachindex(d), [3, 1, 3, 2])
        @test SMLMBaGoL._ks_rayleigh1_curve(d, s2, grid, ix) ==
              [reference_ks(d, s2, τ, ix) for τ in grid]
        @test finder_mstep_plain(d, s2, grid, ix) === reference_mstep(d, s2, grid, ix)
        for interpolate in (false, true)
            @test finder_mstep_rb(d, s2, grid, ix; interpolate) ===
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
    # Each extra grid point allocates far less than one pair-sized buffer (the sort keeps a
    # small fixed workspace of its own).
    @test (fine_bytes - coarse_bytes) / (length(grid) - length(coarse)) < sizeof(d) / 4
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
        boots, n_censored = SMLMBaGoL._bootstrap_msteps(d, s2, pos, grid;
                    n_boot, block_um, seed, ks_noise = 0.01, cap_nm = 50.0)
        @test n_censored == 0
        @test length(boots) == n_boot
        @test boots == expected
        @test SMLMBaGoL.quantile(boots, [0.025, 0.975]) ==
              SMLMBaGoL.quantile(expected, [0.025, 0.975])
    end
    # Resamples whose M-step must extend its grid (planted 16 nm, grid to 10 nm), and at a
    # 12 nm cap are censored: estimates and censor count equal the extending reference.
    rng = MersenneTwister(202)
    s2x = 100 .+ 300 .* rand(rng, 160)
    dx = sqrt.(-2 .* log.(rand(rng, length(s2x)))) .* sqrt.(s2x .+ 2 * 16^2)
    posx = [(4rand(rng) - 2, 3rand(rng) - 1) for _ in eachindex(dx)]
    small = collect(0.0:0.2:10.0)
    for cap in (40.0, 12.0)
        ref_rng = MersenneTwister(5)
        bis = [reference_block_indices(posx, 1.0, ref_rng) for _ in 1:n_boot]
        want = [reference_mstep_rb_ext(dx, s2x, small, bi; cap_nm = cap) for bi in bis]
        @test all(w -> length(w.grid) > length(small), want)
        boots, n_censored = SMLMBaGoL._bootstrap_msteps(dx, s2x, posx, small;
                                n_boot, block_um = 1.0, seed = 5, ks_noise = 0.01, cap_nm = cap)
        @test boots == [w.m for w in want if isfinite(w.m)]
        @test n_censored == count(w -> w.censored, want)
        @test (n_censored > 0) == (cap == 12.0)
    end

    empty_pos = NTuple{2, Float64}[]
    @test isempty(SMLMBaGoL._block_indices(empty_pos, 1.0, MersenneTwister(1)))
    boot_kw = (; ks_noise = 0.01, cap_nm = 50.0)
    @test SMLMBaGoL._bootstrap_msteps(Float64[], Float64[], empty_pos, grid;
              n_boot, block_um = 1.0, seed = 1, boot_kw...) == (Float64[], 0)
    @test SMLMBaGoL._bootstrap_msteps(d, s2, pos, grid;
              n_boot = 0, block_um = 1.0, seed = 1, boot_kw...) == (Float64[], 0)
    @test first(SMLMBaGoL._bootstrap_msteps(d, s2, nothing, grid;
              n_boot = 0, block_um = 0.0, seed = 1, boot_kw...)) isa Vector{Float64}
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
    @test abs(est[] / (SMLMBaGoL._GRAPH_BUILD_PEAK * actual_gb) - 1) < 0.10
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

    # Float32 localizations: the per-call neighbor search returns exactly the cached
    # graph's active pairs for every localization, so labels agree by construction.
    em32 = [SMLMData.Emitter2DFit{Float32}((getfield(e, f) for f in fieldnames(typeof(e)))...)
            for e in em]
    @test em32[1].x isa Float32
    smld32 = SMLMData.BasicSMLD(em32, cam, 1, 1, Dict{String, Any}())
    g32 = SMLMBaGoL._build_finder_neighbor_graph(smld32, 0.030, 3.0; graph_max_gb = Inf)
    for tau in (0.0, 0.008, 0.030)
        infl, _, _ = SMLMBaGoL._maybe_apply_se_adjust(em32, smld32.metadata, (tau, tau), true)
        coords = SMLMBaGoL.coords_matrix(infl)
        σ = Float64[SMLMBaGoL.mean_sigma(e) for e in infl]
        tree = SMLMBaGoL.KDTree(coords)
        max_radius = 3.0 * 2 * maximum(σ)
        same = true
        for i in eachindex(infl)
            r = g32.offsets[i]:(g32.offsets[i + 1] - 1)
            cached = Set(g32.neighbors[t] for t in r
                         if g32.dists[t] < 3.0 * (σ[i] + σ[g32.neighbors[t]]))
            same &= Set(SMLMBaGoL.precision_neighbors(tree, coords, σ, i, 3.0, max_radius)) == cached
        end
        @test same
        @test SMLMBaGoL.precision_dbscan(infl, 3.0, 0; neighbor_graph = g32) ==
              SMLMBaGoL.precision_dbscan(infl, 3.0, 0)
    end

    # A Float32 pair at the threshold, found by search: the earlier per-call test (the
    # precision-weighted distance in Float32) called it apart while the cache's Float64
    # test calls it neighbors. Both paths now call it neighbors.
    mk(x, y, s) = SMLMData.Emitter2DFit{Float32}(x, y, 1000f0, 0f0, s, s, 0f0, 0f0, 0f0, 1, 1, 0, 1)
    pair = [mk(2.31598f0, 1.2200558f0, 0.01874179f0), mk(2.4028454f0, 1.2223487f0, 0.010223437f0)]
    pg = SMLMBaGoL.build_precision_neighbor_graph(SMLMBaGoL.coords_matrix(pair), 1.0)
    @test SMLMBaGoL.precision_dbscan(pair, 3.0, 0; neighbor_graph = pg) == [1, 1]
    @test SMLMBaGoL.precision_dbscan(pair, 3.0, 0) == [1, 1]

    # A pair just inside the finder cache's radius (found by Codex review): two
    # localizations whose σ sets the radius one ulp above their distance, on a grid where
    # an exact-radius KDTree query can prune the pair. Cache and per-call labels agree.
    X = zeros(2, 64 * 64)
    for (i, (x, y)) in enumerate(Iterators.product(1:64, 1:64))
        X[:, i] = [x, y]
    end
    X .*= 0.123123
    lo, hi = 1404, 3889
    r = nextfloat(sqrt((X[1, lo] - X[1, hi])^2 + (X[2, lo] - X[2, hi])^2))
    tight = [SMLMData.Emitter2DFit(X[1, i], X[2, i], 1000.0, 0.0,
                 i in (lo, hi) ? r / 2 : 1e-9, i in (lo, hi) ? r / 2 : 1e-9,
                 0.0, 0.0, 0.0, 1, i, 0, 1) for i in axes(X, 2)]
    tsmld = SMLMData.BasicSMLD(tight, cam, 1, 1, Dict{String, Any}())
    tg = SMLMBaGoL._build_finder_neighbor_graph(tsmld, 0.0, 1.0; graph_max_gb = Inf)
    tinfl, _, _ = SMLMBaGoL._maybe_apply_se_adjust(tight, tsmld.metadata, (0.0, 0.0), true)
    tcached = SMLMBaGoL.precision_dbscan(tinfl, 1.0, 0; neighbor_graph = tg)
    @test tcached == SMLMBaGoL.precision_dbscan(tinfl, 1.0, 0)
    @test tcached[lo] == tcached[hi]

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

    # Partitioning uses the cache only at min_partition_size ≤ 1, so none is built above it.
    for (mps, cached) in ((0, true), (1, true), (2, false), (3, false))
        f = estimate_se_adjust(smld; min_partition_size = mps, graph_max_gb = Inf,
                               max_steps = 0, n_boot = 0)
        @test f.graph_cached == cached
        @test isnan(f.graph_est_gb) == !cached
    end
    @test_throws ArgumentError estimate_se_adjust(smld; graph_max_gb = -1, max_steps = 0)
    @test estimate_se_adjust(smld; graph_max_gb = 8, max_steps = 0, n_boot = 0).graph_cached
end

@testset "finder seeded result does not depend on the cache budget" begin
    # Planted 16 nm unreported error; the search starts below the root under a grid
    # topping at 6 nm, so the second E-step lies above the first cache's 8 nm ceiling and
    # the cache is rebuilt. The middle budget admits the first cache but not the rebuild,
    # so that run hands over from the cache to the per-call search mid-search.
    Random.seed!(20261109)
    photons = 1000.0
    cam = SMLMData.IdealCamera(1:128, 1:128, 0.1)
    em = SMLMData.Emitter2DFit{Float64}[]
    for x in 1:6, y in 1:6, blink in 1:15
        push!(em, SMLMData.Emitter2DFit(Float64(x), Float64(y), photons, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, blink, 1, 0, length(em) + 1))
    end
    truth = SMLMData.BasicSMLD(em, cam, 15, 1, Dict{String,Any}())
    smld = SMLMSim.apply_noise(SMLMSim.apply_noise(truth, 0.016 * sqrt(photons)),
                               0.005 * sqrt(photons))
    est(ceiling_um) = (e = Ref(NaN);
        SMLMBaGoL._build_finder_neighbor_graph(smld, ceiling_um, 3.0; graph_max_gb = Inf,
                                               graph_est_gb = e); e[])
    e8, e16 = est(0.008), est(0.016)
    @test e8 < e16
    kw = (; g_start_um = 0.008, grid_um = 0.0:0.0002:0.006, max_steps = 3,
          n_iterations = 100, burn_in = 50, n_boot = 20)
    runs = map((0.0, (e8 + e16) / 2, Inf)) do budget
        Random.seed!(7)
        @test_logs (:warn, r"finder start is not over-merged") match_mode = :any begin
            estimate_se_adjust(smld; graph_max_gb = budget, kw...)
        end
    end
    off, handover, on = runs
    @test !off.graph_cached && !handover.graph_cached && on.graph_cached
    @test handover.graph_est_gb > (e8 + e16) / 2      # the rebuild went over the budget
    @test maximum(first, on.path_um) > 0.008           # an E-step above the first ceiling
    for f in (handover, on)
        @test f.tau_hat_um === off.tau_hat_um
        @test f.path_um == off.path_um && all(map(===, f.path_um, off.path_um))
        @test f.ci_lo_um === off.ci_lo_um
        @test f.ci_hi_um === off.ci_hi_um
    end
end

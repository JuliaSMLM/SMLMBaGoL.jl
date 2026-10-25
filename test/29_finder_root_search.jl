using Test
using SMLMBaGoL
using Random

# The finder searches for the self-consistent τ g* = m(g*) (E-step + M-step).
# The tests drive the root search through its seam
# `_tau_root_search` with synthetic maps m(g): no BaGoL runs, no sim.
@testset "τ finder root search (synthetic maps)" begin
    grid = collect(0.0:0.1:14.0)
    function search(m, g0; max_steps = 12, tol = 0.1, root = nothing)
        result = SMLMBaGoL._tau_root_search(
            m, g0; grid_nm = grid, stop_tol_nm = tol, max_steps = max_steps)
        @test 0.0 ≤ result.tau_nm ≤ 14.0
        @test all(p -> 0.0 ≤ first(p) ≤ 14.0, result.path)
        @test result.bracket_nm isa Tuple{Float64, Float64}
        a, b = result.bracket_nm
        exact = any(p -> p[1] == p[2] == result.tau_nm, result.path)
        if all(isnan, result.bracket_nm)
            @test !result.converged || exact
        else
            @test 0.0 ≤ a < b ≤ 14.0
            @test a ≤ result.tau_nm ≤ b
            @test any(p -> p[1] == a && p[2] > a, result.path)
            @test any(p -> p[1] == b && p[2] < b, result.path)
            root === nothing || @test a ≤ root ≤ b
        end
        if result.converged && !exact
            @test isfinite(a) && isfinite(b)
            @test b - a ≤ 2tol
            root === nothing || @test abs(result.tau_nm - root) ≤ 2tol
        end
        result
    end
    affine(gs, s) = g -> gs + s * (g - gs)

    # Once bracketed, each new evaluation stays at least tol from both old ends.
    function check_minimum_steps(result, tol)
        for k in 2:length(result.path)
            previous = result.path[1:k - 1]
            a = maximum((g for (g, m) in previous if m > g); init = -Inf)
            b = minimum((g for (g, m) in previous if m < g); init = Inf)
            if isfinite(a) && isfinite(b) && b - a > 2tol
                g = first(result.path[k])
                @test a + tol - 1e-12 ≤ g ≤ b - tol + 1e-12
            end
        end
    end

    # Affine maps with slope s: a residual stop |m−g| ≤ tol sits tol/(1−s) from the
    # fixed point, on the side the iteration came from, so the old answer depended on g_start.
    for s in (0.6, 0.8), gs in (6.3, 3.7)
        results = [search(affine(gs, s), g0; root = gs) for g0 in (1.0, 3.0, 8.0, 14.0)]
        @test all(r -> r.converged, results)
        ts = [r.tau_nm for r in results]
        @test all(t -> abs(t - gs) ≤ 0.05, ts)
        @test maximum(ts) - minimum(ts) ≤ 0.05
    end
    r = search(affine(6.3, 0.8), 14.0)
    @test r.converged
    @test length(r.method_path) == r.n_evals == length(r.path)
    @test r.method_path[1] === :start

    # The bracket secant slope gives no distance bound across this kink. In
    # particular, start 7 used to return 6.507 nm with converged=true (tol = 0.1).
    # The shallow side needs more than the default 12 E-steps to close its bracket.
    kink(g) = min(0.999g + 0.0079, 6.901)
    for g0 in (1.0, 7.0, 14.0)
        rk = search(kink, g0; max_steps = 32, root = 6.901)
        @test rk.converged
        @test abs(rk.tau_nm - 6.901) ≤ 0.2
        check_minimum_steps(rk, 0.1)
    end
    kink_capped = search(kink, 1.0; root = 6.901)
    @test !kink_capped.converged
    @test kink_capped.n_evals == 12

    # Strong curvature, with a unique fixed point in the grid: the other root
    # is -0.5 nm. A root at 6.3 with these coefficients would have two grid roots.
    curved(g) = 1.5 + 0.9 * (g - 1.5) - 0.05 * (g - 1.5)^2
    @test all(g -> 0.0 ≤ curved(g) ≤ 14.0, grid)
    for g0 in (0.0, 1.0, 7.0, 14.0)
        rs = search(curved, g0; root = 1.5)
        @test rs.converged
        @test abs(rs.tau_nm - 1.5) ≤ 0.2
        check_minimum_steps(rs, 0.1)
    end

    # Additive E-step/M-step noise ±0.05 nm: starts below and above still agree.
    noisy(gs, s, seed) = (rng = MersenneTwister(seed); g -> gs + s * (g - gs) + 0.05 * (2rand(rng) - 1))
    for s in (0.6, 0.8), seed in (1, 2, 3)
        ts = [search(noisy(6.3, s, seed), g0).tau_nm for g0 in (3.0, 14.0)]
        @test abs(ts[1] - ts[2]) ≤ 0.2
        @test all(t -> abs(t - 6.3) ≤ 0.35, ts)
    end

    # Zero residual at the start: the start is the root.
    r0 = search(affine(6.0, 0.8), 6.0)
    @test abs(r0.tau_nm - 6.0) ≤ 0.05
    @test r0.converged
    @test all(isnan, r0.bracket_nm)                # exact root before any bracket

    # Root near the grid bottom (τ = 0.3 nm) and at the grid top.
    for g0 in (2.0, 8.0)
        @test abs(search(affine(0.3, 0.8), g0; root = 0.3).tau_nm - 0.3) ≤ 0.05
    end
    top = g -> min(14.0, 15.0 + 0.8 * (g - 15.0))      # fixed point beyond the grid: clamps at 14
    @test abs(search(top, 3.0).tau_nm - 14.0) ≤ 0.05

    # Cap exhaustion is reported, with a finite estimate inside the grid.
    rc = search(affine(6.3, 0.8), 3.0; max_steps = 2)
    @test !rc.converged
    @test rc.n_evals ≤ 2
    @test isfinite(rc.tau_nm) && 0.0 ≤ rc.tau_nm ≤ 14.0
    @test all(isnan, rc.bracket_nm)
    # Even an unconverged bracket is reported, with its interpolated estimate.
    rb = search(kink, 7.0; max_steps = 2, root = 6.901)
    @test !rb.converged
    @test rb.bracket_nm == (6.5, 7.0)
    @test rb.tau_nm ≈ 6.5 + (kink(6.5) - 6.5) /
        ((kink(6.5) - 6.5) - (kink(7.0) - 7.0)) * 0.5

    # A patch of unresolved slope around the root (noise flips the sign of the local slope).
    patch(g) = affine(6.3, 0.8)(g) + (abs(g - 6.3) < 0.3 ? 0.15 * sin(40g) : 0.0)
    for g0 in (3.0, 14.0)
        @test abs(search(patch, g0).tau_nm - 6.3) ≤ 0.3
    end

    # An empty E-step ends the search without a convergence claim.
    re = search(g -> nothing, 8.0)
    @test !re.converged && re.n_evals == 0 && re.tau_nm == 8.0
    for (g0, bounded_start) in ((20.0, 14.0), (-1.0, 0.0))
        bounded = search(affine(6.3, 0.8), g0; root = 6.3)
        @test bounded.converged
        @test first(bounded.path)[1] == bounded_start
        for empty_m in (nothing, NaN)
            seen = Float64[]
            empty_result = search(g -> (push!(seen, g); empty_m), g0)
            @test seen == [bounded_start]
            @test !empty_result.converged && empty_result.n_evals == 0
            @test empty_result.tau_nm == bounded_start
            @test all(isnan, empty_result.bracket_nm)
        end
        zero_seen = Float64[]
        zero = search(g -> (push!(zero_seen, g); g), g0; max_steps = 0)
        @test isempty(zero_seen)
        @test !zero.converged && zero.n_evals == 0
        @test zero.tau_nm == bounded_start
    end
    # A nonfinite M-step is not an estimate.
    calls = Ref(0)
    rn = search(g -> (calls[] += 1; calls[] == 2 ? NaN : affine(6.3, 0.8)(g)), 14.0)
    @test !rn.converged
    @test isfinite(rn.tau_nm)
end

# The CI must pick the same nearest grouping while retaining at most three E-steps.
@testset "τ finder bounded grouping retention" begin
    grid = collect(0.0:0.1:14.0)
    for g0 in (3.0, 14.0), cap in (1, 2, 12)
        evals = NamedTuple[]
        function evalm(g)
            m = g - 0.5 * tanh(g - 6.3)
            e = (; g, m, d = [g], s2 = [g^2], pos = [(g, g)])
            SMLMBaGoL._keep_finder_eval!(evals, e)
            @test 1 ≤ length(evals) ≤ 3
            @test last(evals) === e
            m
        end
        found = SMLMBaGoL._tau_root_search(evalm, g0; grid_nm = grid,
            stop_tol_nm = 0.001, max_steps = cap)
        if cap == 12
            @test length(found.path) > length(evals)
        end
        g_nearest = first(found.path[argmin([abs(g - found.tau_nm) for (g, _) in found.path])])
        kept = evals[argmin([abs(e.g - found.tau_nm) for e in evals])]
        @test kept.g == g_nearest
        @test kept.d == [g_nearest]
        @test kept.s2 == [g_nearest^2]
        @test kept.pos == [(g_nearest, g_nearest)]
    end

    # An exact root keeps both bracket ends plus the latest grouping; the next
    # E-step releases that latest grouping as soon as it is superseded.
    evals = NamedTuple[]
    for (g, m) in ((4.0, 5.0), (8.0, 7.0), (6.0, 6.0))
        SMLMBaGoL._keep_finder_eval!(evals, (; g, m, d = [g]))
    end
    @test [e.g for e in evals] == [4.0, 8.0, 6.0]
    SMLMBaGoL._keep_finder_eval!(evals, (; g = 6.5, m = 6.2, d = [6.5]))
    @test [e.g for e in evals] == [4.0, 6.5]
end

# The M-step locates the KS-threshold crossing between grid points; the grid rule is
# the interpolate=false default.
@testset "right-biased M-step interpolation" begin
    rng = MersenneTwister(20261025)
    n = 4000
    s2 = fill(2 * 13.0^2, n)
    d = sqrt.(-2 .* log.(rand(rng, n))) .* sqrt.(s2 .+ 2 * 6.0^2)
    grid = collect(0.0:0.1:14.0)
    ix = eachindex(d)
    mg = SMLMBaGoL._mstep_rb(d, s2, grid, ix)
    @test SMLMBaGoL._mstep_rb(d, s2, grid, ix; interpolate = false) == mg
    mi = SMLMBaGoL._mstep_rb(d, s2, grid, ix; interpolate = true)
    @test mg ≤ mi ≤ mg + 0.1
    ks = [SMLMBaGoL._ks_rayleigh1(d, s2, τ, ix) for τ in grid]
    thr = minimum(ks) + 0.01
    # the KS curve at the interpolated τ sits on the threshold
    @test abs(SMLMBaGoL._ks_rayleigh1(d, s2, mi, ix) - thr) ≤ 5e-4
    # a coarse grid: the interpolated crossing lies in the interval above the grid rule's
    # point and stays near the fine-grid crossing
    coarse = collect(0.0:0.5:14.0)
    mgc = SMLMBaGoL._mstep_rb(d, s2, coarse, ix)
    mic = SMLMBaGoL._mstep_rb(d, s2, coarse, ix; interpolate = true)
    @test mgc ≤ mic ≤ mgc + 0.5
    @test abs(mic - mi) ≤ 0.25
    # no interpolation when the last qualifying point is the grid end
    @test SMLMBaGoL._mstep_rb(d, s2, grid, ix; ks_noise = 1.0, interpolate = true) == last(grid)
end

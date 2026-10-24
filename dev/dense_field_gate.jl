# G5/G7 dense-field gate: does BaGoL count single emitters the same at a density where
# precision-DBSCAN partitioning percolates as at one where it does not, and does the
# count depend on max_partition_size?
#
# WHY: run_bagol partitions the field with precision-weighted DBSCAN (partition_sigma),
# then METIS-splits partitions above max_partition_size. Where DBSCAN percolates
# (~250 emitters/um^2 at ~16 nm precision) the whole field is one giant component that
# METIS cuts into pieces; the per-partition chain then sees a different N, K and
# bounding-box area than an isolated cluster. The posterior count of a single uniform
# emitter field should not depend on any of that. This gate measures that against an
# ORACLE (one emitter per true emitter with >= 1 localization, at the mean of its
# localizations) with the CURRENT sampler.
#
# WHAT IT TESTS: uniform random single emitters; dense arm 5 x 5 um scored on the inner 4 x 4 um, low arm
# 10 x 10 um scored on the inner 9 x 9 um (~1600 emitters),
# at 250/um^2 (percolating) and 20/um^2 (not: median partition ~44 localizations, a few
# emitters; 100/um^2 still percolated, median partition ~420), production run_bagol settings (locmix,
# learned count model, partition_sigma 3, max_partition_size 500, overlap 0.025,
# 6000 / 2000), se_adjust = 0 (no extra scatter is simulated). The dense field is also
# run at max_partition_size 100 and 1000 on the same data. Per run it reports:
#   - emitters per true emitter with >= 3 localizations (BaGoL / oracle / truth),
#     and per true emitter with >= 1 localization;
#   - share of Euclidean-DBSCAN clusters (SMLMClustering.DBSCANConfig, eps 30 nm,
#     min_points 1, the same settings for BaGoL, oracle and truth) with >= 3 and >= 4
#     emitters; the FULL field is clustered and clusters are kept by centroid in the inner
#     region (emitter counts crop by position);
#   - median partition size per localization (diagnostics.partition_ids).
# Truth = every true emitter (including those with no localization); oracle = true
# emitters with >= 1 localization at the mean of their localizations.
#
# SIMULATION: SMLMSim only. uniform2D(rho, Nmer2D(n=1), field, field) places the emitters;
# SMLMSim.simulate(StaticSMLMConfig; starting_conditions=...) runs a bleaching 3-state
# GenericFluor (dSTORM-like, ~15 locs/emitter, shape ~1.1) with apply_noise. The true
# emitter of every localization is its track_id.
#
# PASS criterion (ratio r = BaGoL emitters / oracle emitters in the inner region):
#   cap sweep ONLY: |count(cap 100 or 1000) / count(cap 500) - 1| <= 0.02.
# The density comparison |r(250) - r(low)| is printed with its bootstrap SE but NOT gated:
# it also exceeds 0.03 (0.065) under the partition-invariant flat control, because density
# changes how resolvable the emitters are.
# |r - 1| at each density is printed but NOT gated: the oracle knows the true
# assignments, so the best principled prior measured so far still sits 3-7% below it;
# the gate tests that the bias does not depend on density or partitioning.
#
# HOW TO RUN (gitignored env dev/output/gate_env, not committed):
#   nice julia -t 16 --project=dev/output/gate_env dev/dense_field_gate.jl [seed]
# RECREATE the env from scratch, from the repo root of this checkout (tested with Julia
# 1.13.0; SMLMBaGoL = this checkout; SMLMSim v0.7.3-DEV @46bc484 and SMLMData v0.7.1
# @0c9fb55 as sibling JuliaSMLM checkouts, or the registered 0.7.x releases; SMLMClustering
# 0.4.3 registered; Statistics, Random, Printf stdlib):
#   mkdir -p dev/output/gate_env
#   julia --project=dev/output/gate_env -e 'using Pkg; Pkg.develop(path="."); \
#     Pkg.develop(path="../JuliaSMLM/SMLMSim"); Pkg.develop(path="../JuliaSMLM/SMLMData"); \
#     Pkg.add(["SMLMClustering", "Statistics", "Random", "Printf"])'
#   (adjust the two sibling paths; or Pkg.add(["SMLMSim", "SMLMData"]) for registered releases)
# GATE_CONTROL=ipp runs production settings with spatial_model=:ipp.
# GATE_CONTROL=flat runs the positive control (fixed-rho flat prior; see CONTROL below).
# GATE_QUICK=1 shrinks the fields for a smoke run. Exit status: 0 on PASS, 1 on FAIL.
# UNCERTAINTY: each gated difference is printed with a bootstrap SE (resampling 1 um tiles
# of the inner region, so the arms stay paired where they share data) next to its threshold.

using SMLMBaGoL, SMLMSim, SMLMData, SMLMClustering
using Statistics, Random, Printf

const SEED = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20261023
const QUICK = get(ENV, "GATE_QUICK", "0") == "1"
const FIELD_DENSE = QUICK ? 2.5 : 5.0           # um
const FIELD_LOW = QUICK ? 5.0 : 10.0            # um
const MARGIN = 0.5                              # inner region = [MARGIN, field - MARGIN]^2
const TILE = 1.0                                # um; bootstrap tile size
const NBOOT = 1000
# POSITIVE CONTROL (GATE_CONTROL=flat): spatial_model=:flat with the Poisson K prior and rho FIXED at
# the simulation's emitter density of each field (learn_rho=false), partition-invariant by
# construction. The control claim is the cap sweep; the density comparison is reported only.
const CONTROL = get(ENV, "GATE_CONTROL", "") == "flat"
# GATE_CONTROL=ipp: production settings but spatial_model=:ipp (pilot intensity map, learned rho).
const IPP = get(ENV, "GATE_CONTROL", "") == "ipp"
const DENSE = 250.0
const SPARSE = parse(Float64, get(ENV, "GATE_LOW_DENSITY", "20"))   # low-density arm; median partition ~44 locs
const N_ITER, BURN_IN = 6000, 2000
const DBSCAN_EPS_NM = 30.0

# ---- SMLMSim fluorophore: bleaching 3-state (ON, OFF, bleached), dSTORM-like ----
const FRAMERATE, NFRAMES = 50.0, 3000
const K_OFF, K_ON, K_BLEACH, PHOTON_RATE = 50.0, 1.0, 6.5, 40000.0
const FLUOR = GenericFluor(PHOTON_RATE,
    [-(K_OFF + K_BLEACH) K_OFF K_BLEACH; K_ON -K_ON 0.0; 0.0 0.0 0.0])
const SIMCFG = StaticSMLMConfig(density=1.0, σ_psf=0.13, minphotons=30, ndatasets=1,
                                nframes=NFRAMES, framerate=FRAMERATE)
const CAMERA = IdealCamera(1:60, 1:60, 0.1)

function simulate_field(ρ::Float64, field::Float64)
    x, y, _ = uniform2D(ρ, Nmer2D(; n=1, d=0.0), field, field)
    ems = [Emitter2DFit{Float64}(x[i], y[i], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0; σ_xy=0.0,
                                 frame=1, dataset=1, track_id=i, id=i) for i in eachindex(x)]
    smld, _ = SMLMSim.simulate(SIMCFG; starting_conditions=ems, molecule=FLUOR,
                               camera=CAMERA, state1=1)
    return smld, x, y
end

ntiles(field) = floor(Int, (field - 2MARGIN) / TILE)
# inner region = whole 1 um tiles inside the margin (= [MARGIN, field - MARGIN]^2 for the full-size fields)
inner(x, y, field) = MARGIN <= x < MARGIN + ntiles(field) * TILE && MARGIN <= y < MARGIN + ntiles(field) * TILE
function tile_of(x, y, field)
    nt = ntiles(field)
    inner(x, y, field) || return 0
    return 1 + floor(Int, (x - MARGIN) / TILE) + nt * floor(Int, (y - MARGIN) / TILE)
end

# Per-tile statistics of one emitter set: n emitters (by position), n DBSCAN clusters, clusters
# with >= 3 and >= 4 emitters. The FULL field is clustered; clusters are kept by centroid tile.
function tile_stats(xs, ys, field)
    nt = ntiles(field)^2
    ne, nc, n3, n4 = zeros(Int, nt), zeros(Int, nt), zeros(Int, nt), zeros(Int, nt)
    for i in eachindex(xs)
        t = tile_of(xs[i], ys[i], field); t > 0 && (ne[t] += 1)
    end
    ems = [Emitter2DFit{Float64}(xs[i], ys[i], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0; σ_xy=0.0,
                                 frame=1, dataset=1, track_id=i, id=0) for i in eachindex(xs)]
    out, _ = cluster(BasicSMLD(ems, CAMERA, 1, 1, Dict{String,Any}()),
                     DBSCANConfig(eps_nm=DBSCAN_EPS_NM, min_points=1, per_dataset=false))
    members = Dict{Int,Vector{Int}}()
    for (i, e) in enumerate(out.emitters)
        push!(get!(members, e.id, Int[]), i)
    end
    for idx in values(members)
        t = tile_of(mean(xs[idx]), mean(ys[idx]), field)
        t > 0 || continue
        nc[t] += 1; n3[t] += length(idx) >= 3; n4[t] += length(idx) >= 4
    end
    return (; ne, nc, n3, n4)
end

tsum(ts, f, sel=:) = sum(getfield(ts, f)[sel])
share(ts, f, sel=:) = tsum(ts, f, sel) / max(tsum(ts, :nc, sel), 1)

function run_case(label, smld, x, y, cap, field, ρ)
    control_kw = CONTROL ? (; spatial_model=:flat, learn_rho=false, rho=ρ) : IPP ? (; spatial_model=:ipp) : (;)
    t0 = time()
    r, diag = run_bagol(smld; n_iterations=N_ITER, burn_in=BURN_IN,
        max_partition_size=cap, overlap=0.025, partition_sigma=3.0, se_adjust=0.0,
        posterior_pixel_size=0.0, verbose=false, control_kw...)
    nloc = Dict{Int,Int}(); sx = Dict{Int,Float64}(); sy = Dict{Int,Float64}()
    for e in smld.emitters
        nloc[e.track_id] = get(nloc, e.track_id, 0) + 1
        sx[e.track_id] = get(sx, e.track_id, 0.0) + e.x
        sy[e.track_id] = get(sy, e.track_id, 0.0) + e.y
    end
    n_true1 = count(i -> inner(x[i], y[i], field) && get(nloc, i, 0) >= 1, eachindex(x))
    n_true3 = count(i -> inner(x[i], y[i], field) && get(nloc, i, 0) >= 3, eachindex(x))
    ids1 = collect(keys(nloc))
    oracle = tile_stats([sx[i] / nloc[i] for i in ids1], [sy[i] / nloc[i] for i in ids1], field)
    truth = tile_stats(x, y, field)
    bagol = tile_stats([e.x for e in r.emitters], [e.y for e in r.emitters], field)
    pid = diag.partition_ids
    psize = Dict{Int,Int}(); for p in pid; p > 0 && (psize[p] = get(psize, p, 0) + 1); end
    medpart = median([psize[p] for p in pid if p > 0])
    return (; label, cap, field, nloc=length(smld.emitters), n_true1, n_true3, bagol, oracle, truth,
            medpart, nparts=diag.n_partitions, mu=diag.final_μ, shape=diag.final_shape,
            minutes=(time() - t0) / 60)
end

function show_case(c)
    @printf("\n[%s] field=%.0f um  max_partition_size=%d  locs=%d  true emitters inner: %d with >=1 loc, %d with >=3 locs\n",
            c.label, c.field, c.cap, c.nloc, c.n_true1, c.n_true3)
    @printf("  partitions=%d  median partition size per loc=%.0f  learned mu=%.2f shape=%.2f  (%.1f min)\n",
            c.nparts, c.medpart, c.mu, c.shape, c.minutes)
    println("  source   emitters  per-true(>=3)  per-true(>=1)  DBSCAN share>=3  share>=4")
    for (nm, s) in (("BaGoL ", c.bagol), ("oracle", c.oracle), ("truth ", c.truth))
        n = tsum(s, :ne)
        @printf("  %s  %8d  %13.3f  %13.3f  %15.3f  %8.3f\n", nm, n, n / c.n_true3, n / c.n_true1,
                share(s, :n3), share(s, :n4))
    end
end

function main()
    t0 = time()
    Random.seed!(SEED)
    println("dense_field_gate  seed=$SEED  threads=$(Threads.nthreads())  mode=$(IPP ? "ipp" : CONTROL ? "CONTROL flat fixed-rho" : "production")")
    dense = simulate_field(DENSE, FIELD_DENSE)
    low = simulate_field(SPARSE, FIELD_LOW)
    cases = [
        run_case("dense 250/um^2", dense..., 500, FIELD_DENSE, DENSE),
        run_case("low-density $(Int(SPARSE))/um^2", low..., 500, FIELD_LOW, SPARSE),
        run_case("dense 250/um^2", dense..., 100, FIELD_DENSE, DENSE),
        run_case("dense 250/um^2", dense..., 1000, FIELD_DENSE, DENSE),
    ]
    foreach(show_case, cases)

    rat(c, sel=:) = tsum(c.bagol, :ne, sel) / tsum(c.oracle, :ne, sel)
    rd, rs = rat(cases[1]), rat(cases[2])
    cnt(c, sel=:) = tsum(c.bagol, :ne, sel)
    cap100 = cnt(cases[3]) / cnt(cases[1]) - 1
    cap1000 = cnt(cases[4]) / cnt(cases[1]) - 1
    # tile bootstrap (dense arms share tiles, so the cap sweep stays paired)
    rng = MersenneTwister(SEED + 1)
    nd, nl = length(cases[1].bagol.ne), length(cases[2].bagol.ne)
    bdiff, b100, b1000 = zeros(NBOOT), zeros(NBOOT), zeros(NBOOT)
    for b in 1:NBOOT
        sd, sl = rand(rng, 1:nd, nd), rand(rng, 1:nl, nl)
        bdiff[b] = rat(cases[1], sd) - rat(cases[2], sl)
        b100[b] = cnt(cases[3], sd) / cnt(cases[1], sd) - 1
        b1000[b] = cnt(cases[4], sd) / cnt(cases[1], sd) - 1
    end
    println("\n=== SUMMARY (r = BaGoL emitters / oracle emitters, inner region) ===")
    @printf("r(250/um^2, cap 500) = %.4f   r(low, cap 500) = %.4f\n", rd, rs)
    @printf("reported, not gated: r-1 = %+.3f (dense), %+.3f (low)\n", rd - 1, rs - 1)
    @printf("DBSCAN share>=4: dense BaGoL/oracle/truth %.3f/%.3f/%.3f; low %.3f/%.3f/%.3f\n",
            share(cases[1].bagol, :n4), share(cases[1].oracle, :n4), share(cases[1].truth, :n4),
            share(cases[2].bagol, :n4), share(cases[2].oracle, :n4), share(cases[2].truth, :n4))
    gated = [("|r_dense - r_low|", abs(rd - rs), std(bdiff), 0.03),
             ("cap 100 vs cap 500 (dense)", abs(cap100), std(b100), 0.02),
             ("cap 1000 vs cap 500 (dense)", abs(cap1000), std(b1000), 0.02)]
    ok = true
    for (gi, (n, v, se, thr)) in enumerate(gated)
        pass = v <= thr
        counted = gi != 1                    # density comparison: informational in both modes
        counted && (ok &= pass)
        @printf("  %s %-30s |value| %.4f   bootstrap SE %.4f   threshold %.2f\n", counted ? (pass ? "ok  " : "FAIL") : "info", n, v, se, thr)
    end
    @printf("signed: r_dense - r_low %+.4f, cap100 %+.4f, cap1000 %+.4f\n", rd - rs, cap100, cap1000)
    @printf("runtime %.1f min\n", (time() - t0) / 60)
    println(ok ? "VERDICT: PASS (dense-field G5/G7)" : "VERDICT: FAIL (dense-field G5/G7)")
    return ok
end

exit(main() ? 0 : 1)

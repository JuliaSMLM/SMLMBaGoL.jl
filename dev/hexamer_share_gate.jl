# G6 hexamer-share gate: in a dense field with 10% of emitters in hexamers, does the
# DBSCAN share of clusters with >= 4 emitters computed on BaGoL's MAP-N match the
# oracle's?
#
# WHY: the downstream readout (oligomer share) is a DBSCAN over BaGoL's emitter
# positions. At ~250 emitters/um^2 precision-DBSCAN partitioning percolates and the
# per-partition chains see a different count/area context than an isolated cluster
# (see dense_field_gate.jl); an undercount of single emitters or a failure to split
# hexamer members both move this share. This gate measures it against an ORACLE (one
# emitter per true emitter with >= 1 localization, at the mean of its localizations)
# and the truth, with the CURRENT sampler.
#
# WHAT IT TESTS: 5 x 5 um field, scored on the inner 4 x 4 um, ~250 emitters/um^2
# equivalent: 90% of emitters single (uniform2D with Nmer2D(n=1)), 10% in hexamers
# (uniform2D with Nmer2D(n=6, d=20 nm), 250 * 0.1 / 6 patterns/um^2). Production
# run_bagol settings (locmix, learned count model, partition_sigma 3,
# max_partition_size 500, overlap 0.025, 6000 / 2000), se_adjust = 0.
# Euclidean DBSCAN (SMLMClustering.DBSCANConfig, eps 30 nm, min_points 1; the same
# settings for BaGoL, oracle and truth); the FULL field is clustered and clusters are kept
# by centroid in the inner region; reports the share of clusters with >= 3 and >= 4 emitters, emitter counts,
# and the median partition size per localization.
# Truth = every true emitter (including those with no localization).
#
# SIMULATION: SMLMSim only (uniform2D, Nmer2D, GenericFluor bleaching 3-state via
# SMLMSim.simulate(StaticSMLMConfig; starting_conditions=...), apply_noise inside
# simulate). The true emitter of every localization is its track_id.
#
# PASS criterion: |share>=4 (BaGoL) - share>=4 (oracle)| <= 0.02.
#
# HOW TO RUN (gitignored env dev/output/gate_env, not committed):
#   nice julia -t 16 --project=dev/output/gate_env dev/hexamer_share_gate.jl [seed]
# RECREATE the env from scratch, from the repo root of this checkout (tested with Julia
# 1.13.0; SMLMBaGoL = this checkout; SMLMSim v0.7.3-DEV @46bc484 and SMLMData v0.7.1
# @0c9fb55 as sibling JuliaSMLM checkouts, or the registered 0.7.x releases; SMLMClustering
# 0.4.3 registered; Statistics, Random, Printf stdlib):
#   mkdir -p dev/output/gate_env
#   julia --project=dev/output/gate_env -e 'using Pkg; Pkg.develop(path="."); \
#     Pkg.develop(path="../JuliaSMLM/SMLMSim"); Pkg.develop(path="../JuliaSMLM/SMLMData"); \
#     Pkg.add(["SMLMClustering", "Statistics", "Random", "Printf"])'
#   (adjust the two sibling paths; or Pkg.add(["SMLMSim", "SMLMData"]) for registered releases)
# GATE_QUICK=1 shrinks the fields for a smoke run. Exit status: 0 on PASS, 1 on FAIL.
# UNCERTAINTY: each gated difference is printed with a bootstrap SE (resampling 1 um tiles
# of the inner region, so the arms stay paired where they share data) next to its threshold.

using SMLMBaGoL, SMLMSim, SMLMData, SMLMClustering
using Statistics, Random, Printf

const SEED = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20261023
const QUICK = get(ENV, "GATE_QUICK", "0") == "1"
const FIELD = QUICK ? 2.5 : 5.0                 # um
const MARGIN = 0.5                              # inner region = [MARGIN, field - MARGIN]^2
const TILE = 1.0                                # um; bootstrap tile size
const NBOOT = 1000
const DENSITY = 250.0                           # emitters / um^2 equivalent
const HEX_FRACTION, HEX_N, HEX_D = 0.10, 6, 0.020   # share of emitters in hexamers; n, diameter (um)
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

function simulate_field()
    xs, ys, _ = uniform2D((1 - HEX_FRACTION) * DENSITY, Nmer2D(; n=1, d=0.0), FIELD, FIELD)
    xh, yh, _ = uniform2D(HEX_FRACTION * DENSITY / HEX_N, Nmer2D(; n=HEX_N, d=HEX_D), FIELD, FIELD)
    x, y = vcat(xs, xh), vcat(ys, yh)
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

# GATE_CONTROL=ipp runs production settings with spatial_model=:ipp.
const IPP = get(ENV, "GATE_CONTROL", "") == "ipp"

function run_case(label, smld, x, y, cap, field)
    ipp_kw = IPP ? (; spatial_model=:ipp) : (;)
    t0 = time()
    r, diag = run_bagol(smld; n_iterations=N_ITER, burn_in=BURN_IN, ipp_kw...,
        max_partition_size=cap, overlap=0.025, partition_sigma=3.0, se_adjust=0.0,
        posterior_pixel_size=0.0, verbose=false)
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
    println("hexamer_share_gate  mode=$(IPP ? "ipp" : "production")  seed=$SEED  threads=$(Threads.nthreads())  field=$(FIELD) um")
    smld, x, y = simulate_field()
    c = run_case("dense + 10% hexamers", smld, x, y, 500, FIELD)
    show_case(c)
    d4 = share(c.bagol, :n4) - share(c.oracle, :n4)
    rng = MersenneTwister(SEED + 1)
    nt = length(c.bagol.ne)
    bd = [(sel = rand(rng, 1:nt, nt); share(c.bagol, :n4, sel) - share(c.oracle, :n4, sel)) for _ in 1:NBOOT]
    println("\n=== SUMMARY ===")
    @printf("DBSCAN share>=4: BaGoL %.3f, oracle %.3f, truth %.3f\n",
            share(c.bagol, :n4), share(c.oracle, :n4), share(c.truth, :n4))
    @printf("DBSCAN share>=3: BaGoL %.3f, oracle %.3f, truth %.3f\n",
            share(c.bagol, :n3), share(c.oracle, :n3), share(c.truth, :n3))
    @printf("emitters (inner): BaGoL %d, oracle %d, truth %d\n",
            tsum(c.bagol, :ne), tsum(c.oracle, :ne), tsum(c.truth, :ne))
    ok = abs(d4) <= 0.02
    @printf("gated: share>=4 BaGoL - oracle = %+.4f   bootstrap SE %.4f   threshold 0.02\n", d4, std(bd))
    @printf("runtime %.1f min\n", (time() - t0) / 60)
    println(ok ? "VERDICT: PASS (hexamer share G6)" : "VERDICT: FAIL (hexamer share G6)")
    return ok
end

exit(main() ? 0 : 1)

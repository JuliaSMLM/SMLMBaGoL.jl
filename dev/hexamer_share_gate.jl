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
# settings for BaGoL, oracle and truth), clusters counted by centroid in the inner
# region; reports the share of clusters with >= 3 and >= 4 emitters, emitter counts,
# and the median partition size per localization.
# Truth = every true emitter (including those with no localization).
#
# SIMULATION: SMLMSim only (uniform2D, Nmer2D, GenericFluor bleaching 3-state via
# SMLMSim.simulate(StaticSMLMConfig; starting_conditions=...), apply_noise inside
# simulate). The true emitter of every localization is its track_id.
#
# PASS criterion: |share>=4 (BaGoL) - share>=4 (oracle)| <= 0.02.
#
# HOW TO RUN (gitignored env at dev/output/gate_env: SMLMBaGoL dev'd to this checkout,
# SMLMSim, SMLMData, SMLMClustering):
#   nice julia -t 16 --project=dev/output/gate_env dev/hexamer_share_gate.jl [seed]
# GATE_QUICK=1 shrinks the field to 2.5 x 2.5 um for a smoke run.

using SMLMBaGoL, SMLMSim, SMLMData, SMLMClustering
using Statistics, Random, Printf

const SEED = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20261023
const QUICK = get(ENV, "GATE_QUICK", "0") == "1"
const FIELD = QUICK ? 2.5 : 5.0                 # um
const MARGIN = QUICK ? 0.25 : 0.5               # inner region = [MARGIN, FIELD - MARGIN]^2
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

inner(x, y) = MARGIN <= x <= FIELD - MARGIN && MARGIN <= y <= FIELD - MARGIN

# DBSCAN shares over a set of emitter positions; clusters counted by centroid in the inner region.
function dbscan_shares(xs, ys)
    n = length(xs)
    ems = [Emitter2DFit{Float64}(xs[i], ys[i], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0; σ_xy=0.0,
                                 frame=1, dataset=1, track_id=i, id=0) for i in 1:n]
    out, _ = cluster(BasicSMLD(ems, CAMERA, 1, 1, Dict{String,Any}()),
                     DBSCANConfig(eps_nm=DBSCAN_EPS_NM, min_points=1, per_dataset=false))
    members = Dict{Int,Vector{Int}}()
    for (i, e) in enumerate(out.emitters)
        push!(get!(members, e.id, Int[]), i)
    end
    sizes = Int[]
    for idx in values(members)
        inner(mean(xs[idx]), mean(ys[idx])) && push!(sizes, length(idx))
    end
    return mean(sizes .>= 3), mean(sizes .>= 4), length(sizes)
end

struct Scored
    count::Int; per3::Float64; per1::Float64
    s3::Float64; s4::Float64
end

function score(xs, ys, n_true1, n_true3)
    sel = [i for i in eachindex(xs) if inner(xs[i], ys[i])]
    s3, s4, _ = dbscan_shares(xs[sel], ys[sel])
    Scored(length(sel), length(sel) / n_true3, length(sel) / n_true1, s3, s4)
end

function run_case(label, smld, x, y, cap)
    t0 = time()
    r, diag = run_bagol(smld; n_iterations=N_ITER, burn_in=BURN_IN,
        max_partition_size=cap, overlap=0.025, partition_sigma=3.0, se_adjust=0.0,
        posterior_pixel_size=0.0, verbose=false)
    # ground truth bookkeeping
    nloc = Dict{Int,Int}(); sx = Dict{Int,Float64}(); sy = Dict{Int,Float64}()
    for e in smld.emitters
        nloc[e.track_id] = get(nloc, e.track_id, 0) + 1
        sx[e.track_id] = get(sx, e.track_id, 0.0) + e.x
        sy[e.track_id] = get(sy, e.track_id, 0.0) + e.y
    end
    in_inner = [inner(x[i], y[i]) for i in eachindex(x)]
    n_true1 = count(i -> in_inner[i] && get(nloc, i, 0) >= 1, eachindex(x))
    n_true3 = count(i -> in_inner[i] && get(nloc, i, 0) >= 3, eachindex(x))
    ids1 = [i for i in keys(nloc)]
    oracle = score([sx[i] / nloc[i] for i in ids1], [sy[i] / nloc[i] for i in ids1], n_true1, n_true3)
    truth = score(x, y, n_true1, n_true3)
    bagol = score([e.x for e in r.emitters], [e.y for e in r.emitters], n_true1, n_true3)
    # median partition size per localization
    pid = diag.partition_ids
    psize = Dict{Int,Int}(); for p in pid; p > 0 && (psize[p] = get(psize, p, 0) + 1); end
    medpart = median([psize[p] for p in pid if p > 0])
    return (; label, cap, nloc=length(smld.emitters), n_true1, n_true3, bagol, oracle, truth,
            medpart, nparts=diag.n_partitions, mu=diag.final_μ, shape=diag.final_shape,
            minutes=(time() - t0) / 60)
end

function show_case(c)
    @printf("\n[%s] max_partition_size=%d  locs=%d  true emitters inner: %d with >=1 loc, %d with >=3 locs\n",
            c.label, c.cap, c.nloc, c.n_true1, c.n_true3)
    @printf("  partitions=%d  median partition size per loc=%.0f  learned mu=%.2f shape=%.2f  (%.1f min)\n",
            c.nparts, c.medpart, c.mu, c.shape, c.minutes)
    println("  source   emitters  per-true(>=3)  per-true(>=1)  DBSCAN share>=3  share>=4")
    for (nm, s) in (("BaGoL ", c.bagol), ("oracle", c.oracle), ("truth ", c.truth))
        @printf("  %s  %8d  %13.3f  %13.3f  %15.3f  %8.3f\n", nm, s.count, s.per3, s.per1, s.s3, s.s4)
    end
end

function main()
    t0 = time()
    Random.seed!(SEED)
    println("hexamer_share_gate  seed=$SEED  threads=$(Threads.nthreads())  field=$(FIELD) um")
    smld, x, y = simulate_field()
    c = run_case("dense + 10% hexamers", smld, x, y, 500)
    show_case(c)
    d4 = c.bagol.s4 - c.oracle.s4
    println("\n=== SUMMARY ===")
    @printf("DBSCAN share>=4: BaGoL %.3f, oracle %.3f, truth %.3f  (BaGoL - oracle = %+.3f)\n",
            c.bagol.s4, c.oracle.s4, c.truth.s4, d4)
    @printf("DBSCAN share>=3: BaGoL %.3f, oracle %.3f, truth %.3f\n", c.bagol.s3, c.oracle.s3, c.truth.s3)
    @printf("emitters (inner): BaGoL %d, oracle %d, truth %d\n", c.bagol.count, c.oracle.count, c.truth.count)
    ok = abs(d4) <= 0.02
    @printf("runtime %.1f min\n", (time() - t0) / 60)
    println(ok ? "VERDICT: PASS (hexamer share G6)" : "VERDICT: FAIL (hexamer share G6)")
    return ok
end

main()

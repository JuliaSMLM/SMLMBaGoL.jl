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
# WHAT IT TESTS: uniform random single emitters, 5 x 5 um, scored on the inner 4 x 4 um,
# at 250/um^2 (percolating) and 100/um^2 (not), production run_bagol settings (locmix,
# learned count model, partition_sigma 3, max_partition_size 500, overlap 0.025,
# 6000 / 2000), se_adjust = 0 (no extra scatter is simulated). The dense field is also
# run at max_partition_size 100 and 1000 on the same data. Per run it reports:
#   - emitters per true emitter with >= 3 localizations (BaGoL / oracle / truth),
#     and per true emitter with >= 1 localization;
#   - share of Euclidean-DBSCAN clusters (SMLMClustering.DBSCANConfig, eps 30 nm,
#     min_points 1, the same settings for BaGoL, oracle and truth) with >= 3 and >= 4
#     emitters, clusters counted by centroid in the inner region;
#   - median partition size per localization (diagnostics.partition_ids).
# Truth = every true emitter (including those with no localization); oracle = true
# emitters with >= 1 localization at the mean of their localizations.
#
# SIMULATION: SMLMSim only. uniform2D(rho, Nmer2D(n=1), 5, 5) places the emitters;
# SMLMSim.simulate(StaticSMLMConfig; starting_conditions=...) runs a bleaching 3-state
# GenericFluor (dSTORM-like, ~15 locs/emitter, shape ~1.1) with apply_noise. The true
# emitter of every localization is its track_id.
#
# PASS criterion (ratio r = BaGoL emitters / oracle emitters in the inner region):
#   |r - 1| <= 0.05 at both densities; |r(250) - r(100)| <= 0.03;
#   cap sweep: |count(cap 100 or 1000) / count(cap 500) - 1| <= 0.02.
#
# HOW TO RUN (gitignored env at dev/output/gate_env: SMLMBaGoL dev'd to this checkout,
# SMLMSim, SMLMData, SMLMClustering):
#   nice julia -t 16 --project=dev/output/gate_env dev/dense_field_gate.jl [seed]
# GATE_QUICK=1 shrinks the field to 2.5 x 2.5 um for a smoke run.

using SMLMBaGoL, SMLMSim, SMLMData, SMLMClustering
using Statistics, Random, Printf

const SEED = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20261023
const QUICK = get(ENV, "GATE_QUICK", "0") == "1"
const FIELD = QUICK ? 2.5 : 5.0                 # um
const MARGIN = QUICK ? 0.25 : 0.5               # inner region = [MARGIN, FIELD - MARGIN]^2
const DENSE, SPARSE = 250.0, 100.0              # emitters / um^2
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

function simulate_field(ρ::Float64)
    x, y, _ = uniform2D(ρ, Nmer2D(; n=1, d=0.0), FIELD, FIELD)
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
    println("dense_field_gate  seed=$SEED  threads=$(Threads.nthreads())  field=$(FIELD) um")
    dense = simulate_field(DENSE)
    sparse = simulate_field(SPARSE)
    cases = [
        run_case("dense 250/um^2", dense..., 500),
        run_case("sparse 100/um^2", sparse..., 500),
        run_case("dense 250/um^2", dense..., 100),
        run_case("dense 250/um^2", dense..., 1000),
    ]
    foreach(show_case, cases)

    ratio(c) = c.bagol.count / c.oracle.count
    rd, rs = ratio(cases[1]), ratio(cases[2])
    cap100 = cases[3].bagol.count / cases[1].bagol.count - 1
    cap1000 = cases[4].bagol.count / cases[1].bagol.count - 1
    println("\n=== SUMMARY (r = BaGoL emitters / oracle emitters, inner region) ===")
    @printf("r(250/um^2, cap 500) = %.4f   r(100/um^2, cap 500) = %.4f   |difference| = %.4f\n", rd, rs, abs(rd - rs))
    @printf("cap sweep on dense: cap100 %+.4f, cap1000 %+.4f vs cap500\n", cap100, cap1000)
    @printf("DBSCAN share>=4: dense BaGoL/oracle/truth %.3f/%.3f/%.3f; sparse %.3f/%.3f/%.3f\n",
            cases[1].bagol.s4, cases[1].oracle.s4, cases[1].truth.s4,
            cases[2].bagol.s4, cases[2].oracle.s4, cases[2].truth.s4)
    checks = [abs(rd - 1) <= 0.05, abs(rs - 1) <= 0.05, abs(rd - rs) <= 0.03,
              abs(cap100) <= 0.02, abs(cap1000) <= 0.02]
    names = ["|r_dense-1|<=5%", "|r_sparse-1|<=5%", "|r_dense-r_sparse|<=3%", "cap100 within 2%", "cap1000 within 2%"]
    for (n, ok) in zip(names, checks)
        println("  ", ok ? "ok  " : "FAIL", " ", n)
    end
    @printf("runtime %.1f min\n", (time() - t0) / 60)
    println(all(checks) ? "VERDICT: PASS (dense-field G5/G7)" : "VERDICT: FAIL (dense-field G5/G7)")
    return all(checks)
end

main()

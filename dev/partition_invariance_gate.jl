# G2 partition-invariance gate: does an isolated pair of emitters resolve the same way
# alone as when it sits in a chain with ~16 neighbouring pairs?
#
# WHY: the collapsed likelihood does not couple spatially distant clusters, so the
# posterior over a pair's partition should not depend on what else is in the chain.
# The count/K prior and the locmix spatial term, however, depend on the whole chain
# (N, K, the bounding-box area), so a pair can resolve differently alone vs grouped.
# Earlier measurement with genmab's simulator: 50% resolution point 16.8 nm alone vs
# 19.6 nm grouped; mean Dahl clusters per pair 2.07 vs 1.74. This gate reproduces
# that comparison with SMLMSim primitives only, against the CURRENT sampler.
#
# WHAT IT TESTS: isolated emitter pairs at separations 0, 5, 10, 15, 20, 25, 30, 40 nm
# (random orientation), 1.1 um apart on a 4x4 grid per block, >= 150 pairs per
# separation. The SAME localizations of every pair are run (a) alone through
# run_collapsed_chain and (b) inside the block of ~16 pairs (~200-400 locs).
# Fixed mu and shape (the simulation's), 6000 / 2000 iterations, Dahl estimate
# (PartitionSamples thin=5 + PSMAccumulator + estimate_dahl, as in run_bagol).
# Per separation it reports the fraction of pairs resolved (each emitter's locs mostly
# in its own Dahl cluster, neither cluster holding >= 20% of the other's locs) and the
# mean number of Dahl clusters per pair, alone vs grouped, plus the 50% resolution point.
#
# SIMULATION: SMLMSim only. Nmer2D(n=2, d=sep) + rotate! give the pair geometry;
# SMLMSim.simulate(StaticSMLMConfig; starting_conditions=...) runs a bleaching 3-state
# GenericFluor (dSTORM-like, ~15 locs/emitter, shape ~1.1) and apply_noise. The true
# emitter of every localization is its track_id. Pairs where either emitter has no
# localization are dropped (same pairs in both arms).
#
# PASS criterion: |50%-point(alone) - 50%-point(grouped)| <= 1.5 nm AND, at every
# separation, mean clusters per pair alone vs grouped agree within 5% (relative).
#
# HOW TO RUN (gitignored env at dev/output/gate_env: SMLMBaGoL dev'd to this checkout,
# SMLMSim, SMLMData, SMLMClustering; see the commit message for how it was built):
#   nice julia -t 16 --project=dev/output/gate_env dev/partition_invariance_gate.jl [seed]
# GATE_BLOCKS=2 gives a quick smoke run. Chains run in parallel across threads, so results vary run to run (seed fixes the data).

using SMLMBaGoL, SMLMSim, SMLMData
using Statistics, Random, Printf

const SEED = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20261023
const SEPS_NM = [0, 5, 10, 15, 20, 25, 30, 40]
const N_BLOCKS = parse(Int, get(ENV, "GATE_BLOCKS", "12"))            # blocks of 16 pairs per separation (~8% dropped => >= 150 pairs)
const GRID_N = 4                # 4x4 pairs per block
const GRID_SPACING = 1.1        # um
const N_ITER, BURN_IN = 6000, 2000
const FIELD_PAD = 0.55          # um margin around the block grid

# ---- SMLMSim fluorophore: bleaching 3-state (ON, OFF, bleached), dSTORM-like ----
const FRAMERATE, NFRAMES = 50.0, 3000
const K_OFF, K_ON, K_BLEACH, PHOTON_RATE = 50.0, 1.0, 6.5, 40000.0
const FLUOR = GenericFluor(PHOTON_RATE,
    [-(K_OFF + K_BLEACH) K_OFF K_BLEACH; K_ON -K_ON 0.0; 0.0 0.0 0.0])
const SIMCFG = StaticSMLMConfig(density=1.0, σ_psf=0.13, minphotons=30, ndatasets=1,
                                nframes=NFRAMES, framerate=FRAMERATE)
const CAMERA = IdealCamera(1:60, 1:60, 0.1)

function sim_emitters(sites::Vector{NTuple{2,Float64}}, id0::Int)
    ems = [Emitter2DFit{Float64}(x, y, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0; σ_xy=0.0,
                                 frame=1, dataset=1, track_id=id0 + i, id=id0 + i)
           for (i, (x, y)) in enumerate(sites)]
    smld, _ = SMLMSim.simulate(SIMCFG; starting_conditions=ems, molecule=FLUOR,
                               camera=CAMERA, state1=1)
    return smld.emitters
end

# Count-model parameters of the simulation (moment match over emitters with >= 1 loc).
function calibrate_count_model(n=20000)
    sites = [(rand() * 10, rand() * 10) for _ in 1:n]
    c = zeros(Int, n)
    for e in sim_emitters(sites, 0)
        c[e.track_id] += 1
    end
    c = c[c .> 0]
    μ, v = mean(c), var(c)
    return μ, v > μ ? μ^2 / (v - μ) : 1000.0
end

# One block: GRID_N^2 pairs of separation sep_nm. Returns locs (track_id = 2p-1, 2p for
# pair p, offset by id0) and the list of pairs kept (both emitters have >= 1 loc).
function make_block(sep_nm::Int, id0::Int)
    pat = Nmer2D(; n=2, d=sep_nm / 1000)
    sites = NTuple{2,Float64}[]
    for gi in 1:GRID_N, gj in 1:GRID_N
        θ = 2π * rand()
        rotate!(pat, θ)                       # rotates the stored offsets cumulatively; random anyway
        cx, cy = FIELD_PAD + (gi - 1) * GRID_SPACING, FIELD_PAD + (gj - 1) * GRID_SPACING
        for m in 1:2
            push!(sites, (cx + pat.x[m], cy + pat.y[m]))
        end
    end
    locs = sim_emitters(sites, id0)
    cnt = Dict{Int,Int}()
    for e in locs
        cnt[e.track_id] = get(cnt, e.track_id, 0) + 1
    end
    keep = [p for p in 1:GRID_N^2 if get(cnt, id0 + 2p - 1, 0) > 0 && get(cnt, id0 + 2p, 0) > 0]
    kept_ids = Set(Iterators.flatten((id0 + 2p - 1, id0 + 2p) for p in keep))
    locs = [e for e in locs if e.track_id in kept_ids]
    return locs, [(id0 + 2p - 1, id0 + 2p) for p in keep]
end

function dahl_labels(locs, μ, shape)
    psm_acc, ps_acc = PSMAccumulator(), PartitionSamples(thin=5)
    run_collapsed_chain(locs; n_iterations=N_ITER, burn_in=BURN_IN,
        shape=shape, μ_prior_shape=1.0, μ_prior_scale=μ,   # initial μ = μ_prior_shape*scale
        learn_distribution=false, learn_rho=false,
        accumulators=AbstractAccumulator[ps_acc, psm_acc])
    psm = SMLMBaGoL.accumulator_result(psm_acc).psm
    _, _, _, z = estimate_dahl(ps_acc.samples, locs, psm)
    return z
end

# resolved?, n clusters  for one pair given labels z over `locs` and the emitters' ids
function score_pair(z, ids_of_loc, ida, idb)
    A = [z[i] for i in eachindex(z) if ids_of_loc[i] == ida]
    B = [z[i] for i in eachindex(z) if ids_of_loc[i] == idb]
    ncl = length(unique(vcat(A, B)))
    major(v) = (u = unique(v); u[argmax([count(==(x), v) for x in u])])
    ca, cb = major(A), major(B)
    res = ca != cb &&
          count(==(ca), A) / length(A) > 0.5 && count(==(cb), B) / length(B) > 0.5 &&
          count(==(ca), B) / length(B) < 0.2 && count(==(cb), A) / length(A) < 0.2
    return res, ncl
end

# Linear interpolation of the first upward crossing of 0.5 (nm); NaN if never crossed.
function p50(seps, frac)
    for i in 2:length(seps)
        if frac[i-1] < 0.5 <= frac[i]
            return seps[i-1] + (0.5 - frac[i-1]) / (frac[i] - frac[i-1]) * (seps[i] - seps[i-1])
        end
    end
    return frac[1] >= 0.5 ? Float64(seps[1]) : NaN
end

function main()
    t0 = time()
    Random.seed!(SEED)
    println("partition_invariance_gate  seed=$SEED  threads=$(Threads.nthreads())")
    μ, shape = calibrate_count_model()
    @printf("simulation count model: mu = %.3f, shape = %.3f\n", μ, shape)

    # data: blocks[(sepidx, b)] = (locs, pairs)
    blocks = Dict{Tuple{Int,Int},Tuple{Vector{Emitter2DFit{Float64}},Vector{NTuple{2,Int}}}}()
    for (si, s) in enumerate(SEPS_NM), b in 1:N_BLOCKS
        blocks[(si, b)] = make_block(s, 100_000 * ((si - 1) * N_BLOCKS + b))
    end
    for si in eachindex(SEPS_NM)
        np = sum(length(blocks[(si, b)][2]) for b in 1:N_BLOCKS)
        np >= 150 || @warn "separation $(SEPS_NM[si]) nm has only $np pairs"
    end

    # jobs: grouped (whole block) and alone (each pair), largest first
    jobs = Tuple[]
    for ((si, b), (locs, pairs)) in blocks
        push!(jobs, (:grouped, si, b, 0, locs))
        for (pi, (ida, idb)) in enumerate(pairs)
            push!(jobs, (:alone, si, b, pi, [e for e in locs if e.track_id == ida || e.track_id == idb]))
        end
    end
    sort!(jobs; by=j -> -length(j[5]))
    labels = Vector{Vector{Int16}}(undef, length(jobs))
    Threads.@threads :dynamic for k in eachindex(jobs)
        labels[k] = dahl_labels(jobs[k][5], μ, shape)
    end
    println(@sprintf("chains done: %d (%.1f min)", length(jobs), (time() - t0) / 60))

    # score
    res = Dict(arm => [(Int[], Int[]) for _ in SEPS_NM] for arm in (:alone, :grouped))
    for (k, (arm, si, b, pi, locs)) in enumerate(jobs)
        ids = [e.track_id for e in locs]
        pairs = blocks[(si, b)][2]
        for (pj, (ida, idb)) in enumerate(pairs)
            (arm === :alone && pj != pi) && continue
            r, n = score_pair(labels[k], ids, ida, idb)
            push!(res[arm][si][1], r)
            push!(res[arm][si][2], n)
        end
    end

    println("\nsep_nm  npairs | resolved alone  grouped | clusters/pair alone  grouped  rel.diff")
    fr = Dict(:alone => Float64[], :grouped => Float64[])
    worst = 0.0
    for (si, s) in enumerate(SEPS_NM)
        ra, na = res[:alone][si]; rg, ng = res[:grouped][si]
        fa, fg = mean(ra), mean(rg)
        push!(fr[:alone], fa); push!(fr[:grouped], fg)
        ca, cg = mean(na), mean(ng)
        worst = max(worst, abs(ca - cg) / ca)
        @printf("%5d  %6d |   %.3f   %.3f   |   %.3f   %.3f   %+.1f%%\n",
                s, length(ra), fa, fg, ca, cg, 100 * (cg - ca) / ca)
    end
    pa, pg = p50(SEPS_NM, fr[:alone]), p50(SEPS_NM, fr[:grouped])
    @printf("\n50%% resolution point: alone %.1f nm, grouped %.1f nm (diff %+.1f nm)\n", pa, pg, pg - pa)
    @printf("max relative difference in mean clusters/pair: %.1f%%\n", 100 * worst)
    ok = isfinite(pa) && isfinite(pg) && abs(pa - pg) <= 1.5 && worst <= 0.05
    @printf("runtime %.1f min\n", (time() - t0) / 60)
    println(ok ? "VERDICT: PASS (partition-invariance G2)" : "VERDICT: FAIL (partition-invariance G2)")
    return ok
end

main()

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
# CONTROL MODE: GATE_CONTROL=flat (see CONTROL below) runs the same gate with the fixed-rho flat prior.
#
# HOW TO RUN (gitignored env dev/output/gate_env, not committed):
#   nice julia -t 16 --project=dev/output/gate_env dev/partition_invariance_gate.jl [seed]
# RECREATE the env from scratch, from the repo root of this checkout (tested with Julia
# 1.13.0; SMLMBaGoL = this checkout; SMLMSim v0.7.3-DEV @46bc484 and SMLMData v0.7.1
# @0c9fb55 as sibling JuliaSMLM checkouts, or the registered 0.7.x releases; SMLMClustering
# 0.4.3 registered; Statistics, Random, Printf stdlib):
#   mkdir -p dev/output/gate_env
#   julia --project=dev/output/gate_env -e 'using Pkg; Pkg.develop(path="."); \
#     Pkg.develop(path="../JuliaSMLM/SMLMSim"); Pkg.develop(path="../JuliaSMLM/SMLMData"); \
#     Pkg.add(["SMLMClustering", "Statistics", "Random", "Printf"])'
#   (adjust the two sibling paths; or Pkg.add(["SMLMSim", "SMLMData"]) for registered releases)
# Exit status: 0 on PASS, 1 on FAIL.
# GATE_BLOCKS=2 gives a quick smoke run. Each chain is seeded from (seed, job identity), so results do
# not depend on thread scheduling.
#
# ESTIMATOR: both arms use a PAIR-LOCAL Dahl estimate: the sample minimizing the Dahl loss
# restricted to that pair's localizations (PSM submatrix). A single block-wide Dahl sample
# would itself shift clusters per pair even for independent pairs. UNCERTAINTY: paired
# bootstrap over pairs (alone/grouped kept together) gives an SE for each gated difference.

using SMLMBaGoL, SMLMSim, SMLMData
using Statistics, Random, Printf

const SEED = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20261023
const SEPS_NM = [0, 5, 10, 15, 20, 25, 30, 40]
const N_BLOCKS = parse(Int, get(ENV, "GATE_BLOCKS", "28"))            # blocks of 16 pairs per separation (~8% dropped => ~400 pairs)
const GRID_N = 4                # 4x4 pairs per block
const GRID_SPACING = 1.1        # um
const N_ITER, BURN_IN = 6000, 2000
# POSITIVE CONTROL (GATE_CONTROL=flat): spatial_model=:flat with the Poisson K prior and rho FIXED
# at the simulation's emitter density (2 emitters per 1.1 x 1.1 um grid cell), a prior that is
# partition-invariant by construction. A gate that passes here and fails in the default
# (production, locmix) mode is calibrated for its purpose.
# GATE_CONTROL=ipp: spatial_model=:ipp. One intensity map per separation is built from ALL that
# separation's pair localizations (the blocks are tiled into one big field) and the same frozen
# map is passed via intensity_map to every chain in both arms; rho is fixed at the field's true
# emitter density (hyperparameters are fixed in this gate). Sensitivity line only: the alone arm
# with a map refitted from each pair's own localizations (rho fixed at 2 / its box area).
const MODE = get(ENV, "GATE_CONTROL", "")
const CONTROL = MODE == "flat"
const IPP = MODE == "ipp"
const CONTROL_KW = CONTROL ? (; spatial_model=:flat, rho=2 / GRID_SPACING^2) : (;)
const TILE_STEP = 5.0           # um between block origins when tiling blocks into one field (ipp)
const TILE_COLS = 6
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

# Pair-local Dahl: among the stored samples, the one minimizing the Dahl loss restricted
# to the localizations `idx` (PSM submatrix). Returns that sample's labels over `idx`.
function local_dahl(samples, psm, idx)
    sub = psm[idx, idx]
    best, bd = 1, Inf
    for (t, smp) in enumerate(samples)
        d = SMLMBaGoL._association_psm_distance(smp[idx], sub)
        d < bd && ((best, bd) = (t, d))
    end
    return samples[best][idx]
end

# Run one chain; score each requested pair (a vector of (ida, idb)) with the pair-local Dahl.
function run_job(locs, pairs, μ, shape, kw=CONTROL_KW)
    psm_acc, ps_acc = PSMAccumulator(), PartitionSamples(thin=5)
    run_collapsed_chain(locs; n_iterations=N_ITER, burn_in=BURN_IN,
        shape=shape, μ_prior_shape=1.0, μ_prior_scale=μ,   # initial μ = μ_prior_shape*scale
        learn_distribution=false, learn_rho=false, kw...,
        accumulators=AbstractAccumulator[ps_acc, psm_acc])
    psm = SMLMBaGoL.accumulator_result(psm_acc).psm
    ids = [e.track_id for e in locs]
    return [score_pair(local_dahl(ps_acc.samples, psm, findall(i -> i == ida || i == idb, ids)),
                       [i for i in ids if i == ida || i == idb], ida, idb) for (ida, idb) in pairs]
end

# resolved?, n clusters for one pair given labels z over the pair's locs and each loc's emitter id
function score_pair(z, ids_of_loc, ida, idb)
    A = [z[i] for i in eachindex(z) if ids_of_loc[i] == ida]
    B = [z[i] for i in eachindex(z) if ids_of_loc[i] == idb]
    ncl = length(unique(z))
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
    println("partition_invariance_gate  seed=$SEED  threads=$(Threads.nthreads())  mode=$(IPP ? "ipp (shared frozen intensity map, fixed rho)" : CONTROL ? "CONTROL flat fixed-rho" : "production")")
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

    # ipp: tile the blocks of each separation into one field (translation only) and build one map
    ipp_kw = Dict{Int,Any}()
    if IPP
        for si in eachindex(SEPS_NM)
            allocs = Emitter2DFit{Float64}[]
            for b in 1:N_BLOCKS
                locs, _ = blocks[(si, b)]
                dx, dy = TILE_STEP * ((b - 1) % TILE_COLS), TILE_STEP * ((b - 1) ÷ TILE_COLS)
                locs = [(e = deepcopy(l); e.x += dx; e.y += dy; e) for l in locs]
                blocks[(si, b)] = (locs, blocks[(si, b)][2])
                append!(allocs, locs)
            end
            box = UniformSpatialPrior(allocs)
            imap = build_intensity_map(allocs, box)
            nem = 2 * sum(length(blocks[(si, b)][2]) for b in 1:N_BLOCKS)
            ipp_kw[si] = (; spatial_model=:ipp, intensity_map=imap, rho=nem / imap.field_area)
            @printf("sep %2d nm: map over %d locs, field area %.0f um^2, rho = %.4f /um^2\n",
                    SEPS_NM[si], length(allocs), imap.field_area, nem / imap.field_area)
        end
    end

    # jobs: grouped (whole block) and alone (each pair); seeded from job identity
    jobs = Tuple[]
    for ((si, b), (locs, pairs)) in blocks
        push!(jobs, (:grouped, si, b, 0, locs, pairs))
        for (pi, (ida, idb)) in enumerate(pairs)
            pl = [e for e in locs if e.track_id == ida || e.track_id == idb]
            push!(jobs, (:alone, si, b, pi, pl, [(ida, idb)]))
            IPP && push!(jobs, (:refit, si, b, pi, pl, [(ida, idb)]))
        end
    end
    sort!(jobs; by=j -> (-length(j[5]), String(j[1]), j[2], j[3], j[4]))
    out = Vector{Any}(undef, length(jobs))
    Threads.@threads :dynamic for k in eachindex(jobs)
        j = jobs[k]
        Random.seed!(UInt(hash((SEED, String(j[1]), j[2], j[3], j[4]))))
        kw = !IPP ? CONTROL_KW : j[1] === :refit ?
             (; spatial_model=:ipp, rho=2 / SMLMBaGoL.area(UniformSpatialPrior(j[5]))) : ipp_kw[j[2]]
        out[k] = run_job(j[5], j[6], μ, shape, kw)
    end
    println(@sprintf("chains done: %d (%.1f min)", length(jobs), (time() - t0) / 60))

    # per separation, per arm: pair key -> (resolved, ncl)
    res = Dict(arm => [Dict{Tuple{Int,Int},Tuple{Bool,Int}}() for _ in SEPS_NM] for arm in (:alone, :grouped, :refit))
    for (k, j) in enumerate(jobs)
        for ((ida, idb), r) in zip(j[6], out[k])
            res[j[1]][j[2]][(ida, idb)] = r
        end
    end
    keys_by_sep = [sort!(collect(keys(res[:alone][si]))) for si in eachindex(SEPS_NM)]
    # grouped pairs share a chain per block, so the bootstrap resamples whole blocks
    blocks_by_sep = [[filter(k -> div(k[1], 100_000) == b, keys_by_sep[si])
                      for b in sort!(unique(div(k[1], 100_000) for k in keys_by_sep[si]))]
                     for si in eachindex(SEPS_NM)]

    # statistic from a (possibly resampled) list of pair keys per separation
    # (all names local to the function: it must not touch the observed fa/fg/rel below)
    function stats(sel)
        pa_, pg_, pr_ = Float64[], Float64[], Float64[]
        for si in eachindex(SEPS_NM)
            ks = sel[si]
            push!(pa_, mean(res[:alone][si][k][1] for k in ks)); push!(pg_, mean(res[:grouped][si][k][1] for k in ks))
            ca = mean(res[:alone][si][k][2] for k in ks); cg = mean(res[:grouped][si][k][2] for k in ks)
            push!(pr_, (cg - ca) / ca)
        end
        return pa_, pg_, pr_, p50(SEPS_NM, pg_) - p50(SEPS_NM, pa_)
    end
    fa, fg, rel, dp = stats(keys_by_sep)
    # block bootstrap (both arms of every pair in a block kept together)
    rng = MersenneTwister(SEED + 1)
    B = 400
    brel = zeros(B, length(SEPS_NM)); bdp = fill(NaN, B); bmax = zeros(B)
    for bi in 1:B
        sel = [reduce(vcat, rand(rng, bl, length(bl))) for bl in blocks_by_sep]
        _, _, r, d = stats(sel)      # bootstrap replicate only; observed fa/fg/rel/dp are untouched
        brel[bi, :] = r; bdp[bi] = d; bmax[bi] = maximum(abs.(r))
    end
    se_rel = vec(std(brel; dims=1)); se_dp = std(filter(isfinite, bdp)); se_max = std(bmax)

    println("\nsep_nm  npairs | resolved alone  grouped | grouped vs alone mean clusters/pair: rel.diff  (block-bootstrap SE)")
    for (si, s) in enumerate(SEPS_NM)
        @printf("%5d  %6d |   %.3f   %.3f   |   %+.1f%%  (%.1f%%)\n", s, length(keys_by_sep[si]),
                fa[si], fg[si], 100 * rel[si], 100 * se_rel[si])
    end
    pa, pg = p50(SEPS_NM, fa), p50(SEPS_NM, fg)
    mx = maximum(abs.(rel))
    @printf("\n50%% resolution point: alone %.1f nm, grouped %.1f nm\n", pa, pg)
    @printf("gated: 50%%-point difference %+.2f nm  (block-bootstrap SE %.2f nm; threshold 1.5 nm)\n", dp, se_dp)
    @printf("gated: max |rel. diff| in mean clusters/pair %.1f%%  (block-bootstrap SE %.1f%%; threshold 5%%)\n", 100 * mx, 100 * se_max)
    if IPP
        fr = [mean(res[:refit][si][k][1] for k in keys_by_sep[si]) for si in eachindex(SEPS_NM)]
        rr = [(mean(res[:grouped][si][k][2] for k in keys_by_sep[si]) - mean(res[:refit][si][k][2] for k in keys_by_sep[si])) /
              mean(res[:refit][si][k][2] for k in keys_by_sep[si]) for si in eachindex(SEPS_NM)]
        @printf("SENSITIVITY (not gated): alone arm with per-pair refitted map: 50%% point %.1f nm (shared-map alone %.1f, grouped %.1f); max |grouped vs refit-alone clusters/pair| %.1f%%\n",
                p50(SEPS_NM, fr), pa, pg, 100 * maximum(abs.(rr)))
    end
    ok = isfinite(pa) && isfinite(pg) && abs(dp) <= 1.5 && mx <= 0.05
    @printf("runtime %.1f min\n", (time() - t0) / 60)
    println(ok ? "VERDICT: PASS (partition-invariance G2)" : "VERDICT: FAIL (partition-invariance G2)")
    return ok
end

exit(main() ? 0 : 1)

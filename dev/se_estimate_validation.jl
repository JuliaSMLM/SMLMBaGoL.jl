# ============================================================================
# estimate_se_adjust — STAGE-1 validation gate (nearest-emitter proxy grouping)
# ============================================================================
#
# Reproduces, in-package, the SMLMClustering-dev validation of the τ finder:
# recover an injected per-blink systematic error on a simulated nanoruler using
# ONLY the simulation output (no ground-truth labels reach the estimator).
#
# Adaptive over-merge descent (proxy grouping). GATE on INVARIANTS, not exact
# values (exact τ̂ depends on seed / BaGoL settings / the proxy):
#   - recovery:           |τ̂ − τ_true| ≤ 0.8 nm   on τ_true = 0 / 4 / 8
#   - cost:               2 ≤ n_bagol ≤ 6  (descent, not a fixed grid)
#   - descent-from-above: path starts at g_start = 12 nm, net-descends, no creep
#   - τ = 0:              lands < 0.8 nm — no false-positive / no stall-high
# Reference values (smlmclustering seed 271, nearest-emitter proxy): 0.3/3.8/8.5.
#
# Run:  julia --threads=auto --project=dev dev/se_estimate_validation.jl
#
# Pipeline mirrors the real one: SMLMSim memoryless blinking -> SMLMFrameConnection
# -> BaGoL grouping. track_id holds the TRUE molecule identity for INJECTION only;
# it is zeroed before frame connection so neither FC nor the estimator can use it.
# ============================================================================
using SMLMBaGoL, SMLMSim, SMLMData, SMLMFrameConnection
using Random, Printf, Statistics

abs_frame(e, nf) = (e.dataset - 1) * nf + e.frame

# ---- simulation: memoryless blinking, true identity in track_id ----
function make_sim(; n_blocks=3, nframes=2500, framerate=50.0, photons=5e4,
                    k_off=10.0, k_on=5e-2, density=1.0, σ_psf=0.13, minphotons=50,
                    pattern_kind=:nanoruler, seed=271)
    Random.seed!(seed)
    camera  = IdealCamera(256, 256, 0.1)
    pattern = pattern_kind === :nanoruler ? Nanoruler2D(spacing=0.02) :
              pattern_kind === :cluster   ? Nmer2D(n=8, d=0.05) :
                                            Nmer2D(n=1, d=0.1)
    fluor   = GenericFluor(photons=photons, k_off=k_off, k_on=k_on)
    params  = StaticSMLMConfig(density=density, σ_psf=σ_psf, minphotons=minphotons,
                               ndatasets=n_blocks, nframes=nframes, framerate=framerate, ndims=2)
    smld, _ = simulate(params; pattern=pattern, molecule=fluor, camera=camera)
    smld
end

# planted per-blink systematic error (constant within a blink run; reported σ stays = CRLB)
function inject_perblink!(smld, τ_nm)
    τ = τ_nm / 1000; nf = smld.n_frames
    bymol = Dict{Int,Vector{Int}}()
    for (i, e) in enumerate(smld.emitters); push!(get!(bymol, e.track_id, Int[]), i); end
    for (_, idxs) in bymol
        sort!(idxs, by=i -> abs_frame(smld.emitters[i], nf))
        runstart = 1
        for k in 2:length(idxs)+1
            isbreak = k > length(idxs) ||
                      abs_frame(smld.emitters[idxs[k]], nf) - abs_frame(smld.emitters[idxs[k-1]], nf) > 1
            if isbreak
                ox = τ * randn(); oy = τ * randn()
                for m in runstart:k-1; smld.emitters[idxs[m]].x += ox; smld.emitters[idxs[m]].y += oy; end
                runstart = k
            end
        end
    end
    smld
end

function gate()
    @printf("%-8s %-9s %-20s %-7s %-6s %s\n", "true τ", "τ̂ (nm)", "95% CI (nm)", "KS", "runs", "descent g→m (nm)")
    println("-"^92)
    pass = true
    for τ_true in (0.0, 4.0, 8.0)
        smld = make_sim(; pattern_kind=:nanoruler, seed=271)
        τ_true > 0 && inject_perblink!(smld, τ_true)
        for (i, e) in enumerate(smld.emitters); e.id = i; e.track_id = 0; end
        combined, _ = frameconnect(smld; max_frame_gap=5, max_sigma_dist=5.0,
                                   n_density_neighbors=2, track_length=nothing)
        r = estimate_se_adjust(combined; n_iterations=1500, burn_in=400, n_boot=150)
        τ̂  = 1000 * r.tau_hat_um
        gs = [1000 * p[1] for p in r.path_um]                 # grouping-τ trajectory (nm)
        ms = [1000 * p[2] for p in r.path_um]                 # M-step trajectory (nm)
        pathstr = join(("$(round(g,digits=1))→$(round(m,digits=1))" for (g, m) in zip(gs, ms)), "  ")
        @printf("%-8.0f %-9.2f [%-6.2f,%-6.2f]    %-7.3f %-6d %s\n",
                τ_true, τ̂, 1000*r.ci_lo_um, 1000*r.ci_hi_um, r.ks_at_hat, r.n_bagol, pathstr)
        # INVARIANTS (not exact values — see header)
        recov   = abs(τ̂ - τ_true) ≤ 0.8
        runs    = 2 ≤ r.n_bagol ≤ 6
        above   = !isempty(gs) && isapprox(gs[1], 12.0; atol=1e-6) && τ̂ ≤ gs[1] + 0.1
        nocreep = length(gs) < 2 || maximum(diff(gs)) ≤ 0.5    # no upward g step > 0.5 nm
        ok = recov && runs && above && nocreep
        ok || @printf("    ^^ INVARIANT FAIL: recovery=%s runs=%s descent-from-above=%s no-creep=%s\n",
                      recov, runs, above, nocreep)
        pass &= ok
    end
    println("-"^92)
    println(pass ? "GATE PASS — adaptive descent recovers 0/4/8 within 0.8 nm; 2–6 runs; descent-from-above; no creep" :
                   "GATE FAIL — an invariant violated; investigate before commit / stage 2")
    pass
end

gate()

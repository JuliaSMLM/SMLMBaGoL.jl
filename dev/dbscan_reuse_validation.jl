# Acceptance test for the DBSCAN neighbor-graph reuse (finder build-once/threshold-many).
# (1) CORRECTNESS: fresh precision_dbscan == cached (prebuilt graph) partition, over
#     nsigma × τ grid — the τ̂-identity proof (identical partition at every E-step ⇒ identical τ̂).
# (2) SPEED: 8 finder-like partition passes, fresh (rebuild KDTree each) vs cached (build once).
# (3) END-TO-END: estimate_se_adjust runs with reuse active, sane τ̂.
using SMLMBaGoL, SMLMData, Random, Printf, Statistics
const S = SMLMBaGoL

# canonical relabel → 1,2,.. by first appearance (partition-equivalence key)
function canonical(labels)
    m = Dict{Int,Int}(); nxt = 0; out = similar(labels)
    for (i, l) in enumerate(labels)
        if l == 0
            out[i] = 0
        else
            if !haskey(m, l); nxt += 1; m[l] = nxt; end
            out[i] = m[l]
        end
    end
    out
end
same_partition(a, b) = canonical(a) == canonical(b)

function run_validation()
    md = Dict{String, Any}()

    # --- synthetic: grid of well-separated 4-mers, realistic per-loc σ (photon CRLB) ---
    sim = S.simulate_nmer_grid(n_per_cluster=4, cluster_diameter=0.02,
                               grid_nx=8, grid_ny=8, grid_spacing=0.3,
                               mean_count=20.0, psf_sigma=0.13, mean_photons=500.0)
    smld = sim.smld
    n = length(smld.emitters)
    σs = [S.mean_sigma(e) for e in smld.emitters]
    @printf("dataset: %d locs, %d emitters; σ_eff range %.1f–%.1f nm\n",
            n, sim.n_emitters, 1000*minimum(σs), 1000*maximum(σs))

    # === (1) CORRECTNESS: fresh vs cached over nsigma × τ ===
    println("\n=== (1) fresh-vs-cached partition-equivalence (τ̂-identity proof) ===")
    grid = 0.0:0.001:0.014
    overall = true
    for nsigma in (2.0, 3.0, 5.0)
        g = S._build_finder_neighbor_graph(smld, maximum(grid), nsigma)
        part_ok = true; id_fails = 0
        for τ in grid
            infl, _, _ = S._maybe_apply_se_adjust(smld.emitters, md, (τ, τ), true)
            fresh  = S.precision_dbscan(infl, nsigma, 0)
            cached = S.precision_dbscan(infl, nsigma, 0; neighbor_graph=g)
            if !same_partition(fresh, cached)
                part_ok = false
                @printf("   MISMATCH nsigma=%.1f τ=%.4f (%d vs %d clusters)\n",
                        nsigma, τ, length(unique(fresh)), length(unique(cached)))
            end
            fresh == cached || (id_fails += 1)   # raw-id identity (ascending-first-appearance)
        end
        overall &= part_ok
        @printf("  nsigma=%.1f: partition-equiv %s over %d τ; raw-label-id identical %d/%d τ\n",
                nsigma, part_ok ? "PASS" : "FAIL", length(grid), length(grid) - id_fails, length(grid))
    end
    println(overall ? "  => CORRECTNESS PASS: partitions bit-identical fresh vs cached at every nsigma×τ" :
                      "  => CORRECTNESS FAIL")

    # === (2) SPEED: finder-like repeated partitioning ===
    println("\n=== (2) speed: 8 partition passes, fresh (rebuild) vs cached (build-once) ===")
    simL = S.simulate_nmer_grid(n_per_cluster=4, cluster_diameter=0.02,
                                grid_nx=20, grid_ny=20, grid_spacing=0.3,
                                mean_count=25.0, psf_sigma=0.13, mean_photons=500.0)
    emsL = simL.smld.emitters; nL = length(emsL)
    τs = collect(0.008:-0.001:0.001)                       # descent-like sequence (8 steps)
    S.precision_dbscan(emsL, 3.0, 0)                       # warmup
    gwarm = S._build_finder_neighbor_graph(simL.smld, 0.014, 3.0)
    S.precision_dbscan(emsL, 3.0, 0; neighbor_graph=gwarm) # warmup cached path
    t_fresh = @elapsed for τ in τs
        infl, _, _ = S._maybe_apply_se_adjust(emsL, md, (τ, τ), true)
        S.precision_dbscan(infl, 3.0, 0)
    end
    t_cached = @elapsed begin
        g2 = S._build_finder_neighbor_graph(simL.smld, 0.014, 3.0)   # build ONCE
        for τ in τs
            infl, _, _ = S._maybe_apply_se_adjust(emsL, md, (τ, τ), true)
            S.precision_dbscan(infl, 3.0, 0; neighbor_graph=g2)
        end
    end
    @printf("  %d locs, %d passes (threads=%d): fresh=%.3fs  cached(build+reuse)=%.3fs  speedup=%.1f×\n",
            nL, length(τs), Threads.nthreads(), t_fresh, t_cached, t_fresh / t_cached)

    # === (3) END-TO-END: finder with reuse active ===
    println("\n=== (3) end-to-end estimate_se_adjust (reuse active) ===")
    t = @elapsed fres = S.estimate_se_adjust(smld; n_iterations=800, burn_in=400,
                                             max_steps=6, n_boot=20)
    @printf("  τ̂ = %.2f nm  (%d BaGoL E-steps, %.1fs)  descent m-path(nm)=%s\n",
            1000*fres.tau_hat_um, fres.n_bagol, t,
            string([round(1000*m, digits=1) for (_, m) in fres.path_um]))
    println("DONE")
end

run_validation()

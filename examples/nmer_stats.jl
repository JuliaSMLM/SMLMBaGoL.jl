# N-mer Statistical Analysis
# ==========================
# Runs N trials of single N-mer BaGoL (no hierarchical) and collects
# partition diagnostics: VI, EPL, K recovery, count-model MAP K.
#
# Run with: julia --threads=auto --project=examples examples/nmer_stats.jl

using Pkg
Pkg.activate(@__DIR__)

using SMLMBaGoL
using SMLMBaGoL: accumulator_result
using SMLMData
using Random
using Statistics
using Distributions
using CairoMakie
using Printf

include(joinpath(@__DIR__, "viz_metrics.jl"))

# =============================================================================
# PARAMETERS
# =============================================================================

const N_TRIALS = 100

# N-mer geometry
const N_EMITTERS = 6
const CLUSTER_DIAMETER = 0.025    # μm (25 nm)

# Photophysics
const PSF_SIGMA = 0.130           # μm (130 nm)
const PHOTON_MEAN = 500.0
const PHOTON_MIN = 100.0
const BLINK_MEAN = 10.0

# Precision filter
const PRECISION_MAX = 0.010       # μm (10 nm)

# BaGoL parameters
const N_ITERATIONS = 20000
const BURN_IN = 4000

# Camera
const CAMERA_PIXELS = 256
const PIXEL_SIZE = 0.100          # μm per pixel

# Count model (fixed, no hierarchical)
const TRUE_SHAPE = 1000.0  # Poisson blinking → large shape

# Output
const OUTPUT_DIR = joinpath(@__DIR__, "output", "nmer_stats")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

function generate_nmer_positions(n_emitters, diameter)
    r = diameter / 2
    fov_size = CAMERA_PIXELS * PIXEL_SIZE
    cx, cy = fov_size / 2, fov_size / 2
    return [(cx + r * cos(2π * (i-1) / n_emitters),
             cy + r * sin(2π * (i-1) / n_emitters))
            for i in 1:n_emitters]
end

function generate_localizations(positions)
    locs = SMLMData.Emitter2DFit[]
    blink_dist = Poisson(BLINK_MEAN)
    photon_dist = Exponential(PHOTON_MEAN)
    loc_id = 1
    for (emitter_idx, (ex, ey)) in enumerate(positions)
        n_target = max(1, rand(blink_dist))
        # Draw photons until we have n_target survivors above PHOTON_MIN.
        # This ensures exactly Poisson(BLINK_MEAN) locs per emitter after
        # filtering, so μ = BLINK_MEAN with no calibration uncertainty.
        n_accepted = 0
        while n_accepted < n_target
            N = rand(photon_dist)
            N < PHOTON_MIN && continue
            n_accepted += 1
            σ = PSF_SIGMA / sqrt(N)
            x = ex + σ * randn()
            y = ey + σ * randn()
            push!(locs, SMLMData.Emitter2DFit(
                x, y, N, 10.0, σ, σ, 0.0, sqrt(N), 1.0,
                loc_id, 1, emitter_idx, loc_id
            ))
            loc_id += 1
        end
    end
    return locs
end

"""
Count-model MAP K: argmax_K P(N|K,μ,shape).
Pure NegBin likelihood — no prior on K.
This is the best K estimate from counting alone (Q-PAINT limit).
"""
function count_model_map_k(N::Int, μ::Float64, shape::Float64; K_max::Int=20)
    best_k = 1
    best_lp = -Inf
    for K in 1:K_max
        lp = SMLMBaGoL.log_prior_total_count(N, K, μ, shape)
        if lp > best_lp
            best_lp = lp
            best_k = K
        end
    end
    return best_k
end

function run_single_trial(true_positions)
    # Generate localizations (rejection sampling ensures exactly Poisson(μ)
    # survivors per emitter — no post-hoc filter needed)
    locs = generate_localizations(true_positions)
    N = length(locs)

    if N < 2
        return nothing  # Skip degenerate trials
    end

    # μ is exactly BLINK_MEAN by construction (no filter bias)
    μ = BLINK_MEAN

    # Count-model MAP K with known μ — unbiased Q-PAINT baseline
    K_count = count_model_map_k(N, μ, TRUE_SHAPE)

    # Run chain
    ps_acc = PartitionSamples(thin=5)
    psm_acc = PSMAccumulator()
    count_hist = EmitterCountHist()

    result = run_collapsed_chain(
        locs;
        n_iterations = N_ITERATIONS,
        burn_in = BURN_IN,
        μ_prior_shape = μ,
        μ_prior_scale = 1.0,
        shape = TRUE_SHAPE,
        learn_shape = false,
        hierarchical_interval = N_ITERATIONS + 1,
        accumulators = AbstractAccumulator[count_hist, ps_acc, psm_acc],
        verbose = false
    )

    # Extract Dahl partition
    samples = accumulator_result(ps_acc)
    psm = accumulator_result(psm_acc).psm
    _, _, _, dahl_z = estimate_dahl(samples, locs, psm)
    K_dahl = length(unique(dahl_z))

    # Oracle partition from track_id
    oracle_z = Int16[loc.track_id for loc in locs]
    K_oracle = length(unique(oracle_z))

    # Posterior K histogram
    posterior_k = result.accumulators[1]
    K_mode = argmax(posterior_k) - 1  # 0-indexed

    # Partition diagnostics (VI + EPL)
    pd = partition_diagnostics(dahl_z, oracle_z, samples)

    # Spatial metrics via Dahl emitters
    dahl_emitters = SMLMBaGoL._emitters_from_assignments(dahl_z, locs)
    m = compute_all_metrics(dahl_emitters, true_positions; threshold=0.020)

    return (
        N = N, μ = μ,
        K_true = N_EMITTERS,
        K_dahl = K_dahl,
        K_count = K_count,
        K_mode = K_mode,
        K_oracle = K_oracle,
        vi_total = pd.vi_total,
        overseg = pd.overseg,
        underseg = pd.underseg,
        epl_dahl = pd.epl_dahl,
        epl_oracle = pd.epl_oracle,
        epl_best = pd.epl_best,
        regret_dahl = pd.regret_dahl,
        jaccard = m.jaccard,
        rmse = m.rmse,
        f1 = m.f1,
    )
end

# =============================================================================
# RUN TRIALS
# =============================================================================

mkpath(OUTPUT_DIR)

println("="^60)
println("N-mer Statistical Analysis")
println("="^60)
println("  N_EMITTERS = $N_EMITTERS")
println("  CLUSTER_DIAMETER = $(CLUSTER_DIAMETER * 1000) nm")
println("  N_TRIALS = $N_TRIALS")
println("  N_ITERATIONS = $N_ITERATIONS")
println()

true_positions = generate_nmer_positions(N_EMITTERS, CLUSTER_DIAMETER)

results = []
t_start = time()
for trial in 1:N_TRIALS
    r = run_single_trial(true_positions)
    if r !== nothing
        push!(results, r)
    end
    if trial % 10 == 0
        elapsed = time() - t_start
        rate = trial / elapsed
        eta = (N_TRIALS - trial) / rate
        @printf("  Trial %3d/%d  (%.1f trials/min, ETA %.0fs)\n",
                trial, N_TRIALS, rate * 60, eta)
    end
end

elapsed = time() - t_start
println("\nCompleted $(length(results))/$N_TRIALS trials in $(round(elapsed, digits=0))s")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

println("\n" * "="^60)
println("Results Summary")
println("="^60)

# K recovery rates
K_dahls = [r.K_dahl for r in results]
K_counts = [r.K_count for r in results]
K_modes = [r.K_mode for r in results]

pct_dahl = 100 * count(==(N_EMITTERS), K_dahls) / length(results)
pct_count = 100 * count(==(N_EMITTERS), K_counts) / length(results)
pct_mode = 100 * count(==(N_EMITTERS), K_modes) / length(results)

println("\nK Recovery (% correct = $N_EMITTERS):")
println("  Count-only MAP:    $(round(pct_count, digits=1))%  (mean K = $(round(mean(K_counts), digits=1)))")
println("  Posterior mode:    $(round(pct_mode, digits=1))%  (mean K = $(round(mean(K_modes), digits=1)))")
println("  Dahl consensus:   $(round(pct_dahl, digits=1))%  (mean K = $(round(mean(K_dahls), digits=1)))")

# VI statistics
vis = [r.vi_total for r in results]
oversegs = [r.overseg for r in results]
undersegs = [r.underseg for r in results]
println("\nVariation of Information (Dahl vs Oracle):")
println("  VI:      $(round(mean(vis), digits=3)) ± $(round(std(vis), digits=3))")
println("  Over:    $(round(mean(oversegs), digits=3)) ± $(round(std(oversegs), digits=3))")
println("  Under:   $(round(mean(undersegs), digits=3)) ± $(round(std(undersegs), digits=3))")

# EPL statistics
epl_dahls = [r.epl_dahl for r in results]
epl_oracles = [r.epl_oracle for r in results]
epl_bests = [r.epl_best for r in results]
regrets = [r.regret_dahl for r in results]
pct_dahl_better = 100 * count(r -> r.epl_dahl <= r.epl_oracle, results) / length(results)

println("\nExpected Posterior Loss (VI):")
println("  EPL(Dahl):    $(round(mean(epl_dahls), digits=3)) ± $(round(std(epl_dahls), digits=3))")
println("  EPL(Oracle):  $(round(mean(epl_oracles), digits=3)) ± $(round(std(epl_oracles), digits=3))")
println("  EPL(Best):    $(round(mean(epl_bests), digits=3)) ± $(round(std(epl_bests), digits=3))")
println("  Dahl regret:  $(round(mean(regrets), digits=3)) ± $(round(std(regrets), digits=3))")
println("  Dahl ≤ Oracle: $(round(pct_dahl_better, digits=1))%")

# Spatial metrics
jaccards = [r.jaccard for r in results]
rmses = [r.rmse * 1000 for r in results]  # nm
f1s = [r.f1 for r in results]

println("\nSpatial Metrics:")
println("  Jaccard:  $(round(mean(jaccards), digits=3)) ± $(round(std(jaccards), digits=3))")
println("  RMSE:     $(round(mean(rmses), digits=1)) ± $(round(std(rmses), digits=1)) nm")
println("  F1:       $(round(mean(f1s), digits=3)) ± $(round(std(f1s), digits=3))")

# Localization stats
Ns = [r.N for r in results]
println("\nLocalizations per trial: $(round(mean(Ns), digits=1)) ± $(round(std(Ns), digits=1))")

# =============================================================================
# VISUALIZATION
# =============================================================================

println("\nGenerating figures...")

fig = Figure(size=(1400, 900))

# Row 1: K distributions
ax1 = Axis(fig[1, 1], title="K Recovery", xlabel="K", ylabel="Count")
k_range = min(minimum(K_dahls), minimum(K_counts)):max(maximum(K_dahls), maximum(K_counts))
counts_dahl = [count(==(k), K_dahls) for k in k_range]
counts_count = [count(==(k), K_counts) for k in k_range]
barplot!(ax1, collect(k_range) .- 0.15, counts_dahl, width=0.3, color=:red, label="Dahl")
barplot!(ax1, collect(k_range) .+ 0.15, counts_count, width=0.3, color=:steelblue, label="Count-only")
vlines!(ax1, [N_EMITTERS], color=:black, linewidth=2, linestyle=:dash, label="True K=$N_EMITTERS")
axislegend(ax1, position=:rt, framevisible=false)

# Row 1: VI histogram
ax2 = Axis(fig[1, 2], title="VI(Dahl, Oracle)", xlabel="VI (nats)", ylabel="Count")
hist!(ax2, vis, bins=20, color=:steelblue)
vlines!(ax2, [mean(vis)], color=:red, linewidth=2, label="Mean=$(round(mean(vis), digits=2))")
axislegend(ax2, position=:rt, framevisible=false)

# Row 1: VI decomposition
ax3 = Axis(fig[1, 3], title="VI Decomposition", xlabel="nats", ylabel="Count")
hist!(ax3, oversegs, bins=15, color=(:orange, 0.7), label="Over-seg")
hist!(ax3, undersegs, bins=15, color=(:purple, 0.7), label="Under-seg")
axislegend(ax3, position=:rt, framevisible=false)

# Row 2: EPL comparison
ax4 = Axis(fig[2, 1], title="EPL: Dahl vs Oracle", xlabel="EPL(Dahl)", ylabel="EPL(Oracle)")
scatter!(ax4, epl_dahls, epl_oracles, color=:steelblue, markersize=6, alpha=0.6)
lo = min(minimum(epl_dahls), minimum(epl_oracles))
hi = max(maximum(epl_dahls), maximum(epl_oracles))
lines!(ax4, [lo, hi], [lo, hi], color=:black, linestyle=:dash, linewidth=1, label="y=x")
axislegend(ax4, position=:rb, framevisible=false)

# Row 2: Regret histogram
ax5 = Axis(fig[2, 2], title="Dahl Regret", xlabel="EPL(Dahl) - EPL(Best)", ylabel="Count")
hist!(ax5, regrets, bins=20, color=:steelblue)
vlines!(ax5, [0], color=:red, linewidth=1, linestyle=:dash)

# Row 2: Jaccard and RMSE
ax6 = Axis(fig[2, 3], title="Spatial Quality", xlabel="Jaccard", ylabel="RMSE (nm)")
scatter!(ax6, jaccards, rmses, color=:steelblue, markersize=6, alpha=0.6)

# Row 3: K_dahl vs K_count scatter (jittered)
ax7 = Axis(fig[3, 1], title="K: BaGoL vs Count-only",
    xlabel="K (Count-only)", ylabel="K (Dahl)")
jitter = () -> randn() * 0.08
scatter!(ax7, [r.K_count + jitter() for r in results],
         [r.K_dahl + jitter() for r in results],
         color=:steelblue, markersize=5, alpha=0.5)
lines!(ax7, [0, 12], [0, 12], color=:black, linestyle=:dash, linewidth=1)
hlines!(ax7, [N_EMITTERS], color=:red, linestyle=:dot, linewidth=1)
vlines!(ax7, [N_EMITTERS], color=:red, linestyle=:dot, linewidth=1)

# Row 3: N (locs) vs K_dahl
ax8 = Axis(fig[3, 2], title="K vs N locs", xlabel="N (filtered locs)", ylabel="K (Dahl)")
scatter!(ax8, Ns, K_dahls, color=:steelblue, markersize=5, alpha=0.5)
hlines!(ax8, [N_EMITTERS], color=:red, linestyle=:dash, linewidth=1, label="True K")
axislegend(ax8, position=:rt, framevisible=false)

# Row 3: EPL(Dahl) < EPL(Oracle) fraction as text summary
ax9 = Axis(fig[3, 3])
hidedecorations!(ax9)
hidespines!(ax9)
summary_text = """N = $N_TRIALS trials
$(N_EMITTERS)-mer, $(Int(CLUSTER_DIAMETER*1000))nm diameter

K recovery:
  Count-only: $(round(pct_count, digits=0))%
  BaGoL Dahl: $(round(pct_dahl, digits=0))%

EPL(Dahl) ≤ EPL(Oracle):
  $(round(pct_dahl_better, digits=0))% of trials

Mean VI: $(round(mean(vis), digits=2)) nats
Mean RMSE: $(round(mean(rmses), digits=1)) nm
Mean Jaccard: $(round(mean(jaccards), digits=2))"""
text!(ax9, 0.05, 0.95, text=summary_text, align=(:left, :top), fontsize=13)

save(joinpath(OUTPUT_DIR, "nmer_stats.png"), fig)
println("Saved: $(joinpath(OUTPUT_DIR, "nmer_stats.png"))")

println("\n" * "="^60)
println("Done.")

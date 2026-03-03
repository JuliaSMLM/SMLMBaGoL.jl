# Collapsed Gibbs vs RJMCMC Comparison
# =====================================
# Head-to-head comparison on identical data: speed and quality metrics.
#
# Run with: julia --threads=auto --project=examples examples/sampler_comparison.jl

using Pkg
Pkg.activate(@__DIR__)

using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions
using CairoMakie

include(joinpath(@__DIR__, "viz_metrics.jl"))

# =============================================================================
# PARAMETERS
# =============================================================================

const SEED = 42

# Grid: 4×4 identical 8-mers
const GRID_SIZE = 4
const N_EMITTERS = 8
const GRID_SPACING = 1.0
const CLUSTER_DIAMETER = 0.050
const FOV_SIZE = (GRID_SIZE + 1) * GRID_SPACING
const PIXEL_SIZE = 0.100
const CAMERA_PIXELS = round(Int, FOV_SIZE / PIXEL_SIZE)

# Photophysics
const PSF_SIGMA = 0.130
const PHOTON_MEAN = 500.0
const PHOTON_MIN = 100.0
const BLINK_MEAN = 10.0
const PRECISION_MAX = 0.010

# MCMC — same for both
const N_ITERATIONS = 15000
const BURN_IN = 3000
const NSIGMA = 3.0
const MAX_PARTITION_SIZE = 1000

const OUTPUT_DIR = joinpath(@__DIR__, "output", "comparison")

# =============================================================================
# GENERATE DATA (once, shared by both samplers)
# =============================================================================

function generate_grid_data(; seed=SEED)
    Random.seed!(seed)

    true_positions = Tuple{Float64, Float64}[]
    grid_extent = (GRID_SIZE - 1) * GRID_SPACING
    offset = (FOV_SIZE - grid_extent) / 2
    cluster_radius = CLUSTER_DIAMETER / 2

    for i in 1:GRID_SIZE, j in 1:GRID_SIZE
        cx = offset + (i - 1) * GRID_SPACING
        cy = offset + (j - 1) * GRID_SPACING
        for k in 1:N_EMITTERS
            θ = 2π * (k - 1) / N_EMITTERS
            push!(true_positions, (cx + cluster_radius * cos(θ),
                                    cy + cluster_radius * sin(θ)))
        end
    end

    blink_dist = Poisson(BLINK_MEAN)
    photon_dist = Exponential(PHOTON_MEAN)
    locs = SMLMData.Emitter2DFit[]
    loc_id = 1

    for (ex, ey) in true_positions
        for _ in 1:max(1, rand(blink_dist))
            N = rand(photon_dist)
            N < PHOTON_MIN && continue
            σ = PSF_SIGMA / sqrt(N)
            push!(locs, SMLMData.Emitter2DFit(
                ex + σ * randn(), ey + σ * randn(), N, 10.0,
                σ, σ, 0.0, sqrt(N), 1.0, loc_id, 1, 0, loc_id))
            loc_id += 1
        end
    end

    locs = filter(l -> max(l.σ_x, l.σ_y) <= PRECISION_MAX, locs)
    camera = SMLMData.IdealCamera(CAMERA_PIXELS, CAMERA_PIXELS, PIXEL_SIZE)
    smld = SMLMData.BasicSMLD(locs, camera, 1, 1)

    return smld, true_positions
end

# =============================================================================
# RUN COMPARISON
# =============================================================================

mkpath(OUTPUT_DIR)

smld, true_positions = generate_grid_data()
println("="^70)
println("Sampler Comparison: Collapsed Gibbs vs RJMCMC")
println("="^70)
println("Data: $(GRID_SIZE)×$(GRID_SIZE) grid of $(N_EMITTERS)-mers = $(length(true_positions)) true emitters")
println("Localizations: $(length(smld.emitters))")
println("Iterations: $N_ITERATIONS (burn-in: $BURN_IN)")
println()

# --- Collapsed Gibbs ---
println("-"^70)
println("Running COLLAPSED GIBBS...")
t_collapsed = @elapsed begin
    result_c, diag_c = run_bagol(smld;
        sampler = :collapsed,
        nsigma = NSIGMA,
        max_partition_size = MAX_PARTITION_SIZE,
        n_iterations = N_ITERATIONS,
        burn_in = BURN_IN,
        verbose = false)
end
metrics_c = compute_all_metrics(result_c.emitters, true_positions; threshold=0.020)
println("  Time: $(round(t_collapsed, digits=2))s")
println("  Emitters: $(length(result_c.emitters))")
println("  Jaccard: $(round(metrics_c.jaccard, digits=3))")
println("  RMSE: $(round(metrics_c.rmse * 1000, digits=1)) nm")

# --- RJMCMC ---
println("\n" * "-"^70)
println("Running RJMCMC...")
t_rjmcmc = @elapsed begin
    result_r, diag_r = run_bagol(smld;
        sampler = :rjmcmc,
        nsigma = NSIGMA,
        max_partition_size = MAX_PARTITION_SIZE,
        n_iterations = N_ITERATIONS,
        burn_in = BURN_IN,
        verbose = false)
end
metrics_r = compute_all_metrics(result_r.emitters, true_positions; threshold=0.020)
println("  Time: $(round(t_rjmcmc, digits=2))s")
println("  Emitters: $(length(result_r.emitters))")
println("  Jaccard: $(round(metrics_r.jaccard, digits=3))")
println("  RMSE: $(round(metrics_r.rmse * 1000, digits=1)) nm")

# --- Multiple runs for timing statistics ---
println("\n" * "-"^70)
println("Timing statistics (3 additional runs each)...")

t_collapsed_runs = Float64[t_collapsed]
t_rjmcmc_runs = Float64[t_rjmcmc]

for run in 1:3
    smld_r, _ = generate_grid_data(seed=SEED + run)

    tc = @elapsed run_bagol(smld_r; sampler=:collapsed,
        nsigma=NSIGMA, max_partition_size=MAX_PARTITION_SIZE,
        n_iterations=N_ITERATIONS, burn_in=BURN_IN, verbose=false)
    push!(t_collapsed_runs, tc)

    tr = @elapsed run_bagol(smld_r; sampler=:rjmcmc,
        nsigma=NSIGMA, max_partition_size=MAX_PARTITION_SIZE,
        n_iterations=N_ITERATIONS, burn_in=BURN_IN, verbose=false)
    push!(t_rjmcmc_runs, tr)

    println("  Run $(run+1): collapsed=$(round(tc, digits=2))s, rjmcmc=$(round(tr, digits=2))s")
end

# --- Quality comparison across seeds ---
println("\n" * "-"^70)
println("Quality comparison across 5 random seeds...")

jaccards_c = Float64[]
jaccards_r = Float64[]
rmses_c = Float64[]
rmses_r = Float64[]
n_est_c = Int[]
n_est_r = Int[]

for seed in [42, 123, 456, 789, 1024]
    smld_q, tp_q = generate_grid_data(seed=seed)

    res_c, _ = run_bagol(smld_q; sampler=:collapsed,
        nsigma=NSIGMA, max_partition_size=MAX_PARTITION_SIZE,
        n_iterations=N_ITERATIONS, burn_in=BURN_IN, verbose=false)
    m_c = compute_all_metrics(res_c.emitters, tp_q; threshold=0.020)

    res_r, _ = run_bagol(smld_q; sampler=:rjmcmc,
        nsigma=NSIGMA, max_partition_size=MAX_PARTITION_SIZE,
        n_iterations=N_ITERATIONS, burn_in=BURN_IN, verbose=false)
    m_r = compute_all_metrics(res_r.emitters, tp_q; threshold=0.020)

    push!(jaccards_c, m_c.jaccard)
    push!(jaccards_r, m_r.jaccard)
    push!(rmses_c, m_c.rmse * 1000)
    push!(rmses_r, m_r.rmse * 1000)
    push!(n_est_c, length(res_c.emitters))
    push!(n_est_r, length(res_r.emitters))

    println("  Seed $seed: collapsed J=$(round(m_c.jaccard, digits=3)) RMSE=$(round(m_c.rmse*1000, digits=1))nm N=$(length(res_c.emitters))  |  rjmcmc J=$(round(m_r.jaccard, digits=3)) RMSE=$(round(m_r.rmse*1000, digits=1))nm N=$(length(res_r.emitters))")
end

# =============================================================================
# SUMMARY TABLE
# =============================================================================

speedup = mean(t_rjmcmc_runs) / mean(t_collapsed_runs)

println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println()
println("                    Collapsed Gibbs       RJMCMC")
println("                    ───────────────       ──────")
println("  Time (mean):      $(lpad(round(mean(t_collapsed_runs), digits=2), 8))s          $(lpad(round(mean(t_rjmcmc_runs), digits=2), 8))s")
println("  Time (std):       $(lpad(round(std(t_collapsed_runs), digits=2), 8))s          $(lpad(round(std(t_rjmcmc_runs), digits=2), 8))s")
println("  Speedup:          $(round(speedup, digits=1))×")
println()
println("  Jaccard (mean):   $(lpad(round(mean(jaccards_c), digits=3), 8))           $(lpad(round(mean(jaccards_r), digits=3), 8))")
println("  Jaccard (std):    $(lpad(round(std(jaccards_c), digits=3), 8))           $(lpad(round(std(jaccards_r), digits=3), 8))")
println("  RMSE nm (mean):   $(lpad(round(mean(rmses_c), digits=1), 8))           $(lpad(round(mean(rmses_r), digits=1), 8))")
println("  RMSE nm (std):    $(lpad(round(std(rmses_c), digits=1), 8))           $(lpad(round(std(rmses_r), digits=1), 8))")
println("  N_est (mean):     $(lpad(round(mean(n_est_c), digits=1), 8))           $(lpad(round(mean(n_est_r), digits=1), 8))")
println("  N_true:           $(lpad(128, 8))           $(lpad(128, 8))")
println()
println("  Final μ:          $(lpad(round(diag_c.final_μ, digits=2), 8))           $(lpad(round(diag_r.final_μ, digits=2), 8))")
println("  Final shape:      $(lpad(round(diag_c.final_shape, digits=2), 8))           $(lpad(round(diag_r.final_shape, digits=2), 8))")
println()

# =============================================================================
# COMPARISON PLOT
# =============================================================================

fig = Figure(size=(1400, 500))

# 1. Timing comparison
ax1 = Axis(fig[1, 1], title="Runtime (4 runs)",
           ylabel="Time (s)", xticks=(1:2, ["Collapsed", "RJMCMC"]))
barplot!(ax1, [1, 2], [mean(t_collapsed_runs), mean(t_rjmcmc_runs)],
         color=[:steelblue, :coral])
errorbars!(ax1, [1, 2], [mean(t_collapsed_runs), mean(t_rjmcmc_runs)],
           [std(t_collapsed_runs), std(t_rjmcmc_runs)], color=:black, whiskerwidth=10)
text!(ax1, 1.5, maximum([mean(t_collapsed_runs), mean(t_rjmcmc_runs)]) * 0.9,
      text="$(round(speedup, digits=1))× speedup",
      align=(:center, :top), fontsize=14, font=:bold)

# 2. Jaccard comparison
ax2 = Axis(fig[1, 2], title="Jaccard Index (5 seeds)",
           ylabel="Jaccard", xticks=(1:2, ["Collapsed", "RJMCMC"]))
# Individual points
scatter!(ax2, fill(0.85, length(jaccards_c)), jaccards_c,
         color=(:steelblue, 0.6), markersize=10)
scatter!(ax2, fill(2.15, length(jaccards_r)), jaccards_r,
         color=(:coral, 0.6), markersize=10)
# Means
barplot!(ax2, [1, 2], [mean(jaccards_c), mean(jaccards_r)],
         color=[(:steelblue, 0.3), (:coral, 0.3)])
errorbars!(ax2, [1, 2], [mean(jaccards_c), mean(jaccards_r)],
           [std(jaccards_c), std(jaccards_r)], color=:black, whiskerwidth=10)

# 3. RMSE comparison
ax3 = Axis(fig[1, 3], title="RMSE (5 seeds)",
           ylabel="RMSE (nm)", xticks=(1:2, ["Collapsed", "RJMCMC"]))
scatter!(ax3, fill(0.85, length(rmses_c)), rmses_c,
         color=(:steelblue, 0.6), markersize=10)
scatter!(ax3, fill(2.15, length(rmses_r)), rmses_r,
         color=(:coral, 0.6), markersize=10)
barplot!(ax3, [1, 2], [mean(rmses_c), mean(rmses_r)],
         color=[(:steelblue, 0.3), (:coral, 0.3)])
errorbars!(ax3, [1, 2], [mean(rmses_c), mean(rmses_r)],
           [std(rmses_c), std(rmses_r)], color=:black, whiskerwidth=10)

# 4. N_estimated comparison
ax4 = Axis(fig[1, 4], title="Emitter Count (5 seeds)",
           ylabel="N_estimated", xticks=(1:2, ["Collapsed", "RJMCMC"]))
scatter!(ax4, fill(0.85, length(n_est_c)), n_est_c,
         color=(:steelblue, 0.6), markersize=10)
scatter!(ax4, fill(2.15, length(n_est_r)), n_est_r,
         color=(:coral, 0.6), markersize=10)
barplot!(ax4, [1, 2], [mean(n_est_c), mean(n_est_r)],
         color=[(:steelblue, 0.3), (:coral, 0.3)])
hlines!(ax4, [128], color=:black, linestyle=:dash, linewidth=2, label="True N=128")
axislegend(ax4, position=:rt)

save(joinpath(OUTPUT_DIR, "sampler_comparison.png"), fig)
println("Plot saved: $(joinpath(OUTPUT_DIR, "sampler_comparison.png"))")

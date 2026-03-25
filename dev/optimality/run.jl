# Unified Optimality Workflow — Entry Point
#
# Run: julia --threads=auto --project=dev dev/optimality/run.jl
# With secondary sweeps: RUN_SECONDARY=true julia --threads=auto --project=dev dev/optimality/run.jl

using SMLMBaGoL
using SMLMBaGoL: accumulator_result, _emitters_from_assignments,
    log_prior_total_count, log_prior_k
using SMLMData
using CairoMakie
using Statistics
using Random
using Distributions
using Hungarian
using Printf

include(joinpath(dirname(dirname(@__DIR__)), "examples", "viz_metrics.jl"))

const OPTIM_DIR = @__DIR__
include(joinpath(OPTIM_DIR, "config.jl"))
include(joinpath(OPTIM_DIR, "helpers.jl"))
include(joinpath(OPTIM_DIR, "scorecard.jl"))
include(joinpath(OPTIM_DIR, "dashboard.jl"))
include(joinpath(OPTIM_DIR, "secondary.jl"))

# =============================================================================
# Calibration
# =============================================================================

rm(OUTPUT_DIR; force=true, recursive=true)
mkpath(OUTPUT_DIR)

ref_locs, _ = simulate_locs([(FOV_CENTER)]; μ=100.0, seed=999)
ref_σ = median([(l.σ_x + l.σ_y) / 2 for l in ref_locs])
println("Reference σ_loc: $(round(ref_σ * 1000, digits=1)) nm")

calib_locs_all = SMLMData.Emitter2DFit[]
for s in 1:200
    ll, _ = simulate_locs([FOV_CENTER]; μ=BLINK_MEAN, seed=50000+s)
    append!(calib_locs_all, ll)
end
μ_calibrated = length(calib_locs_all) / 200
println("Calibrated μ (post-filter): $(round(μ_calibrated, digits=2))")

# =============================================================================
# Primary Sweep
# =============================================================================

n_d = length(D_OVER_SIGMA)

count_K = zeros(Int, N_TRIALS, n_d)
count_posterior = [Vector{Vector{Float64}}() for _ in 1:n_d]
nohier_results = [Vector{Any}() for _ in 1:n_d]
hier_results = [Vector{Any}() for _ in 1:n_d]
locs_N = zeros(Int, N_TRIALS, n_d)
median_sigmas = zeros(n_d)

println("\n" * "="^70)
println("PRIMARY SWEEP: d/σ = $(D_OVER_SIGMA)")
println("  $N_TRIALS trials × $(n_d) d/σ × 3 layers")
println("="^70)

t_start = time()

for (di, d_over_σ) in enumerate(D_OVER_SIGMA)
    d = d_over_σ * ref_σ
    true_pos = dimer_positions(d)

    @printf("\n  d/σ = %5.2f (d = %5.1f nm)\n", d_over_σ, d * 1000)
    trial_sigmas = Float64[]

    for trial in 1:N_TRIALS
        seed = 10000 * di + trial
        locs, _ = simulate_locs(true_pos; seed=seed)
        N = length(locs)
        locs_N[trial, di] = N

        if N < 2
            count_K[trial, di] = 0
            push!(count_posterior[di], Float64[])
            push!(nohier_results[di], nothing)
            push!(hier_results[di], nothing)
            continue
        end

        push!(trial_sigmas, median([(l.σ_x + l.σ_y) / 2 for l in locs]))

        # Layer 1: Count-only
        map_k, posterior = count_model_posterior(N, μ_calibrated, TRUE_ALPHA)
        count_K[trial, di] = map_k
        push!(count_posterior[di], posterior)

        # Layer 2: Nohier (with locmix prior)
        μ_fix = Float64(N) / K_TRUE
        push!(nohier_results[di], run_sampler_trial(locs, true_pos;
            μ_fix=μ_fix, shape_fix=TRUE_ALPHA, hierarchical=false, K_true=K_TRUE,
            use_locmix_prior=true))

        # Layer 3: Hier (with locmix prior)
        push!(hier_results[di], run_sampler_trial(locs, true_pos;
            μ_fix=μ_fix, shape_fix=TRUE_ALPHA, hierarchical=true, K_true=K_TRUE,
            use_locmix_prior=true))

        if trial % 10 == 0
            elapsed = time() - t_start
            done = (di - 1) * N_TRIALS + trial
            total = n_d * N_TRIALS
            @printf("    trial %2d/%d  (%.1f/min, ETA %.0fs)\n",
                    trial, N_TRIALS, done / elapsed * 60, (total - done) / (done / elapsed))
        end
    end

    median_sigmas[di] = isempty(trial_sigmas) ? NaN : median(trial_sigmas)
end

elapsed_primary = time() - t_start
@printf("\nPrimary sweep complete: %.0fs (%.1f min)\n", elapsed_primary, elapsed_primary / 60)

# =============================================================================
# Metric Extraction
# =============================================================================

recovery_count = [count(==(K_TRUE), count_K[:, di]) / N_TRIALS for di in 1:n_d]
recovery_nohier = [let v = filter(!isnothing, nohier_results[di]);
    isempty(v) ? 0.0 : count(r -> r.K_dahl == K_TRUE, v) / N_TRIALS end for di in 1:n_d]
recovery_hier = [let v = filter(!isnothing, hier_results[di]);
    isempty(v) ? 0.0 : count(r -> r.K_dahl == K_TRUE, v) / N_TRIALS end for di in 1:n_d]

# =============================================================================
# Scorecard
# =============================================================================

tests, passes = evaluate_scorecard()
all_pass = all(passes)

println("\n" * "="^70)
println("SCORECARD")
println("="^70)
for (t, p) in zip(tests, passes)
    println("  [$(p ? "PASS" : "FAIL")] $t")
end
println("\nOverall: $(all_pass ? "PASS" : "FAIL")")

# =============================================================================
# Dashboard + Secondary
# =============================================================================

println("\nGenerating dashboard...")
build_dashboard(; ref_σ, μ_calibrated, tests, passes)

if RUN_SECONDARY
    println("\n" * "="^70)
    println("SECONDARY SWEEPS")
    println("="^70)
    run_secondary_sweeps(; ref_σ, μ_calibrated)
end

# =============================================================================
# Summary
# =============================================================================

println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println("  Primary: $(n_d) d/σ × $N_TRIALS trials × 3 layers in $(round(elapsed_primary/60, digits=1)) min")
println("  Output: $(OUTPUT_DIR)/dashboard.png")
RUN_SECONDARY && println("  Secondary: $(OUTPUT_DIR)/secondary_sweeps.png")
println()
println("  d/σ=0:  count=$(round(recovery_count[1]*100))%  nohier=$(round(recovery_nohier[1]*100))%  hier=$(round(recovery_hier[1]*100))%")
idx10 = findfirst(==(10.0), D_OVER_SIGMA)
idx10 !== nothing && println("  d/σ=10: count=$(round(recovery_count[idx10]*100))%  nohier=$(round(recovery_nohier[idx10]*100))%  hier=$(round(recovery_hier[idx10]*100))%")
println()
for (t, p) in zip(tests, passes)
    println("  [$(p ? "PASS" : "FAIL")] $t")
end
println("\n  Overall: $(all_pass ? "PASS" : "FAIL")")
println("="^70)

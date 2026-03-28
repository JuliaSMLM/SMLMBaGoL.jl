# Optimality sweep and speed test
#
# run_optimality_sweep() — systematic d/σ sweep across conditions
# run_speed_test() — throughput benchmark
# Plotting via BaGoLMakieExt (plot_sweep, plot_speed)

# ============================================================================
# Count-only MAP-K (Q-PAINT baseline)
# ============================================================================

"""
    count_model_map_k(n_locs, mu, shape) -> Int

MAP estimate of K from count model alone (no spatial information).
Maximizes P(K | N, μ, shape) ∝ Gamma(N; K*shape, μ/shape).

This is the Q-PAINT baseline: best you can do without resolving emitters.
"""
function count_model_map_k(n_locs::Int, mu::Float64, shape::Float64)
    n_locs == 0 && return 0
    best_k = 1
    best_ll = -Inf
    # Search K from 1 to 3× expected
    k_max = max(10, ceil(Int, 3 * n_locs / mu))
    for k in 1:k_max
        α = k * shape
        θ = mu / shape
        ll = (α - 1) * log(n_locs) - n_locs / θ - α * log(θ) - loggamma(α)
        if ll > best_ll
            best_ll = ll
            best_k = k
        end
    end
    return best_k
end

# ============================================================================
# Single trial helper
# ============================================================================

function _run_sweep_trial(;
    n::Int, mu::Float64, d_sigma::Float64, shape::Float64,
    fixed_sigma::Float64, n_iterations::Int, burn_in::Int, trial::Int
)
    diameter = d_sigma * fixed_sigma

    sim = simulate_nmer(;
        n=n, diameter=diameter,
        mean_count=mu, count_model=:negbin, count_shape=shape,
        fixed_sigma=fixed_sigma
    )

    n_locs = length(sim.smld.emitters)
    k_true = n

    # Count-only MAP-K
    k_count = count_model_map_k(n_locs, mu, shape)

    # Nohier: fixed μ, shape at true values, single partition
    result_nohier, diag_nohier = run_bagol(sim.smld;
        n_iterations=n_iterations, burn_in=burn_in,
        shape=shape, learn_shape=false,
        nsigma=Inf, sync_interval=n_iterations + 1,
        μ_prior_shape=mu, verbose=false
    )

    # Hier: learn μ and shape
    result_hier, diag_hier = run_bagol(sim.smld;
        n_iterations=n_iterations, burn_in=burn_in,
        shape=shape, learn_shape=true,
        nsigma=Inf, verbose=false
    )

    # Compute reports for both
    report_nohier = compute_report(result_nohier, diag_nohier;
        true_positions=sim.true_positions, locs_smld=sim.smld)
    report_hier = compute_report(result_hier, diag_hier;
        true_positions=sim.true_positions, locs_smld=sim.smld)

    return (
        n = n,
        mu = mu,
        d_sigma = d_sigma,
        trial = trial,
        n_locs = n_locs,
        k_true = k_true,
        k_count = k_count,
        k_nohier = diag_nohier.n_emitters,
        k_hier = diag_hier.n_emitters,
        rmse_nohier = report_nohier.rmse,
        rmse_hier = report_hier.rmse,
        rmse_oracle = report_nohier.rmse_oracle,
        jaccard_nohier = report_nohier.jaccard,
        jaccard_hier = report_hier.jaccard,
        mahal_d2_hier = isempty(report_hier.calibration.mahal_d2) ? NaN :
                        mean(report_hier.calibration.mahal_d2),
        mu_learned = diag_hier.final_μ,
        shape_learned = diag_hier.final_shape,
    )
end

# ============================================================================
# Aggregation helpers
# ============================================================================

function _aggregate_curves(trials, n_values, mu_values, d_sigma_range)
    curves = Dict{Tuple{Int, Float64}, NamedTuple}()

    for n in n_values, mu in mu_values
        cond_trials = filter(t -> t.n == n && t.mu == mu, trials)
        isempty(cond_trials) && continue

        k_rec_count = Float64[]
        k_rec_count_std = Float64[]
        k_rec_nohier = Float64[]
        k_rec_nohier_std = Float64[]
        k_rec_hier = Float64[]
        k_rec_hier_std = Float64[]
        rmse_ratio_nohier = Float64[]
        rmse_ratio_hier = Float64[]

        for ds in d_sigma_range
            ds_trials = filter(t -> t.d_sigma == ds, cond_trials)
            isempty(ds_trials) && continue
            nt = length(ds_trials)

            # K-recovery rates
            rc = count(t -> t.k_count == t.k_true, ds_trials) / nt
            rn = count(t -> t.k_nohier == t.k_true, ds_trials) / nt
            rh = count(t -> t.k_hier == t.k_true, ds_trials) / nt
            push!(k_rec_count, rc)
            push!(k_rec_count_std, sqrt(rc * (1 - rc) / nt))
            push!(k_rec_nohier, rn)
            push!(k_rec_nohier_std, sqrt(rn * (1 - rn) / nt))
            push!(k_rec_hier, rh)
            push!(k_rec_hier_std, sqrt(rh * (1 - rh) / nt))

            # RMSE ratio (filter NaN)
            valid_nohier = filter(t -> !isnan(t.rmse_nohier) && !isnan(t.rmse_oracle) &&
                                       t.rmse_oracle > 0, ds_trials)
            valid_hier = filter(t -> !isnan(t.rmse_hier) && !isnan(t.rmse_oracle) &&
                                      t.rmse_oracle > 0, ds_trials)
            push!(rmse_ratio_nohier, isempty(valid_nohier) ? NaN :
                  mean(t.rmse_nohier / t.rmse_oracle for t in valid_nohier))
            push!(rmse_ratio_hier, isempty(valid_hier) ? NaN :
                  mean(t.rmse_hier / t.rmse_oracle for t in valid_hier))
        end

        curves[(n, mu)] = (
            d_sigma = collect(d_sigma_range),
            k_recovery_count = k_rec_count,
            k_recovery_count_std = k_rec_count_std,
            k_recovery_nohier = k_rec_nohier,
            k_recovery_nohier_std = k_rec_nohier_std,
            k_recovery_hier = k_rec_hier,
            k_recovery_hier_std = k_rec_hier_std,
            rmse_ratio_nohier = rmse_ratio_nohier,
            rmse_ratio_hier = rmse_ratio_hier,
        )
    end

    return curves
end

# ============================================================================
# Scorecard
# ============================================================================

function _evaluate_scorecard(trials, curves, n_values, mu_values)
    checks = NamedTuple[]

    # 1. Q-PAINT floor: nohier ≥ 0.85× count-only at all d/σ (all conditions)
    qpaint_pass = true
    for (key, c) in curves
        for i in eachindex(c.k_recovery_nohier)
            baseline = c.k_recovery_count[i]
            if baseline > 0 && c.k_recovery_nohier[i] < 0.85 * baseline
                qpaint_pass = false
            end
        end
    end
    push!(checks, (name="Q-PAINT floor", pass=qpaint_pass,
          detail="nohier ≥ 0.85× count-only at all d/σ"))

    # 2. Resolved: K recovery > 90% at d/σ=10 (all conditions)
    resolved_pass = true
    for (key, c) in curves
        idx = findfirst(==(10.0), c.d_sigma)
        if idx !== nothing && c.k_recovery_hier[idx] < 0.90
            resolved_pass = false
        end
    end
    push!(checks, (name="Resolved regime", pass=resolved_pass,
          detail="K recovery > 90% at d/σ=10"))

    # 3. Spatial benefit: recovery(d/σ=5) > recovery(d/σ=0) (all conditions)
    spatial_pass = true
    for (key, c) in curves
        idx0 = findfirst(==(0.0), c.d_sigma)
        idx5 = findfirst(==(5.0), c.d_sigma)
        if idx0 !== nothing && idx5 !== nothing
            if c.k_recovery_hier[idx5] <= c.k_recovery_hier[idx0]
                spatial_pass = false
            end
        end
    end
    push!(checks, (name="Spatial benefit", pass=spatial_pass,
          detail="recovery(d/σ=5) > recovery(d/σ=0)"))

    # 4. Monotonicity: no >5% drops in K-recovery curve (hier)
    mono_pass = true
    for (key, c) in curves
        for i in 2:length(c.k_recovery_hier)
            if c.k_recovery_hier[i] < c.k_recovery_hier[i-1] - 0.05
                mono_pass = false
            end
        end
    end
    push!(checks, (name="Monotonicity", pass=mono_pass,
          detail="no >5% drops in K-recovery curve"))

    # 5. Calibration: scale factor c ∈ [0.7, 1.3]
    valid_mahal = filter(t -> !isnan(t.mahal_d2_hier), trials)
    c_val = isempty(valid_mahal) ? NaN : sqrt(mean(t.mahal_d2_hier for t in valid_mahal) / 2)
    cal_pass = !isnan(c_val) && 0.7 ≤ c_val ≤ 1.3
    push!(checks, (name="Calibration", pass=cal_pass,
          detail="c = $(isnan(c_val) ? "NaN" : round(c_val, digits=2)) ∈ [0.7, 1.3]"))

    # 6. Position accuracy: RMSE/oracle < 1.3 at d/σ ≥ 5
    acc_pass = true
    for (key, c) in curves
        for i in eachindex(c.d_sigma)
            c.d_sigma[i] >= 5.0 || continue
            if !isnan(c.rmse_ratio_hier[i]) && c.rmse_ratio_hier[i] > 1.3
                acc_pass = false
            end
        end
    end
    push!(checks, (name="Position accuracy", pass=acc_pass,
          detail="RMSE/oracle < 1.3 at d/σ ≥ 5"))

    return checks
end

# ============================================================================
# run_optimality_sweep
# ============================================================================

"""
    run_optimality_sweep(; kwargs...) -> NamedTuple

Run systematic d/σ sweep across (n, μ) conditions with parallel trials.

# Conditions
- `n_values=[2, 8]`: cluster sizes (dimer, octamer)
- `mu_values=[5.0, 10.0]`: mean blinks per emitter
- `d_sigma_range`: separation / localization precision grid

# Per condition, per d/σ
Runs `n_trials` independent simulations with count-only, nohier, and hier methods.
Trials run in parallel via `Threads.@threads`.

# Returns
NamedTuple with:
- `trials`: Vector of per-trial results
- `curves`: Dict{(n, μ) => aggregated curves}
- `scorecard`: Vector of pass/fail checks
- `config`: sweep parameters
"""
function run_optimality_sweep(;
    n_values::Vector{Int} = [2, 8],
    mu_values::Vector{Float64} = [5.0, 10.0],
    d_sigma_range::Vector{Float64} = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0],
    n_trials::Int = 50,
    n_iterations::Int = 10_000,
    burn_in::Int = 2000,
    fixed_sigma::Float64 = 0.010,
    shape::Float64 = 5.0,
    verbose::Bool = true
)
    config = (n_values=n_values, mu_values=mu_values, d_sigma_range=d_sigma_range,
              n_trials=n_trials, n_iterations=n_iterations, burn_in=burn_in,
              fixed_sigma=fixed_sigma, shape=shape)

    # Build flat work list: (n, mu, d_sigma, trial)
    work = Tuple{Int, Float64, Float64, Int}[]
    for n in n_values, mu in mu_values, ds in d_sigma_range, trial in 1:n_trials
        push!(work, (n, mu, ds, trial))
    end

    n_total = length(work)
    if verbose
        n_threads = Threads.nthreads()
        println("Optimality sweep: $(length(n_values))×$(length(mu_values)) conditions, " *
                "$(length(d_sigma_range)) d/σ points, $n_trials trials = $n_total runs")
        println("Using $n_threads threads")
    end

    results = Vector{NamedTuple}(undef, n_total)
    done = Threads.Atomic{Int}(0)
    t0 = time()

    Threads.@threads for idx in 1:n_total
        n, mu, ds, trial = work[idx]
        results[idx] = _run_sweep_trial(;
            n=n, mu=mu, d_sigma=ds, shape=shape,
            fixed_sigma=fixed_sigma, n_iterations=n_iterations,
            burn_in=burn_in, trial=trial
        )
        n_done = Threads.atomic_add!(done, 1) + 1
        if verbose && (n_done % max(1, n_total ÷ 20) == 0 || n_done == n_total)
            elapsed = round(time() - t0, digits=1)
            pct = round(100 * n_done / n_total, digits=0)
            println("  $(Int(pct))% ($n_done/$n_total) in $(elapsed)s")
        end
    end

    trials = collect(results)
    curves = _aggregate_curves(trials, n_values, mu_values, d_sigma_range)
    scorecard = _evaluate_scorecard(trials, curves, n_values, mu_values)

    if verbose
        elapsed = round(time() - t0, digits=1)
        println("Sweep complete in $(elapsed)s")
        println()
        for sc in scorecard
            status = sc.pass ? "PASS" : "FAIL"
            println("  $status  $(sc.name): $(sc.detail)")
        end
    end

    return (trials=trials, curves=curves, scorecard=scorecard, config=config)
end

# ============================================================================
# Speed test
# ============================================================================

"""
    run_speed_test(; kwargs...) -> Vector{NamedTuple}

Benchmark BaGoL throughput at different dataset sizes.

Generates grids of 8-mers at varying density to control localization count.
Measures wall time for full `run_bagol` pipeline.

# Returns
Vector of `(n_locs_target, n_locs_actual, n_partitions, wall_time_s, locs_per_s, times)`.
"""
function run_speed_test(;
    n_locs_range::Vector{Int} = [100, 500, 1000, 5000, 10_000],
    n_repeats::Int = 3,
    n_iterations::Int = 5000,
    burn_in::Int = 1000,
    n_per_cluster::Int = 8,
    cluster_diameter::Float64 = 0.050,
    mu::Float64 = 10.0,
    fixed_sigma::Float64 = 0.010,
    shape::Float64 = 5.0,
    verbose::Bool = true
)
    results = NamedTuple[]

    if verbose
        println("Speed test: $(length(n_locs_range)) sizes × $n_repeats repeats")
        println("  $(n_iterations) iterations, $(Threads.nthreads()) threads")
    end

    for n_target in n_locs_range
        # Compute grid to get ~n_target locs
        n_clusters = max(1, round(Int, n_target / (n_per_cluster * mu)))
        grid_side = max(1, ceil(Int, sqrt(n_clusters)))

        times = Float64[]
        n_actual = 0
        n_parts = 0

        for rep in 1:n_repeats
            sim = simulate_nmer_grid(;
                n_per_cluster=n_per_cluster, cluster_diameter=cluster_diameter,
                grid_nx=grid_side, grid_ny=grid_side, grid_spacing=1.0,
                mean_count=mu, count_model=:negbin, count_shape=shape,
                fixed_sigma=fixed_sigma
            )
            n_actual = length(sim.smld.emitters)

            t = @elapsed begin
                _, diag = run_bagol(sim.smld;
                    n_iterations=n_iterations, burn_in=burn_in,
                    shape=shape, verbose=false
                )
                n_parts = diag.n_partitions
            end
            push!(times, t)
        end

        med_t = median(times)
        push!(results, (
            n_locs_target = n_target,
            n_locs_actual = n_actual,
            n_partitions = n_parts,
            wall_time_s = med_t,
            locs_per_s = n_actual / med_t,
            times = times,
        ))

        if verbose
            println("  N=$(lpad(n_actual, 6)): $(round(med_t, digits=2))s, " *
                    "$(round(n_actual / med_t, digits=0)) locs/s, " *
                    "$n_parts partitions")
        end
    end

    return results
end

# ============================================================================
# Write sweep/speed results
# ============================================================================

"""
    write_sweep(sweep; output_dir="output")

Write sweep results to disk: `sweep_data.csv` and `scorecard.txt`.
"""
function write_sweep(sweep; output_dir::String = "output")
    mkpath(output_dir)

    # CSV
    csv_path = joinpath(output_dir, "sweep_data.csv")
    open(csv_path, "w") do io
        println(io, "n,mu,d_sigma,trial,n_locs,k_true,k_count,k_nohier,k_hier," *
                    "rmse_nohier,rmse_hier,rmse_oracle,jaccard_nohier,jaccard_hier," *
                    "mahal_d2_hier,mu_learned,shape_learned")
        for t in sweep.trials
            vals = [t.n, t.mu, t.d_sigma, t.trial, t.n_locs,
                    t.k_true, t.k_count, t.k_nohier, t.k_hier,
                    t.rmse_nohier, t.rmse_hier, t.rmse_oracle,
                    t.jaccard_nohier, t.jaccard_hier,
                    t.mahal_d2_hier, t.mu_learned, t.shape_learned]
            println(io, join(vals, ","))
        end
    end
    println("Saved: $csv_path ($(length(sweep.trials)) trials)")

    # Scorecard
    sc_path = joinpath(output_dir, "scorecard.txt")
    open(sc_path, "w") do io
        println(io, "SMLMBaGoL Optimality Scorecard")
        println(io, "=" ^ 40)
        println(io)
        all_pass = all(sc.pass for sc in sweep.scorecard)
        println(io, "Overall: $(all_pass ? "ALL PASS" : "FAILURES DETECTED")")
        println(io)
        for sc in sweep.scorecard
            status = sc.pass ? "PASS" : "FAIL"
            println(io, "  $status  $(rpad(sc.name * ":", 22)) $(sc.detail)")
        end
        println(io)
        println(io, "Config:")
        println(io, "  n_values:     $(sweep.config.n_values)")
        println(io, "  mu_values:    $(sweep.config.mu_values)")
        println(io, "  d/σ range:    $(sweep.config.d_sigma_range)")
        println(io, "  trials/point: $(sweep.config.n_trials)")
        println(io, "  iterations:   $(sweep.config.n_iterations)")
        println(io, "  fixed_sigma:  $(sweep.config.fixed_sigma * 1000) nm")
    end
    println("Saved: $sc_path")
end

"""
    write_speed(speed; output_dir="output")

Write speed test results to `speed_data.csv`.
"""
function write_speed(speed; output_dir::String = "output")
    mkpath(output_dir)
    path = joinpath(output_dir, "speed_data.csv")
    open(path, "w") do io
        println(io, "n_locs_target,n_locs_actual,n_partitions,wall_time_s,locs_per_s")
        for r in speed
            println(io, "$(r.n_locs_target),$(r.n_locs_actual),$(r.n_partitions)," *
                        "$(round(r.wall_time_s, digits=3)),$(round(r.locs_per_s, digits=1))")
        end
    end
    println("Saved: $path ($(length(speed)) sizes)")
end

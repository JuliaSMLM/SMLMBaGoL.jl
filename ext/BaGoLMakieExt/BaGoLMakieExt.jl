module BaGoLMakieExt

using SMLMBaGoL
using CairoMakie
using Statistics
using Random: MersenneTwister
using SpecialFunctions: gamma, loggamma

# ============================================================================
# plot_report — Category 1 & 2 figures
# ============================================================================

"""
    plot_report(report; output_dir="output")

Generate standard report figures using CairoMakie.

Category 1 (always): partition_k, acceptance_rates, cluster_sizes
Category 2 (if has_gt): k_recovery, calibration
"""
function SMLMBaGoL.plot_report(report; output_dir::String = "output")
    mkpath(output_dir)

    _plot_partition_k(report, output_dir)
    _plot_acceptance_rates(report, output_dir)
    _plot_cluster_sizes(report, output_dir)
    if !isempty(report.nn_distances)
        _plot_nn_distances(report, output_dir)
    end

    if hasproperty(report, :convergence_trace) && !isempty(report.convergence_trace.iters)
        _plot_convergence(report, output_dir)
    end

    if hasproperty(report, :motion) && report.motion !== nothing && !isempty(report.motion.axis_mean)
        _plot_motion_velocities(report, output_dir)
    end

    if report.has_gt
        _plot_k_recovery(report, output_dir)
        if !isnan(report.calibration.scale_factor)
            _plot_calibration(report, output_dir)
        end
    end
end

# Histogram of the recovered per-emitter velocities (motion=:linear), one panel per axis.
# Restricted to emitters with velocity leverage (n ≥ 2) — n=1 emitters have v̂≡0 (no time
# information) and would just pile a spike at 0. Per-emitter `velocity_var`/`n` are in
# diag.motion for any downstream gating/weighting.
function _plot_motion_velocities(report, output_dir)
    m = report.motion
    V = m.velocities                       # N×D (μm)
    isempty(V) && return
    keep = m.n .>= 2
    any(keep) || return
    V = V[keep, :]
    D = size(V, 2)
    axn = ("x", "y", "z")
    fig = Figure(size = (430 * D, 430))
    Label(fig[0, 1:D], "Recovered per-emitter drift (motion=:linear) — $(size(V, 1)) emitters (n ≥ 2), end-to-end",
          fontsize = 14, font = :bold)
    for d in 1:D
        vd = 1000 .* V[:, d]               # nm
        mn = length(vd) >= 1 ? sum(vd) / length(vd) : 0.0
        ax = Axis(fig[1, d], xlabel = "v_$(axn[d]) (nm)", ylabel = "emitters",
                  title = "v_$(axn[d]) (mean $(round(mn, digits = 1)) nm)")
        hist!(ax, vd; bins = max(8, round(Int, sqrt(length(vd)))), color = (:steelblue, 0.7))
        vlines!(ax, [mn]; color = :red, linewidth = 2, label = "mean")
        vlines!(ax, [0.0]; color = :gray50, linestyle = :dash, label = "0")
    end
    save(joinpath(output_dir, "motion_velocities.png"), fig)
end

function _plot_convergence(report, output_dir)
    tr = report.convergence_trace
    isempty(tr.iters) && return
    its = Float64.(tr.iters)
    fig = Figure(size = (950, 680))
    Label(fig[0, 1:2],
          "MCMC convergence trace (resolution = sync_interval) — a still-trending curve ⇒ still in burn-in",
          fontsize = 13, font = :bold)
    panels = [("emitter count K", Float64.(tr.K)), ("μ (mean locs/emitter)", tr.mu),
              ("shape", tr.shape), ("ρ (emitters/μm²)", tr.rho)]
    for (i, (lab, vals)) in enumerate(panels)
        r = (i - 1) ÷ 2 + 1; c = (i - 1) % 2 + 1
        ax = Axis(fig[r, c], xlabel = "iteration", ylabel = lab, title = lab)
        scatterlines!(ax, its, vals; color = :steelblue, linewidth = 2, markersize = 7)
        if tr.burn_in > 0   # samples LEFT of this line are discarded as burn-in (transient)
            vlines!(ax, [Float64(tr.burn_in)]; color = :gray40, linestyle = :dash, linewidth = 1.5,
                    label = "burn-in = $(tr.burn_in)")
            i == 1 && axislegend(ax; position = :rb, framevisible = false, labelsize = 9)
        end
        # flag a monotone trend over the last half (a crude "still climbing/falling" cue)
        if length(vals) ≥ 4
            h = length(vals) ÷ 2
            d = vals[end] - vals[h]
            rng = maximum(vals) - minimum(vals)
            if rng > 0 && abs(d) > 0.05 * rng
                text!(ax, its[end], vals[end]; text = d > 0 ? "still ↑" : "still ↓",
                      align = (:right, d > 0 ? :bottom : :top), fontsize = 10, color = :crimson)
            end
        end
    end
    save(joinpath(output_dir, "convergence.png"), fig, px_per_unit = 2)
    println("Saved: $(joinpath(output_dir, "convergence.png"))")
end

function _plot_partition_k(report, output_dir)
    pk = report.partition_k
    isempty(pk) && return
    fig = Figure(size=(500, 350))
    ax = Axis(fig[1, 1], xlabel="Emitters found per partition", ylabel="Count",
              title="Per-partition K ($(length(pk)) partitions, $(sum(pk)) emitters total)")
    lo, hi = extrema(pk)
    counts = zeros(Int, hi - lo + 1)
    for v in pk; counts[v - lo + 1] += 1; end
    barplot!(ax, collect(lo:hi), counts; color=:steelblue)
    vlines!(ax, [median(pk)]; color=:red, linewidth=2, linestyle=:dash,
            label="Median = $(round(median(pk), digits=1))")
    axislegend(ax; position=:rt, framevisible=false)
    save(joinpath(output_dir, "partition_k.png"), fig, px_per_unit=2)
    println("Saved: $(joinpath(output_dir, "partition_k.png"))")
end

function _plot_acceptance_rates(report, output_dir)
    ar = report.acceptance_rates
    isempty(ar) && return
    names = String[]
    vals = Float64[]
    for k in sort(collect(keys(ar)))
        push!(names, string(k))
        push!(vals, ar[k] * 100)
    end
    fig = Figure(size=(400, 300))
    ax = Axis(fig[1, 1], xlabel="Move type", ylabel="Acceptance rate (%)",
              title="MCMC acceptance rates", xticks=(1:length(names), names))
    barplot!(ax, 1:length(vals), vals; color=:steelblue)
    ylims!(ax, 0, 100)
    save(joinpath(output_dir, "acceptance_rates.png"), fig, px_per_unit=2)
    println("Saved: $(joinpath(output_dir, "acceptance_rates.png"))")
end

function _plot_cluster_sizes(report, output_dir)
    cs = report.cluster_sizes
    isempty(cs) && return

    fig = Figure(size=(500, 350))
    ax = Axis(fig[1, 1], xlabel="Localizations per emitter", ylabel="Probability",
              title="Count distribution (locs/emitter)")

    # Ground truth empirical counts: only show when true_positions was
    # provided (simulation). For real SMLM data, track_id is often set by
    # FrameConnect or upstream pipelines and is NOT ground truth — do not
    # present it as such.
    has_gt_counts = report.has_gt && !isempty(report.empirical_counts)
    ec = has_gt_counts ? report.empirical_counts : Int[]

    # BaGoL's inferred cluster sizes — cap x-axis at 99th percentile for readability
    k_max = min(maximum(cs) + 5, round(Int, quantile(Float64.(cs), 0.99)) + 10)
    if has_gt_counts
        k_max = max(k_max, min(maximum(ec) + 5, round(Int, quantile(Float64.(ec), 0.99)) + 10))
    end
    k_range = 1:k_max

    if has_gt_counts
        gt_data = zeros(k_max)
        for c in ec
            1 <= c <= k_max && (gt_data[c] += 1)
        end
        gt_data ./= sum(gt_data)
        barplot!(ax, collect(k_range), gt_data; color=(:gray70, 0.5), label="GT (empirical)")
    end

    hist_data = zeros(k_max)
    for c in cs
        1 <= c <= k_max && (hist_data[c] += 1)
    end
    hist_data ./= sum(hist_data)
    barplot!(ax, collect(k_range), hist_data; color=(:steelblue, 0.6), label="Chain (inferred)")

    # NegBin PMF: P(X=k) = Γ(k+α) / (k! Γ(α)) × p^α × (1-p)^k, k=0,1,2,...
    # where p = α/(α+μ). Clamped to k≥1, renormalized.
    function _negbin_pmf(μ_val, α_val, ks)
        p = α_val / (α_val + μ_val)
        raw = [exp(loggamma(k + α_val) - loggamma(k + 1.0) - loggamma(α_val) +
                   α_val * log(p) + k * log(1 - p)) for k in Float64.(ks)]
        total = sum(raw)
        total > 0 ? raw ./ total : zeros(length(ks))
    end

    # True NegBin (known simulation params)
    tcp = report.true_count_params
    if tcp !== nothing && tcp.μ > 0 && tcp.shape > 0
        negbin_true = _negbin_pmf(tcp.μ, tcp.shape, k_range)
        lines!(ax, collect(k_range), negbin_true; color=:blue, linewidth=2,
               label="Known: μ=$(round(tcp.μ, digits=1)), α=$(round(tcp.shape, digits=1))")
    end

    # Learned NegBin (found by BaGoL)
    μ = report.final_mu
    α = report.final_shape
    if μ > 0 && α > 0
        negbin_learned = _negbin_pmf(μ, α, k_range)
        lines!(ax, collect(k_range), negbin_learned; color=:red, linewidth=2,
               label="Found: μ=$(round(μ, digits=1)), α=$(round(α, digits=1))")
    end

    axislegend(ax; position=:rt, framevisible=false)
    save(joinpath(output_dir, "count_distribution.png"), fig, px_per_unit=2)
    println("Saved: $(joinpath(output_dir, "count_distribution.png"))")
end

function _plot_nn_distances(report, output_dir)
    dists = report.nn_distances .* 1000  # nm
    has_gt = hasproperty(report, :gt_nn_distances) && !isempty(report.gt_nn_distances)
    gt_dists = has_gt ? report.gt_nn_distances .* 1000 : Float64[]

    # Truncate at 99th percentile for readability (use max of both distributions)
    cutoff = quantile(dists, 0.99)
    if has_gt
        cutoff = max(cutoff, quantile(gt_dists, 0.99))
    end
    dists_trunc = filter(d -> d <= cutoff, dists)
    n_bins = 30

    fig = Figure(size=(500, 350))
    ax = Axis(fig[1, 1], xlabel="Nearest-neighbor distance (nm)", ylabel="Count",
              title="NN distances between emitters ($(length(dists)) total)")

    # GT histogram (behind, if available)
    if has_gt
        gt_trunc = filter(d -> d <= cutoff, gt_dists)
        hist!(ax, gt_trunc; bins=n_bins, color=(:gray70, 0.4), label="GT ($(length(gt_dists)))")
    end

    # BaGoL histogram
    hist!(ax, dists_trunc; bins=n_bins, color=(:steelblue, 0.6), label="BaGoL ($(length(dists)))")

    # Shared bin edges for mode computation
    if length(dists_trunc) >= 2
        edges = range(minimum(dists_trunc), maximum(dists_trunc); length=n_bins + 1)

        # BaGoL mode
        counts = zeros(Int, n_bins)
        for d in dists
            idx = clamp(searchsortedlast(edges, d), 1, n_bins)
            counts[idx] += 1
        end
        max_idx = argmax(counts)
        mode_val = (edges[max_idx] + edges[max_idx + 1]) / 2
        vlines!(ax, [mode_val]; color=:orange, linewidth=2, linestyle=:solid,
                label="Mode = $(round(mode_val, digits=1)) nm")

        # GT mode
        if has_gt
            gt_counts = zeros(Int, n_bins)
            for d in gt_dists
                idx = clamp(searchsortedlast(edges, d), 1, n_bins)
                gt_counts[idx] += 1
            end
            gt_max_idx = argmax(gt_counts)
            gt_mode = (edges[gt_max_idx] + edges[gt_max_idx + 1]) / 2
            vlines!(ax, [gt_mode]; color=:green3, linewidth=2, linestyle=:solid,
                    label="GT mode = $(round(gt_mode, digits=1)) nm")
        end
    end

    # Median lines
    vlines!(ax, [median(dists)]; color=:red, linewidth=2, linestyle=:dash,
            label="Median = $(round(median(dists), digits=1)) nm")
    if has_gt
        vlines!(ax, [median(gt_dists)]; color=:green3, linewidth=2, linestyle=:dashdot,
                label="GT median = $(round(median(gt_dists), digits=1)) nm")
    end

    axislegend(ax; position=:rt, framevisible=false)
    save(joinpath(output_dir, "nn_distances.png"), fig, px_per_unit=2)
    println("Saved: $(joinpath(output_dir, "nn_distances.png"))")
end

function _plot_k_recovery(report, output_dir)
    pk = report.partition_k
    isempty(pk) && return
    n_parts = length(pk)

    # For n-mer grids: K_true per partition = total true / n_partitions
    # (assumes equal-size clusters, one per partition)
    k_true_per_part = n_parts > 0 ? report.k_true / n_parts : 0
    n_correct = count(==(round(Int, k_true_per_part)), pk)
    pct = round(100 * n_correct / n_parts, digits=0)

    fig = Figure(size=(500, 350))
    ax = Axis(fig[1, 1], xlabel="Emitters found per partition", ylabel="Count",
              title="K recovery: $(Int(pct))% correct ($n_correct/$n_parts)")
    lo, hi = extrema(pk)
    counts = zeros(Int, hi - lo + 1)
    for v in pk; counts[v - lo + 1] += 1; end
    barplot!(ax, collect(lo:hi), counts; color=:steelblue)
    if k_true_per_part > 0 && isinteger(k_true_per_part)
        vlines!(ax, [k_true_per_part]; color=:red, linewidth=2, linestyle=:dash,
                label="True K/partition = $(Int(k_true_per_part))")
        axislegend(ax; position=:rt, framevisible=false)
    end
    save(joinpath(output_dir, "k_recovery.png"), fig, px_per_unit=2)
    println("Saved: $(joinpath(output_dir, "k_recovery.png"))")
end

function _plot_calibration(report, output_dir)
    cal = report.calibration
    isempty(cal.mahal_d2) && return

    fig = Figure(size=(900, 400))

    # Panel 1: estimated σ vs actual error
    emitters = report.emitters
    n_true = report.k_true
    # We have mahal_d2 for matched emitters — plot σ_est vs |error|
    ax1 = Axis(fig[1, 1], xlabel="Estimated σ (nm)", ylabel="Actual error (nm)",
               title="Uncertainty calibration", aspect=1)
    # Reconstruct matched pairs from report data
    # Use simple proxy: sqrt(σ_x² + σ_y²)/√2 as "estimated σ"
    sigma_est = Float64[]
    errors = Float64[]
    for (d2, e) in zip(cal.mahal_d2, emitters)
        σ_eff = sqrt(e.σ_x^2 + e.σ_y^2) / sqrt(2) * 1000  # nm
        # |error| = σ_eff * sqrt(d2) approximately
        err = σ_eff * sqrt(d2)
        push!(sigma_est, σ_eff)
        push!(errors, err)
    end
    if !isempty(sigma_est)
        scatter!(ax1, sigma_est, errors; color=:steelblue, markersize=5)
        max_val = max(maximum(sigma_est), maximum(errors)) * 1.1
        lines!(ax1, [0, max_val], [0, max_val]; color=:red, linewidth=1.5,
               linestyle=:dash, label="y = x")
        axislegend(ax1; position=:lt, framevisible=false)
    end

    # Panel 2: coverage probability
    ax2 = Axis(fig[1, 2], xlabel="Radius (nσ)", ylabel="Coverage probability",
               title="Coverage (scale = $(round(cal.scale_factor, digits=2)))")
    n_sigmas = [1, 2, 3]
    observed = [cal.coverage_1σ, cal.coverage_2σ, cal.coverage_3σ]
    expected = [cal.expected_1σ, cal.expected_2σ, cal.expected_3σ]
    barplot!(ax2, Float64.(n_sigmas) .- 0.15, observed; width=0.3, color=:steelblue,
             label="Observed")
    barplot!(ax2, Float64.(n_sigmas) .+ 0.15, expected; width=0.3, color=:lightgray,
             label="Expected (χ²₂)")
    ylims!(ax2, 0, 1.05)
    axislegend(ax2; position=:lt, framevisible=false)

    save(joinpath(output_dir, "calibration.png"), fig, px_per_unit=2)
    println("Saved: $(joinpath(output_dir, "calibration.png"))")
end

# ============================================================================
# plot_sweep — Category 3 sweep figures
# ============================================================================

"""
    plot_sweep(sweep; output_dir="output")

Generate optimality sweep figures:
- `sweep_krecovery.png` — 2×2 grid: K-recovery vs d/σ per condition
- `sweep_collated.png` — all conditions overlaid
- `sweep_rmse.png` — RMSE/oracle ratio vs d/σ
- `k_true_vs_k_found.png` — scatter across all runs
"""
function SMLMBaGoL.plot_sweep(sweep; output_dir::String = "output")
    mkpath(output_dir)

    _plot_sweep_krecovery(sweep, output_dir)
    _plot_sweep_collated(sweep, output_dir)
    _plot_sweep_rmse(sweep, output_dir)
    _plot_k_scatter(sweep, output_dir)
end

function _plot_sweep_krecovery(sweep, output_dir)
    n_vals = sweep.config.n_values
    mu_vals = sweep.config.mu_values
    n_rows = length(n_vals)
    n_cols = length(mu_vals)

    fig = Figure(size=(450 * n_cols, 350 * n_rows))
    for (ri, n) in enumerate(n_vals), (ci, mu) in enumerate(mu_vals)
        key = (n, mu)
        haskey(sweep.curves, key) || continue
        c = sweep.curves[key]

        ax = Axis(fig[ri, ci],
            xlabel = ri == n_rows ? "d/σ" : "",
            ylabel = ci == 1 ? "K recovery rate" : "",
            title = "n=$n, μ=$mu")

        ds = c.d_sigma

        # Count-only baseline
        band!(ax, ds, c.k_recovery_count .- c.k_recovery_count_std,
              c.k_recovery_count .+ c.k_recovery_count_std;
              color=(:gray, 0.2))
        lines!(ax, ds, c.k_recovery_count; color=:gray, linewidth=1.5,
               linestyle=:dash, label="Count-only")

        # Nohier
        band!(ax, ds, c.k_recovery_nohier .- c.k_recovery_nohier_std,
              c.k_recovery_nohier .+ c.k_recovery_nohier_std;
              color=(:dodgerblue, 0.2))
        lines!(ax, ds, c.k_recovery_nohier; color=:dodgerblue, linewidth=2,
               label="Fixed μ,α")

        # Hier
        band!(ax, ds, c.k_recovery_hier .- c.k_recovery_hier_std,
              c.k_recovery_hier .+ c.k_recovery_hier_std;
              color=(:crimson, 0.2))
        lines!(ax, ds, c.k_recovery_hier; color=:crimson, linewidth=2,
               label="Hierarchical")

        ylims!(ax, -0.05, 1.05)

        if ri == 1 && ci == n_cols
            axislegend(ax; position=:rb, framevisible=false)
        end
    end

    save(joinpath(output_dir, "sweep_krecovery.png"), fig, px_per_unit=2)
    println("Saved: $(joinpath(output_dir, "sweep_krecovery.png"))")
end

function _plot_sweep_collated(sweep, output_dir)
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1], xlabel="d/σ", ylabel="K recovery rate (hier)",
              title="Collated K-recovery across conditions")

    colors = [:crimson, :dodgerblue, :forestgreen, :darkorange]
    markers = [:circle, :rect, :utriangle, :diamond]
    ci = 1
    for (key, c) in sort(collect(sweep.curves))
        n, mu = key
        lines!(ax, c.d_sigma, c.k_recovery_hier;
               color=colors[mod1(ci, length(colors))], linewidth=2,
               label="n=$n, μ=$mu")
        scatter!(ax, c.d_sigma, c.k_recovery_hier;
                 color=colors[mod1(ci, length(colors))],
                 marker=markers[mod1(ci, length(markers))], markersize=8)
        ci += 1
    end
    ylims!(ax, -0.05, 1.05)
    axislegend(ax; position=:rb, framevisible=false)

    save(joinpath(output_dir, "sweep_collated.png"), fig, px_per_unit=2)
    println("Saved: $(joinpath(output_dir, "sweep_collated.png"))")
end

function _plot_sweep_rmse(sweep, output_dir)
    n_vals = sweep.config.n_values
    mu_vals = sweep.config.mu_values
    n_rows = length(n_vals)
    n_cols = length(mu_vals)

    fig = Figure(size=(450 * n_cols, 350 * n_rows))
    for (ri, n) in enumerate(n_vals), (ci, mu) in enumerate(mu_vals)
        key = (n, mu)
        haskey(sweep.curves, key) || continue
        c = sweep.curves[key]

        ax = Axis(fig[ri, ci],
            xlabel = ri == n_rows ? "d/σ" : "",
            ylabel = ci == 1 ? "RMSE / Oracle RMSE" : "",
            title = "n=$n, μ=$mu")

        ds = c.d_sigma
        valid_nh = [isnan(v) ? NaN : v for v in c.rmse_ratio_nohier]
        valid_h = [isnan(v) ? NaN : v for v in c.rmse_ratio_hier]

        lines!(ax, ds, valid_nh; color=:dodgerblue, linewidth=2, label="Fixed μ,α")
        lines!(ax, ds, valid_h; color=:crimson, linewidth=2, label="Hierarchical")
        hlines!(ax, [1.0]; color=:gray, linewidth=1, linestyle=:dash, label="Oracle")
        ylims!(ax, 0.5, 3.0)

        if ri == 1 && ci == n_cols
            axislegend(ax; position=:rt, framevisible=false)
        end
    end

    save(joinpath(output_dir, "sweep_rmse.png"), fig, px_per_unit=2)
    println("Saved: $(joinpath(output_dir, "sweep_rmse.png"))")
end

function _plot_k_scatter(sweep, output_dir)
    fig = Figure(size=(500, 500))
    ax = Axis(fig[1, 1], xlabel="K true", ylabel="K found (hier)",
              title="K true vs K found", aspect=1)

    # Collect all (k_true, k_hier) pairs
    kt = [t.k_true for t in sweep.trials]
    kf = [t.k_hier for t in sweep.trials]

    # Jitter for visibility
    jitter = 0.15
    kt_j = kt .+ jitter .* (rand(length(kt)) .- 0.5)
    kf_j = kf .+ jitter .* (rand(length(kf)) .- 0.5)

    scatter!(ax, kt_j, kf_j; color=(:steelblue, 0.3), markersize=4)

    # Diagonal
    k_range = [minimum(kt) - 0.5, maximum(kt) + 0.5]
    lines!(ax, k_range, k_range; color=:red, linewidth=1.5, linestyle=:dash)

    save(joinpath(output_dir, "k_true_vs_k_found.png"), fig, px_per_unit=2)
    println("Saved: $(joinpath(output_dir, "k_true_vs_k_found.png"))")
end

# ============================================================================
# plot_speed — speed test figure
# ============================================================================

"""
    plot_speed(speed; output_dir="output")

Generate speed test figure: throughput vs dataset size (log-log).
"""
function SMLMBaGoL.plot_speed(speed; output_dir::String = "output")
    mkpath(output_dir)

    fig = Figure(size=(500, 400))
    ax = Axis(fig[1, 1], xlabel="Localizations", ylabel="Throughput (locs/s)",
              title="BaGoL speed test", xscale=log10, yscale=log10)

    n_locs = [r.n_locs_actual for r in speed]
    throughput = [r.locs_per_s for r in speed]

    scatterlines!(ax, n_locs, throughput; color=:steelblue, linewidth=2, markersize=8)

    # Add wall time annotations
    for r in speed
        text!(ax, r.n_locs_actual, r.locs_per_s;
              text="$(round(r.wall_time_s, digits=1))s",
              fontsize=10, align=(:left, :bottom), offset=(5, 5))
    end

    save(joinpath(output_dir, "speed_test.png"), fig, px_per_unit=2)
    println("Saved: $(joinpath(output_dir, "speed_test.png"))")
end

# ============================================================================
# plot_se_adjust — τ-finder diagnostics (ported from SMLMClustering diagnose_tau.jl v2)
# ============================================================================
# STANDARD output: the FROZEN grouping (the finder's returned d_nm/s2_nm2) RE-SCORED
# over the grid — zero extra BaGoL. The expensive re-group-per-τ circularity and the
# sim-only ground-truth-label curve are intentionally dropped (opt-in deep diagnostics).
# On REAL data these are FIT-QUALITY-per-cell panels, not a proof τ is "right".

"""
    plot_se_adjust(result; output_dir="output", n_boot=300, seed=1)

τ-finder diagnostics from `estimate_se_adjust(...; return_diagnostics=true)`.
"""
function SMLMBaGoL.plot_se_adjust(result; output_dir::String = "output",
                                  n_boot::Int = 300, seed::Int = 1)
    dg = result.diagnostics
    dg === nothing && error("plot_se_adjust requires estimate_se_adjust(...; return_diagnostics=true)")
    mkpath(output_dir)
    d = dg.d_nm; s2 = dg.s2_nm2; pos = dg.pos_um; grid = collect(dg.grid_nm)
    τh = dg.tau_hat_nm; ks_path = dg.ks_path
    ci_lo = 1000 * result.ci_lo_um; ci_hi = 1000 * result.ci_hi_um
    ray1_cdf(z) = 1 - exp(-z^2 / 2); ray1_q(q) = sqrt(-2 * log(1 - q))
    zτ(t) = [d[k] / sqrt(s2[k] + 2t^2) for k in eachindex(d)]

    fig = Figure(size = (1250, 950))
    Label(fig[0, 1:2],
          "se_adjust τ-finder diagnostics — τ̂ = $(round(τh, digits=2)) nm  " *
          "($(length(d)) pairs, $(result.n_bagol) BaGoL runs) [frozen grouping]",
          fontsize = 18, font = :bold)

    # Panel 1 — frozen KS(τ) landscape + descent path + bootstrap CI
    ax1 = Axis(fig[1, 1], title = "1. KS(τ) to Rayleigh(1) — frozen grouping",
               xlabel = "assumed τ (nm)", ylabel = "KS")
    (isfinite(ci_lo) && isfinite(ci_hi)) &&
        vspan!(ax1, ci_lo, ci_hi, color = (:seagreen, 0.15), label = "95% CI")
    lines!(ax1, grid, ks_path, color = :seagreen, linewidth = 2.5, label = "frozen KS(τ)")
    vlines!(ax1, [τh], color = :black, linestyle = :dash, linewidth = 2,
            label = "τ̂ = $(round(τh, digits=2)) nm")
    pg = [1000 * p[1] for p in result.path_um]
    isempty(pg) || scatter!(ax1, pg, [ks_path[argmin(abs.(grid .- g))] for g in pg],
                            color = :crimson, markersize = 9, label = "descent g→")
    axislegend(ax1, position = :rt, framevisible = false, labelsize = 10)

    # Panel 2 — empirical CDF − Rayleigh(1) at τ̂ + SPATIAL-BLOCK bootstrap band
    # (block = independent unit, NOT raw pairs: a raw-pair band is spuriously tight)
    ax2 = Axis(fig[1, 2], title = "2. empirical CDF − Rayleigh(1) at τ̂ (block-bootstrap 95%)",
               xlabel = "z = d/√(σ²+2τ̂²)", ylabel = "CDF − Rayleigh(1)")
    zgrid = collect(0.05:0.05:4.0)
    ediff(z) = [count(<=(r), z) / length(z) - ray1_cdf(r) for r in zgrid]
    rng = MersenneTwister(seed); boots = Vector{Vector{Float64}}()
    for _ in 1:n_boot
        bi = SMLMBaGoL._block_indices(pos, 1.0, rng)
        isempty(bi) || push!(boots, ediff([d[i] / sqrt(s2[i] + 2τh^2) for i in bi]))
    end
    if !isempty(boots)
        lo = [quantile([bt[j] for bt in boots], 0.025) for j in eachindex(zgrid)]
        hi = [quantile([bt[j] for bt in boots], 0.975) for j in eachindex(zgrid)]
        band!(ax2, zgrid, lo, hi, color = (:seagreen, 0.22), label = "block-bootstrap 95%")
    end
    hlines!(ax2, [0.0], color = :black, linestyle = :dash)
    lines!(ax2, zgrid, ediff(zτ(τh)), color = :seagreen, linewidth = 2.5, label = "τ̂")
    lines!(ax2, zgrid, ediff(zτ(max(0.5, τh / 2))), color = :dodgerblue, linestyle = :dash,
           linewidth = 1.5, label = "τ̂/2 (under)")
    lines!(ax2, zgrid, ediff(zτ(1.5τh)), color = :crimson, linestyle = :dot,
           linewidth = 1.5, label = "1.5·τ̂ (over)")
    axislegend(ax2, position = :rt, framevisible = false, labelsize = 10)

    # Panel 3 — Q-Q vs Rayleigh(1) at τ̂ + upper-tail inset
    ax3 = Axis(fig[2, 1], title = "3. Q-Q vs Rayleigh(1) at τ̂ (tails = non-Gaussian flag)",
               xlabel = "Rayleigh(1) quantile", ylabel = "sample quantile")
    qs = collect(0.02:0.02:0.98); theo = ray1_q.(qs); zh = zτ(τh)
    lines!(ax3, theo, theo, color = :black, linestyle = :dash)
    scatter!(ax3, theo, [quantile(zh, q) for q in qs], color = :seagreen, markersize = 5)
    ax3i = Axis(fig[2, 1], width = Relative(0.34), height = Relative(0.34),
                halign = 0.12, valign = 0.9, title = "upper tail", titlesize = 9,
                xticklabelsize = 7, yticklabelsize = 7)
    qt = collect(0.90:0.005:0.995); tt = ray1_q.(qt)
    lines!(ax3i, tt, tt, color = :black, linestyle = :dash)
    scatter!(ax3i, tt, [quantile(zh, q) for q in qt], color = :seagreen, markersize = 4)

    # Panel 4 — σ-stratified ⟨z²⟩/2 (=1 at correct τ; high-σ dip flagged, not hidden)
    ax4 = Axis(fig[2, 2], title = "4. σ-stratification: ⟨z²⟩/2 by σ-bin (=1 at correct τ)",
               xlabel = "σ-pair bin (low → high)", ylabel = "⟨z²⟩/2")
    hlines!(ax4, [1.0], color = :black, linestyle = :dash, label = "target = 1")
    function strat(t)
        q = quantile(s2, range(0, 1, length = 5))
        [ (idx = [i for i in eachindex(s2) if q[k] <= s2[i] <= q[k + 1]];
           isempty(idx) ? NaN :
           sum(abs2, [d[i] / sqrt(s2[i] + 2t^2) for i in idx]) / (2 * length(idx)))
          for k in 1:4 ]
    end
    v = strat(τh)
    lines!(ax4, 1:4, v, color = :seagreen, linewidth = 2, label = "τ̂")
    scatter!(ax4, 1:4, v, color = :seagreen)
    fin = filter(isfinite, v)
    isempty(fin) || text!(ax4, 4, v[end], text = "4-bin mean = $(round(mean(fin), digits=2))",
                          align = (:right, :top), fontsize = 9, color = :gray30)
    axislegend(ax4, position = :lb, framevisible = false, labelsize = 10)

    out = joinpath(output_dir, "se_adjust_diagnostics.png")
    save(out, fig, px_per_unit = 2)
    println("Saved: $out")
    out
end

end # module

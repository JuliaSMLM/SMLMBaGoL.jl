module BaGoLMakieExt

using SMLMBaGoL
using CairoMakie
using Statistics
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

    if report.has_gt
        _plot_k_recovery(report, output_dir)
        if !isnan(report.calibration.scale_factor)
            _plot_calibration(report, output_dir)
        end
    end
end

function _plot_partition_k(report, output_dir)
    pk = report.partition_k
    isempty(pk) && return
    fig = Figure(size=(500, 350))
    ax = Axis(fig[1, 1], xlabel="Emitters found per partition", ylabel="Count",
              title="Per-partition K ($(length(pk)) partitions, $(sum(pk)) emitters total)")
    hist!(ax, Float64.(pk); bins=max(1, maximum(pk) - minimum(pk) + 1), color=:steelblue)
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
    # Use empirical counts (from track_id) if available, else BaGoL cluster sizes
    has_empirical = !isempty(report.empirical_counts)
    cs = has_empirical ? report.empirical_counts : report.cluster_sizes
    isempty(cs) && return

    fig = Figure(size=(500, 350))
    ax = Axis(fig[1, 1], xlabel="Localizations per emitter", ylabel="Probability",
              title="Count distribution (locs/emitter)")

    # Histogram (normalized to probability)
    k_max = maximum(cs) + 5
    k_range = 1:k_max
    hist_data = zeros(k_max)
    for c in cs
        1 <= c <= k_max && (hist_data[c] += 1)
    end
    hist_data ./= sum(hist_data)
    lbl = has_empirical ? "Data (track_id)" : "BaGoL clusters"
    barplot!(ax, collect(k_range), hist_data; color=(:steelblue, 0.6), label=lbl)

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
    fig = Figure(size=(500, 350))
    ax = Axis(fig[1, 1], xlabel="Nearest-neighbor distance (nm)", ylabel="Count",
              title="NN distances between emitters")
    hist!(ax, dists; bins=30, color=:steelblue)
    vlines!(ax, [median(dists)]; color=:red, linewidth=2, linestyle=:dash,
            label="Median = $(round(median(dists), digits=1)) nm")
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
    hist!(ax, Float64.(pk); bins=max(1, maximum(pk) - minimum(pk) + 1), color=:steelblue)
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

end # module

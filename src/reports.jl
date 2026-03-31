# Standard report computation and output
#
# compute_report() returns a NamedTuple with all metrics.
# write_report() writes machine-readable outputs (CSV, JSON, TXT, PNG).
# Plotting is handled by extensions (BaGoLMakieExt, BaGoLRenderExt).

# ============================================================================
# FOV computation (shared by posterior image, renders, and reports)
# ============================================================================

"""
    compute_fov(smld; margin=nothing) -> (x_min, x_max, y_min, y_max)

Compute standard field of view from localizations. Bounds cover all
localization 1σ circles plus `margin` (μm) on each side.

If `margin` is not specified, defaults to 5× median σ.

Use this to set consistent bounds for `run_bagol(; posterior_xlim, posterior_ylim)`
and `render_report(; fov=...)`.
"""
function compute_fov(smld::SMLMData.SMLD; margin::Union{Nothing, Real} = nothing)
    emitters = smld.emitters
    x_min = minimum(e.x - e.σ_x for e in emitters)
    x_max = maximum(e.x + e.σ_x for e in emitters)
    y_min = minimum(e.y - e.σ_y for e in emitters)
    y_max = maximum(e.y + e.σ_y for e in emitters)

    pad = if margin !== nothing
        Float64(margin)
    else
        5 * median(mean_sigma(e) for e in emitters)
    end
    pad = max(pad, 0.010)  # minimum 10 nm

    return (x_min - pad, x_max + pad, y_min - pad, y_max + pad)
end

# ============================================================================
# Extension stubs (implemented by BaGoLMakieExt / BaGoLRenderExt)
# ============================================================================

"""
    plot_report(report; output_dir="output")

Plot standard report figures. Requires `using CairoMakie`.
"""
function plot_report end

"""
    plot_sweep(sweep; output_dir="output")

Plot optimality sweep results. Requires `using CairoMakie`.
"""
function plot_sweep end

"""
    plot_speed(speed; output_dir="output")

Plot speed test results. Requires `using CairoMakie`.
"""
function plot_speed end

"""
    render_report(locs_smld, bagol_smld; output_dir="output", true_positions=...)

Render SMLMRender visualization suite. Requires `using SMLMRender`.
"""
function render_report end

# ============================================================================
# Position matching (Hungarian algorithm)
# ============================================================================

"""
    match_positions(estimated, true_positions; threshold=0.020) -> NamedTuple

Match estimated emitters to true positions using Hungarian algorithm.
Threshold in μm (default 20 nm).

Returns `(assignments, matched_distances, cost_matrix)` where
`assignments[i]` = matched true index for estimated emitter i (0 if unmatched).
"""
function match_positions(
    estimated::Vector{<:SMLMData.AbstractEmitter},
    true_positions::Vector{Tuple{Float64, Float64}};
    threshold::Float64 = 0.020
)
    n_est = length(estimated)
    n_true = length(true_positions)

    if n_est == 0 || n_true == 0
        return (assignments=zeros(Int, n_est), matched_distances=Float64[],
                cost_matrix=zeros(0, 0))
    end

    cost = zeros(n_est, n_true)
    for i in 1:n_est, j in 1:n_true
        dx = estimated[i].x - true_positions[j][1]
        dy = estimated[i].y - true_positions[j][2]
        cost[i, j] = sqrt(dx^2 + dy^2)
    end

    assignment, _ = Hungarian.hungarian(cost)

    assignments = zeros(Int, n_est)
    matched_distances = Float64[]
    for (i, j) in enumerate(assignment)
        if 0 < j <= n_true && cost[i, j] <= threshold
            assignments[i] = j
            push!(matched_distances, cost[i, j])
        end
    end

    return (assignments=assignments, matched_distances=matched_distances,
            cost_matrix=cost)
end

# ============================================================================
# Calibration
# ============================================================================

function _compute_calibration(
    emitters::Vector{<:SMLMData.AbstractEmitter},
    true_positions::Vector{Tuple{Float64, Float64}},
    assignments::Vector{Int}
)
    mahal_d2 = Float64[]
    for (i, j) in enumerate(assignments)
        j == 0 && continue
        e = emitters[i]
        tx, ty = true_positions[j]
        dx = e.x - tx
        dy = e.y - ty
        σ_xx = e.σ_x^2
        σ_yy = e.σ_y^2
        σ_xy = e.σ_xy  # covariance term
        det = σ_xx * σ_yy - σ_xy^2
        det > 0 || continue
        d2 = (σ_yy * dx^2 - 2 * σ_xy * dx * dy + σ_xx * dy^2) / det
        push!(mahal_d2, d2)
    end

    if isempty(mahal_d2)
        return (mahal_d2=Float64[], scale_factor=NaN,
                coverage_1σ=NaN, coverage_2σ=NaN, coverage_3σ=NaN,
                expected_1σ=1 - exp(-0.5), expected_2σ=1 - exp(-2.0),
                expected_3σ=1 - exp(-4.5))
    end

    scale_factor = sqrt(mean(mahal_d2) / 2)  # ~1.0 for χ²(2)
    n = length(mahal_d2)
    # 2D Gaussian contours: P(d² ≤ r²) = 1 - exp(-r²/2) for χ²(2)
    coverage_1σ = count(d -> d ≤ 1.0, mahal_d2) / n
    coverage_2σ = count(d -> d ≤ 4.0, mahal_d2) / n
    coverage_3σ = count(d -> d ≤ 9.0, mahal_d2) / n

    return (mahal_d2=mahal_d2, scale_factor=scale_factor,
            coverage_1σ=coverage_1σ, coverage_2σ=coverage_2σ, coverage_3σ=coverage_3σ,
            expected_1σ=1 - exp(-0.5), expected_2σ=1 - exp(-2.0),
            expected_3σ=1 - exp(-4.5))
end

# ============================================================================
# Oracle RMSE (best achievable from true assignments)
# ============================================================================

function _oracle_rmse(locs::Vector{<:SMLMData.AbstractEmitter},
                      true_positions::Vector{Tuple{Float64, Float64}};
                      threshold::Float64 = 0.020)
    oracle_assignments = Int16[loc.track_id for loc in locs]
    any(a != 0 for a in oracle_assignments) || return NaN

    oracle_emitters = _emitters_from_assignments(oracle_assignments, locs)
    m = match_positions(oracle_emitters, true_positions; threshold)
    isempty(m.matched_distances) && return NaN
    return sqrt(mean(m.matched_distances .^ 2))
end

# ============================================================================
# NN distances from result emitters
# ============================================================================

function _nn_distances(emitters::Vector{<:SMLMData.AbstractEmitter})
    n = length(emitters)
    n < 2 && return Float64[]
    dists = Float64[]
    for i in 1:n
        min_d = Inf
        for j in 1:n
            i == j && continue
            dx = emitters[i].x - emitters[j].x
            dy = emitters[i].y - emitters[j].y
            d = sqrt(dx^2 + dy^2)
            d < min_d && (min_d = d)
        end
        push!(dists, min_d)
    end
    return dists
end

function _nn_distances(positions::Vector{Tuple{Float64, Float64}})
    n = length(positions)
    n < 2 && return Float64[]
    dists = Float64[]
    for i in 1:n
        min_d = Inf
        for j in 1:n
            i == j && continue
            dx = positions[i][1] - positions[j][1]
            dy = positions[i][2] - positions[j][2]
            d = sqrt(dx^2 + dy^2)
            d < min_d && (min_d = d)
        end
        push!(dists, min_d)
    end
    return dists
end

# ============================================================================
# compute_report
# ============================================================================

"""
    compute_report(result_smld, diagnostics; true_positions=nothing, locs_smld=nothing)

Compute standard analysis report from BaGoL results.

# Category 1 (no GT): always computed
Returns metrics for convergence assessment, cluster statistics, and emitter output.

# Category 2 (with GT): when `true_positions` is provided
Adds matching metrics (Jaccard, RMSE, F1), calibration, and oracle comparison.

# Returns
NamedTuple with all computed metrics. Pass to `write_report()` for disk output,
`plot_report()` (CairoMakie extension) for figures, or `render_report()` (SMLMRender
extension) for spatial visualizations.
"""
function compute_report(
    result_smld::SMLMData.SMLD,
    diagnostics::BaGoLDiagnostics;
    true_positions::Union{Nothing, Vector{Tuple{Float64, Float64}}} = nothing,
    locs_smld::Union{Nothing, SMLMData.SMLD} = nothing,
    count_params::Union{Nothing, NamedTuple} = nothing
)
    emitters = result_smld.emitters
    n_emitters = diagnostics.n_emitters
    n_locs = locs_smld !== nothing ? length(locs_smld.emitters) : 0

    nn_dists = _nn_distances(emitters)

    # Empirical locs/emitter from track_id (simulation GT)
    empirical_counts = if locs_smld !== nothing && any(e.track_id != 0 for e in locs_smld.emitters)
        counts = Dict{Int, Int}()
        for e in locs_smld.emitters
            e.track_id == 0 && continue
            counts[e.track_id] = get(counts, e.track_id, 0) + 1
        end
        sort(collect(values(counts)))
    else
        Int[]
    end

    base = (
        n_locs = n_locs,
        n_emitters = n_emitters,
        grouping_ratio = n_locs > 0 ? n_locs / max(n_emitters, 1) : NaN,
        final_mu = diagnostics.final_μ,
        final_shape = diagnostics.final_shape,
        final_rho = diagnostics.final_ρ,
        n_partitions = diagnostics.n_partitions,
        acceptance_rates = diagnostics.acceptance_rates,
        posterior_k = diagnostics.posterior_k,
        partition_k = diagnostics.partition_k,
        partition_ids = diagnostics.partition_ids,
        cluster_sizes = diagnostics.cluster_sizes,
        empirical_counts = empirical_counts,
        posterior_image = diagnostics.posterior_image,
        nn_distances = nn_dists,
        true_count_params = count_params,
        emitters = emitters,
        has_gt = true_positions !== nothing,
    )

    true_positions === nothing && return base

    # --- Category 2: Ground truth comparison ---
    n_true = length(true_positions)
    m = match_positions(emitters, true_positions)
    n_matched = length(m.matched_distances)

    jaccard = n_matched / max(n_emitters + n_true - n_matched, 1)
    precision = n_emitters > 0 ? n_matched / n_emitters : 0.0
    recall = n_true > 0 ? n_matched / n_true : 0.0
    f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0.0
    rmse = isempty(m.matched_distances) ? NaN : sqrt(mean(m.matched_distances .^ 2))

    calibration = _compute_calibration(emitters, true_positions, m.assignments)

    oracle_rmse = if locs_smld !== nothing
        _oracle_rmse(locs_smld.emitters, true_positions)
    else
        NaN
    end

    gt_nn_dists = _nn_distances(true_positions)

    gt_fields = (
        k_true = n_true,
        n_matched = n_matched,
        jaccard = jaccard,
        precision = precision,
        recall = recall,
        f1 = f1,
        rmse = rmse,
        rmse_oracle = oracle_rmse,
        calibration = calibration,
        gt_nn_distances = gt_nn_dists,
    )

    return merge(base, gt_fields)
end

# ============================================================================
# write_report — disk output (no plotting deps)
# ============================================================================

"""
    write_report(report; output_dir="output")

Write standard report files to disk. No plotting dependencies required.

Creates:
- `summary.txt` — human-readable summary
- `emitters.csv` — emitter positions and uncertainties
- `metrics.json` — machine-readable metrics (if GT available)
- `posterior_image.png` — posterior density image (if available)
"""
function write_report(report; output_dir::String = "output")
    mkpath(output_dir)
    _write_summary(report, joinpath(output_dir, "summary.txt"))
    _write_emitters_csv(report, joinpath(output_dir, "emitters.csv"))
    if report.has_gt
        _write_metrics_json(report, joinpath(output_dir, "metrics.json"))
    end
    if report.posterior_image !== nothing
        save_posterior_png(joinpath(output_dir, "posterior_image.png"),
                          report.posterior_image; colormap=:inferno)
    end
end

function _write_summary(report, path)
    open(path, "w") do io
        println(io, "SMLMBaGoL Analysis Summary")
        println(io, "=" ^ 40)
        println(io)
        println(io, "Localizations:    $(report.n_locs)")
        println(io, "Emitters found:   $(report.n_emitters)")
        if report.n_locs > 0
            println(io, "Grouping ratio:   $(round(report.grouping_ratio, digits=1))")
        end
        println(io, "Partitions:       $(report.n_partitions)")
        if !isempty(report.partition_k)
            pk = report.partition_k
            println(io, "  K/partition:    median=$(round(median(pk), digits=1)), " *
                        "range=$(minimum(pk))-$(maximum(pk))")
        end
        println(io)
        println(io, "Learned parameters:")
        println(io, "  μ (mean count):  $(round(report.final_mu, digits=2))")
        println(io, "  shape:           $(round(report.final_shape, digits=2))")
        println(io)
        println(io, "Acceptance rates:")
        for (move, rate) in sort(collect(report.acceptance_rates))
            println(io, "  $(rpad(move, 14)) $(round(rate * 100, digits=1))%")
        end
        if !isempty(report.nn_distances)
            med_nn = median(report.nn_distances) * 1000  # nm
            println(io)
            println(io, "Median NN distance: $(round(med_nn, digits=1)) nm")
            if hasproperty(report, :gt_nn_distances) && !isempty(report.gt_nn_distances)
                gt_med = median(report.gt_nn_distances) * 1000
                println(io, "GT median NN dist:  $(round(gt_med, digits=1)) nm")
            end
        end
        if report.has_gt
            println(io)
            println(io, "Ground Truth Comparison")
            println(io, "-" ^ 40)
            println(io, "True emitters:    $(report.k_true)")
            println(io, "Matched:          $(report.n_matched)")
            println(io, "Jaccard:          $(round(report.jaccard, digits=3))")
            println(io, "Precision:        $(round(report.precision, digits=3))")
            println(io, "Recall:           $(round(report.recall, digits=3))")
            println(io, "F1:               $(round(report.f1, digits=3))")
            println(io, "RMSE:             $(round(report.rmse * 1000, digits=1)) nm")
            if !isnan(report.rmse_oracle)
                println(io, "Oracle RMSE:      $(round(report.rmse_oracle * 1000, digits=1)) nm")
                ratio = report.rmse / report.rmse_oracle
                println(io, "RMSE/Oracle:      $(round(ratio, digits=2))")
            end
            c = report.calibration
            if !isnan(c.scale_factor)
                println(io)
                println(io, "Calibration:")
                println(io, "  Scale factor:   $(round(c.scale_factor, digits=2)) (ideal: 1.0)")
                println(io, "  Coverage 1σ:    $(round(c.coverage_1σ * 100, digits=1))% (expected: $(round(c.expected_1σ * 100, digits=1))%)")
                println(io, "  Coverage 2σ:    $(round(c.coverage_2σ * 100, digits=1))% (expected: $(round(c.expected_2σ * 100, digits=1))%)")
                println(io, "  Coverage 3σ:    $(round(c.coverage_3σ * 100, digits=1))% (expected: $(round(c.expected_3σ * 100, digits=1))%)")
            end
        end
    end
    println("Saved: $path")
end

function _write_emitters_csv(report, path)
    open(path, "w") do io
        println(io, "x_um,y_um,sigma_x_um,sigma_y_um,sigma_xy,photons")
        for e in report.emitters
            println(io, "$(e.x),$(e.y),$(e.σ_x),$(e.σ_y),$(e.σ_xy),$(e.photons)")
        end
    end
    println("Saved: $path ($(length(report.emitters)) emitters)")
end

function _write_metrics_json(report, path)
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"n_locs\": $(report.n_locs),")
        println(io, "  \"n_emitters\": $(report.n_emitters),")
        println(io, "  \"k_true\": $(report.k_true),")
        println(io, "  \"n_matched\": $(report.n_matched),")
        println(io, "  \"jaccard\": $(report.jaccard),")
        println(io, "  \"precision\": $(report.precision),")
        println(io, "  \"recall\": $(report.recall),")
        println(io, "  \"f1\": $(report.f1),")
        println(io, "  \"rmse_um\": $(report.rmse),")
        println(io, "  \"rmse_nm\": $(round(report.rmse * 1000, digits=2)),")
        println(io, "  \"rmse_oracle_nm\": $(isnan(report.rmse_oracle) ? "null" : round(report.rmse_oracle * 1000, digits=2)),")
        println(io, "  \"final_mu\": $(round(report.final_mu, digits=3)),")
        println(io, "  \"final_shape\": $(round(report.final_shape, digits=3)),")
        println(io, "  \"n_partitions\": $(report.n_partitions),")
        c = report.calibration
        println(io, "  \"calibration_scale_factor\": $(isnan(c.scale_factor) ? "null" : round(c.scale_factor, digits=3)),")
        println(io, "  \"coverage_1sigma\": $(isnan(c.coverage_1σ) ? "null" : round(c.coverage_1σ, digits=3)),")
        println(io, "  \"coverage_2sigma\": $(isnan(c.coverage_2σ) ? "null" : round(c.coverage_2σ, digits=3)),")
        println(io, "  \"coverage_3sigma\": $(isnan(c.coverage_3σ) ? "null" : round(c.coverage_3σ, digits=3)),")
        ar = report.acceptance_rates
        println(io, "  \"acceptance_gibbs\": $(round(get(ar, :gibbs_sweep, 0.0), digits=3)),")
        println(io, "  \"acceptance_split\": $(round(get(ar, :split, 0.0), digits=3)),")
        println(io, "  \"acceptance_merge\": $(round(get(ar, :merge, 0.0), digits=3)),")
        println(io, "  \"acceptance_birth\": $(round(get(ar, :birth, 0.0), digits=3)),")
        println(io, "  \"acceptance_death\": $(round(get(ar, :death, 0.0), digits=3))")
        println(io, "}")
    end
    println("Saved: $path")
end

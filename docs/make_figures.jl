# docs/make_figures.jl
# =====================
# Regenerate the documentation figures into docs/src/assets/.
#
# Run from the repo root under the EXAMPLES project (it carries SMLMSim / SMLMRender /
# CairoMakie, which the lightweight docs project deliberately does not):
#
#     julia --threads=auto --project=examples docs/make_figures.jl
#
# The PNGs are committed (see the docs/src/assets/ exception in .gitignore) so the docs CI
# build stays fast and free of the heavy sim/render deps. Re-run this after changing the
# model or the example pipeline.
#
# Each figure is wrapped in `figure(...)`, which logs and continues on error, so one broken
# figure never aborts the whole run. Figures are tagged by source in the doc placeholders:
#   [pipeline]   — harvested from render_report / plot_report
#   [data-plot]  — a Makie plot of an object the sampler already computes (PSM, locmix grid)
#   [schematic]  — a drawn conceptual diagram (TODO stubs below)

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "examples"))

using SMLMBaGoL
using SMLMData
using SMLMSim
using SMLMRender
using CairoMakie
using Statistics
using Random

const ASSETS = joinpath(@__DIR__, "src", "assets")
mkpath(ASSETS)
Random.seed!(20260625)

"Run figure `name`; log and continue on failure (so one bad figure can't abort the run)."
function figure(name::AbstractString, f)
    try
        @info "building figure" name
        f()
    catch err
        @warn "figure FAILED (skipped)" name exception = (err, catch_backtrace())
    end
end

"Copy a pipeline output PNG into the assets dir under its documentation filename."
harvest(src, dst) = isfile(src) ? cp(src, joinpath(ASSETS, dst); force = true) :
                    @warn("expected pipeline output missing", src)

# =============================================================================
# 1. Pipeline — single N-mer  (intro, collapsed, mapn, hierarchical, priors, moves)
# =============================================================================
function pipeline_single()
    sim = simulate_nmer(; n = 6, diameter = 0.025, mean_count = 10.0, psf_sigma = 0.130,
                          mean_photons = 500.0, min_photons = 100.0,
                          pixel_size = 0.100, field_size = 25.6)
    fov = compute_fov(sim.smld)
    result, diag = run_bagol(sim.smld; n_iterations = 20000, burn_in = 4000,
                             partition_sigma = Inf, posterior_pixel_size = 0.001,
                             posterior_xlim = (fov[1], fov[2]),
                             posterior_ylim = (fov[3], fov[4]))

    out = mktempdir()
    report = compute_report(result, diag; true_positions = sim.true_positions,
                            locs_smld = sim.smld, count_params = sim.count_params)
    plot_report(report; output_dir = out)
    render_report(sim.smld, result; output_dir = out, true_positions = sim.true_positions,
                  partition_ids = report.partition_ids, fov = fov, prefix = "render")

    # Harvest the individual pipeline PNGs to their documentation names.
    harvest(joinpath(out, "render_sr_gaussian.png"),         "intro_pre.png")
    harvest(joinpath(out, "render_mapn_gaussian.png"),       "intro_post.png")
    harvest(joinpath(out, "render_mapn_gaussian.png"),       "results_posterior.png")
    harvest(joinpath(out, "render_circles.png"),             "collapsed_cluster.png")
    harvest(joinpath(out, "render_circles_groundtruth.png"), "mapn_ellipses.png")
    harvest(joinpath(out, "render_circles_groundtruth.png"), "results_groundtruth.png")
    harvest(joinpath(out, "convergence.png"),                "hier_convergence.png")
    harvest(joinpath(out, "count_distribution.png"),         "priors_negbin.png")
    harvest(joinpath(out, "count_distribution.png"),         "hier_countdist.png")
    harvest(joinpath(out, "acceptance_rates.png"),           "moves_acceptance.png")

    return (; sim, result, diag, report, fov)
end

# =============================================================================
# 2. Pipeline — grid of N-mers  (partitioning)
# =============================================================================
function pipeline_grid()
    # TODO: verify kwargs against examples/nmer_grid_test.jl on first run.
    sim = simulate_nmer_grid(; n = 6, n_per_side = 3, spacing = 0.30, diameter = 0.025,
                               mean_count = 10.0, psf_sigma = 0.130,
                               mean_photons = 500.0, min_photons = 100.0,
                               pixel_size = 0.100, field_size = 25.6)
    fov = compute_fov(sim.smld)
    result, diag = run_bagol(sim.smld; n_iterations = 8000, burn_in = 2000,
                             partition_sigma = 3.0)   # finite → real partitioning
    out = mktempdir()
    report = compute_report(result, diag; true_positions = sim.true_positions,
                            locs_smld = sim.smld, count_params = sim.count_params)
    render_report(sim.smld, result; output_dir = out, true_positions = sim.true_positions,
                  partition_ids = report.partition_ids, fov = fov, prefix = "grid")
    harvest(joinpath(out, "grid_partitions.png"), "partition_field.png")
    harvest(joinpath(out, "grid_partitions.png"), "results_partitions.png")
end

# =============================================================================
# 3. Data-plots — objects the sampler already computes
# =============================================================================

"PSM (co-assignment) heatmap from a short chain with a PSMAccumulator. [data-plot]"
function fig_psm(sim)
    psm_acc = PSMAccumulator()
    res = run_collapsed_chain(sim.smld.emitters; n_iterations = 6000, burn_in = 2000,
                              accumulators = AbstractAccumulator[psm_acc])
    C = accumulator_result(psm_acc, res)          # T×T co-assignment matrix in [0,1]
    fig = Figure(size = (480, 420))
    ax = Axis(fig[1, 1], title = "Posterior similarity matrix",
              xlabel = "localization", ylabel = "localization", aspect = 1)
    hm = heatmap!(ax, C; colormap = :viridis, colorrange = (0, 1))
    Colorbar(fig[1, 2], hm, label = "co-assignment frequency")
    save(joinpath(ASSETS, "mapn_psm.png"), fig; px_per_unit = 2)
end

"Localization-mixture prior density heatmap over the loc field. [data-plot]"
function fig_locmix_grid(sim)
    # TODO: build the grid via SMLMBaGoL.build_locmix_grid on sim.smld.emitters and
    # evaluate log_prior_locmix over a mesh; render as a heatmap with the locs overlaid.
    # Also produce marginal_flat_vs_locmix.png (constant sheet vs data-adaptive density).
    @info "TODO: fig_locmix_grid → assets/marginal_locmix_grid.png, marginal_flat_vs_locmix.png"
end

# =============================================================================
# 4. Schematics — drawn conceptual diagrams (Makie). TODO stubs.
# =============================================================================
# Each is a small hand-built Makie figure; implement as the docs mature. Keeping them as
# logged stubs means the script runs clean and reports exactly what is still to draw.
fig_generative()    = @info "TODO: schematic → assets/intro_generative.png (emitter → loc scatter)"
fig_precision()     = @info "TODO: schematic → assets/collapsed_precision.png (σ-weighted mean)"
fig_allocation()    = @info "TODO: schematic → assets/priors_allocation.png (K=1,2,3 under DM)"
fig_splitmerge()    = @info "TODO: schematic → assets/moves_splitmerge.png"
fig_birthdeath()    = @info "TODO: schematic → assets/moves_birthdeath.png"
fig_dedup()         = @info "TODO: schematic → assets/partition_dedup.png"

# =============================================================================
# 5. Composites & se_adjust — TODO (need image assembly / a τ-finder run)
# =============================================================================
fig_intro_prepost() = @info "TODO: composite intro_pre|intro_post → assets/intro_prepost.png, results_prepost.png"
fig_diagnostics()   = @info "TODO: 2×2 panel of plot_report outputs → assets/results_diagnostics.png"
fig_se_adjust()     = @info "TODO: plot_se_adjust → assets/se_cdf.png; τ=0 vs τ̂ renders → assets/se_prepost.png"

# =============================================================================
# Driver
# =============================================================================
function main()
    local single
    figure("pipeline_single", () -> (single = pipeline_single()))
    figure("pipeline_grid",   pipeline_grid)
    if @isdefined(single) && single !== nothing
        figure("psm",          () -> fig_psm(single.sim))
        figure("locmix_grid",  () -> fig_locmix_grid(single.sim))
    end
    for f in (fig_generative, fig_precision, fig_allocation, fig_splitmerge,
              fig_birthdeath, fig_dedup, fig_intro_prepost, fig_diagnostics, fig_se_adjust)
        figure(string(f), f)
    end
    @info "done — figures written" dir = ASSETS
end

main()

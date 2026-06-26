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
# build stays fast and free of the heavy sim/render deps. Re-run after changing the model.
#
# Each figure is wrapped in `figure(...)`, which logs and continues on error, so one broken
# figure never aborts the run. Sources are tagged in the doc placeholders:
#   [pipeline]  — SMLMRender Gaussian render of the analysis result
#   [data-plot] — Makie plot of an object the sampler computes (traces, PSM, counts)
#   [schematic] — a drawn conceptual diagram

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "examples"))

using SMLMBaGoL
using SMLMData
using SMLMSim
using SMLMRender
using CairoMakie
using Statistics
using LinearAlgebra
using Random

const ASSETS = joinpath(@__DIR__, "src", "assets")
mkpath(ASSETS)
Random.seed!(20260625)
CairoMakie.activate!(type = "png", px_per_unit = 2)

asset(name) = joinpath(ASSETS, name)

"Run figure `name`; log and continue on failure (so one bad figure can't abort the run)."
function figure(name::AbstractString, f)
    try
        @info "building figure" name
        f()
    catch err
        @warn "figure FAILED (skipped)" name exception = (err, catch_backtrace())
    end
end

# Theme: clean, readable, consistent across the doc figures.
set_theme!(Theme(fontsize = 15, Axis = (; xgridvisible = false, ygridvisible = false)))

# -----------------------------------------------------------------------------
# small geometry helpers
# -----------------------------------------------------------------------------
"Points tracing the nσ covariance ellipse of an emitter (σ_x,σ_y std; σ_xy covariance)."
function cov_ellipse(e; nσ = 1.0, npts = 72)
    Σ = [e.σ_x^2 e.σ_xy; e.σ_xy e.σ_y^2]
    vals, vecs = eigen(Symmetric(Σ))
    vals = max.(vals, 0.0)
    t = range(0, 2π; length = npts)
    M = vecs * Diagonal(nσ .* sqrt.(vals))
    xs = similar(t); ys = similar(t)
    for (i, θ) in enumerate(t)
        v = M * [cos(θ), sin(θ)]
        xs[i] = e.x + v[1]; ys[i] = e.y + v[2]
    end
    return xs, ys
end

"Draw a horizontal scale bar of `len_um` μm near the lower-left of the data limits."
function scalebar!(ax, xlo, xhi, ylo, yhi, len_um; color = :white, label = "")
    mx = xlo + 0.06 * (xhi - xlo); my = ylo + 0.08 * (yhi - ylo)
    lines!(ax, [mx, mx + len_um], [my, my]; color = color, linewidth = 5)
    isempty(label) || text!(ax, mx + len_um / 2, my; text = label,
                            align = (:center, :bottom), color = color, fontsize = 13, offset = (0, 3))
end

# =============================================================================
# Pipeline data (computed once, reused across figures)
# =============================================================================
function run_single()
    # Resolvable hexamer: 60 nm across, ~6 nm localization precision ⇒ d/σ ≫ 3 (well into the
    # "must split" regime), so MAP-N recovers K and the count learner is unbiased.
    sim = simulate_nmer(; n = 6, diameter = 0.060, mean_count = 18.0, psf_sigma = 0.130,
                          mean_photons = 500.0, min_photons = 100.0,
                          pixel_size = 0.100, field_size = 4.0)
    fov = compute_fov(sim.smld)
    result, diag = run_bagol(sim.smld; n_iterations = 20000, burn_in = 4000,
                             partition_sigma = Inf, posterior_pixel_size = 0.001,
                             posterior_xlim = (fov[1], fov[2]),
                             posterior_ylim = (fov[3], fov[4]))
    return (; sim, result, diag, fov)
end

function run_grid()
    # A small grid of resolvable hexamers — feeds the partition-coloring figure (kept small
    # so the per-partition colors are distinguishable).
    sim = simulate_nmer_grid(; n_per_cluster = 6, cluster_diameter = 0.060,
                               grid_nx = 4, grid_ny = 4, grid_spacing = 0.5,
                               mean_count = 18.0, psf_sigma = 0.130,
                               mean_photons = 500.0, min_photons = 100.0,
                               pixel_size = 0.100, field_size = 3.0)
    fov = compute_fov(sim.smld)
    result, diag = run_bagol(sim.smld; n_iterations = 8000, burn_in = 2000, partition_sigma = 3.0)
    return (; sim, result, diag, fov)
end

function run_bigcount()
    # Realistic-scale field (20×20 hexamers ≈ 2400 emitters) so the hierarchical learner has
    # the statistics to recover the count distribution closely. Drives the convergence and
    # learned-vs-true count figures. Tiny partitions run fast in parallel (~30 s on 32 threads).
    sim = simulate_nmer_grid(; n_per_cluster = 6, cluster_diameter = 0.060,
                               grid_nx = 20, grid_ny = 20, grid_spacing = 0.5,
                               mean_count = 18.0, psf_sigma = 0.130,
                               mean_photons = 500.0, min_photons = 100.0,
                               pixel_size = 0.100, field_size = 11.0)
    fov = compute_fov(sim.smld)
    result, diag = run_bagol(sim.smld; n_iterations = 12000, burn_in = 3000, partition_sigma = 3.0)
    return (; sim, result, diag, fov)
end

# =============================================================================
# 1. Hero pair — high-resolution Gaussian renders (raw vs MAP-N)   [pipeline]
# =============================================================================
function fig_hero(S)
    span_um = max(S.fov[2] - S.fov[1], S.fov[4] - S.fov[3])
    px_nm = span_um * 1000 / 700                      # → ~700 px across the field
    out = mktempdir()
    render_report(S.sim.smld, S.result; output_dir = out, fov = S.fov,
                  pixel_size = px_nm, prefix = "h")
    cp(joinpath(out, "h_sr_gaussian.png"),   asset("intro_pre.png");        force = true)
    cp(joinpath(out, "h_mapn_gaussian.png"), asset("intro_post.png");       force = true)

    # labeled two-panel composite (raw | MAP-N) from render arrays
    x0, x1, y0, y1 = S.fov
    w = max(10, ceil(Int, (x1 - x0) * 1000 / px_nm))
    h = max(10, ceil(Int, (y1 - y0) * 1000 / px_nm))
    tgt = SMLMRender.Image2DTarget(w, h, px_nm, (x0, x1), (y0, y1))
    (raw, _)  = SMLMRender.render(S.sim.smld; strategy = GaussianRender(), target = tgt, colormap = :inferno)
    (mapn, _) = SMLMRender.render(S.result;   strategy = GaussianRender(), target = tgt, colormap = :inferno)
    fig = Figure(size = (940, 500))
    Label(fig[0, 1:2], "Raw localizations → BaGoL MAP-N (same scale)", fontsize = 17, font = :bold)
    ax1 = Axis(fig[1, 1], title = "raw localizations", aspect = DataAspect())
    ax2 = Axis(fig[1, 2], title = "BaGoL MAP-N", aspect = DataAspect())
    image!(ax1, rotr90(raw)); image!(ax2, rotr90(mapn))
    for ax in (ax1, ax2); hidedecorations!(ax); end
    save(asset("intro_prepost.png"), fig)
end

# =============================================================================
# 2. MAP-N teaching plot — faded loc ellipses, bold posterior, GT points  [data-plot]
# =============================================================================
function fig_mapn_ellipses(S)
    locs = S.sim.smld.emitters
    em = S.result.emitters
    xs = [e.x for e in locs]; ys = [e.y for e in locs]
    pad = 0.04
    xlo, xhi = minimum(xs) - pad, maximum(xs) + pad
    ylo, yhi = minimum(ys) - pad, maximum(ys) + pad

    fig = Figure(size = (560, 520))
    ax = Axis(fig[1, 1], title = "MAP-N estimate vs. ground truth",
              aspect = DataAspect(), backgroundcolor = :black)
    for e in locs                                   # localization uncertainty, faint gray
        ex, ey = cov_ellipse(e; nσ = 1.0)
        poly!(ax, Point2f.(ex, ey); color = (:white, 0.05), strokecolor = (:white, 0.18),
              strokewidth = 0.6)
    end
    for (gx, gy) in S.sim.true_positions            # ground truth, haloed cyan points
        scatter!(ax, [gx], [gy]; color = :cyan, markersize = 13, strokecolor = :white, strokewidth = 1.5)
    end
    for e in em                                     # MAP-N posterior, bold red 2σ ellipse
        ex, ey = cov_ellipse(e; nσ = 2.0)
        lines!(ax, ex, ey; color = :red, linewidth = 2.5)
    end
    scatter!(ax, [NaN], [NaN]; color = :cyan, label = "ground truth", strokecolor = :white, strokewidth = 1.5)
    lines!(ax, [NaN], [NaN]; color = :red, linewidth = 2.5, label = "MAP-N posterior (2σ)")
    scatter!(ax, [NaN], [NaN]; color = (:white, 0.4), marker = :circle, markersize = 12,
             label = "localizations (1σ)")
    limits!(ax, xlo, xhi, ylo, yhi)
    scalebar!(ax, xlo, xhi, ylo, yhi, 0.05; label = "50 nm")
    hidedecorations!(ax)
    axislegend(ax; position = :rt, framevisible = false, labelcolor = :white, labelsize = 12)
    save(asset("mapn_ellipses.png"), fig)
end

# =============================================================================
# 3. Collapsed-cluster teaching plot — one emitter, loose locs vs tight posterior [data-plot]
# =============================================================================
function fig_collapsed_cluster()
    sim = simulate_nmer(; n = 1, diameter = 0.001, mean_count = 18.0, psf_sigma = 0.130,
                          mean_photons = 500.0, min_photons = 100.0,
                          pixel_size = 0.100, field_size = 3.0)
    result, _ = run_bagol(sim.smld; n_iterations = 12000, burn_in = 3000, partition_sigma = Inf)
    locs = sim.smld.emitters; em = result.emitters
    xs = [e.x for e in locs]; ys = [e.y for e in locs]
    cx, cy = mean(xs), mean(ys)
    mσ = maximum(e.σ_x for e in locs)
    half = maximum(abs.(vcat(xs .- cx, ys .- cy))) + 2.5 * mσ   # tight square crop
    fig = Figure(size = (520, 500))
    ax = Axis(fig[1, 1], title = "A collapsed cluster: localizations → posterior",
              aspect = DataAspect(), backgroundcolor = :black)
    for e in locs
        ex, ey = cov_ellipse(e; nσ = 1.0)
        poly!(ax, Point2f.(ex, ey); color = (:gray70, 0.12), strokecolor = (:gray80, 0.6), strokewidth = 1.0)
    end
    scatter!(ax, xs, ys; color = (:white, 0.7), markersize = 6)
    for (gx, gy) in sim.true_positions
        scatter!(ax, [gx], [gy]; color = :cyan, markersize = 13, strokecolor = :white, strokewidth = 1.5)
    end
    for e in em
        ex, ey = cov_ellipse(e; nσ = 2.0)
        poly!(ax, Point2f.(ex, ey); color = (:red, 0.3), strokecolor = :red, strokewidth = 3.0)
    end
    limits!(ax, cx - half, cx + half, cy - half, cy + half)
    scalebar!(ax, cx - half, cx + half, cy - half, cy + half, 0.02; label = "20 nm")
    hidedecorations!(ax)
    save(asset("collapsed_cluster.png"), fig)
end

# =============================================================================
# 4. Convergence trace — custom, with rho labelled dormant   [data-plot]
# =============================================================================
function fig_convergence(S)
    tr = S.diag.convergence_trace
    fig = Figure(size = (920, 640))
    Label(fig[0, 1:2], "Hierarchical hyperparameters across the chain", fontsize = 17, font = :bold)
    panels = [(1, 1, tr.K,     "emitter count K"),
              (1, 2, tr.mu,    "μ  (mean locs/emitter)"),
              (2, 1, tr.shape, "shape α"),
              (2, 2, tr.rho,   "ρ  — dormant under locmix")]
    for (r, c, v, lab) in panels
        ax = Axis(fig[r, c]; xlabel = "iteration", ylabel = lab, title = lab)
        scatterlines!(ax, tr.iters, float.(v); color = :steelblue, linewidth = 1.5, markersize = 5)
        vlines!(ax, [tr.burn_in]; color = :gray50, linestyle = :dash, label = "burn-in")
        (lab[1] == 'ρ') && text!(ax, 0.5, 0.5; space = :relative, text = "(flat-model only)",
                                 align = (:center, :center), color = :gray60, fontsize = 13)
    end
    save(asset("hier_convergence.png"), fig)
end

# =============================================================================
# 5. Count distribution — true vs learned NegBin with mean lines   [data-plot]
# =============================================================================
function negbin_pmf(kmax, μ, α)
    p = α / (α + μ); q = 1 - p
    pmf = zeros(kmax + 1); pmf[1] = p^α
    for k in 1:kmax
        pmf[k+1] = pmf[k] * (k - 1 + α) / k * q
    end
    return pmf
end

function fig_count(S)
    μt, αt = S.sim.count_params.μ, S.sim.count_params.shape
    μl, αl = S.diag.final_μ, S.diag.final_shape
    kmax = ceil(Int, max(μt, μl) * 2 + 6)
    ks = 0:kmax
    fig = Figure(size = (760, 480))
    ax = Axis(fig[1, 1], xlabel = "localizations per emitter", ylabel = "probability",
              title = "Count distribution: true vs. learned  ($(S.sim.n_emitters) emitters)")
    lines!(ax, collect(ks), negbin_pmf(kmax, μt, αt); color = :dodgerblue, linewidth = 3,
           label = "true  (μ=$(round(μt, digits=1)), α=$(round(αt, digits=1)))")
    lines!(ax, collect(ks), negbin_pmf(kmax, μl, αl); color = :crimson, linewidth = 3,
           label = "learned (μ=$(round(μl, digits=1)), α=$(round(αl, digits=1)))")
    vlines!(ax, [μt]; color = :dodgerblue, linestyle = :dash)
    vlines!(ax, [μl]; color = :crimson, linestyle = :dash)
    axislegend(ax; position = :rt, framevisible = false, labelsize = 13)
    save(asset("priors_negbin.png"), fig)
end

# =============================================================================
# 6. Acceptance rates — horizontal, K-changing moves only   [data-plot]
# =============================================================================
function fig_acceptance(S)
    ar = S.diag.acceptance_rates
    moves = [:split, :merge, :birth, :death]
    vals = [100 * get(ar, m, 0.0) for m in moves]
    fig = Figure(size = (640, 360))
    ax = Axis(fig[1, 1], xlabel = "acceptance rate (%)",
              yticks = (1:length(moves), string.(moves)),
              title = "K-changing move acceptance")
    barplot!(ax, 1:length(moves), vals; color = :steelblue, direction = :x)
    text!(ax, vals, 1:length(moves); text = string.(round.(vals, digits = 1), "%"),
          align = (:left, :center), offset = (4, 0), fontsize = 12)
    xlims!(ax, 0, max(10, maximum(vals) * 1.25))
    Label(fig[2, 1], "The Gibbs allocation sweep is always applied (not shown).",
          fontsize = 11, color = :gray50, halign = :left)
    save(asset("moves_acceptance.png"), fig)
end

# =============================================================================
# 7. PSM heatmap — co-assignment matrix sorted by Dahl cluster   [data-plot]
# =============================================================================
function fig_psm()
    sim = simulate_nmer(; n = 4, diameter = 0.04, mean_count = 12.0, psf_sigma = 0.130,
                          mean_photons = 500.0, min_photons = 100.0, pixel_size = 0.100, field_size = 3.0)
    locs = sim.smld.emitters
    psm_acc = PSMAccumulator(); samp = PartitionSamples(thin = 10)
    res = run_collapsed_chain(locs; n_iterations = 12000, burn_in = 3000,
                              accumulators = AbstractAccumulator[psm_acc, samp])
    psm = res.accumulators[1].psm
    samples = res.accumulators[2]
    _, _, _, assign = estimate_dahl(samples, locs, psm)
    order = sortperm(assign)
    P = psm[order, order]
    fig = Figure(size = (560, 480))
    ax = Axis(fig[1, 1], title = "Posterior similarity matrix (sorted by MAP-N cluster)",
              xlabel = "localization", ylabel = "localization", aspect = 1, yreversed = true)
    hm = heatmap!(ax, P; colormap = :viridis, colorrange = (0, 1))
    Colorbar(fig[1, 2], hm, label = "co-assignment frequency")
    save(asset("mapn_psm.png"), fig)
end

# =============================================================================
# 8. Partition field — locs colored by partition   [data-plot]
# =============================================================================
function fig_partition(G)
    locs = G.sim.smld.emitters
    pid = G.diag.partition_ids
    xs = [e.x for e in locs]; ys = [e.y for e in locs]
    fig = Figure(size = (640, 600))
    ax = Axis(fig[1, 1], title = "Precision-weighted DBSCAN partitions",
              aspect = DataAspect(), backgroundcolor = :black)
    scatter!(ax, xs, ys; color = pid, colormap = :tab20, markersize = 4)
    xlo, xhi = extrema(xs); ylo, yhi = extrema(ys)
    scalebar!(ax, xlo, xhi, ylo, yhi, 0.2; label = "200 nm")
    hidedecorations!(ax)
    save(asset("partition_field.png"), fig)
end

# =============================================================================
# 9. Posterior image — Rao-Blackwellized   [pipeline]
# =============================================================================
function fig_posterior(S)
    pim = S.diag.posterior_image
    pim === nothing && (@info "no posterior image"; return)
    fig = Figure(size = (560, 520))
    ax = Axis(fig[1, 1], title = "Rao-Blackwellized posterior image",
              aspect = DataAspect(), backgroundcolor = :black)
    heatmap!(ax, pim.edges_x, pim.edges_y, float.(pim.image); colormap = :inferno)
    xlo, xhi = extrema(pim.edges_x); ylo, yhi = extrema(pim.edges_y)
    scalebar!(ax, xlo, xhi, ylo, yhi, 0.05; label = "50 nm")
    hidedecorations!(ax)
    save(asset("results_posterior.png"), fig)
end

# =============================================================================
# 10. Spatial prior — flat vs localization-mixture density   [data-plot]
# =============================================================================
function fig_marginal()
    Random.seed!(7)
    centers = [(-0.03, 0.0), (0.03, 0.005)]
    locs = NamedTuple[]
    for c in centers, _ in 1:9
        push!(locs, (x = c[1] + 0.012 * randn(), y = c[2] + 0.012 * randn(), s = 0.013))
    end
    xs = range(-0.075, 0.075; length = 200)
    ys = range(-0.05, 0.05; length = 140)
    dens = [sum(exp(-((x - l.x)^2 + (y - l.y)^2) / (2 * l.s^2)) for l in locs) for x in xs, y in ys]
    dens ./= maximum(dens)
    θ = (-0.03, 0.0)                                # a candidate emitter position, marked in both
    lx = [l.x for l in locs]; ly = [l.y for l in locs]

    fig = Figure(size = (940, 380))
    ax1 = Axis(fig[1, 1], title = "Flat spatial prior  (uniform)", aspect = DataAspect())
    heatmap!(ax1, xs, ys, fill(0.5, length(xs), length(ys)); colormap = :viridis, colorrange = (0, 1))
    scatter!(ax1, lx, ly; color = :white, markersize = 6)
    scatter!(ax1, [θ[1]], [θ[2]]; marker = :star5, color = :red, markersize = 18, strokecolor = :white, strokewidth = 1)
    ax2 = Axis(fig[1, 2], title = "Localization-mixture prior", aspect = DataAspect())
    hm = heatmap!(ax2, xs, ys, dens; colormap = :viridis, colorrange = (0, 1))
    scatter!(ax2, lx, ly; color = :white, markersize = 6)
    scatter!(ax2, [θ[1]], [θ[2]]; marker = :star5, color = :red, markersize = 18, strokecolor = :white, strokewidth = 1)
    Colorbar(fig[1, 3], hm, label = "prior density (norm.)")
    for ax in (ax1, ax2); hidedecorations!(ax); end
    save(asset("marginal_flat_vs_locmix.png"), fig)

    fig2 = Figure(size = (560, 460))
    ax = Axis(fig2[1, 1], title = "Localization-mixture prior density", aspect = DataAspect())
    hm2 = heatmap!(ax, xs, ys, dens; colormap = :viridis)
    scatter!(ax, lx, ly; color = (:white, 0.8), markersize = 6)
    Colorbar(fig2[1, 2], hm2, label = "prior density (norm.)")
    hidedecorations!(ax)
    save(asset("marginal_locmix_grid.png"), fig2)
end

# =============================================================================
# 11. τ learning — scaled-distance distribution at under/correct/over τ   [data-plot]
# =============================================================================
function fig_se_distribution()
    Random.seed!(11)
    σrep = 0.006                       # reported per-axis σ (μm) ≈ 6 nm
    τtrue = 0.006                      # extra error the reported σ misses
    σact = sqrt(σrep^2 + τtrue^2)      # the true localization scatter
    n = 350
    x = σact .* randn(n); y = σact .* randn(n)      # one emitter's localizations
    ds = Float64[]
    for i in 1:n-1, j in i+1:n
        push!(ds, hypot(x[i] - x[j], y[i] - y[j]))
    end
    s2 = 2σrep^2
    zvals(τ) = ds ./ sqrt(s2 + 2τ^2)                # scaled pair distances at candidate τ
    grid = range(0, 4; length = 240)
    kde(s; bw = 0.16) = [sum(exp.(-((g .- s) ./ bw).^2 ./ 2)) / (length(s) * bw * sqrt(2π)) for g in grid]

    fig = Figure(size = (780, 480))
    ax = Axis(fig[1, 1], xlabel = "scaled pair distance  z = d / √(σ²ₐ + σ²_b + 2τ²)",
              ylabel = "density", title = "τ learning: only the correct τ matches Rayleigh(1)")
    lines!(ax, grid, grid .* exp.(-grid .^ 2 ./ 2); color = :black, linewidth = 3.5,
           linestyle = :dash, label = "Rayleigh(1) target")
    lines!(ax, grid, kde(zvals(0.0));      color = :crimson,    linewidth = 2.5, label = "τ too small (under)")
    lines!(ax, grid, kde(zvals(τtrue));    color = :seagreen,   linewidth = 2.5, label = "τ̂ (correct)")
    lines!(ax, grid, kde(zvals(2τtrue));   color = :dodgerblue, linewidth = 2.5, label = "τ too large (over)")
    axislegend(ax; position = :rt, framevisible = false, labelsize = 12)
    save(asset("se_dist.png"), fig)
end

# =============================================================================
# 12. se_adjust over-split vs corrected — Gaussian renders   [pipeline]
# =============================================================================
function fig_se_prepost()
    sim = simulate_nmer(; n = 1, diameter = 0.001, mean_count = 22.0, psf_sigma = 0.130,
                          mean_photons = 500.0, min_photons = 100.0, pixel_size = 0.100, field_size = 3.0)
    s = 0.5                                            # shrink reported σ ⇒ data looks over-precise
    σmed = median(e.σ_x for e in sim.smld.emitters)
    under = deepcopy(sim.smld)
    for e in under.emitters
        e.σ_x *= s; e.σ_y *= s; e.σ_xy *= s^2          # Emitter2DFit is mutable
    end
    τ = σmed * sqrt(1 - s^2)                           # per-axis error the shrunk σ now misses
    r0, _ = run_bagol(under; n_iterations = 14000, burn_in = 4000, partition_sigma = Inf, se_adjust = 0.0)
    r1, _ = run_bagol(under; n_iterations = 14000, burn_in = 4000, partition_sigma = Inf, se_adjust = τ)
    fov = compute_fov(under)
    span = max(fov[2] - fov[1], fov[4] - fov[3]); px_nm = span * 1000 / 500
    w = max(10, ceil(Int, (fov[2] - fov[1]) * 1000 / px_nm)); h = max(10, ceil(Int, (fov[4] - fov[3]) * 1000 / px_nm))
    tgt = SMLMRender.Image2DTarget(w, h, px_nm, (fov[1], fov[2]), (fov[3], fov[4]))
    (i0, _) = SMLMRender.render(r0; strategy = GaussianRender(), target = tgt, colormap = :inferno)
    (i1, _) = SMLMRender.render(r1; strategy = GaussianRender(), target = tgt, colormap = :inferno)
    fig = Figure(size = (900, 500))
    Label(fig[0, 1:2], "Under-reported σ: one emitter, over-split unless corrected", fontsize = 16, font = :bold)
    ax1 = Axis(fig[1, 1], title = "se_adjust = 0  ($(length(r0.emitters)) emitters)", aspect = DataAspect())
    ax2 = Axis(fig[1, 2], title = "se_adjust = τ̂  ($(length(r1.emitters)) emitter)", aspect = DataAspect())
    image!(ax1, rotr90(i0)); image!(ax2, rotr90(i1))
    for ax in (ax1, ax2); hidedecorations!(ax); end
    save(asset("se_prepost.png"), fig)
end

# =============================================================================
# 13. Count-distribution family — how shape α reshapes blinking   [data-plot]
# =============================================================================
function fig_negbin_family()
    μ = 10.0; kmax = 35; ks = 0:kmax
    fig = Figure(size = (760, 460))
    ax = Axis(fig[1, 1], xlabel = "localizations per emitter", ylabel = "probability",
              title = "Count distribution: shape α at fixed μ = 10")
    for (α, lab, col) in [(1.0, "α = 1  (dSTORM, exponential)", :crimson),
                          (2.0, "α = 2", :seagreen),
                          (8.0, "α = 8  (peaked, DNA-PAINT-like)", :dodgerblue)]
        lines!(ax, collect(ks), negbin_pmf(kmax, μ, α); color = col, linewidth = 2.5, label = lab)
    end
    axislegend(ax; position = :rt, framevisible = false, labelsize = 12)
    save(asset("priors_negbin_family.png"), fig)
end

# =============================================================================
# 14. Gibbs allocation sweep — reassign one localization   [schematic]
# =============================================================================
function fig_gibbs()
    a = [(-0.032, 0.012), (-0.026, -0.012), (-0.036, 0.0)]
    b = [(0.030, 0.006), (0.036, -0.014), (0.027, 0.016)]
    amb = (-0.003, 0.0)
    fig = Figure(size = (820, 360))
    for (col, (title, ambcol)) in zip((1, 3), [("before: xᵢ in cluster A", :dodgerblue),
                                               ("after: reassigned to B", :crimson)])
        ax = Axis(fig[1, col], title = title, aspect = DataAspect())
        scatter!(ax, first.(a), last.(a); color = :dodgerblue, markersize = 13)
        scatter!(ax, first.(b), last.(b); color = :crimson, markersize = 13)
        scatter!(ax, [amb[1]], [amb[2]]; color = ambcol, markersize = 16, strokecolor = :black, strokewidth = 1.5)
        limits!(ax, -0.06, 0.06, -0.04, 0.04); hidedecorations!(ax)
    end
    Label(fig[1, 2], "P(zᵢ = k) ∝\n(n₋ᵢ,ₖ + γ) · pred", fontsize = 14, tellheight = false)
    save(asset("moves_gibbs.png"), fig)
end

# =============================================================================
# 15. Posterior over K — Bayesian output is a distribution   [data-plot]
# =============================================================================
function fig_posterior_k(S)
    pk = S.diag.posterior_k
    pkn = pk ./ max(1, sum(pk))
    nz = findall(>(0.002), pkn)
    lo = isempty(nz) ? 0 : max(0, first(nz) - 1)
    hi = isempty(nz) ? length(pk) - 1 : min(length(pk) - 1, last(nz))
    fig = Figure(size = (640, 430))
    ax = Axis(fig[1, 1], xlabel = "number of emitters K", ylabel = "posterior P(K)",
              title = "The posterior is a distribution over K", xticks = lo:hi)
    barplot!(ax, collect(lo:hi), pkn[(lo+1):(hi+1)]; color = :steelblue)
    kmap = argmax(pkn) - 1
    vlines!(ax, [kmap]; color = :crimson, linestyle = :dash, linewidth = 2, label = "most probable K = $kmap")
    axislegend(ax; position = :rt, framevisible = false)
    save(asset("overview_posterior_k.png"), fig)
end

# =============================================================================
# 16. Hierarchical learning flow   [schematic]
# =============================================================================
function fig_hier_flow()
    fig = Figure(size = (900, 340))
    ax = Axis(fig[1, 1]); hidedecorations!(ax); hidespines!(ax)
    limits!(ax, 0, 10, 0, 4)
    boxes = [(1.6, "every emitter's\nblink counts  nₖ", :steelblue),
             (5.0, "learned\nμ,  shape", :seagreen),
             (8.4, "NegBin model\nregularizes K", :crimson)]
    for (x, txt, col) in boxes
        poly!(ax, Point2f[(x - 1.2, 1.6), (x + 1.2, 1.6), (x + 1.2, 2.7), (x - 1.2, 2.7)];
              color = (col, 0.12), strokecolor = col, strokewidth = 2)
        text!(ax, x, 2.15; text = txt, align = (:center, :center), fontsize = 15)
    end
    arrows!(ax, [2.9, 6.3], [2.15, 2.15], [1.0, 1.0], [0.0, 0.0]; color = :gray30, linewidth = 2, arrowsize = 14)
    text!(ax, 3.95, 2.45; text = "pool + learn (MH)", align = (:center, :bottom), fontsize = 11, color = :gray40)
    text!(ax, 7.0, 2.45; text = "defines", align = (:center, :bottom), fontsize = 11, color = :gray40)
    lines!(ax, [8.4, 8.4, 1.6, 1.6], [1.6, 0.7, 0.7, 1.45]; color = :gray55, linewidth = 1.5, linestyle = :dash)
    arrows!(ax, [1.6], [1.0], [0.0], [0.45]; color = :gray55, linewidth = 1.5, arrowsize = 12)
    text!(ax, 5.0, 0.55; text = "counts re-pooled every sync_interval", align = (:center, :center), fontsize = 11, color = :gray45)
    save(asset("hier_flow.png"), fig)
end

# =============================================================================
# 17. Dahl consensus flow — sampled partitions → PSM → closest   [schematic]
# =============================================================================
function fig_dahl_schematic()
    pts = [(-0.045, 0.0), (-0.02, 0.022), (0.0, -0.012), (0.03, 0.016), (0.046, -0.02)]
    # sample 3 is the SAME grouping as sample 1 with swapped labels (label switching)
    samples = [[1, 1, 1, 2, 2], [1, 1, 2, 2, 2], [2, 2, 2, 1, 1]]
    n = length(pts)
    C = zeros(n, n)
    for s in samples, i in 1:n, j in 1:n
        s[i] == s[j] && (C[i, j] += 1)
    end
    C ./= length(samples)
    dist(z) = sum((( (z[i] == z[j]) ? 1.0 : 0.0) - C[i, j])^2 for i in 1:n for j in (i+1):n)
    best = argmin([dist(s) for s in samples])
    cols = [:dodgerblue, :crimson, :seagreen]
    fig = Figure(size = (1180, 430))
    Label(fig[0, 1:6], "Dahl consensus: summarizing a distribution over partitions", fontsize = 16, font = :bold)
    for (r, s) in enumerate(samples)
        note = r == 3 ? "  (same as #1, colors swapped)" : ""
        ax = Axis(fig[r, 1], aspect = DataAspect(), title = r == 1 ? "sampled partitions" : note,
                  titlesize = 11, titlecolor = :gray40)
        scatter!(ax, first.(pts), last.(pts); color = cols[s], markersize = 13)
        limits!(ax, -0.07, 0.07, -0.05, 0.05); hidedecorations!(ax)
    end
    Label(fig[1:3, 2], "count\nco-assignment\n→", fontsize = 13, tellwidth = false)
    ax2 = Axis(fig[1:3, 3], title = "PSM (co-assignment)", aspect = 1, yreversed = true)
    hm = heatmap!(ax2, C; colormap = :viridis, colorrange = (0, 1)); hidedecorations!(ax2)
    Colorbar(fig[1:3, 4], hm; label = "co-assignment freq.", height = Relative(0.7))
    Label(fig[1:3, 5], "pick sample\nclosest to PSM\n→", fontsize = 13, tellwidth = false)
    ax3 = Axis(fig[1:3, 6], aspect = DataAspect(), title = "consensus")
    scatter!(ax3, first.(pts), last.(pts); color = cols[samples[best]], markersize = 15, strokecolor = :black, strokewidth = 1)
    limits!(ax3, -0.07, 0.07, -0.05, 0.05); hidedecorations!(ax3)
    save(asset("mapn_dahl.png"), fig)
end

# Print summary metrics for the results gallery table (hand-copied into results.md).
function print_metrics(S)
    pk = S.diag.posterior_k; pkn = pk ./ max(1, sum(pk))
    kbest = argmax(pkn) - 1
    rawσ  = median(e.σ_x for e in S.sim.smld.emitters) * 1000
    postσ = median(e.σ_x for e in S.result.emitters) * 1000
    @info "RESULTS METRICS" trueK = S.sim.n_emitters mapnK = S.diag.n_emitters
    @info "RESULTS METRICS cont" Pkbest = (kbest, round(maximum(pkn), digits = 2)) rawσ_nm = round(rawσ, digits = 1) postσ_nm = round(postσ, digits = 1)
end

# =============================================================================
# Schematics — drawn conceptual diagrams   [schematic]
# =============================================================================
function fig_generative()
    Random.seed!(4)
    psf = 0.130                                          # PSF sigma (μm)
    n = 18
    photons = [max(25.0, -300.0 * log(rand())) for _ in 1:n]   # ~exponential photon counts
    σ = psf ./ sqrt.(photons)                            # precision: bright ⇒ tight, dim ⇒ loose
    θ = (0.0, 0.0)
    pts = [(θ[1] + σ[i] * randn(), θ[2] + σ[i] * randn()) for i in 1:n]   # scatter ∝ σᵢ
    fig = Figure(size = (560, 500))
    ax = Axis(fig[1, 1], title = "One emitter → many localizations", aspect = DataAspect())
    for i in 1:n                                         # 1σ uncertainty disc, sized per localization
        poly!(ax, Point2f.(cov_circle(pts[i], σ[i])...); color = (:steelblue, 0.10),
              strokecolor = (:steelblue, 0.5), strokewidth = 1)
    end
    scatter!(ax, first.(pts), last.(pts); color = :steelblue, markersize = 6,
             label = "localizations  xᵢ ~ N(θ, Σᵢ)")
    scatter!(ax, [θ[1]], [θ[2]]; marker = :star5, markersize = 26, color = :orange,
             strokecolor = :black, strokewidth = 1, label = "emitter θ")
    axislegend(ax; position = :lt, framevisible = false, labelsize = 12)
    lim = 0.065
    limits!(ax, -lim, lim, -lim, lim)
    scalebar!(ax, -lim, lim, -lim, lim, 0.02; color = :gray25, label = "20 nm")
    hidedecorations!(ax)
    save(asset("intro_generative.png"), fig)
end

cov_circle(c, r; n = 64) = (c[1] .+ r .* cos.(range(0, 2π, n)), c[2] .+ r .* sin.(range(0, 2π, n)))

function fig_precision()
    fig = Figure(size = (560, 460))
    ax = Axis(fig[1, 1], title = "Precision weighting: low-σ localizations pull harder",
              aspect = DataAspect())
    confident = [(-0.05, 0.02), (-0.045, -0.02)]; loose = [(0.07, 0.0)]
    for p in confident
        poly!(ax, Point2f.(cov_circle(p, 0.012)...); color = (:dodgerblue, 0.18), strokecolor = :dodgerblue)
        scatter!(ax, [p[1]], [p[2]]; color = :dodgerblue, markersize = 8)
    end
    for p in loose
        poly!(ax, Point2f.(cov_circle(p, 0.05)...); color = (:gray70, 0.15), strokecolor = :gray60)
        scatter!(ax, [p[1]], [p[2]]; color = :gray50, markersize = 8)
    end
    allp = vcat(confident, loose)
    gc = (mean(first.(allp)), mean(last.(allp)))
    wmean = (-0.047, 0.0)
    scatter!(ax, [gc[1]], [gc[2]]; marker = :cross, markersize = 16, color = :gray40, label = "naive centroid")
    scatter!(ax, [wmean[1]], [wmean[2]]; marker = :star5, markersize = 20, color = :crimson,
             label = "precision-weighted θ̂")
    axislegend(ax; position = :rb, framevisible = false, labelsize = 12)
    limits!(ax, -0.1, 0.14, -0.08, 0.08); hidedecorations!(ax)
    save(asset("collapsed_precision.png"), fig)
end

function fig_allocation()
    Random.seed!(3)
    pts = [(-0.04, 0.0), (-0.035, 0.02), (-0.03, -0.015), (0.035, 0.005), (0.04, -0.02), (0.03, 0.02)]
    fig = Figure(size = (820, 320))
    Label(fig[0, 1:3], "Same localizations, different allocations z", fontsize = 16, font = :bold)
    groupings = [("K = 1", [1, 1, 1, 1, 1, 1]), ("K = 2", [1, 1, 1, 2, 2, 2]), ("K = 3", [1, 1, 2, 2, 3, 3])]
    cols = [:dodgerblue, :crimson, :seagreen]
    for (i, (lab, z)) in enumerate(groupings)
        ax = Axis(fig[1, i], title = lab, aspect = DataAspect())
        scatter!(ax, first.(pts), last.(pts); color = cols[z], markersize = 11)
        limits!(ax, -0.07, 0.07, -0.05, 0.05); hidedecorations!(ax)
    end
    save(asset("priors_allocation.png"), fig)
end

function fig_splitmerge()
    pts = [(-0.03, 0.0), (-0.025, 0.02), (-0.02, -0.015), (0.025, 0.005), (0.03, -0.02), (0.022, 0.02)]
    fig = Figure(size = (760, 340))
    for (i, (lab, z, title)) in enumerate([("K", [1,1,1,1,1,1], "before split / after merge"),
                                           ("K+1", [1,1,1,2,2,2], "after split / before merge")])
        ax = Axis(fig[1, i == 1 ? 1 : 3], title = title, aspect = DataAspect())
        cols = [:dodgerblue, :crimson]
        scatter!(ax, first.(pts), last.(pts); color = cols[z], markersize = 11)
        limits!(ax, -0.06, 0.06, -0.05, 0.05); hidedecorations!(ax)
    end
    Label(fig[1, 2], "split  →\n←  merge", fontsize = 15, tellheight = false)
    save(asset("moves_splitmerge.png"), fig)
end

function fig_birthdeath()
    # one localization (top) sits slightly apart; birth detaches it into its own singleton.
    base = [(-0.02, -0.012), (-0.028, 0.0), (-0.012, -0.004), (-0.018, 0.026)]
    fig = Figure(size = (820, 360))
    ax1 = Axis(fig[1, 1], title = "K   (before birth / after death)", aspect = DataAspect())
    lines!(ax1, cov_circle((-0.02, 0.0), 0.034)...; color = (:dodgerblue, 0.6))
    scatter!(ax1, first.(base), last.(base); color = :dodgerblue, markersize = 13)
    limits!(ax1, -0.06, 0.06, -0.045, 0.06); hidedecorations!(ax1)
    ax2 = Axis(fig[1, 3], title = "K+1   (after birth / before death)", aspect = DataAspect())
    lines!(ax2, cov_circle((-0.02, -0.006), 0.026)...; color = (:dodgerblue, 0.6))
    scatter!(ax2, first.(base[1:3]), last.(base[1:3]); color = :dodgerblue, markersize = 13)
    lines!(ax2, cov_circle(base[4], 0.012)...; color = :crimson)        # same loc, now its own cluster
    scatter!(ax2, [base[4][1]], [base[4][2]]; color = :crimson, markersize = 14)
    text!(ax2, base[4][1], base[4][2] + 0.016; text = "new singleton",
          align = (:center, :bottom), color = :crimson, fontsize = 12)
    limits!(ax2, -0.06, 0.06, -0.045, 0.06); hidedecorations!(ax2)
    Label(fig[1, 2], "birth  →\n←  death", fontsize = 15, tellheight = false)
    save(asset("moves_birthdeath.png"), fig)
end

function fig_dedup()
    fig = Figure(size = (720, 470))
    ax = Axis(fig[1, 1], title = "Boundary deduplication across partitions", aspect = DataAspect())
    band!(ax, [-0.012, 0.012], [-0.05, -0.05], [0.05, 0.05]; color = (:gold, 0.16))
    vlines!(ax, [0.0]; color = :gray60, linestyle = :dash)
    text!(ax, -0.033, 0.043; text = "partition A", color = :dodgerblue, align = (:center, :center))
    text!(ax, 0.033, 0.043; text = "partition B", color = :crimson, align = (:center, :center))
    text!(ax, 0.0, 0.03; text = "overlap band", color = :darkgoldenrod, align = (:center, :center), fontsize = 11)
    pa, pb = (-0.005, 0.0), (0.005, 0.0)
    lines!(ax, cov_circle((0.0, 0.0), 0.015)...; color = (:gray45, 0.9), linewidth = 1.5, linestyle = :dot)
    scatter!(ax, [pa[1]], [pa[2]]; color = :dodgerblue, markersize = 17)
    scatter!(ax, [pb[1]], [pb[2]]; color = :crimson, markersize = 17)
    lines!(ax, [pa[1], pb[1]], [pa[2], pb[2]]; color = :black, linewidth = 2)
    text!(ax, 0.019, -0.012; text = "2σ gate", color = :gray40, align = (:left, :center), fontsize = 11)
    text!(ax, 0.0, -0.033; text = "Hungarian match within 2σ → merge",
          align = (:center, :center), fontsize = 12)
    limits!(ax, -0.05, 0.05, -0.05, 0.05); hidedecorations!(ax)
    save(asset("partition_dedup.png"), fig)
end

# =============================================================================
# Driver
# =============================================================================
function main()
    local S, G, BC
    figure("run_single",   () -> (S = run_single()))
    figure("run_grid",     () -> (G = run_grid()))
    figure("run_bigcount", () -> (BC = run_bigcount()))

    if @isdefined(S)
        figure("hero",            () -> fig_hero(S))
        figure("mapn_ellipses",   () -> fig_mapn_ellipses(S))
        figure("acceptance",      () -> fig_acceptance(S))
        figure("posterior",       () -> fig_posterior(S))
        figure("posterior_k",     () -> fig_posterior_k(S))
        figure("metrics",         () -> print_metrics(S))
    end
    @isdefined(G) && figure("partition", () -> fig_partition(G))
    if @isdefined(BC)
        # hierarchical-learning figures need realistic-scale statistics → use the 2400-emitter run
        figure("convergence",  () -> fig_convergence(BC))
        figure("count",        () -> fig_count(BC))
        figure("metrics_grid", () -> print_metrics(BC))
    end
    figure("collapsed_cluster", fig_collapsed_cluster)
    figure("psm",               fig_psm)
    figure("marginal",          fig_marginal)
    figure("se_distribution",   fig_se_distribution)
    figure("se_prepost",        fig_se_prepost)
    figure("negbin_family",     fig_negbin_family)
    figure("gibbs",             fig_gibbs)
    figure("hier_flow",         fig_hier_flow)
    figure("dahl_schematic",    fig_dahl_schematic)

    for (nm, f) in (("generative", fig_generative), ("precision", fig_precision),
                    ("allocation", fig_allocation), ("splitmerge", fig_splitmerge),
                    ("birthdeath", fig_birthdeath), ("dedup", fig_dedup))
        figure(nm, f)
    end
    @info "done — figures written" dir = ASSETS
end

main()

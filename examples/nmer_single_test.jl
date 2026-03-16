# Single N-mer
# ============
# Runs BaGoL on a single N-mer cluster using run_collapsed_chain directly.
# Emitter positions come from ClusterStats posterior.
#
# Run with: julia --project=examples examples/nmer_single_test.jl

using Pkg
Pkg.activate(@__DIR__)

using SMLMBaGoL
using SMLMBaGoL: accumulator_result
using SMLMData
using Random
using Statistics
using Distributions
using CairoMakie

# Include shared visualization functions (metrics + render suite)
include(joinpath(@__DIR__, "viz_metrics.jl"))
include(joinpath(@__DIR__, "viz_smlmrender.jl"))

# =============================================================================
# ADJUSTABLE PARAMETERS
# =============================================================================

const SEED = nothing
# const SEED = 42

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
const CALLBACK_INTERVAL = 50

# Camera
const CAMERA_PIXELS = 256
const PIXEL_SIZE = 0.100          # μm per pixel

# Output
const OUTPUT_DIR = joinpath(@__DIR__, "output", "nmer_single")

# =============================================================================
# SETUP
# =============================================================================

mkpath(OUTPUT_DIR)
if SEED !== nothing
    Random.seed!(SEED)
end

println("="^60)
println("Single $(N_EMITTERS)-mer — Collapsed Gibbs Sampler")
println("="^60)
println("\nParameters:")
println("  Cluster diameter: $(CLUSTER_DIAMETER * 1000) nm")
println("  PSF sigma: $(PSF_SIGMA * 1000) nm")
println("  Photon mean: $PHOTON_MEAN (min: $PHOTON_MIN)")
println("  Blink distribution: Poisson($(BLINK_MEAN))")

# =============================================================================
# GENERATE N-MER
# =============================================================================

println("\n" * "-"^60)
println("Generating $(N_EMITTERS)-mer...")

cluster_radius = CLUSTER_DIAMETER / 2
fov_size = CAMERA_PIXELS * PIXEL_SIZE
center_x, center_y = fov_size / 2, fov_size / 2

true_positions = Tuple{Float64, Float64}[]
for i in 1:N_EMITTERS
    θ = 2π * (i - 1) / N_EMITTERS
    ex = center_x + cluster_radius * cos(θ)
    ey = center_y + cluster_radius * sin(θ)
    push!(true_positions, (ex, ey))
end

# Generate localizations
blink_dist = Poisson(BLINK_MEAN)
photon_dist = Exponential(PHOTON_MEAN)

function generate_localizations(positions, blink_dist, photon_dist)
    locs = SMLMData.Emitter2DFit[]
    blink_counts = Int[]
    loc_id = 1
    for (emitter_idx, (ex, ey)) in enumerate(positions)
        n_blinks = max(1, rand(blink_dist))
        actual_blinks = 0
        for _ in 1:n_blinks
            N = rand(photon_dist)
            N < PHOTON_MIN && continue
            σ = PSF_SIGMA / sqrt(N)
            x = ex + σ * randn()
            y = ey + σ * randn()
            push!(locs, SMLMData.Emitter2DFit(
                x, y, N, 10.0,
                σ, σ, 0.0,
                sqrt(N), 1.0,
                loc_id, 1, emitter_idx, loc_id
            ))
            loc_id += 1
            actual_blinks += 1
        end
        push!(blink_counts, actual_blinks)
    end
    return locs, blink_counts
end

locs, true_blink_counts = generate_localizations(true_positions, blink_dist, photon_dist)

println("  Generated $(length(locs)) localizations from $(N_EMITTERS) emitters")
println("  Blinks per emitter: $(true_blink_counts)")
println("  Mean blinks: $(round(mean(true_blink_counts), digits=1))")

sigmas = [loc.σ_x for loc in locs]
println("  Localization precision: $(round(mean(sigmas)*1000, digits=1)) ± $(round(std(sigmas)*1000, digits=1)) nm")

# Precision filter
n_before = length(locs)
locs = filter(loc -> max(loc.σ_x, loc.σ_y) <= PRECISION_MAX, locs)
println("  Precision filter (σ ≤ $(PRECISION_MAX*1000) nm): $(n_before) → $(length(locs)) locs")

camera = SMLMData.IdealCamera(CAMERA_PIXELS, CAMERA_PIXELS, PIXEL_SIZE)
locs_smld = SMLMData.BasicSMLD(locs, camera, 1, 1)

# =============================================================================
# RUN COLLAPSED GIBBS WITH CALLBACK
# =============================================================================

println("\n" * "-"^60)
println("Running collapsed Gibbs sampler...")

# Trace collector for diagnostics + animation frames
mutable struct CollapsedTraceCollector
    iters::Vector{Int}
    ks::Vector{Int}
    mus::Vector{Float64}
    shapes::Vector{Float64}
    # Animation frames: assignment snapshots + cluster positions
    frame_iters::Vector{Int}
    frame_assignments::Vector{Vector{Int16}}
    frame_positions::Vector{Vector{Tuple{Float64, Float64}}}
end
CollapsedTraceCollector() = CollapsedTraceCollector(
    Int[], Int[], Float64[], Float64[],
    Int[], Vector{Int16}[], Vector{Tuple{Float64, Float64}}[])

trace = CollapsedTraceCollector()
const ANIM_INTERVAL = 10  # capture every 10 iterations for animation

function trace_callback(iter, state, μ, shape)
    push!(trace.iters, iter)
    push!(trace.ks, state.n_active)
    push!(trace.mus, μ)
    push!(trace.shapes, shape)

    # Capture animation frame at higher frequency
    if iter % ANIM_INTERVAL == 0
        push!(trace.frame_iters, iter)
        push!(trace.frame_assignments, copy(state.assignments))
        # Extract posterior mean positions for active clusters
        positions = Tuple{Float64, Float64}[]
        for (j, cs) in enumerate(state.clusters)
            state.active[j] || continue
            cs.n == 0 && continue
            mx, my = SMLMBaGoL.posterior_mean(cs)
            push!(positions, (mx, my))
        end
        push!(trace.frame_positions, positions)
    end
end

# Create accumulators
count_hist = EmitterCountHist()
post_img = PosteriorImage(pixel_size=0.002)
nn_hist = NNDistHist(max_dist=0.100, n_bins=100)
ps_acc = PartitionSamples(thin=5)
psm_acc = PSMAccumulator()

result = run_collapsed_chain(
    locs;
    λ_K = Float64(N_EMITTERS),
    n_iterations = N_ITERATIONS,
    burn_in = BURN_IN,
    accumulators = AbstractAccumulator[count_hist, post_img, nn_hist, ps_acc, psm_acc],
    callback = trace_callback,
    callback_interval = CALLBACK_INTERVAL,
    verbose = true
)

# Extract emitters via Dahl+overlap MAP-N (Dahl consensus + overlap Hungarian)
samples = accumulator_result(ps_acc)
psm = accumulator_result(psm_acc).psm
_, _, _, dahl_assignments = estimate_dahl(samples, locs, psm)
emitters, _ = estimate_mapn_overlap(samples, locs, dahl_assignments)

# Build posterior_k from EmitterCountHist
posterior_k = result.accumulators[1]  # EmitterCountHist result = Vector{Int}

println("\n" * "-"^60)
println("Results:")
println("  True emitters: $(N_EMITTERS)")
println("  Estimated emitters: $(length(emitters))")

σ_values = [sqrt(e.σ_x^2 + e.σ_y^2) for e in emitters]
println("  Mean σ: $(round(mean(σ_values)*1000, digits=2)) nm")

# =============================================================================
# METRICS
# =============================================================================

println("\n" * "-"^60)
println("Evaluation Metrics:")
metrics = print_metrics(emitters, true_positions; threshold=0.020)

# =============================================================================
# PARTITION DIAGNOSTICS: Dahl vs Oracle
# =============================================================================

println("\n" * "-"^60)
println("Partition diagnostics (Dahl vs Oracle):")

oracle_z = Int16[loc.track_id for loc in locs]
pd = partition_diagnostics(dahl_assignments, oracle_z, samples)

println("  VI(Dahl, Oracle) = $(round(pd.vi_total, digits=3)) nats")
println("    Over-segmentation  H(Dahl|Oracle) = $(round(pd.overseg, digits=3))")
println("    Under-segmentation H(Oracle|Dahl) = $(round(pd.underseg, digits=3))")
println("  K: Dahl=$(pd.K_dahl), Oracle=$(pd.K_oracle)")
println("  Expected Posterior Loss (VI):")
println("    EPL(Dahl)   = $(round(pd.epl_dahl, digits=3))")
println("    EPL(Oracle) = $(round(pd.epl_oracle, digits=3))")
println("    EPL(Best)   = $(round(pd.epl_best, digits=3))  (sample #$(pd.best_sample_idx))")
println("  Regret:")
println("    Dahl regret   = $(round(pd.regret_dahl, digits=3))")
println("    Oracle regret = $(round(pd.regret_oracle, digits=3))")
if pd.epl_dahl <= pd.epl_oracle
    println("  → Dahl is a better posterior summary than oracle")
else
    println("  → Oracle is a better posterior summary (sampler may be missing modes)")
end

# =============================================================================
# GENERATE VISUALIZATIONS
# =============================================================================

println("\n" * "-"^60)
println("Generating visualizations...")

# Helper functions
function draw_circle!(ax, x, y, r; color=:black, linewidth=1.0, alpha=1.0)
    poly!(ax, Circle(Point2f(x, y), Float32(r)),
          color=(:white, 0.0), strokecolor=(color, alpha), strokewidth=linewidth)
end

function draw_ellipse!(ax, x, y, rx, ry; color=:black, linewidth=1.0, alpha=1.0)
    θ = range(0, 2π, length=50)
    ex = x .+ rx .* cos.(θ)
    ey = y .+ ry .* sin.(θ)
    lines!(ax, ex, ey, color=(color, alpha), linewidth=linewidth)
end

function draw_x!(ax, x, y, r; color=:blue, linewidth=2.0, alpha=1.0)
    lines!(ax, [x - r, x + r], [y + r, y - r], color=(color, alpha), linewidth=linewidth)
    lines!(ax, [x - r, x + r], [y - r, y + r], color=(color, alpha), linewidth=linewidth)
end

# 1. Collapsed Gibbs diagnostics (K trace, μ trace, shape trace, posterior K)
println("  [1/7] Collapsed diagnostics...")
fig1 = Figure(size=(1200, 800))

# K trace
ax1 = Axis(fig1[1, 1], title="Emitter Count K", xlabel="Iteration", ylabel="K")
lines!(ax1, trace.iters, trace.ks, color=:steelblue, linewidth=0.5)
vlines!(ax1, [BURN_IN], color=:red, linestyle=:dash, linewidth=2, label="Burn-in")
post_burn_ks = trace.ks[trace.iters .> BURN_IN]
if !isempty(post_burn_ks)
    hlines!(ax1, [mean(post_burn_ks)], color=:green, linewidth=2,
            label="Mean = $(round(mean(post_burn_ks), digits=1))")
end
axislegend(ax1, position=:rt)

# μ trace
ax2 = Axis(fig1[1, 2], title="Hierarchical μ", xlabel="Iteration", ylabel="μ")
lines!(ax2, trace.iters, trace.mus, color=:darkorange, linewidth=0.8)
vlines!(ax2, [BURN_IN], color=:red, linestyle=:dash, linewidth=2)
# Mark true mean blinks
true_mean_blinks = mean(true_blink_counts)
hlines!(ax2, [true_mean_blinks], color=:blue, linestyle=:dot, linewidth=2,
        label="True mean = $(round(true_mean_blinks, digits=1))")
axislegend(ax2, position=:rt)

# Shape trace
ax3 = Axis(fig1[2, 1], title="Gamma Shape", xlabel="Iteration", ylabel="shape")
lines!(ax3, trace.iters, trace.shapes, color=:purple, linewidth=0.8)
vlines!(ax3, [BURN_IN], color=:red, linestyle=:dash, linewidth=2)

# Posterior K histogram
ax4 = Axis(fig1[2, 2], title="Posterior K", xlabel="K", ylabel="Count")
k_values = 0:(length(posterior_k) - 1)
barplot!(ax4, collect(k_values), posterior_k, color=:steelblue)
vlines!(ax4, [N_EMITTERS], color=:red, linewidth=3, label="True K = $N_EMITTERS")
axislegend(ax4, position=:rt)

save(joinpath(OUTPUT_DIR, "collapsed_diagnostics.png"), fig1)

# 2. Result plot (localizations + emitters + GT)
println("  [2/7] Result plot...")
fig2 = Figure(size=(800, 800))
ax = Axis(fig2[1, 1],
    title="Collapsed Gibbs Result (K=$(length(emitters)))",
    xlabel="x (μm)", ylabel="y (μm)",
    aspect=DataAspect(), yreversed=true)

# Zoom to cluster region
loc_xs = [l.x for l in locs]
loc_ys = [l.y for l in locs]
pad = 0.030
xlims!(ax, minimum(loc_xs) - pad, maximum(loc_xs) + pad)
ylims!(ax, minimum(loc_ys) - pad, maximum(loc_ys) + pad)

# Localizations as 1σ circles
for loc in locs
    σ = mean([loc.σ_x, loc.σ_y])
    draw_circle!(ax, loc.x, loc.y, σ; color=:gray30, linewidth=0.8, alpha=0.5)
end

# True positions as blue X
for (tx, ty) in true_positions
    draw_x!(ax, tx, ty, 0.003; color=:blue, linewidth=2.0)
end

# BaGoL emitters as red dots + uncertainty ellipses
for e in emitters
    scatter!(ax, [e.x], [e.y], color=:red, markersize=8)
    if e.σ_x > 0 && e.σ_y > 0
        draw_ellipse!(ax, e.x, e.y, e.σ_x, e.σ_y;
                     color=:red, linewidth=1.0, alpha=0.7)
    end
end

save(joinpath(OUTPUT_DIR, "result.png"), fig2)

# 3. Posterior image (Rao-Blackwellized)
println("  [3/7] Rao-Blackwellized posterior image...")
post = result.accumulators[2]  # PosteriorImage result
if post isa NamedTuple && haskey(post, :image)
    # Save raw PNG
    SMLMBaGoL.save_posterior_png(joinpath(OUTPUT_DIR, "posterior_image.png"), post; percentile=0.99)
    println("    Image size: $(size(post.image, 1))×$(size(post.image, 2))")

    # Also plot with CairoMakie colormap
    fig3 = Figure(size=(700, 600))
    ax3 = Axis(fig3[1, 1], title="Rao-Blackwellized Posterior Image",
               xlabel="x (μm)", ylabel="y (μm)", aspect=DataAspect(), yreversed=true)
    img = post.image
    # Clip at 99th percentile
    sorted_vals = sort(vec(img))
    nonzero = filter(>(0), sorted_vals)
    vmax = isempty(nonzero) ? 1.0 : nonzero[min(end, round(Int, length(nonzero) * 0.99))]
    clipped = min.(img, vmax)
    heatmap!(ax3, post.edges_x[1:end-1], post.edges_y[1:end-1], clipped',
             colormap=:inferno)
    # Overlay true positions
    for (tx, ty) in true_positions
        scatter!(ax3, [tx], [ty], color=:cyan, markersize=6, marker=:xcross, strokewidth=1)
    end
    Colorbar(fig3[1, 2], limits=(0, vmax), colormap=:inferno, label="Density")
    save(joinpath(OUTPUT_DIR, "posterior_heatmap.png"), fig3)
end

# 4. Nearest-neighbor distance histogram
println("  [4/7] NN distance histogram...")
nn_result = result.accumulators[3]  # NNDistHist result
if nn_result isa NamedTuple && haskey(nn_result, :counts)
    fig4 = Figure(size=(600, 400))
    ax4 = Axis(fig4[1, 1], title="Nearest-Neighbor Distances",
               xlabel="Distance (nm)", ylabel="Count")
    bin_centers = [(nn_result.bin_edges[i] + nn_result.bin_edges[i+1]) / 2 * 1000
                   for i in 1:length(nn_result.counts)]
    barplot!(ax4, bin_centers, nn_result.counts, color=:steelblue)
    # Mark true NN distance: chord length between adjacent emitters on circle
    nn_true = CLUSTER_DIAMETER * sin(π / N_EMITTERS) * 1000  # nm
    vlines!(ax4, [nn_true], color=:red, linewidth=2,
            label="True NN = $(round(nn_true, digits=1)) nm")
    axislegend(ax4, position=:rt)
    save(joinpath(OUTPUT_DIR, "nn_distances.png"), fig4)
end

# 5. Acceptance rates
println("  [5/7] Acceptance rates...")
fig5 = Figure(size=(600, 400))
ax5 = Axis(fig5[1, 1], title="Acceptance Rates", ylabel="Rate (%)")

move_names = String[]
rates = Float64[]
for (move, (acc, tot)) in result.acceptance
    push!(move_names, string(move))
    push!(rates, tot > 0 ? 100.0 * acc / tot : 0.0)
end
barplot!(ax5, 1:length(move_names), rates, color=:steelblue)
ax5.xticks = (1:length(move_names), move_names)
save(joinpath(OUTPUT_DIR, "acceptance_rates.png"), fig5)

# 6. Partition diagnostics figure
println("  [6/9] Partition diagnostics...")
fig6 = Figure(size=(900, 350))

# Panel 1: EPL comparison (bar chart)
ax6a = Axis(fig6[1, 1], title="Expected Posterior Loss (VI)",
    ylabel="EPL (nats)", xticks=(1:3, ["Dahl", "Oracle", "Best\nsample"]))
barplot!(ax6a, [1, 2, 3], [pd.epl_dahl, pd.epl_oracle, pd.epl_best],
    color=[:red, :green, :steelblue])

# Panel 2: VI decomposition (stacked bar)
ax6b = Axis(fig6[1, 2], title="VI(Dahl, Oracle) = $(round(pd.vi_total, digits=2))",
    ylabel="nats", xticks=(1:2, ["Over-seg\n(splitting)", "Under-seg\n(merging)"]))
barplot!(ax6b, [1, 2], [pd.overseg, pd.underseg],
    color=[:orange, :purple])

# Panel 3: P(K|data) with markers
ax6c = Axis(fig6[1, 3], title="Posterior P(K|data)",
    xlabel="K", ylabel="Frequency")
k_hist = posterior_k
if !isempty(k_hist)
    k_min, k_max = extrema(k_hist)
    k_range = k_min:k_max
    counts = [count(==(k), k_hist) for k in k_range]
    barplot!(ax6c, collect(k_range), counts, color=:gray70)
    vlines!(ax6c, [pd.K_dahl], color=:red, linewidth=2, label="Dahl K=$(pd.K_dahl)")
    vlines!(ax6c, [pd.K_oracle], color=:green, linewidth=2, linestyle=:dash,
        label="Oracle K=$(pd.K_oracle)")
    axislegend(ax6c, position=:rt, framevisible=false, labelsize=10)
end

save(joinpath(OUTPUT_DIR, "partition_diagnostics.png"), fig6)
println("Saved: partition_diagnostics.png")

# 7. Posterior image (raw PNG)
println("  [7/9] Posterior image PNG (alt method via save_posterior_png)...")
# Already saved above

# 8. SMLMRender suite
println("  [8/9] SMLMRender suite...")
bagol_smld = SMLMData.BasicSMLD(emitters, camera, 1, 1)
render_bagol_suite(locs_smld, bagol_smld;
    true_positions = true_positions,
    prefix = "render",
    output_dir = OUTPUT_DIR,
    pixel_size = 1.0)

# 9. Chain animation (collapsed Gibbs)
println("  [9/9] Chain animation...")

const PALETTE = [
    colorant"#e41a1c", colorant"#377eb8", colorant"#4daf4a",
    colorant"#984ea3", colorant"#ff7f00", colorant"#ffff33",
    colorant"#a65628", colorant"#f781bf", colorant"#999999",
    colorant"#66c2a5", colorant"#fc8d62", colorant"#8da0cb",
]

loc_xs = [l.x for l in locs]
loc_ys = [l.y for l in locs]
pad = 0.030
xlim = (minimum(loc_xs) - pad, maximum(loc_xs) + pad)
ylim = (minimum(loc_ys) - pad, maximum(loc_ys) + pad)
median_σ = median([mean([l.σ_x, l.σ_y]) for l in locs])

n_frames = length(trace.frame_iters)
if n_frames > 0
    fig_anim = Figure(size=(1200, 600))
    ax_sp = Axis(fig_anim[1:2, 1], title="Collapsed Gibbs Evolution",
                 xlabel="x (μm)", ylabel="y (μm)",
                 aspect=DataAspect())
    xlims!(ax_sp, xlim...)
    ylims!(ax_sp, ylim...)

    ax_kt = Axis(fig_anim[1, 2], title="K trace",
                 xlabel="Iteration", ylabel="K")
    xlims!(ax_kt, 0, maximum(trace.frame_iters))
    ylims!(ax_kt, 0, maximum(trace.ks) + 2)

    ax_inf = Axis(fig_anim[2, 2])
    hidedecorations!(ax_inf)
    hidespines!(ax_inf)

    record(fig_anim, joinpath(OUTPUT_DIR, "chain_animation.mp4"),
           1:n_frames; framerate=30) do fi
        empty!(ax_sp)
        empty!(ax_kt)
        empty!(ax_inf)

        assignments = trace.frame_assignments[fi]
        positions = trace.frame_positions[fi]
        iter = trace.frame_iters[fi]

        # Build slot → color index mapping
        slot_to_color = Dict{Int16, Int}()
        cidx = 0
        for s in sort(unique(assignments))
            s <= 0 && continue
            cidx += 1
            slot_to_color[s] = cidx
        end

        # Draw localizations colored by cluster assignment
        for (i, loc) in enumerate(locs)
            σ = mean([loc.σ_x, loc.σ_y])
            s = assignments[i]
            if s > 0 && haskey(slot_to_color, s)
                c = PALETTE[mod1(slot_to_color[s], length(PALETTE))]
                poly!(ax_sp, Circle(Point2f(loc.x, loc.y), Float32(σ)),
                      color=(:white, 0.0), strokecolor=(c, 0.6), strokewidth=0.8)
            end
        end

        # Draw true positions
        for (tx, ty) in true_positions
            scatter!(ax_sp, [tx], [ty], marker=:xcross,
                     color=:black, markersize=12, strokewidth=2)
        end

        # Draw cluster posterior means
        for (ci, (mx, my)) in enumerate(positions)
            c = PALETTE[mod1(ci, length(PALETTE))]
            scatter!(ax_sp, [mx], [my], color=c, markersize=10,
                     strokecolor=:black, strokewidth=1)
            poly!(ax_sp, Circle(Point2f(mx, my), Float32(median_σ)),
                  color=(:white, 0.0), strokecolor=(c, 0.9), strokewidth=2.5)
        end

        # K trace up to current frame
        idx_up_to = searchsortedlast(trace.iters, iter)
        if idx_up_to > 0
            lines!(ax_kt, trace.iters[1:idx_up_to], trace.ks[1:idx_up_to],
                   color=:steelblue, linewidth=1)
        end
        hlines!(ax_kt, [N_EMITTERS], color=:red, linestyle=:dash, linewidth=1.5)
        K_now = length(positions)
        scatter!(ax_kt, [iter], [K_now], color=:red, markersize=8)

        # Info
        text!(ax_inf, 0.5, 0.7, text="Iter $iter / $(N_ITERATIONS)\nK = $K_now",
              align=(:center, :center), fontsize=16)
    end
    println("    Saved chain_animation.mp4 ($n_frames frames)")
end

# =============================================================================
# SUMMARY
# =============================================================================

println("\n" * "="^60)
println("Output files saved to: $OUTPUT_DIR")
println("="^60)
println("\nCollapsed Gibbs diagnostics:")
println("  - collapsed_diagnostics.png   (K, μ, shape traces + posterior K)")
println("  - result.png                  (emitters + GT + localizations)")
println("  - posterior_heatmap.png        (Rao-Blackwellized posterior image)")
println("  - posterior_image.png          (raw grayscale posterior)")
println("  - nn_distances.png            (nearest-neighbor distance histogram)")
println("  - acceptance_rates.png        (move acceptance rates)")
println("  - chain_animation.mp4         (collapsed Gibbs chain animation)")
println("\nSMLMRender outputs:")
println("  - render_mapn_gaussian.png    (Gaussian render of emitters)")
println("  - render_sr_gaussian.png      (Gaussian SR of input locs)")
println("  - render_circles.png          (locs gray + emitters red)")
println("  - render_comparison.png       (locs gray + emitters red + GT blue)")
println("\nMetrics:")
println("  Jaccard Index: $(round(metrics.jaccard, digits=3))")
println("  F1 Score: $(round(metrics.f1, digits=3))")
println("  RMSE: $(round(metrics.rmse * 1000, digits=1)) nm")

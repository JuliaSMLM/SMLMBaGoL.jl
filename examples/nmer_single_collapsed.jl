# Single N-mer — Collapsed Gibbs Sampler
# =======================================
# Collapsed Gibbs version of nmer_single_test.jl.
# Uses run_collapsed_chain with accumulators instead of RJMCMC chain.
# Emitter positions come from ClusterStats posterior (no MAP-N step).
#
# Run with: julia --project=examples examples/nmer_single_collapsed.jl

using Pkg
Pkg.activate(@__DIR__)

using SMLMBaGoL
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
const N_EMITTERS = 8
const CLUSTER_DIAMETER = 0.050    # μm (50 nm)

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
const OUTPUT_DIR = joinpath(@__DIR__, "output", "nmer_single_collapsed")

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
    for (ex, ey) in positions
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
                loc_id, 1, 0, loc_id
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

# Trace collector for diagnostics
mutable struct CollapsedTraceCollector
    iters::Vector{Int}
    ks::Vector{Int}
    mus::Vector{Float64}
    shapes::Vector{Float64}
end
CollapsedTraceCollector() = CollapsedTraceCollector(Int[], Int[], Float64[], Float64[])

trace = CollapsedTraceCollector()

function trace_callback(iter, state, μ, shape)
    push!(trace.iters, iter)
    push!(trace.ks, state.n_active)
    push!(trace.mus, μ)
    push!(trace.shapes, shape)
end

# Create accumulators
count_hist = EmitterCountHist()
post_img = PosteriorImage(pixel_size=0.002)
nn_hist = NNDistHist(max_dist=0.100, n_bins=100)

result = run_collapsed_chain(
    locs;
    λ_K = Float64(N_EMITTERS),
    n_iterations = N_ITERATIONS,
    burn_in = BURN_IN,
    accumulators = AbstractAccumulator[count_hist, post_img, nn_hist],
    callback = trace_callback,
    callback_interval = CALLBACK_INTERVAL,
    verbose = true
)

# Extract emitters from final state
emitters = SMLMBaGoL.extract_emitters(result.state, locs)

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
# GENERATE VISUALIZATIONS
# =============================================================================

println("\n" * "-"^60)
println("Generating visualizations...")

# Helper functions
function draw_circle!(ax, x, y, r; color=:black, linewidth=1.0, alpha=1.0)
    θ = range(0, 2π, length=50)
    cx = x .+ r .* cos.(θ)
    cy = y .+ r .* sin.(θ)
    lines!(ax, cx, cy, color=(color, alpha), linewidth=linewidth)
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
    # Mark true cluster diameter
    vlines!(ax4, [CLUSTER_DIAMETER * 1000], color=:red, linewidth=2,
            label="Cluster d = $(CLUSTER_DIAMETER * 1000) nm")
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

# 6. Posterior image (raw PNG)
println("  [6/7] Posterior image PNG (alt method via save_posterior_png)...")
# Already saved above

# 7. SMLMRender suite
println("  [7/7] SMLMRender suite...")
bagol_smld = SMLMData.BasicSMLD(emitters, camera, 1, 1)
render_bagol_suite(locs_smld, bagol_smld;
    true_positions = true_positions,
    prefix = "render",
    output_dir = OUTPUT_DIR,
    pixel_size = 1.0)

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
println("\nSMLMRender outputs:")
println("  - render_mapn_gaussian.png    (Gaussian render of emitters)")
println("  - render_sr_gaussian.png      (Gaussian SR of input locs)")
println("  - render_circles.png          (locs gray + emitters red)")
println("  - render_comparison.png       (locs gray + emitters red + GT blue)")
println("\nMetrics:")
println("  Jaccard Index: $(round(metrics.jaccard, digits=3))")
println("  F1 Score: $(round(metrics.f1, digits=3))")
println("  RMSE: $(round(metrics.rmse * 1000, digits=1)) nm")

# BaGoL on GenMAb HexaBody RGY — FULL FIELD (Cell_01)
# =====================================================
# Full 25×25 μm FOV analysis. Saves all outputs incrementally.
# Run with: julia --threads=auto --project=dev -e 'include("dev/genmab_fullfield.jl")'

using Pkg
Pkg.develop(path=joinpath(@__DIR__, "..", "..", "SMLMRender"))

using MAT, SMLMData, SMLMBaGoL, SMLMRender, CairoMakie, Statistics, Colors
using Serialization, Dates

function create_target(x_min, x_max, y_min, y_max; pixel_size=1.0)
    w = max(10, ceil(Int, (x_max - x_min) * 1000 / pixel_size))
    h = max(10, ceil(Int, (y_max - y_min) * 1000 / pixel_size))
    SMLMRender.Image2DTarget(w, h, Float64(pixel_size), (x_min, x_max), (y_min, y_max))
end

OUTPUT_DIR = joinpath(@__DIR__, "output", "genmab_fullfield")
rm(OUTPUT_DIR; force=true, recursive=true)
mkpath(OUTPUT_DIR)

println("Output directory: $OUTPUT_DIR")
println("Threads: $(Threads.nthreads())")
flush(stdout)

## ── Load SMITE results ──────────────────────────────────────────────────────

println("\n── Loading MAT file ──")
flush(stdout)

mat_path = "/mnt/nas/cellpath/Genmab/Data/20250603_A431_SaturatingIgG10min+C1q/" *
    "A431_IgG1-2F8-RGY-AF647_5ugml_10min+C1q/Results/Cell_01/Label_01/" *
    "Data_2025-6-9-19-40-7/Data_2025-6-9-19-40-7_Results.mat"

t_load = @elapsed begin
    f = matopen(mat_path)
    smd = read(f, "SMD")
    close(f)
end

pixel_size = smd["PixelSize"]

X_um = vec(Float64.(smd["X"])) .* pixel_size
Y_um = vec(Float64.(smd["Y"])) .* pixel_size
X_SE_um = vec(Float64.(smd["X_SE"])) .* pixel_size
Y_SE_um = vec(Float64.(smd["Y_SE"])) .* pixel_size

println("  Loaded in $(round(t_load, digits=2)) s")
println("  Total localizations: $(length(X_um))")
println("  Field: $(round(maximum(X_um)-minimum(X_um), digits=1)) × $(round(maximum(Y_um)-minimum(Y_um), digits=1)) μm")
println("  Median σ_x: $(round(median(X_SE_um)*1000, digits=1)) nm")
flush(stdout)

## ── Build SMLD ──────────────────────────────────────────────────────────────

locs = [SMLMData.Emitter2DFit(
    X_um[i], Y_um[i], 100.0, 10.0, X_SE_um[i], Y_SE_um[i], 0.0, 0.0, 0.0, 1, 1, 0, i
) for i in eachindex(X_um)]

cam = SMLMData.IdealCamera(256, 256, pixel_size)
smld = SMLMData.BasicSMLD(locs, cam, 1, 1)

xmin, xmax = extrema(X_um)
ymin, ymax = extrema(Y_um)

## ── Estimate μ from partitioning ─────────────────────────────────────────────

println("\n── Pre-partitioning ──")
flush(stdout)

t_partition = @elapsed begin
    pre_partitions, _ = partition_locs(locs; nsigma=2.0, min_size=0, max_size=1000)
end
partition_sizes = [length(p.locs) for p in pre_partitions]
median_locs = median(Float64.(partition_sizes))
est_μ = median_locs / 5.0

println("  $(length(pre_partitions)) partitions in $(round(t_partition, digits=1)) s")
println("  Locs/partition: median=$(round(median_locs, digits=0)), mean=$(round(mean(partition_sizes), digits=1)), max=$(maximum(partition_sizes))")
println("  Estimated μ: $(round(est_μ, digits=1))")
flush(stdout)

# Save partition size distribution
open(joinpath(OUTPUT_DIR, "partition_sizes.txt"), "w") do io
    for s in partition_sizes
        println(io, s)
    end
end
println("  Saved: partition_sizes.txt")
flush(stdout)

## ── Run BaGoL (full field) ───────────────────────────────────────────────────

println("\n── Running BaGoL (full field) ──")
println("  n_iterations=10000, burn_in=2000, nsigma=2.0")
println("  posterior_pixel_size=0.002")
println("  START: $(Dates.now())")
flush(stdout)

GC.gc()
println("  Memory before BaGoL: $(round(Sys.total_memory()/1024^3, digits=1)) GiB total, $(round(Sys.free_memory()/1024^3, digits=1)) GiB free")
flush(stdout)
t_bagol = @elapsed begin
    result_smld, diag = try
        run_bagol(smld;
            n_iterations=10_000,
            burn_in=2_000,
            nsigma=2.0,
            shape=2.0,
            learn_shape=true,
            sync_interval=500,
            μ_prior_shape=2.0,
            μ_prior_scale=max(est_μ / 2.0, 5.0),
            posterior_pixel_size=0.002,
            posterior_xlim=(xmin, xmax),
            posterior_ylim=(ymin, ymax),
            progress_file=joinpath(OUTPUT_DIR, "progress.log"),
            verbose=true)
    catch e
        println(stderr, "\nERROR in run_bagol:")
        println(stderr, sprint(showerror, e, catch_backtrace()))
        flush(stderr)
        rethrow()
    end
end

println("\n  FINISH: $(Dates.now())")
println("  run_bagol: $(round(t_bagol, digits=1)) s ($(round(t_bagol/60, digits=1)) min)")
flush(stdout)

emitters = result_smld.emitters

## ── Save results immediately ─────────────────────────────────────────────────

println("\n── Saving results ──")
flush(stdout)

# Serialize full result for later analysis
serialize(joinpath(OUTPUT_DIR, "result_smld.jls"), result_smld)
serialize(joinpath(OUTPUT_DIR, "diagnostics.jls"), diag)
println("  Saved: result_smld.jls, diagnostics.jls")

# Human-readable summary with timing (separate from standard report summary)
open(joinpath(OUTPUT_DIR, "run_summary.txt"), "w") do io
    println(io, "GenMAb Full-Field BaGoL Results")
    println(io, "="^50)
    println(io, "Date: $(Dates.now())")
    println(io, "Threads: $(Threads.nthreads())")
    println(io, "")
    println(io, "Input")
    println(io, "  Localizations: $(length(locs))")
    println(io, "  Field: $(round(xmax-xmin, digits=1)) × $(round(ymax-ymin, digits=1)) μm")
    println(io, "  Median σ: $(round(median(X_SE_um)*1000, digits=1)) nm")
    println(io, "")
    println(io, "Parameters")
    println(io, "  n_iterations: 10000")
    println(io, "  burn_in: 2000")
    println(io, "  nsigma: 2.0")
    println(io, "  posterior_pixel_size: 0.002")
    println(io, "")
    println(io, "Results")
    println(io, "  Output emitters: $(length(emitters))")
    println(io, "  Grouping ratio: $(round(length(locs)/max(length(emitters),1), digits=1))×")
    println(io, "  Partitions: $(diag.n_partitions)")
    println(io, "  Final μ: $(round(diag.final_μ, digits=2))")
    println(io, "  Final shape: $(round(diag.final_shape, digits=2))")
    println(io, "")
    println(io, "Acceptance Rates")
    for (move, rate) in sort(collect(diag.acceptance_rates))
        println(io, "  $move: $(round(100*rate, digits=1))%")
    end
    println(io, "")
    println(io, "Timing")
    println(io, "  MAT load: $(round(t_load, digits=2)) s")
    println(io, "  Partitioning: $(round(t_partition, digits=1)) s")
    println(io, "  run_bagol: $(round(t_bagol, digits=1)) s ($(round(t_bagol/60, digits=1)) min)")
    println(io, "  Throughput: $(round(length(locs)/t_bagol, digits=0)) locs/s")
end
println("  Saved: run_summary.txt")
flush(stdout)

## ── Print results ────────────────────────────────────────────────────────────

println("\n── Results ──")
println("  Input localizations: $(length(locs))")
println("  Output emitters: $(length(emitters))")
println("  Grouping ratio: $(round(length(locs)/max(length(emitters),1), digits=1))×")
println("  Partitions: $(diag.n_partitions)")
println("  Final μ: $(round(diag.final_μ, digits=2))")
println("  Final shape: $(round(diag.final_shape, digits=2))")

println("\n── Acceptance Rates ──")
for (move, rate) in sort(collect(diag.acceptance_rates))
    println("  $move: $(round(100*rate, digits=1))%")
end

println("\n── Timing ──")
println("  MAT load:     $(round(t_load, digits=2)) s")
println("  Partitioning: $(round(t_partition, digits=1)) s")
println("  run_bagol:    $(round(t_bagol, digits=1)) s ($(round(t_bagol/60, digits=1)) min)")
println("  Throughput:   $(round(length(locs)/t_bagol, digits=0)) locs/s")
flush(stdout)

## ── Standard report + visualizations ─────────────────────────────────────────

bagol_smld = SMLMData.BasicSMLD(emitters, cam, 1, 1)
locs_smld = SMLMData.BasicSMLD(locs, cam, 1, 1)

println("\n── Standard report ──")
flush(stdout)
report = compute_report(result_smld, diag; locs_smld=locs_smld)
write_report(report; output_dir=OUTPUT_DIR)
plot_report(report; output_dir=OUTPUT_DIR)

fov = (xmin, xmax, ymin, ymax)
render_report(locs_smld, bagol_smld;
    output_dir=OUTPUT_DIR, prefix="genmab_ff", pixel_size=2.0, fov=fov)
flush(stdout)

# (compose now handled by render_report above)
flush(stdout)

# Zoomed ROI — densest 2×2 μm region (1 nm/px for detail)
println("\nZoomed ROI...")
flush(stdout)
# Find densest 2×2 μm region via 2D histogram binning (fast)
bin_size = 0.5
nx = ceil(Int, (xmax - xmin) / bin_size) + 1
ny = ceil(Int, (ymax - ymin) / bin_size) + 1
density_grid = zeros(Int, nx, ny)
for l in locs
    ix = clamp(ceil(Int, (l.x - xmin) / bin_size), 1, nx)
    iy = clamp(ceil(Int, (l.y - ymin) / bin_size), 1, ny)
    density_grid[ix, iy] += 1
end
# Convolve with 2×2 μm window (4 bins at 0.5 μm)
w = ceil(Int, 1.0 / bin_size)  # half-window in bins
best_cx, best_cy, best_n = let _cx=0.0, _cy=0.0, _n=0
    for ix in (w+1):(nx-w)
        for iy in (w+1):(ny-w)
            n = sum(density_grid[ix-w:ix+w, iy-w:iy+w])
            if n > _n
                _cx = xmin + (ix - 0.5) * bin_size
                _cy = ymin + (iy - 0.5) * bin_size
                _n = n
            end
        end
    end
    _cx, _cy, _n
end
zoom_roi = (best_cx - 1.0, best_cx + 1.0, best_cy - 1.0, best_cy + 1.0)
zoom_target = create_target(zoom_roi...; pixel_size=1.0)
(zbg, _) = render(locs_smld;
    strategy = CircleRender(), color = :white,
    target = zoom_target, clip_percentile = nothing)
(zfg, _) = render(bagol_smld;
    strategy = CircleRender(), color = :red,
    target = zoom_target, clip_percentile = nothing)
zoomed = compose(zbg, zfg; blend=:replace)
save_image(joinpath(OUTPUT_DIR, "zoom_compose.png"), zoomed)
println("  Saved: zoom_compose.png")
flush(stdout)

println("\n" * "="^60)
println("All outputs saved to: $OUTPUT_DIR")
println("Total wall time: $(round((t_load + t_partition + t_bagol)/60, digits=1)) min (excl. viz)")
println("="^60)

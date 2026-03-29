# Viz-only for GenMAb fullfield — loads saved results, skips MCMC
# Run with: julia --threads=auto --project=dev dev/genmab_fullfield_viz.jl

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

## ── Load saved results ──────────────────────────────────────────────────────

println("Loading saved results...")
result_smld = deserialize(joinpath(OUTPUT_DIR, "result_smld.jls"))
diag = deserialize(joinpath(OUTPUT_DIR, "diagnostics.jls"))
emitters = result_smld.emitters
println("  $(length(emitters)) emitters loaded")

## ── Reload raw data for viz ─────────────────────────────────────────────────

println("Loading MAT file...")
mat_path = "/mnt/nas/cellpath/Genmab/Data/20250603_A431_SaturatingIgG10min+C1q/" *
    "A431_IgG1-2F8-RGY-AF647_5ugml_10min+C1q/Results/Cell_01/Label_01/" *
    "Data_2025-6-9-19-40-7/Data_2025-6-9-19-40-7_Results.mat"

f = matopen(mat_path)
smd = read(f, "SMD")
close(f)

pixel_size = smd["PixelSize"]
X_um = vec(Float64.(smd["X"])) .* pixel_size
Y_um = vec(Float64.(smd["Y"])) .* pixel_size
X_SE_um = vec(Float64.(smd["X_SE"])) .* pixel_size
Y_SE_um = vec(Float64.(smd["Y_SE"])) .* pixel_size

locs = [SMLMData.Emitter2DFit(
    X_um[i], Y_um[i], 100.0, 10.0, X_SE_um[i], Y_SE_um[i], 0.0, 0.0, 0.0, 1, 1, 0, i
) for i in eachindex(X_um)]

cam = SMLMData.IdealCamera(256, 256, pixel_size)
xmin, xmax = extrema(X_um)
ymin, ymax = extrema(Y_um)

println("  $(length(locs)) localizations, $(round(xmax-xmin, digits=1)) × $(round(ymax-ymin, digits=1)) μm")

bagol_smld = SMLMData.BasicSMLD(emitters, cam, 1, 1)
locs_smld = SMLMData.BasicSMLD(locs, cam, 1, 1)

## ── Standard report + visualizations ──────────────────────────────────────────

println("\n── Standard report ──")
report = compute_report(result_smld, diag; locs_smld=locs_smld)
write_report(report; output_dir=OUTPUT_DIR)
plot_report(report; output_dir=OUTPUT_DIR)

fov = (xmin, xmax, ymin, ymax)
render_report(locs_smld, bagol_smld;
    output_dir=OUTPUT_DIR, prefix="genmab_ff", pixel_size=2.0, fov=fov)

# (compose now handled by render_report above)

# Zoomed ROI — densest 2×2 μm region (1 nm/px for detail)
println("\nZoomed ROI...")
flush(stdout)
bin_size = 0.5
nx = ceil(Int, (xmax - xmin) / bin_size) + 1
ny = ceil(Int, (ymax - ymin) / bin_size) + 1
density_grid = zeros(Int, nx, ny)
for l in locs
    ix = clamp(ceil(Int, (l.x - xmin) / bin_size), 1, nx)
    iy = clamp(ceil(Int, (l.y - ymin) / bin_size), 1, ny)
    density_grid[ix, iy] += 1
end
w = ceil(Int, 1.0 / bin_size)
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

println("\n" * "="^60)
println("All outputs saved to: $OUTPUT_DIR")
println("="^60)

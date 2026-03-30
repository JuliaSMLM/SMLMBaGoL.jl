# BaGoL on GenMAb HexaBody RGY data (Cell_01)
# Load SMITE results, extract ROI, run collapsed Gibbs sampler
# Outputs: SMLMRender suite, posterior heatmap, μ/shape diagnostics

using Pkg
Pkg.develop(path=joinpath(@__DIR__, "..", "..", "SMLMRender"))

using MAT, SMLMData, SMLMBaGoL, SMLMRender, CairoMakie, Statistics

OUTPUT_DIR = joinpath(@__DIR__, "output", "genmab")
rm(OUTPUT_DIR; force=true, recursive=true)
mkpath(OUTPUT_DIR)

## ── Load SMITE results ──────────────────────────────────────────────────────

mat_path = "/mnt/nas/cellpath/Genmab/Data/20250603_A431_SaturatingIgG10min+C1q/" *
    "A431_IgG1-2F8-RGY-AF647_5ugml_10min+C1q/Results/Cell_01/Label_01/" *
    "Data_2025-6-9-19-40-7/Data_2025-6-9-19-40-7_Results.mat"

f = matopen(mat_path)
smd = read(f, "SMD")
close(f)

pixel_size = smd["PixelSize"]  # 0.0978 μm
println("Pixel size: $(pixel_size * 1000) nm")

# Convert pixels → μm
X_um = vec(Float64.(smd["X"])) .* pixel_size
Y_um = vec(Float64.(smd["Y"])) .* pixel_size
X_SE_um = vec(Float64.(smd["X_SE"])) .* pixel_size
Y_SE_um = vec(Float64.(smd["Y_SE"])) .* pixel_size

println("Total localizations: $(length(X_um))")
println("Field: $(round(maximum(X_um) - minimum(X_um), digits=1)) × " *
        "$(round(maximum(Y_um) - minimum(Y_um), digits=1)) μm")
println("Median σ_x: $(round(median(X_SE_um) * 1000, digits=1)) nm")

## ── Extract ROI ─────────────────────────────────────────────────────────────

# 2×2 μm ROI near center of field
roi_center_x = median(X_um)
roi_center_y = median(Y_um)
roi_half = 1.0  # μm → 2×2 μm ROI

roi_mask = (abs.(X_um .- roi_center_x) .< roi_half) .&
           (abs.(Y_um .- roi_center_y) .< roi_half)

x = X_um[roi_mask]
y = Y_um[roi_mask]
σ_x = X_SE_um[roi_mask]
σ_y = Y_SE_um[roi_mask]

println("\nROI: $(round(roi_center_x, digits=2)) ± $roi_half μm × " *
        "$(round(roi_center_y, digits=2)) ± $roi_half μm")
println("ROI localizations: $(length(x))")

## ── Build SMLD ──────────────────────────────────────────────────────────────

# Emitter2DFit: x, y, photons, bg, σ_x, σ_y, σ_xy, σ_photons, σ_bg, frame, dataset, track_id, id
locs = [SMLMData.Emitter2DFit(
    x[i], y[i], 100.0, 10.0, σ_x[i], σ_y[i], 0.0, 0.0, 0.0, 1, 1, 0, i
) for i in eachindex(x)]

cam = SMLMData.IdealCamera(256, 256, pixel_size)
smld = SMLMData.BasicSMLD(locs, cam, 1, 1)
println("SMLD built: $(length(smld.emitters)) localizations")

## ── Run BaGoL ───────────────────────────────────────────────────────────────

# ROI bounds for posterior image
roi_xmin = roi_center_x - roi_half
roi_xmax = roi_center_x + roi_half
roi_ymin = roi_center_y - roi_half
roi_ymax = roi_center_y + roi_half

# Estimate μ from data: partition, count median locs/partition as proxy
pre_partitions, _ = partition_locs(locs; partition_sigma=2.0, min_size=0, max_size=1000)
median_locs_per_partition = median(Float64[length(p.locs) for p in pre_partitions])
# Rough estimate: assume ~5 emitters per partition → μ ≈ locs/5
est_μ = median_locs_per_partition / 5.0
println("\nEstimated μ from data: $(round(est_μ, digits=1)) (median partition size=$(round(median_locs_per_partition, digits=0)))")

println("\nRunning BaGoL (collapsed Gibbs, hierarchical)...")
result_smld, diag = run_bagol(smld;
    n_iterations=10_000,
    burn_in=2_000,
    partition_sigma=2.0,
    μ=est_μ,
    shape=2.0,
    learn_distribution=true,
    sync_interval=500,
    posterior_pixel_size=0.002,  # 2 nm pixels
    posterior_xlim=(roi_xmin, roi_xmax),
    posterior_ylim=(roi_ymin, roi_ymax),
)

emitters = result_smld.emitters

println("\n── Results ──")
println("Input localizations: $(length(locs))")
println("Output emitters: $(length(emitters))")
println("Partitions: $(diag.n_partitions)")
println("Final μ: $(round(diag.final_μ, digits=2))")
println("Final shape: $(round(diag.final_shape, digits=2))")

## ── Setup rendering ─────────────────────────────────────────────────────────

bagol_smld = SMLMData.BasicSMLD(emitters, cam, 1, 1)
locs_smld = SMLMData.BasicSMLD(locs, cam, 1, 1)

## ── Standard report (summary, CSVs, posterior image, Makie plots) ────────────

println("\n── Standard report ──")
report = compute_report(result_smld, diag; locs_smld=locs_smld)
write_report(report; output_dir=OUTPUT_DIR)
plot_report(report; output_dir=OUTPUT_DIR)

## ── SMLMRender suite ─────────────────────────────────────────────────────────

fov = (roi_xmin, roi_xmax, roi_ymin, roi_ymax)
render_report(locs_smld, bagol_smld;
    output_dir=OUTPUT_DIR, pixel_size=2.0, fov=fov,
    partition_ids=report.partition_ids)

println("\nAll outputs saved to: $OUTPUT_DIR")

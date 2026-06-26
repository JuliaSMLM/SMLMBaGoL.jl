module BaGoLRenderExt

using SMLMBaGoL
using SMLMData
using SMLMRender

# ============================================================================
# render_report — SMLMRender visualization suite
# ============================================================================

"""
    render_report(locs_smld, bagol_smld; output_dir, true_positions, partition_ids,
                  pixel_size=1.0, zoom=nothing, prefix="render", ...)

Render the standard visualization suite using SMLMRender.

**Every panel is rendered into one shared target** — identical physical bounds and identical
`width × height` in pixels — so the outputs overlay exactly and pan/zoom in lockstep in the
viewer (e.g. an FE pin comparing two of them).

Always creates:
- `{prefix}_mapn.png`       — Gaussian SR render of the BaGoL MAP-N result
- `{prefix}_sr.png`         — Gaussian SR render of the input localizations
- `{prefix}_circles.png`    — white localizations + red MAP-N emitters (uncertainty ellipses)
- `{prefix}_partitions.png` — partition-colored localizations (with `partition_ids`)

With ground truth:
- `{prefix}_groundtruth.png` — white locs + blue GT + green oracle + red found

Resolution is `pixel_size` nm per output pixel (default `1.0`). Alternatively pass `zoom`
(camera-pixel relative, like SMLMRender's `zoom`): output pixels are
`camera_pixel_size / zoom` nm, so e.g. `zoom=50` renders at 50× the camera sampling. When
`zoom` is given it overrides `pixel_size`.
"""
function SMLMBaGoL.render_report(
    locs_smld::SMLMData.SMLD,
    bagol_smld::SMLMData.SMLD;
    output_dir::String = "output",
    true_positions::Vector{Tuple{Float64, Float64}} = Tuple{Float64, Float64}[],
    partition_ids::Vector{Int} = Int[],
    pixel_size::Real = 1.0,
    zoom::Union{Nothing, Real} = nothing,
    prefix::String = "render",
    fov::Union{Nothing, Tuple{Float64, Float64, Float64, Float64}} = nothing,
    se_adjust = 0.0,
    force_se_adjust::Bool = false
)
    mkpath(output_dir)

    # When se_adjust is in effect, draw the localization ellipses at the σ BaGoL
    # actually used (σ² + τ²) so circle diameters reflect the correction. No-op
    # (returns locs_smld) when se_adjust=0 or the SMLD is already σ-corrected.
    locs_render = SMLMBaGoL.apply_se_adjust(locs_smld, se_adjust; force_se_adjust=force_se_adjust)

    # Calculate bounds
    if fov !== nothing
        x_min, x_max, y_min, y_max = fov
    else
        x_min, x_max, y_min, y_max = SMLMBaGoL.compute_fov(locs_smld)
    end

    # Resolution: `zoom` (camera-pixel relative) overrides `pixel_size` (nm/pixel) when set,
    # mirroring SMLMRender's `zoom` — output pixels are camera_pixel_size/zoom nm.
    px_nm = Float64(pixel_size)
    if zoom !== nothing
        px_nm = SMLMRender.get_camera_pixel_size(locs_smld.camera) / zoom
    end

    # One shared target for every panel ⇒ identical bounds and identical pixel dimensions,
    # so the outputs overlay and pan/zoom in lockstep (FE pin compatibility).
    target = _create_target(x_min, x_max, y_min, y_max; pixel_size = px_nm)

    println("  Render bounds: x=[$(round(x_min*1000, digits=1)), $(round(x_max*1000, digits=1))] nm, " *
            "y=[$(round(y_min*1000, digits=1)), $(round(y_max*1000, digits=1))] nm")
    println("  Image size: $(target.width) x $(target.height) pixels at $(round(px_nm, digits=3)) nm/pixel" *
            (zoom === nothing ? "" : " (zoom=$(zoom)×)"))

    # 1. Gaussian render of BaGoL MAP-N result
    mapn_path = joinpath(output_dir, "$(prefix)_mapn.png")
    render(bagol_smld; strategy=GaussianRender(), target=target,
           colormap=:inferno, filename=mapn_path)
    println("Saved: $mapn_path")

    # 2. Gaussian SR render of input localizations
    sr_path = joinpath(output_dir, "$(prefix)_sr.png")
    render(locs_smld; strategy=GaussianRender(), target=target,
           colormap=:inferno, filename=sr_path)
    println("Saved: $sr_path")

    # 3. Circles: white localizations + red MAP-N emitters
    circles_path = joinpath(output_dir, "$(prefix)_circles.png")
    (bg_img, _) = render(locs_render; strategy=EllipseRender(), color=:white,
                         target=target, clip_percentile=nothing)
    (fg_img, _) = render(bagol_smld; strategy=EllipseRender(), color=:red,
                         target=target, clip_percentile=nothing)
    combined = compose(bg_img, fg_img; blend=:replace)
    save_image(circles_path, combined)
    println("Saved: $circles_path")

    # 4. Partition-colored localizations (categorical coloring on single SMLD)
    if !isempty(partition_ids) && length(partition_ids) == length(locs_smld.emitters)
        part_path = joinpath(output_dir, "$(prefix)_partitions.png")
        # Set dataset field to partition ID for categorical coloring
        part_emitters = [SMLMData.Emitter2DFit{Float64}(
            Float64(e.x), Float64(e.y), Float64(e.photons), Float64(e.bg),
            Float64(e.σ_x), Float64(e.σ_y), Float64(e.σ_photons), Float64(e.σ_bg);
            σ_xy=Float64(e.σ_xy), frame=e.frame, dataset=partition_ids[i],
            track_id=e.track_id, id=e.id
        ) for (i, e) in enumerate(locs_render.emitters)]
        part_smld = SMLMData.BasicSMLD(part_emitters, locs_smld.camera, 1, 1)
        render(part_smld; strategy=EllipseRender(), color_by=:dataset,
               categorical=true, target=target, filename=part_path)
        println("Saved: $part_path")
    end

    # 5. Ground truth overlay: white locs + blue GT + green oracle + red found
    if !isempty(true_positions)
        gt_path = joinpath(output_dir, "$(prefix)_groundtruth.png")

        (base_img, _) = render(locs_render; strategy=EllipseRender(), color=:white,
                               target=target, clip_percentile=nothing)

        # GT (blue)
        gt_smld = _positions_to_smld(true_positions, locs_smld.camera)
        (gt_img, _) = render(gt_smld; strategy=EllipseRender(), color=:blue,
                             target=target, clip_percentile=nothing)
        comp = compose(base_img, gt_img; blend=:replace)

        # Oracle MAP-N (green) if track_id available
        has_oracle = any(e.track_id != 0 for e in locs_smld.emitters)
        if has_oracle
            oracle_assignments = Int32[loc.track_id for loc in locs_smld.emitters]
            oracle_emitters = SMLMBaGoL._emitters_from_assignments(
                oracle_assignments, locs_smld.emitters)
            oracle_smld = SMLMData.BasicSMLD(oracle_emitters, locs_smld.camera, 1, 1)
            (oracle_img, _) = render(oracle_smld; strategy=EllipseRender(), color=:green,
                                     target=target, clip_percentile=nothing)
            comp = compose(comp, oracle_img; blend=:replace)
        end

        # Found MAP-N (red)
        (found_img, _) = render(bagol_smld; strategy=EllipseRender(), color=:red,
                                target=target, clip_percentile=nothing)
        comp = compose(comp, found_img; blend=:replace)

        save_image(gt_path, comp)
        println("Saved: $gt_path")
    end

    return target
end

# ============================================================================
# Helpers
# ============================================================================

function _create_target(x_min, x_max, y_min, y_max; pixel_size::Real=1.0)
    width = max(10, ceil(Int, (x_max - x_min) * 1000 / pixel_size))
    height = max(10, ceil(Int, (y_max - y_min) * 1000 / pixel_size))
    return SMLMRender.Image2DTarget(width, height, Float64(pixel_size),
                                     (x_min, x_max), (y_min, y_max))
end

function _positions_to_smld(positions::Vector{Tuple{Float64, Float64}},
                             camera::SMLMData.AbstractCamera; σ::Float64=0.005)
    emitters = [SMLMData.Emitter2DFit(
        pos[1], pos[2], 1000.0, 0.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, i
    ) for (i, pos) in enumerate(positions)]
    return SMLMData.BasicSMLD(emitters, camera, 1, 1)
end

end # module

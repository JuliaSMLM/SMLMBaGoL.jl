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

With `posterior_image` (the `diagnostics.posterior_image` NamedTuple):
- `{prefix}_posterior.png`  — Rao-Blackwellized posterior image, resampled onto the shared
  target so it has identical pixel dimensions to every other panel (pans/zooms in lockstep).

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
    posterior_image::Union{Nothing, NamedTuple} = nothing,
    fov::Union{Nothing, Tuple{Float64, Float64, Float64, Float64}} = nothing,
    se_adjust = 0.0,
    mapn_clip_percentile::Real = 0.99,
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

    # 0. Rao-Blackwellized posterior image — resampled onto the shared target (identical pixel
    #    dims to every other panel ⇒ pans/zooms in lockstep) and colorized + saved through
    #    SMLMRender's libpng-backed save_image. (The core pure-Julia save_posterior_png writer is
    #    UNCOMPRESSED — ~1 GB at 1 nm full-field — so it is deliberately not used here.)
    if posterior_image !== nothing
        post_path = joinpath(output_dir, "$(prefix)_posterior.png")
        resampled = _resample_posterior(posterior_image, x_min, x_max, y_min, y_max,
                                        target.width, target.height)
        save_image(post_path, _posterior_rgb(resampled))
        println("Saved: $post_path")
    end

    # 1. Gaussian render of BaGoL MAP-N result
    mapn_path = joinpath(output_dir, "$(prefix)_mapn.png")
    render(bagol_smld; strategy=GaussianRender(), target=target,
           colormap=:inferno, clip_percentile=mapn_clip_percentile, filename=mapn_path)
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

# Resample a posterior-image NamedTuple (image[nx,ny] + edges_x/edges_y/pixel_size in μm) onto a
# W×H grid spanning [x_min,x_max]×[y_min,y_max] μm (nearest-neighbour, mapped by physical
# coordinate so it is robust to a bounds/pixel-size mismatch between the posterior grid and the
# render target). Returns Int[W,H] in the [ix,iy] layout save_posterior_png expects, so the
# posterior PNG lands at EXACTLY the render target's pixel dimensions and overlays every panel.
function _resample_posterior(post::NamedTuple, x_min, x_max, y_min, y_max, W::Int, H::Int)
    src = post.image
    nx, ny = size(src)
    sx0 = post.edges_x[1]
    sy0 = post.edges_y[1]
    spx = post.pixel_size
    px = (x_max - x_min) / W
    py = (y_max - y_min) / H
    out = zeros(Int, W, H)
    @inbounds for iy in 1:H
        yc = y_min + (iy - 0.5) * py
        siy = floor(Int, (yc - sy0) / spx) + 1
        (1 <= siy <= ny) || continue
        for ix in 1:W
            xc = x_min + (ix - 0.5) * px
            six = floor(Int, (xc - sx0) / spx) + 1
            (1 <= six <= nx) || continue
            out[ix, iy] = src[six, siy]
        end
    end
    return out
end

# Colorize a resampled posterior count image (Int[nx,ny] = [ix,iy]) into a Matrix{RGB}[ny,nx]
# (row=y from top, col=x — matching the other panels' orientation) with percentile-scaled
# `:inferno`, using SMLMRender's colormap. Returned to save_image ⇒ libpng-compressed PNG.
function _posterior_rgb(img::Matrix{Int}; percentile::Float64 = 0.99, colormap::Symbol = :inferno)
    nx, ny = size(img)
    lut = SMLMRender.get_colormap_lut(colormap)
    zero_color = SMLMRender.colormap_lookup(lut, 0.0)   # empty pixels → colormap floor (as in save_posterior_png)
    out = fill(zero_color, ny, nx)
    nz = filter(>(0), vec(img))
    isempty(nz) && return out
    s = sort(nz)
    vmax = Float64(s[clamp(round(Int, percentile * length(s)), 1, length(s))])
    vmax <= 0 && return out
    @inbounds for iy in 1:ny, ix in 1:nx
        v = img[ix, iy]
        v > 0 && (out[iy, ix] = SMLMRender.colormap_lookup(lut, min(Float64(v) / vmax, 1.0)))
    end
    return out
end

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

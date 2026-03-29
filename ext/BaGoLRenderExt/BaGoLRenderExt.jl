module BaGoLRenderExt

using SMLMBaGoL
using SMLMData
using SMLMRender

# ============================================================================
# render_report — SMLMRender visualization suite
# ============================================================================

"""
    render_report(locs_smld, bagol_smld; output_dir="output", true_positions=[], pixel_size=1.0, prefix="render")

Render standard visualization suite using SMLMRender.

Always creates:
- `{prefix}_sr_gaussian.png` — Gaussian SR render of input localizations
- `{prefix}_mapn_gaussian.png` — Gaussian render of BaGoL MAP-N result
- `{prefix}_circles.png` — Circle overlay (gray locs + red BaGoL emitters)

With ground truth:
- `{prefix}_comparison.png` — Multi-layer (gray locs + blue GT + green oracle + red found)
"""
function SMLMBaGoL.render_report(
    locs_smld::SMLMData.SMLD,
    bagol_smld::SMLMData.SMLD;
    output_dir::String = "output",
    true_positions::Vector{Tuple{Float64, Float64}} = Tuple{Float64, Float64}[],
    pixel_size::Real = 1.0,
    prefix::String = "render",
    fov::Union{Nothing, Tuple{Float64, Float64, Float64, Float64}} = nothing
)
    mkpath(output_dir)

    # Calculate bounds
    if fov !== nothing
        x_min, x_max, y_min, y_max = fov
    else
        x_min, x_max, y_min, y_max = SMLMBaGoL.compute_fov(locs_smld)
    end

    target = _create_target(x_min, x_max, y_min, y_max; pixel_size)

    println("  Render bounds: x=[$(round(x_min*1000, digits=1)), $(round(x_max*1000, digits=1))] nm, " *
            "y=[$(round(y_min*1000, digits=1)), $(round(y_max*1000, digits=1))] nm")
    println("  Image size: $(target.width) x $(target.height) pixels at $(pixel_size) nm/pixel")

    # 1. Gaussian render of BaGoL MAP-N result
    mapn_path = joinpath(output_dir, "$(prefix)_mapn_gaussian.png")
    render(bagol_smld; strategy=GaussianRender(), target=target,
           colormap=:inferno, filename=mapn_path)
    println("Saved: $mapn_path")

    # 2. Gaussian SR render of input localizations
    sr_path = joinpath(output_dir, "$(prefix)_sr_gaussian.png")
    render(locs_smld; strategy=GaussianRender(), target=target,
           colormap=:inferno, filename=sr_path)
    println("Saved: $sr_path")

    # 3. Circles: white localizations + red MAP-N emitters
    circles_path = joinpath(output_dir, "$(prefix)_circles.png")
    (bg_img, _) = render(locs_smld; strategy=EllipseRender(), color=:white,
                         target=target, clip_percentile=nothing)
    (fg_img, _) = render(bagol_smld; strategy=EllipseRender(), color=:red,
                         target=target, clip_percentile=nothing)
    combined = compose(bg_img, fg_img; blend=:replace)
    save_image(circles_path, combined)
    println("Saved: $circles_path")

    # 4. Ground truth overlay: white locs + blue GT + green oracle + red found
    if !isempty(true_positions)
        gt_path = joinpath(output_dir, "$(prefix)_circles_groundtruth.png")

        (base_img, _) = render(locs_smld; strategy=EllipseRender(), color=:white,
                               target=target, clip_percentile=nothing)

        # GT (blue)
        gt_smld = _positions_to_smld(true_positions, locs_smld.camera)
        (gt_img, _) = render(gt_smld; strategy=EllipseRender(), color=:blue,
                             target=target, clip_percentile=nothing)
        comp = compose(base_img, gt_img; blend=:replace)

        # Oracle MAP-N (green) if track_id available
        has_oracle = any(e.track_id != 0 for e in locs_smld.emitters)
        if has_oracle
            oracle_assignments = Int16[loc.track_id for loc in locs_smld.emitters]
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

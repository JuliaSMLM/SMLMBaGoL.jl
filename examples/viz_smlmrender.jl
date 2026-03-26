# BaGoL SMLMRender Visualization
# ===============================
# Publication-quality rendering of BaGoL results using SMLMRender.
# These functions require only SMLMRender (no CairoMakie dependency).
#
# Usage:
#   include("viz_smlmrender.jl")
#   render_bagol_suite(locs_smld, bagol_smld; output_dir="output")

using SMLMRender
using SMLMData

"""
Calculate render bounds from localizations, expanded by factor.

Returns (x_min, x_max, y_min, y_max) in μm, covering all localizations
plus their 1σ uncertainty, expanded by the given factor.
"""
function calculate_render_bounds(locs_smld::SMLMData.SMLD; expand_factor::Real=2.0)
    emitters = locs_smld.emitters

    # Get bounds including 1σ circles
    x_min = minimum(e.x - e.σ_x for e in emitters)
    x_max = maximum(e.x + e.σ_x for e in emitters)
    y_min = minimum(e.y - e.σ_y for e in emitters)
    y_max = maximum(e.y + e.σ_y for e in emitters)

    # Calculate center and span
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_span = x_max - x_min
    y_span = y_max - y_min

    # Expand by factor (ensure minimum span of 20nm = 0.020 μm)
    x_span = max(x_span * expand_factor, 0.020)
    y_span = max(y_span * expand_factor, 0.020)

    return (
        x_center - x_span/2,
        x_center + x_span/2,
        y_center - y_span/2,
        y_center + y_span/2
    )
end

"""
Create a render target from bounds at specified pixel size.
"""
function create_target(x_min, x_max, y_min, y_max; pixel_size::Real=1.0)
    # Calculate dimensions (pixel_size is in nm, bounds are in μm)
    width = ceil(Int, (x_max - x_min) * 1000 / pixel_size)
    height = ceil(Int, (y_max - y_min) * 1000 / pixel_size)

    # Ensure minimum size
    width = max(width, 10)
    height = max(height, 10)

    return SMLMRender.Image2DTarget(width, height, Float64(pixel_size),
                                     (x_min, x_max), (y_min, y_max))
end

"""
Convert ground truth positions to a BasicSMLD for rendering.

Creates Emitter2DFit entries from (x, y) position tuples.
Uses small default uncertainty for visualization.

# Arguments
- `positions`: Vector of (x, y) tuples in micrometers
- `camera`: Camera from original SMLD
- `σ`: Position uncertainty for rendering circles (default: 0.005 μm = 5 nm)
"""
function positions_to_smld(
    positions::Vector{Tuple{Float64, Float64}},
    camera::SMLMData.AbstractCamera;
    σ::Float64 = 0.005
)
    emitters = [
        SMLMData.Emitter2DFit(
            pos[1], pos[2],     # x, y position
            1000.0, 0.0,        # photons, bg (placeholder)
            σ, σ, 0.0,          # σ_x, σ_y, σ_xy
            0.0, 0.0,           # σ_photons, σ_bg
            1, 1, 0, i          # frame, dataset, track_id, id
        )
        for (i, pos) in enumerate(positions)
    ]
    return SMLMData.BasicSMLD(emitters, camera, 1, 1)
end

"""
    oracle_mapn(locs) -> Vector{Emitter2DFit}

Compute oracle MAP-N emitters using the known true clustering stored in `track_id`.
Each localization's `track_id` field must contain the parent emitter index (set during simulation).
Groups locs by `track_id`, computes ClusterStats posterior for each group.
"""
function oracle_mapn(locs::Vector{<:SMLMData.AbstractEmitter})
    assignments = Int16[loc.track_id for loc in locs]
    return SMLMBaGoL._emitters_from_assignments(assignments, locs)
end

"""
Render complete BaGoL visualization suite.

Calculates render bounds from localizations (2x extent of 1σ circles),
then renders all outputs at 1nm pixel size using consistent bounds.

# Arguments
- `locs_smld`: Input localizations
- `bagol_smld`: BaGoL result
- `true_positions`: Optional ground truth positions (default: empty)
- `prefix`: Filename prefix for output (default: "render")
- `output_dir`: Directory for output files (default: current directory)
- `pixel_size`: Pixel size in nm (default: 1.0)
- `expand_factor`: How much to expand beyond 1σ bounds (default: 2.0)

# Creates files
- `{prefix}_mapn_gaussian.png`: Gaussian render of BaGoL MAP-N result
- `{prefix}_sr_gaussian.png`: Gaussian SR render of input localizations
- `{prefix}_circles.png`: Circle overlay (locs gray + BaGoL red)
- `{prefix}_comparison.png`: Multi-channel (locs gray + BaGoL red + GT blue + oracle green)

# Returns
- `Image2DTarget`: The render target for consistency with other renders
"""
function render_bagol_suite(
    locs_smld::SMLMData.SMLD,
    bagol_smld::SMLMData.SMLD;
    true_positions::Vector{Tuple{Float64, Float64}} = Tuple{Float64, Float64}[],
    prefix::String = "render",
    output_dir::String = ".",
    pixel_size::Real = 1.0,
    expand_factor::Real = 2.0,
    fov::Union{Nothing, Tuple{Float64, Float64, Float64, Float64}} = nothing
)
    # Use explicit FOV bounds if provided, otherwise calculate from data
    if fov !== nothing
        x_min, x_max, y_min, y_max = fov
    else
        x_min, x_max, y_min, y_max = calculate_render_bounds(locs_smld; expand_factor=expand_factor)
    end

    # Create common target for all renders
    target = create_target(x_min, x_max, y_min, y_max; pixel_size=pixel_size)

    println("  Render bounds: x=[$(round(x_min*1000, digits=1)), $(round(x_max*1000, digits=1))] nm, " *
            "y=[$(round(y_min*1000, digits=1)), $(round(y_max*1000, digits=1))] nm")
    println("  Image size: $(target.width) x $(target.height) pixels at $(pixel_size) nm/pixel")

    # 1. Gaussian render of BaGoL MAP-N result
    mapn_path = joinpath(output_dir, "$(prefix)_mapn_gaussian.png")
    render(bagol_smld;
        strategy = GaussianRender(),
        target = target,
        colormap = :inferno,
        filename = mapn_path
    )
    println("Saved: $mapn_path")

    # 2. Gaussian SR render of input localizations
    sr_path = joinpath(output_dir, "$(prefix)_sr_gaussian.png")
    render(locs_smld;
        strategy = GaussianRender(),
        target = target,
        colormap = :inferno,
        filename = sr_path
    )
    println("Saved: $sr_path")

    # 3. Circle overlay: localizations (gray) + BaGoL (red) via compose
    circles_path = joinpath(output_dir, "$(prefix)_circles.png")
    (bg_img, _) = render(locs_smld;
        strategy = EllipseRender(), color = :gray,
        target = target, clip_percentile = nothing)
    (fg_img, _) = render(bagol_smld;
        strategy = EllipseRender(), color = :red,
        target = target, clip_percentile = nothing)
    combined = compose(bg_img, fg_img; blend=:replace)
    save_image(circles_path, combined)
    println("Saved: $circles_path")

    # 4. Comparison with GT and oracle MAP-N if GT provided
    #    Layer order (bottom to top): locs (gray) → GT (blue) → oracle (green) → found (red)
    if !isempty(true_positions)
        comparison_path = joinpath(output_dir, "$(prefix)_comparison.png")

        # Start with locs (gray) as base
        (base_img, _) = render(locs_smld;
            strategy = EllipseRender(), color = :gray,
            target = target, clip_percentile = nothing)

        # GT true positions (blue)
        gt_smld = positions_to_smld(true_positions, locs_smld.camera)
        (gt_img, _) = render(gt_smld;
            strategy = EllipseRender(), color = :blue,
            target = target, clip_percentile = nothing)
        comp = compose(base_img, gt_img; blend=:replace)

        # Oracle MAP-N (green) if track_id info is available
        has_oracle = any(e.track_id != 0 for e in locs_smld.emitters)
        if has_oracle
            oracle_emitters = oracle_mapn(locs_smld.emitters)
            oracle_smld = SMLMData.BasicSMLD(oracle_emitters, locs_smld.camera, 1, 1)
            (oracle_img, _) = render(oracle_smld;
                strategy = EllipseRender(), color = :green,
                target = target, clip_percentile = nothing)
            comp = compose(comp, oracle_img; blend=:replace)
            println("  Oracle MAP-N: $(length(oracle_emitters)) emitters (green)")
        end

        # Found MAP-N (red) on top
        (found_img, _) = render(bagol_smld;
            strategy = EllipseRender(), color = :red,
            target = target, clip_percentile = nothing)
        comp = compose(comp, found_img; blend=:replace)

        save_image(comparison_path, comp)
        println("Saved: $comparison_path")
    end

    return target  # Return target for use by other renders
end

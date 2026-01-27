# BaGoL SMLMRender Visualization
# ===============================
# Publication-quality rendering of BaGoL results using SMLMRender.
# These functions require only SMLMRender (no CairoMakie dependency).
#
# Usage:
#   include("viz_smlmrender.jl")
#   render_bagol_gaussian(bagol_smld; filename="result.png")

using SMLMRender
using SMLMData

"""
Render BaGoL result as Gaussian blobs.

Creates a publication-quality image with Gaussian-rendered emitter positions.

# Arguments
- `bagol_smld`: BasicSMLD from run_bagol with estimated emitter positions
- `pixel_size`: Pixel size in nm (default: 2.0 for ~2nm resolution)
- `colormap`: Color scheme (default: :inferno)
- `filename`: Output filename (optional)
- `clip_percentile`: Intensity clip percentile (default: 0.995)

# Returns
- RenderResult2D with the rendered image
"""
function render_bagol_gaussian(
    bagol_smld::SMLMData.SMLD;
    pixel_size::Real = 2.0,
    colormap::Symbol = :inferno,
    filename::Union{String, Nothing} = nothing,
    clip_percentile::Real = 0.995
)
    return render(bagol_smld;
        strategy = GaussianRender(),
        pixel_size = pixel_size,
        colormap = colormap,
        clip_percentile = clip_percentile,
        filename = filename
    )
end

"""
Render localizations and BaGoL result as circle overlay.

Two-color overlay: localizations (cyan) + MAP-N emitters (red).

# Arguments
- `locs_smld`: BasicSMLD with input localizations
- `bagol_smld`: BasicSMLD from run_bagol with estimated positions
- `pixel_size`: Pixel size in nm (default: 1.0 for 1nm resolution)
- `filename`: Output filename (optional)

# Returns
- Overlaid RGB image
"""
function render_bagol_circles(
    locs_smld::SMLMData.SMLD,
    bagol_smld::SMLMData.SMLD;
    pixel_size::Real = 1.0,
    filename::Union{String, Nothing} = nothing
)
    return render([locs_smld, bagol_smld];
        colors = [:cyan, :red],
        strategy = CircleRender(),
        pixel_size = pixel_size,
        filename = filename
    )
end

"""
Render three-channel comparison: localizations, BaGoL, ground truth.

Creates a 3-color overlay image:
- Gray: Input localizations
- Red: BaGoL MAP-N emitters
- Blue: Ground truth positions

# Arguments
- `locs_smld`: BasicSMLD with input localizations
- `bagol_smld`: BasicSMLD from run_bagol with estimated positions
- `gt_smld`: BasicSMLD with ground truth positions
- `pixel_size`: Pixel size in nm (default: 1.0 for 1nm resolution)
- `filename`: Output filename (optional)

# Returns
- Overlaid RGB image
"""
function render_comparison(
    locs_smld::SMLMData.SMLD,
    bagol_smld::SMLMData.SMLD,
    gt_smld::SMLMData.SMLD;
    pixel_size::Real = 1.0,
    filename::Union{String, Nothing} = nothing
)
    return render([locs_smld, bagol_smld, gt_smld];
        colors = [:gray, :red, :blue],
        strategy = CircleRender(),
        pixel_size = pixel_size,
        filename = filename
    )
end

"""
Convert ground truth positions to a BasicSMLD for rendering.

Creates fake Emitter2DFit entries from (x, y) position tuples.
Uses small default uncertainty for visualization.

# Arguments
- `positions`: Vector of (x, y) tuples in micrometers
- `camera`: Camera from original SMLD
- `σ`: Position uncertainty for rendering circles (default: 0.005 μm = 5 nm)

# Returns
- BasicSMLD suitable for rendering
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
Render complete BaGoL comparison (localizations, result, optional GT).

Convenience function that wraps the full workflow.

# Arguments
- `locs_smld`: Input localizations
- `bagol_smld`: BaGoL result
- `true_positions`: Optional ground truth positions (default: empty)
- `prefix`: Filename prefix for output (default: "bagol")
- `output_dir`: Directory for output files (default: current directory)
- `pixel_size`: Pixel size in nm (default: 1.0)

# Creates files
- `{prefix}_gaussian.png`: Gaussian render of result
- `{prefix}_circles.png`: Circle overlay (locs + result)
- `{prefix}_comparison.png`: Three-channel comparison (if GT provided)
"""
function render_bagol_suite(
    locs_smld::SMLMData.SMLD,
    bagol_smld::SMLMData.SMLD;
    true_positions::Vector{Tuple{Float64, Float64}} = Tuple{Float64, Float64}[],
    prefix::String = "bagol",
    output_dir::String = ".",
    pixel_size::Real = 1.0
)
    # Gaussian render of result (use slightly coarser pixel for Gaussian)
    gaussian_path = joinpath(output_dir, "$(prefix)_gaussian.png")
    render_bagol_gaussian(bagol_smld; pixel_size=max(2.0, pixel_size), filename=gaussian_path)
    println("Saved: $gaussian_path")

    # Circle overlay
    circles_path = joinpath(output_dir, "$(prefix)_circles.png")
    render_bagol_circles(locs_smld, bagol_smld; pixel_size=pixel_size, filename=circles_path)
    println("Saved: $circles_path")

    # Three-channel comparison if GT provided
    if !isempty(true_positions)
        gt_smld = positions_to_smld(true_positions, locs_smld.camera)
        comparison_path = joinpath(output_dir, "$(prefix)_comparison.png")
        render_comparison(locs_smld, bagol_smld, gt_smld; pixel_size=pixel_size, filename=comparison_path)
        println("Saved: $comparison_path")
    end

    return nothing
end

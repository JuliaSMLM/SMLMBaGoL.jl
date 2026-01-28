# BaGoL SMLMRender Visualization
# ===============================
# Publication-quality rendering of BaGoL results using SMLMRender.
# These functions require only SMLMRender (no CairoMakie dependency).
#
# Usage:
#   include("viz_smlmrender.jl")
#   render_bagol_suite(locs_smld, bagol_smld; output_dir="output")
#   render_posterior_histogram(chain, locs_smld; output_dir="output")

using SMLMRender
using SMLMData
using SMLMBaGoL: RJMCMCChain

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
Convert chain samples to a BasicSMLD for histogram rendering.

Extracts all (x, y) positions from chain samples and creates
an SMLD where each sample emitter becomes one localization.
Use with HistogramRender to visualize posterior density.

# Arguments
- `chain`: RJMCMCChain with samples
- `camera`: Camera from original SMLD
- `filter_k`: If provided, only include samples with this K (default: nothing = all samples)
"""
function chain_to_smld(chain::RJMCMCChain, camera::SMLMData.AbstractCamera; filter_k::Union{Int, Nothing} = nothing)
    emitters = SMLMData.Emitter2DFit[]
    loc_id = 1

    for sample in chain.samples
        # Skip if filtering by K and this sample doesn't match
        if filter_k !== nothing && length(sample.emitters) != filter_k
            continue
        end

        for emitter in sample.emitters
            push!(emitters, SMLMData.Emitter2DFit(
                Float64(emitter.x), Float64(emitter.y),
                1000.0, 0.0,           # photons, bg
                0.001, 0.001, 0.0,     # tiny σ for histogram binning
                0.0, 0.0,              # σ_photons, σ_bg
                loc_id, 1, 0, loc_id   # frame, dataset, track_id, id
            ))
            loc_id += 1
        end
    end

    return SMLMData.BasicSMLD(emitters, camera, 1, 1)
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
- `{prefix}_circles.png`: Circle overlay (locs cyan + BaGoL red)
- `{prefix}_comparison.png`: Three-channel (locs gray + BaGoL red + GT blue)

# Returns
- `Image2DTarget`: The render target (pass to render_posterior_histogram for consistency)
"""
function render_bagol_suite(
    locs_smld::SMLMData.SMLD,
    bagol_smld::SMLMData.SMLD;
    true_positions::Vector{Tuple{Float64, Float64}} = Tuple{Float64, Float64}[],
    prefix::String = "render",
    output_dir::String = ".",
    pixel_size::Real = 1.0,
    expand_factor::Real = 2.0
)
    # Calculate bounds from localizations
    x_min, x_max, y_min, y_max = calculate_render_bounds(locs_smld; expand_factor=expand_factor)

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

    # 3. Circle overlay: localizations (cyan) + BaGoL (red)
    circles_path = joinpath(output_dir, "$(prefix)_circles.png")
    render([locs_smld, bagol_smld];
        colors = [:cyan, :red],
        strategy = CircleRender(),
        target = target,
        filename = circles_path
    )
    println("Saved: $circles_path")

    # 4. Three-channel comparison if GT provided
    if !isempty(true_positions)
        gt_smld = positions_to_smld(true_positions, locs_smld.camera)
        comparison_path = joinpath(output_dir, "$(prefix)_comparison.png")
        render([locs_smld, bagol_smld, gt_smld];
            colors = [:gray, :red, :blue],
            strategy = CircleRender(),
            target = target,
            filename = comparison_path
        )
        println("Saved: $comparison_path")
    end

    return target  # Return target for use by posterior histogram
end

"""
Render posterior density histogram from MCMC chain samples.

Each emitter position from each chain sample adds 1 count to the pixel
it falls in, producing a 2D histogram of the posterior distribution.

# Arguments
- `chain`: RJMCMCChain with samples
- `locs_smld`: Input localizations SMLD (for camera and bounds)
- `target`: Optional Image2DTarget (use output from render_bagol_suite for consistency)
- `prefix`: Filename prefix (default: "render")
- `output_dir`: Directory for output file (default: current directory)
- `pixel_size`: Pixel size in nm if target not provided (default: 1.0)
- `expand_factor`: Bound expansion factor if target not provided (default: 2.0)
"""
function render_posterior_histogram(
    chain::RJMCMCChain,
    locs_smld::SMLMData.SMLD;
    target::Union{SMLMRender.Image2DTarget, Nothing} = nothing,
    prefix::String = "render",
    output_dir::String = ".",
    pixel_size::Real = 1.0,
    expand_factor::Real = 2.0
)
    # Create target if not provided
    if target === nothing
        x_min, x_max, y_min, y_max = calculate_render_bounds(locs_smld; expand_factor=expand_factor)
        target = create_target(x_min, x_max, y_min, y_max; pixel_size=pixel_size)
    end

    # Convert chain samples to SMLD
    chain_smld = chain_to_smld(chain, locs_smld.camera)

    n_samples = length(chain.samples)
    n_positions = length(chain_smld.emitters)
    println("  Posterior histogram: $(n_positions) positions from $(n_samples) samples")

    # Render with HistogramRender - each position adds 1 to pixel count
    posterior_path = joinpath(output_dir, "$(prefix)_posterior.png")
    render(chain_smld;
        strategy = HistogramRender(),
        target = target,
        colormap = :inferno,
        filename = posterior_path
    )
    println("Saved: $posterior_path")

    return nothing
end

"""
Render histogram from only MAP-N samples (K = modal K).

This allows direct comparison with MAP-N Gaussian blobs to isolate
σ computation issues from sample selection issues.

# Arguments
- `chain`: RJMCMCChain with samples
- `locs_smld`: Input localizations SMLD (for camera and bounds)
- `target`: Optional Image2DTarget (use output from render_bagol_suite for consistency)
- `prefix`: Filename prefix (default: "render")
- `output_dir`: Directory for output file (default: current directory)

# Returns
- `map_n`: The modal K value used for filtering
"""
function render_mapn_histogram(
    chain::RJMCMCChain,
    locs_smld::SMLMData.SMLD;
    target::Union{SMLMRender.Image2DTarget, Nothing} = nothing,
    prefix::String = "render",
    output_dir::String = ".",
    pixel_size::Real = 1.0,
    expand_factor::Real = 2.0
)
    # Create target if not provided
    if target === nothing
        x_min, x_max, y_min, y_max = calculate_render_bounds(locs_smld; expand_factor=expand_factor)
        target = create_target(x_min, x_max, y_min, y_max; pixel_size=pixel_size)
    end

    # Find MAP-N (modal K)
    ks = [length(s.emitters) for s in chain.samples]
    k_max = maximum(ks)
    posterior_k = zeros(Int, k_max + 1)
    for k in ks
        posterior_k[k + 1] += 1
    end
    map_n = argmax(posterior_k) - 1

    # Convert chain samples to SMLD, filtering to only K=MAP-N
    chain_smld = chain_to_smld(chain, locs_smld.camera; filter_k=map_n)

    n_mapn_samples = count(s -> length(s.emitters) == map_n, chain.samples)
    n_positions = length(chain_smld.emitters)
    println("  MAP-N histogram (K=$map_n only): $(n_positions) positions from $(n_mapn_samples) samples")

    # Render with HistogramRender
    mapn_hist_path = joinpath(output_dir, "$(prefix)_mapn_histogram.png")
    render(chain_smld;
        strategy = HistogramRender(),
        target = target,
        colormap = :inferno,
        filename = mapn_hist_path
    )
    println("Saved: $mapn_hist_path")

    return map_n
end

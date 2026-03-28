# Simulation for BaGoL testing and benchmarking
#
# Generates synthetic SMLM data with configurable count and photon models.
# All positions in micrometers (μm). Pixel size default 100 nm.

"""
    SimulationResult

Result of SMLM simulation containing localizations, SMLD, and ground truth.

Fields:
- `smld`: Ready-to-use BasicSMLD with auto-sized camera
- `true_positions`: Ground truth emitter positions (μm)
- `true_counts`: Localizations generated per emitter (after min_photons filter)
- `n_emitters`: Number of true emitters
"""
struct SimulationResult
    smld::SMLMData.BasicSMLD
    true_positions::Vector{Tuple{Float64, Float64}}
    true_counts::Vector{Int}
    n_emitters::Int
end

# ============================================================================
# Position generators
# ============================================================================

"""
    nmer_positions(n, diameter; center=(0.0, 0.0))

Place `n` emitters on a circle of given `diameter` (μm).
`n=1` returns a single point at `center`.
"""
function nmer_positions(n::Int, diameter::Float64;
                         center::Tuple{Float64, Float64} = (0.0, 0.0))
    positions = Vector{Tuple{Float64, Float64}}(undef, n)
    if n == 1
        positions[1] = center
    else
        r = diameter / 2
        for k in 1:n
            θ = 2π * (k - 1) / n
            positions[k] = (center[1] + r * cos(θ), center[2] + r * sin(θ))
        end
    end
    return positions
end

"""
    nmer_grid_positions(; n_per_cluster, cluster_diameter, grid_nx, grid_ny, grid_spacing)

Place n-mer clusters on an `grid_nx × grid_ny` grid with `grid_spacing` (μm) between centers.
Grid is centered so that the center of mass is at the field center.

Returns `(all_positions, cluster_centers)`.
"""
function nmer_grid_positions(;
    n_per_cluster::Int,
    cluster_diameter::Float64,
    grid_nx::Int,
    grid_ny::Int,
    grid_spacing::Float64
)
    cluster_centers = Tuple{Float64, Float64}[]
    all_positions = Tuple{Float64, Float64}[]

    # Center the grid: offset so center of mass is at grid center
    ox = grid_spacing * (grid_nx + 1) / 2
    oy = grid_spacing * (grid_ny + 1) / 2

    for i in 1:grid_nx
        for j in 1:grid_ny
            cx = ox + (i - 1 - (grid_nx - 1) / 2) * grid_spacing
            cy = oy + (j - 1 - (grid_ny - 1) / 2) * grid_spacing
            push!(cluster_centers, (cx, cy))
            append!(all_positions, nmer_positions(n_per_cluster, cluster_diameter; center=(cx, cy)))
        end
    end

    return all_positions, cluster_centers
end

"""
    nmer_random_positions(; n_per_cluster, cluster_diameter, density, field_size)

Place n-mer clusters randomly at given `density` (clusters/μm²) within a square field.
Number of clusters = round(Int, density * field_size²).

Returns `(all_positions, cluster_centers)`.
"""
function nmer_random_positions(;
    n_per_cluster::Int,
    cluster_diameter::Float64,
    density::Float64,
    field_size::Float64
)
    n_clusters = max(1, round(Int, density * field_size^2))
    margin = field_size * 0.05  # keep away from edges

    cluster_centers = Tuple{Float64, Float64}[]
    all_positions = Tuple{Float64, Float64}[]

    for _ in 1:n_clusters
        cx = margin + rand() * (field_size - 2 * margin)
        cy = margin + rand() * (field_size - 2 * margin)
        push!(cluster_centers, (cx, cy))
        append!(all_positions, nmer_positions(n_per_cluster, cluster_diameter; center=(cx, cy)))
    end

    return all_positions, cluster_centers
end

# ============================================================================
# Camera/field utilities
# ============================================================================

"""
    make_camera(field_size, pixel_size) -> IdealCamera

Create a square camera with pixels of `pixel_size` (μm) covering `field_size` (μm).
"""
function make_camera(field_size::Float64, pixel_size::Float64)
    n_pixels = ceil(Int, field_size / pixel_size)
    return SMLMData.IdealCamera(n_pixels, n_pixels, pixel_size)
end

"""
    auto_field_size(positions, pixel_size; margin_factor=0.2) -> Float64

Compute field size from bounding box of positions, with margin, rounded to pixel grid.
"""
function auto_field_size(positions::Vector{Tuple{Float64, Float64}},
                          pixel_size::Float64;
                          margin_factor::Float64 = 0.2)
    xs = [p[1] for p in positions]
    ys = [p[2] for p in positions]
    span = max(maximum(xs) - minimum(xs), maximum(ys) - minimum(ys))
    # Add margin proportional to span, minimum 1 μm
    margin = max(span * margin_factor, 0.5)
    raw = max(maximum(xs), maximum(ys)) + margin
    # Also ensure minimum(xs) - margin/2 > 0 by shifting if needed
    min_coord = min(minimum(xs), minimum(ys))
    if min_coord < margin / 2
        raw += margin / 2 - min_coord
    end
    # Round up to pixel grid
    n_pixels = ceil(Int, raw / pixel_size)
    return n_pixels * pixel_size
end

# ============================================================================
# Core simulation
# ============================================================================

"""
    simulate_localizations(true_positions; kwargs...) -> SimulationResult

Generate synthetic SMLM localizations from known emitter positions.

# Count models (`count_model`)
- `:poisson` (default) — `Poisson(mean_count)`, clamped to ≥ 1
- `:negbin` — `NegativeBinomial(count_shape, p)` where `p = count_shape/(count_shape+mean_count)`
- `:fixed` — exactly `round(Int, mean_count)` per emitter

# Photophysics modes
- Default: photons from `Exponential(mean_photons)`, filtered by `min_photons`,
  precision `σ = psf_sigma / √photons`
- `fixed_sigma` mode: constant σ and photons, no photon sampling

# Keywords
- `mean_count=10.0`: mean localizations per emitter
- `count_model=:poisson`: `:poisson`, `:negbin`, or `:fixed`
- `count_shape=5.0`: NegBin shape parameter (only for `:negbin`)
- `psf_sigma=0.130`: PSF standard deviation (μm)
- `mean_photons=500.0`: mean photons per localization
- `min_photons=100.0`: discard localizations below this
- `fixed_sigma=nothing`: if set, use this constant σ (skips photophysics)
- `fixed_photons=1000.0`: photon count used with `fixed_sigma`
- `background=10.0`: background photons per pixel
- `pixel_size=0.100`: camera pixel size (μm)
- `field_size=nothing`: field of view (μm); auto-computed if `nothing`
"""
function simulate_localizations(
    true_positions::Vector{Tuple{Float64, Float64}};
    mean_count::Float64 = 10.0,
    count_model::Symbol = :poisson,
    count_shape::Float64 = 5.0,
    psf_sigma::Float64 = 0.130,
    mean_photons::Float64 = 500.0,
    min_photons::Float64 = 100.0,
    fixed_sigma::Union{Nothing, Float64} = nothing,
    fixed_photons::Float64 = 1000.0,
    background::Float64 = 10.0,
    pixel_size::Float64 = 0.100,
    field_size::Union{Nothing, Float64} = nothing
)
    n_emitters = length(true_positions)

    # Count distribution
    count_dist = if count_model == :poisson
        Poisson(mean_count)
    elseif count_model == :negbin
        p = count_shape / (count_shape + mean_count)
        NegativeBinomial(count_shape, p)
    elseif count_model == :fixed
        nothing  # handled below
    else
        error("Unknown count_model: $count_model. Use :poisson, :negbin, or :fixed.")
    end

    # Photon distribution (only used when fixed_sigma is nothing)
    use_fixed = fixed_sigma !== nothing
    photon_dist = use_fixed ? nothing : Exponential(mean_photons)

    locs = SMLMData.Emitter2DFit[]
    true_counts = zeros(Int, n_emitters)
    loc_id = 1

    for (emitter_idx, (ex, ey)) in enumerate(true_positions)
        # Sample count
        n_j = if count_model == :fixed
            round(Int, mean_count)
        else
            max(1, rand(count_dist))
        end

        actual_count = 0
        for _ in 1:n_j
            if use_fixed
                σ = fixed_sigma
                photons = fixed_photons
            else
                photons = rand(photon_dist)
                photons < min_photons && continue
                σ = psf_sigma / sqrt(photons)
            end

            x = ex + randn() * σ
            y = ey + randn() * σ

            push!(locs, SMLMData.Emitter2DFit(
                x, y,
                photons, background,
                σ, σ,           # σ_x, σ_y
                0.0,            # σ_xy
                sqrt(photons),  # σ_photons
                sqrt(background), # σ_bg
                loc_id, 1,      # frame, dataset
                emitter_idx,    # track_id = parent emitter
                loc_id          # id
            ))
            loc_id += 1
            actual_count += 1
        end
        true_counts[emitter_idx] = actual_count
    end

    # Camera
    fs = if field_size !== nothing
        field_size
    else
        auto_field_size(true_positions, pixel_size)
    end
    camera = make_camera(fs, pixel_size)

    smld = SMLMData.BasicSMLD(locs, camera, length(locs), 1)

    return SimulationResult(smld, true_positions, true_counts, n_emitters)
end

# ============================================================================
# Convenience wrappers
# ============================================================================

"""
    simulate_nmer(; n, diameter, center=(0.0, 0.0), kwargs...) -> SimulationResult

Simulate a single n-mer cluster. All other kwargs passed to `simulate_localizations`.
"""
function simulate_nmer(;
    n::Int,
    diameter::Float64,
    center::Tuple{Float64, Float64} = (0.0, 0.0),
    kwargs...
)
    positions = nmer_positions(n, diameter; center)
    return simulate_localizations(positions; kwargs...)
end

"""
    simulate_nmer_grid(; n_per_cluster, cluster_diameter, grid_nx, grid_ny,
                         grid_spacing, kwargs...) -> SimulationResult

Simulate a grid of identical n-mer clusters. All other kwargs passed to `simulate_localizations`.
"""
function simulate_nmer_grid(;
    n_per_cluster::Int,
    cluster_diameter::Float64,
    grid_nx::Int,
    grid_ny::Int,
    grid_spacing::Float64,
    kwargs...
)
    all_positions, _ = nmer_grid_positions(;
        n_per_cluster, cluster_diameter, grid_nx, grid_ny, grid_spacing)
    return simulate_localizations(all_positions; kwargs...)
end

"""
    print_simulation_summary(result::SimulationResult)

Print summary statistics of simulation result.
"""
function print_simulation_summary(result::SimulationResult)
    locs = result.smld.emitters
    counts = result.true_counts
    n_locs = length(locs)

    println("Simulation Summary:")
    println("  Emitters: $(result.n_emitters)")
    println("  Localizations: $n_locs")
    println("  Mean count/emitter: $(round(mean(counts), digits=1))")

    if length(counts) > 1
        println("  Count range: $(minimum(counts)) - $(maximum(counts))")
    end

    # Precision statistics (in nm)
    σs = [mean([loc.σ_x, loc.σ_y]) * 1000 for loc in locs]
    println("  Precision (nm): mean=$(round(mean(σs), digits=1)), range=$(round(minimum(σs), digits=1))-$(round(maximum(σs), digits=1))")

    # Camera
    cam = result.smld.camera
    nx = length(cam.pixel_edges_x) - 1
    ny = length(cam.pixel_edges_y) - 1
    px_size = cam.pixel_edges_x[2] - cam.pixel_edges_x[1]
    println("  Camera: $(nx)×$(ny) pixels, $(px_size) μm/px")
end

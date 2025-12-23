# Simulation for BaGoL testing
#
# Generates synthetic SMLM data with proper count statistics.
# Count distribution: n_j ~ NegBin(α, p) where p = α/(α+μ)
#   - E[n_j] = μ
#   - Var[n_j] = μ(1 + μ/α)
#   - α → ∞: Poisson-like (DNA-PAINT)
#   - α ≈ 1: Exponential-like (dSTORM)
#
# Localization precision from CRLB:
#   σ_loc ≈ σ_psf / √N * √(1 + background/N)

"""
    SimulationResult

Result of SMLM simulation containing localizations and ground truth.
"""
struct SimulationResult
    localizations::Vector{SMLMData.Emitter2DFit}
    true_positions::Vector{Tuple{Float64, Float64}}
    true_counts::Vector{Int}
    μ::Float64
    α::Float64
    n_frames::Int
end

"""
    crlb_precision(σ_psf, photons, background)

Calculate localization precision from Cramér-Rao Lower Bound.

Approximation: σ_loc ≈ σ_psf / √N * √(1 + background/N)

# Arguments
- `σ_psf`: PSF standard deviation (μm)
- `photons`: Number of detected photons
- `background`: Background photons per pixel

# Returns
- Localization precision σ_loc (μm)
"""
function crlb_precision(σ_psf::Float64, photons::Float64, background::Float64)
    # CRLB approximation including background contribution
    # σ² ≈ σ_psf² / N * (1 + background/N)
    N = max(photons, 1.0)  # Avoid division by zero
    return σ_psf / sqrt(N) * sqrt(1.0 + background / N)
end

"""
    simulate_smlm(true_positions; μ, α, σ_psf, mean_photons, background, n_frames)

Generate synthetic SMLM localizations from known emitter positions.

Localization precision is calculated from CRLB using sampled photon counts.

# Arguments
- `true_positions`: Vector of (x, y) tuples for true emitter locations (in μm)
- `μ`: Mean localizations per emitter (default: 10.0)
- `α`: Shape parameter controlling count variance (default: 2.0)
  - α ≈ 1: Exponential-like (high variance, dSTORM/photobleaching)
  - α → ∞: Poisson-like (variance ≈ mean, DNA-PAINT)
- `σ_psf`: PSF standard deviation in μm (default: 0.130 = 130 nm)
- `mean_photons`: Mean photons per localization (default: 500)
- `photon_shape`: Shape parameter for photon distribution (default: 2.0)
  - Lower = more variable brightness, higher = more uniform
- `background`: Background photons per pixel (default: 10.0)
- `n_frames`: Total number of acquisition frames (default: 1000)

# Returns
- `SimulationResult` containing localizations and ground truth

# Precision Model
Photons are sampled from Gamma(shape, mean/shape), giving:
- E[N] = mean_photons
- Var[N] = mean_photons² / shape

Precision is calculated from CRLB:
- σ_loc ≈ σ_psf / √N * √(1 + background/N)

# Example
```julia
positions = [(0.1, 0.1), (0.2, 0.1), (0.15, 0.2)]
result = simulate_smlm(positions; μ=8.0, α=1.0, mean_photons=800)
```
"""
function simulate_smlm(
    true_positions::Vector{Tuple{Float64, Float64}};
    μ::Float64 = 10.0,
    α::Float64 = 2.0,
    σ_psf::Float64 = 0.130,          # 130 nm PSF width
    mean_photons::Float64 = 500.0,   # Mean photons per loc
    photon_shape::Float64 = 2.0,     # Gamma shape for photon variability
    background::Float64 = 10.0,      # Background per pixel
    n_frames::Int = 1000
)
    # Count distribution: NegBin(α, p) with p = α/(α+μ)
    # This gives E[n] = μ and Var[n] = μ + μ²/α
    p = α / (α + μ)
    count_dist = NegativeBinomial(α, p)

    # Photon distribution: Gamma(shape, scale) with mean = shape * scale
    # Using shape and scale = mean/shape gives E[N] = mean_photons
    photon_dist = Gamma(photon_shape, mean_photons / photon_shape)

    localizations = SMLMData.Emitter2DFit[]
    true_counts = Int[]
    loc_id = 1

    for (emitter_idx, (ex, ey)) in enumerate(true_positions)
        # Sample count from NegBin (ensures n ≥ 0)
        n_j = rand(count_dist)

        # Ensure at least 1 localization per emitter for identifiability
        n_j = max(1, n_j)
        push!(true_counts, n_j)

        # Sample frame numbers uniformly across acquisition
        frames = rand(1:n_frames, n_j)
        sort!(frames)  # Sort for realistic temporal ordering

        for frame in frames
            # Sample photon count
            photons = rand(photon_dist)

            # Calculate precision from CRLB
            σ_loc = crlb_precision(σ_psf, photons, background)

            # Add Gaussian position noise based on precision
            x = ex + randn() * σ_loc
            y = ey + randn() * σ_loc

            push!(localizations, SMLMData.Emitter2DFit(
                x, y,           # position
                photons,        # photons (now realistic)
                background,     # background
                σ_loc, σ_loc,   # σ_x, σ_y from CRLB
                50.0, photons,  # llr, frame_photons
                frame, 1,       # frame, dataset_idx
                0, loc_id       # connect_idx, id
            ))
            loc_id += 1
        end
    end

    return SimulationResult(localizations, true_positions, true_counts, μ, α, n_frames)
end

"""
    simulate_grid(; nx, ny, spacing, μ, α, σ_loc, n_frames, offset)

Generate emitters on a regular grid for testing.

# Arguments
- `nx`, `ny`: Grid dimensions (default: 3x3)
- `spacing`: Distance between emitters in μm (default: 0.050 = 50 nm)
- `offset`: Grid offset (default: (0.1, 0.1))
- Other arguments passed to `simulate_smlm`

# Example
```julia
result = simulate_grid(nx=4, ny=4, spacing=0.040, μ=10.0, α=2.0)
```
"""
function simulate_grid(;
    nx::Int = 3,
    ny::Int = 3,
    spacing::Float64 = 0.050,
    offset::Tuple{Float64, Float64} = (0.1, 0.1),
    kwargs...
)
    positions = Tuple{Float64, Float64}[]
    for i in 1:nx
        for j in 1:ny
            x = offset[1] + (i - 1) * spacing
            y = offset[2] + (j - 1) * spacing
            push!(positions, (x, y))
        end
    end
    return simulate_smlm(positions; kwargs...)
end

"""
    simulate_nmers(; n_dimers, n_trimers, emitter_spacing, cluster_spacing, kwargs...)

Generate n-mers (dimers, trimers) for resolution testing.

# Arguments
- `n_dimers`: Number of dimers (default: 4)
- `n_trimers`: Number of trimers (default: 2)
- `emitter_spacing`: Distance between emitters within cluster (default: 0.040 = 40 nm)
- `cluster_spacing`: Distance between cluster centers (default: 0.200 = 200 nm)
- Other arguments passed to `simulate_smlm`

# Example
```julia
result = simulate_nmers(n_dimers=5, n_trimers=3, μ=8.0, α=1.5)
```
"""
function simulate_nmers(;
    n_dimers::Int = 4,
    n_trimers::Int = 2,
    emitter_spacing::Float64 = 0.040,
    cluster_spacing::Float64 = 0.200,
    offset::Tuple{Float64, Float64} = (0.1, 0.1),
    kwargs...
)
    positions = Tuple{Float64, Float64}[]

    n_clusters = n_dimers + n_trimers
    cols = ceil(Int, sqrt(n_clusters))

    for i in 1:n_clusters
        n = i <= n_dimers ? 2 : 3
        row = (i - 1) ÷ cols
        col = (i - 1) % cols
        cx = offset[1] + col * cluster_spacing
        cy = offset[2] + row * cluster_spacing

        # Arrange emitters: line for dimer, triangle for trimer
        for j in 1:n
            if n == 2
                ex = cx + (j == 1 ? -emitter_spacing/2 : emitter_spacing/2)
                ey = cy
            else
                θ = 2π * (j - 1) / n - π/2
                r = emitter_spacing / (2 * sin(π / n))
                ex = cx + r * cos(θ)
                ey = cy + r * sin(θ)
            end
            push!(positions, (ex, ey))
        end
    end

    return simulate_smlm(positions; kwargs...)
end

"""
    print_simulation_summary(result::SimulationResult)

Print summary statistics of simulation result.
"""
function print_simulation_summary(result::SimulationResult)
    n_emitters = length(result.true_positions)
    n_locs = length(result.localizations)
    counts = result.true_counts
    locs = result.localizations

    println("Simulation Summary:")
    println("  Emitters: $n_emitters")
    println("  Localizations: $n_locs")
    println("  Frames: $(result.n_frames)")
    println("  Parameters: μ=$(result.μ), α=$(result.α)")

    println("  Count statistics:")
    println("    Mean: $(round(mean(counts), digits=1))")
    println("    Std:  $(round(std(counts), digits=1))")
    println("    Range: $(minimum(counts)) - $(maximum(counts))")

    # Theoretical values
    theoretical_var = result.μ * (1 + result.μ / result.α)
    println("    Theoretical Var: $(round(theoretical_var, digits=1))")
    println("    Observed Var:    $(round(var(counts), digits=1))")

    # Photon statistics
    photons = [loc.photons for loc in locs]
    println("  Photon statistics:")
    println("    Mean: $(round(mean(photons), digits=0))")
    println("    Std:  $(round(std(photons), digits=0))")
    println("    Range: $(round(minimum(photons), digits=0)) - $(round(maximum(photons), digits=0))")

    # Precision statistics (in nm for readability)
    σs = [mean([loc.σ_x, loc.σ_y]) * 1000 for loc in locs]  # Convert to nm
    println("  Precision statistics (nm):")
    println("    Mean: $(round(mean(σs), digits=1))")
    println("    Std:  $(round(std(σs), digits=1))")
    println("    Range: $(round(minimum(σs), digits=1)) - $(round(maximum(σs), digits=1))")
end

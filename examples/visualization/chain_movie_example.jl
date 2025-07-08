"""
Example script demonstrating MCMC chain movie generation.

This script:
1. Simulates a simple 3-mer structure
2. Runs MCMC sampling
3. Generates a movie showing chain evolution with allocation colors
"""

using SMLMBaGoL
using SMLMData
using Random

# Set random seed for reproducibility
Random.seed!(42)

# Generate synthetic data: 3-mer with well-separated emitters
println("Generating synthetic 3-mer data...")
emitter_positions = [
    (2.0, 2.0),
    (4.0, 2.0),
    (3.0, 4.0)
]

# Generate localizations with realistic parameters
localizations = Localization2D[]
localizations_per_emitter = 50
σ_loc = 0.15  # 150 nm localization uncertainty

for (x_true, y_true) in emitter_positions
    for _ in 1:localizations_per_emitter
        x_meas = x_true + randn() * σ_loc
        y_meas = y_true + randn() * σ_loc
        push!(localizations, Localization2D(x_meas, y_meas, σ_loc, σ_loc, 1))
    end
end

# Shuffle localizations
shuffle!(localizations)
println("Generated $(length(localizations)) localizations from $(length(emitter_positions)) emitters")

# Create priors using default settings
spatial_prior, count_prior = create_default_prior(localizations)

# Initialize chain
println("\nInitializing MCMC chain...")
chain = initialize_chain(
    localizations,
    Emitter2D{Float64},
    spatial_prior,
    count_prior;
    burn_in = 500,
    thin = 10
)

# Run MCMC
println("Running MCMC sampling...")
println("  Burn-in: $(chain.burn_in) iterations")
println("  Thinning: every $(chain.thin) iterations")
println("  Target samples: 100")

run_rjmcmc!(chain, 1500)  # 500 burn-in + 1000 post-burn-in (100 samples with thin=10)

println("\nCollected $(length(chain.samples)) samples")

# Create output directory if it doesn't exist
output_dir = "../output"
if !isdir(output_dir)
    mkdir(output_dir)
end

# Generate movie with default settings
println("\nGenerating chain evolution movie...")
generate_chain_movie(
    chain,
    joinpath(output_dir, "chain_evolution_default.mp4")
)
println("Created: $(joinpath(output_dir, "chain_evolution_default.mp4"))")

# Generate movie with custom settings
println("\nGenerating movie with custom settings...")
generate_chain_movie(
    chain,
    joinpath(output_dir, "chain_evolution_custom.mp4"),
    fps = 30,
    figsize = (1000, 800),
    color_palette = :Dark2_8,
    localization_alpha = 0.7,
    emitter_markersize = 20
)
println("Created: $(joinpath(output_dir, "chain_evolution_custom.mp4"))")

# Generate movie of just the first 50 samples to see early dynamics
println("\nGenerating movie of early chain dynamics...")
generate_chain_movie(
    chain,
    joinpath(output_dir, "chain_early_dynamics.mp4"),
    sample_range = 1:min(50, length(chain.samples)),
    fps = 5  # Slower playback to see details
)
println("Created: $(joinpath(output_dir, "chain_early_dynamics.mp4"))")

println("\nMovie generation complete!")
println("The movies show:")
println("  - Localizations colored by their allocated emitter")
println("  - Emitters as X markers with matching colors")
println("  - Metrics display with frame number, K, and log-likelihood")
println("  - How allocations change as the chain evolves")
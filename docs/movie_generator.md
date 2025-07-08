# MCMC Chain Movie Generator

The `generate_chain_movie()` function creates MP4 movies showing the evolution of MCMC chains over time.

## Features

- **Allocation visualization**: Localizations are colored by their allocated emitter
- **Emitter tracking**: Emitters shown as X markers with consistent colors
- **Metrics display**: Shows frame number, iteration, K (number of emitters), and log-likelihood
- **Configurable output**: Adjustable frame rate, figure size, and visual parameters

## Basic Usage

```julia
using SMLMBaGoL, SMLMData

# Run your BaGoL analysis
chain = run_bagol(localizations, n_iterations=2000)

# Generate movie with default settings (saves to examples/output/)
generate_chain_movie(chain)

# Or specify custom path
generate_chain_movie(chain, "my_analysis/chain_movie.mp4")
```

## Advanced Options

```julia
generate_chain_movie(
    chain,
    "examples/output/custom_movie.mp4",
    fps = 30,                    # Frame rate
    figsize = (1200, 800),       # Figure size
    color_palette = :Dark2_8,    # Color scheme
    localization_alpha = 0.7,    # Localization transparency
    emitter_markersize = 20,     # Emitter X marker size
    sample_range = 1:100         # Animate only first 100 samples
)
```

## Visual Elements

- **Localizations**: Colored circles showing uncertainty, color indicates allocated emitter
- **Emitters**: X markers with colors matching their allocated localizations
- **Metrics**: Top banner showing current frame, iteration, K, and log-likelihood
- **Color consistency**: Emitters maintain consistent colors across frames when possible

## Performance Notes

- Each frame corresponds to one saved sample from the chain
- For long chains, consider using `sample_range` to focus on specific portions
- Default settings work well for most analyses
- Movie generation time scales with number of samples and figure size

## Example Output

The movies show:
1. How localizations switch between emitters (allocation changes)
2. Emitter birth and death events (K changes)
3. Convergence behavior and chain mixing
4. Uncertainty quantification through localization circles

See `examples/visualization/chain_movie_example.jl` for a complete working example.
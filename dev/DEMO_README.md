# BaGoL Demo Scripts

This directory contains demonstration scripts showing how to use SMLMBaGoL with synthetic data from SMLMSim.

## Files

- `bagol_demo_simple.jl` - Complete demo with synthetic data generation and posterior analysis
- `output/` - Directory for output files (git-ignored)

## Running the Demo

```bash
# Run the BaGoL demo
julia --project=dev dev/bagol_demo_simple.jl
```

## What the Demo Shows

The demo demonstrates:

1. **Synthetic Data Generation**: Using SMLMSim to create realistic SMLM datasets
2. **BaGoL Clustering**: Applying Bayesian Grouping of Localizations to reduce noise
3. **Results Analysis**: Comparing input localizations to output emitters
4. **Posterior Image Generation**: Creating probability density maps for visualization

## Output Files

- `simple_demo_results.txt` - Summary of clustering results
- `posterior_viridis.png` - Posterior probability image with Viridis colormap
- `posterior_hot.png` - Posterior probability image with Hot colormap  
- `posterior_gray.png` - Posterior probability image in grayscale
- `posterior_data.csv` - Raw posterior probability data for analysis

## Key Results

- **Input**: Raw noisy localizations from fluorophore blinking
- **Output**: High-precision emitter positions with reduced noise  
- **Posterior**: Probability density image showing emitter likelihood
- **Precision**: Improved localization accuracy through Bayesian inference

## Example Output

```
=== Simple BaGoL Demo ===

1. Creating synthetic SMLM data...
   - Generated 148 true emitters
   - Generated 21518 localizations

2. Running BaGoL clustering...
   - Found 21675 MAP-N emitters

4. Saving results and posterior image...
   - Posterior images saved:
     • Viridis colormap: dev/output/posterior_viridis.png
     • Hot colormap: dev/output/posterior_hot.png
     • Grayscale: dev/output/posterior_gray.png
   - Image size: 2836 x 2940 pixels

✓ BaGoL successfully processed synthetic SMLM data
```

This shows BaGoL processing ~21,000 noisy localizations and producing high-quality emitter positions plus beautiful posterior probability images with multiple colormaps for visualization and analysis.
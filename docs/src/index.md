# SMLMBaGoL.jl

Bayesian Grouping of Localizations (BaGoL) for single-molecule localization microscopy (SMLM) data.

## Overview

SMLMBaGoL groups raw localizations from repeated observations of the same emitter into higher-precision emitter positions using Reversible Jump Markov Chain Monte Carlo (RJMCMC). The algorithm simultaneously estimates:

- The number of emitters (K)
- Their positions with uncertainties
- The count distribution parameters (μ and shape)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaSMLM/SMLMBaGoL.jl")
```

## Quick Start

```julia
using SMLMBaGoL
using SMLMData

# Load or create SMLD data
camera = IdealCamera(64, 64, 0.1)  # 64x64 pixels, 100nm pixel size
smld = BasicSMLD(locs, camera, n_frames, n_datasets)

# Run BaGoL
result_smld, diagnostics = run_bagol(smld;
    n_iterations = 10000,
    burn_in = 2000,
    shape = 2.0,
    learn_shape = true
)

# Access results
result_smld.emitters  # Vector{Emitter2DFit} with grouped positions
diagnostics.n_emitters
diagnostics.posterior_k
```

## Count Model

The count distribution for localizations per emitter uses:

```
n_j ~ Gamma(shape, μ/shape)
```

Where:
- `μ` = mean localizations per emitter
- `shape` = 1: exponential (dSTORM photobleaching)
- `shape` > 1: peaked distribution (DNA-PAINT-like)

## Features

- **Automatic partitioning**: Precision-weighted DBSCAN clusters localizations for parallel processing
- **Hierarchical Bayes**: Learns μ and shape from the data
- **Robust MAP-N estimation**: Iterative Hungarian matching handles label switching
- **MAD-based uncertainties**: Robust to outliers

## Units

All positions and uncertainties are in **micrometers (μm)**.

## Citation

If you use this package, please cite:

> Bayesian Grouping of Localizations (BaGoL). https://doi.org/10.1101/752287

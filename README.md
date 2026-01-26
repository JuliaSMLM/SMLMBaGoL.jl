# SMLMBaGoL

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliasmlm.github.io/SMLMBaGoL.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliasmlm.github.io/SMLMBaGoL.jl/dev)
[![Build Status](https://github.com/juliasmlm/SMLMBaGoL.jl/workflows/CI/badge.svg)](https://github.com/juliasmlm/SMLMBaGoL.jl/actions)
[![Coverage](https://codecov.io/gh/juliasmlm/SMLMBaGoL.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/juliasmlm/SMLMBaGoL.jl)

## Overview

SMLMBaGoL performs Bayesian Grouping of Localizations (BaGoL) on single-molecule localization microscopy (SMLM) data organized in an SMLMData.SMLD structure (https://github.com/JuliaSMLM/SMLMData.jl). Specifically, SMLMBaGoL groups raw localizations into association with individual emitters, yielding higher precision localizations of those single emitters. The algorithm(s) implemented are presented in https://doi.org/10.1101/752287.

## Quick Start

```julia
using SMLMBaGoL
using SMLMData

# Run BaGoL on an SMLD - returns (BasicSMLD, BaGoLDiagnostics)
result_smld, diagnostics = run_bagol(smld)

# Access results
result_smld.emitters  # Vector of Emitter2DFit with grouped positions
diagnostics.n_emitters
diagnostics.posterior_k
```

## Interface

The main entry point is `run_bagol()`, which accepts an `SMLMData.SMLD` and returns grouped emitter positions with uncertainties.

```julia
result_smld, diagnostics = run_bagol(smld;
    n_iterations = 10000,      # Total MCMC iterations
    burn_in = 2000,            # Burn-in before recording samples
    partition_threshold = 500, # Auto-partition if n_locs > threshold (0 = never)
    α = 2.0,                   # Shape parameter (or :auto to estimate)
    learn_α = false,           # Update α during MCMC
    verbose = true
)
```

**Returns:**
- `result_smld::BasicSMLD` - Grouped emitter positions as `Emitter2DFit` with uncertainties
- `diagnostics::BaGoLDiagnostics` - Contains `n_emitters`, `posterior_k`, `acceptance_rates`, `final_μ`, `final_α`

### Advanced: Chain Access

For visualization and detailed diagnostics, use `run_bagol_chain()` to get the full MCMC chain:

```julia
chain = run_bagol_chain(locs; n_iterations=10000, burn_in=2000)
emitters, posterior_k = estimate_mapn(chain)

# Visualization
plot_bagol(chain, emitters, posterior_k, locs)
plot_hierarchical_diagnostics(chain)
```

### Large Datasets

For datasets exceeding `partition_threshold` localizations, `run_bagol` automatically partitions the data using precision-weighted DBSCAN clustering, runs MCMC in parallel on each partition with synchronized global hierarchical updates, and merges results with boundary deduplication.

```julia
# Explicit partitioning control
result_smld, diagnostics = run_bagol(smld;
    partition_threshold = 500,
    max_partition_size = 1000,
    sync_interval = 500  # Iterations between global μ/α updates
)
```

## Examples

See the `examples/` directory for complete workflows:
- `01_basic_demo.jl` - Core API usage
- `02_nmer_demo.jl` - N-mer analysis with known ground truth
- `08_partitioned_demo.jl` - Large dataset partitioning

Run examples with threading enabled:
```bash
julia --threads=auto --project=examples examples/02_nmer_demo.jl
```

## Citation

https://doi.org/10.1101/752287

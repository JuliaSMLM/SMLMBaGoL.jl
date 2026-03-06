# Testing Guidelines for test/

## Structure

All tests live in `runtests.jl` as inline `@testset` blocks. There are no separate included test files.

### Important Rules
- **No `using` statements in included files** — all imports at the top of `runtests.jl`
- **Aim for simplicity** — good coverage without bloating tests
- **Avoid pedantic edge cases** — focus on meaningful tests that aid development
- **Maintainability first** — tests should be easy to update as code evolves

## Running Tests

```bash
# From shell
julia --project=. -e "using Pkg; Pkg.test()"

# From REPL (with project activated)
include("test/runtests.jl")

# With specific seed for reproducibility
julia --project=. -e "using Random; Random.seed!(123); using Pkg; Pkg.test()"
```

## Current Testsets

- **Basic Types** — Emitter construction
- **ClusterStats** — add/remove loc, posterior mean/cov, marginal likelihood, predictive
- **Collapsed Sampler - 2 Emitters** — integration test with `run_collapsed_chain`
- **run_bagol Collapsed Integration** — full pipeline with SMLD input
- **run_bagol RJMCMC Legacy** — legacy sampler + `estimate_mapn`
- **Accumulators** — EmitterCountHist merge, NNDistHist init
- **Collapsed MAP-N** — PartitionSamples + `estimate_mapn_collapsed` with position validation
- **Spatial Utilities** — get_coords, get_sigma, mean_sigma, precision_weighted_distance
- **Partitioning** — partition_locs with cluster separation + index tracking
- **Oversized Cluster Splitting** — max_size enforcement + skip_size
- **Posterior Image (RJMCMC)** — posterior_image from chain
- **Posterior Image (Collapsed)** — via run_bagol with posterior_pixel_size
- **Partitioned BaGoL Collapsed** — multi-partition with sync
- **Archive Write/Read** — mmap archive round-trip

## Notes

- MCMC tests use `Random.seed!()` for reproducibility but are statistical — if a test flakes, check whether the seed still produces the expected clustering
- Use `julia --threads=auto` when testing parallel partitioned functionality
- Tests create `Emitter2DFit` manually with 13-arg constructor: `(x, y, photons, bg, σ_x, σ_y, Δx, Δy, Δz, frame, dataset, channel, id)`

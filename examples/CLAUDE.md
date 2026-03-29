# Examples

Separate project environment (`--project=examples`). Uses CairoMakie and SMLMRender
which activate the BaGoLMakieExt and BaGoLRenderExt package extensions.

## Running

```bash
# Single-partition scripts
julia --project=examples examples/nmer_single_test.jl

# Multi-partition scripts (threading needed)
julia --threads=auto --project=examples examples/nmer_grid_test.jl

# Optimality sweep (parallel trials, benefits from many threads)
julia --threads=auto --project=examples examples/nmer_stats.jl
```

## Standard workflow pattern

All examples follow the same structure:

```julia
sim = simulate_nmer(; ...)                    # or simulate_nmer_grid, SMLMSim
fov = compute_fov(sim.smld; margin=...)       # shared FOV for posterior + renders
result_smld, diagnostics = run_bagol(sim.smld; posterior_xlim/ylim from fov, ...)
report = compute_report(result_smld, diagnostics;
    true_positions=..., locs_smld=..., count_params=sim.count_params)
write_report(report; output_dir=...)          # summary.txt, emitters.csv, metrics.json, posterior_image.png
plot_report(report; output_dir=...)           # CairoMakie plots
render_report(sim.smld, result_smld; fov=...) # SMLMRender spatial images
```

## Nohier convention

Fixed-parameter (no hierarchical learning) scripts pass simulation's actual count params:

```julia
run_bagol(sim.smld; μ=sim.count_params.μ, shape=sim.count_params.shape,
    learn_shape=false, sync_interval=N_ITERATIONS+1)
```

For SMLMSim (which doesn't use our count model), compute empirical μ directly:
`μ = n_locs / n_true_emitters`, `shape=1000.0` (Poisson approximation).

## SMLMSim scripts

SMLMSim.simulate requires explicit pattern and fluorophore kwargs:

```julia
pattern = SMLMSim.Nmer2D(n=8, d=0.050)
fluor = SMLMSim.GenericFluor(photons=50000.0, k_off=50.0, k_on=0.5)
smld, info = SMLMSim.simulate(params; pattern=pattern, molecule=fluor, camera=camera)
true_positions = unique([(e.x, e.y) for e in info.smld_true.emitters])
```

Without these kwargs SMLMSim uses defaults that produce very few localizations.

## Standard output files

**Category 1 (always):**
- `summary.txt`, `emitters.csv`, `posterior_image.png`
- `partition_k.png`, `count_distribution.png`, `acceptance_rates.png`, `nn_distances.png`
- `render_mapn_gaussian.png`, `render_sr_gaussian.png`, `render_circles.png`

**Category 2 (with GT, adds):**
- `metrics.json`, `k_recovery.png`, `calibration.png`
- `render_circles_groundtruth.png`

**Category 3 (optimality sweep):**
- `sweep_data.csv`, `scorecard.txt`, `speed_data.csv`
- `sweep_krecovery.png`, `sweep_collated.png`, `sweep_rmse.png`, `k_true_vs_k_found.png`, `speed_test.png`

## Scripts

| Script | Type | Clusters | Partitioned | Hierarchical |
|--------|------|----------|-------------|--------------|
| `nmer_single_test.jl` | Single 6-mer | 1 | No (nsigma=Inf) | Yes |
| `nmer_single_test_nohier.jl` | Single 6-mer | 1 | No | No |
| `nmer_grid_test.jl` | 4x4 grid of 8-mers | 16 | Yes | Yes |
| `nmer_grid_test_nohier.jl` | 4x4 grid of 8-mers | 16 | Yes | No |
| `smlmsim_test.jl` | SMLMSim 8-mers | ~86 | Yes | Yes |
| `smlmsim_test_nohier.jl` | SMLMSim 8-mers | ~86 | Yes | No |
| `nmer_stats.jl` | Optimality sweep + speed | d/σ sweep | Yes | Both |

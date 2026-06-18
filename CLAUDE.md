# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Run all tests
julia --project=. -e "using Pkg; Pkg.test()"

# Run examples (threads needed for parallel partition processing)
julia --threads=auto --project=examples examples/nmer_single_test.jl

# Interactive development
julia --project=.
using SMLMBaGoL

# Run specific testset interactively (from REPL)
include("test/runtests.jl")  # runs all tests

# Run tests with specific seed for reproducibility
julia --project=. -e "using Random; Random.seed!(123); using Pkg; Pkg.test()"
```

**Examples environment:** `examples/Project.toml` is a separate project with local path deps
to sibling JuliaSMLM repos (`SMLMData`, `SMLMSim`, `SMLMRender`, `SMLMFrameConnection`).
Always use `--project=examples` when running examples.

**Threading:** Partitioned BaGoL uses `Threads.@threads` for parallel partition processing.
Use `julia --threads=auto` for any workload with multiple partitions.

## Code Conventions

- **All `using`/`import` statements must be in `src/SMLMBaGoL.jl` only** - included files have no imports
- Units: positions and uncertainties in micrometers (μm)
- Uncertainty correction: in the integrated pipeline σ is corrected upstream (SMLMClustering, stamped `metadata["sigma_corrected"]=true`); for standalone runs use `run_bagol(...; se_adjust=τ)` — quadrature σ²+τ², per-axis, self-guards against double-applying to an already-corrected SMLD

## Architecture

### Source File Organization
```
src/
├── SMLMBaGoL.jl          # Module entry: all imports + exports
├── spatial.jl             # Coordinate utilities (get_cov_xy, mean_sigma)
├── cluster_stats.jl       # ClusterStats: sufficient statistics for collapsed sampler
├── types.jl               # Types: BaGoLDiagnostics, CollapsedState, BaGoLResult, CollapsedChainResult
├── priors.jl              # UniformSpatialPrior, log_prior_k, log_prior_total_count
├── hierarchical.jl        # Hierarchical Bayes MH updates for μ and shape
├── mapn.jl                # MAP-N estimation: Dahl, PSM, VI-greedy, Hungarian matching
├── accumulators.jl        # Accumulator interface + EmitterCountHist, PosteriorImage, NNDistHist
├── collapsed_moves.jl     # Gibbs sweep, split/merge moves
├── collapsed_sampler.jl   # run_collapsed_chain() - collapsed Gibbs sampler
├── partition.jl           # Precision-weighted DBSCAN clustering
├── partitioned.jl         # Boundary emitter deduplication
├── rjmcmc.jl              # run_bagol() main entry point
├── posterior_image.jl     # Posterior image histogram + PNG writer
├── archive.jl             # Mmap-based binary chain archive
├── simulation.jl          # simulate_localizations, nmer_positions, simulate_nmer, simulate_nmer_grid
├── reports.jl             # compute_report, write_report, match_positions (standard outputs)
├── optimality.jl          # run_optimality_sweep, run_speed_test (Cat 3)
└── diagnostics/           # Sampler validation and correctness tools
    ├── target.jl          # AbstractTargetDensity: DecoupledTarget, DMFlatTarget, DirectNegBinFlatTarget
    ├── finite_state.jl    # Brute-force enumeration, kernel invariance, exact posterior
    ├── mixing.jl          # ESS, autocorrelation, R-hat, indicator ESS
    └── detailed_balance.jl # DetailedBalanceResult, check_detailed_balance

ext/
├── BaGoLMakieExt/         # CairoMakie extension: plot_report, plot_sweep, plot_speed
└── BaGoLRenderExt/        # SMLMRender extension: render_report
```

**Include order matters:** Files are included in dependency order in `SMLMBaGoL.jl`.
`spatial.jl` before `cluster_stats.jl`, `partition.jl` and `partitioned.jl` before `rjmcmc.jl`.

**Package extensions:** CairoMakie and SMLMRender are weak dependencies.
Plotting/rendering functions activate when the user loads these packages.

**Other directories:**
- `examples/` — Complete workflow scripts (separate project environment)
- `dev/` — Validation scripts: brute-force enumeration, detailed balance, experimental data, high-density tests
- `test/` — All tests in `runtests.jl` (single file, all testsets inline). See `test/CLAUDE.md` for testing guidelines.

### Collapsed Gibbs Sampler

- State = allocation vector only (which locs belong to which cluster)
- Emitter positions integrated out analytically via ClusterStats
- Moves: DM-weighted Gibbs allocation sweep (50%), RJMCMC split/merge (25%), birth/death (25% × 5 substeps)
- Gibbs: P(z_i=k|rest) ∝ (n_{-i,k}+γ) × predictive (Dirichlet-Multinomial partition prior)
- Split/merge: |ΔK|=1 random proposals with Jain-Neal restricted Gibbs scans
- MH acceptance: Δ_spatial + Δ_partition + Δ_proposal + Δ_count + Δ_move_type
- K changes by ±1 (coin flip split/merge, boundary-aware)
- Rao-Blackwellized posterior image (Gaussian blobs, not point deltas)
- MAP-N estimation via Dahl+Hungarian matching on stored assignment samples

### Core Types

- `ClusterStats` - Immutable sufficient statistics (precision matrix, natural params, quadratic form)
- `CollapsedState` - Assignment vector + ClusterStats cache + active bitvector
- `CollapsedChainResult` - Final state + accumulator results
- `BaGoLDiagnostics` - n_emitters, posterior_k, acceptance_rates, final_μ, final_shape, n_partitions
- `Partition` - Spatial cluster with locs, original_indices, and boundary flags

### Main API

```julia
# 1. Standard workflow
result_smld, diagnostics = run_bagol(smld; n_iterations=10000, burn_in=2000)

# Also accepts Vector{Emitter2DFit} directly:
result_smld, diagnostics = run_bagol(locs; camera=camera, n_iterations=10000)

# Key parameters:
#   μ=10.0                        # Mean locs per emitter (via kwargs)
#   shape=2.0                     # Count distribution shape (1=exp, >1=peaked)
#   learn_distribution=true       # true/false/:mu/:shape — control count distribution learning
#   partition_sigma=3.0           # DBSCAN threshold (Inf = no partitioning)
#   overlap=:auto                 # Overlap fraction for boundary dedup (:auto or Float64)
#   sync_interval=500             # Iterations between global μ/shape updates
#   posterior_pixel_size=0.002    # Rao-Blackwellized posterior image (0.0 to disable)
#
# Advanced kwargs (forwarded to chain):
#   spatial_model=:locmix         # :locmix or :flat
#   allocation_model=:dm          # :dm (Dirichlet-Multinomial) or :decoupled
#   n_restricted_scans=5          # Jain-Neal restricted Gibbs scans for split/merge
#   n_bd_substeps=3               # Birth/death substeps per iteration

# 2. Standard report (compute metrics + write files + plot + render)
report = compute_report(result_smld, diagnostics;
    true_positions=sim.true_positions, locs_smld=sim.smld)
write_report(report; output_dir="output")
plot_report(report; output_dir="output")     # requires CairoMakie
render_report(sim.smld, result_smld;          # requires SMLMRender
    output_dir="output", true_positions=sim.true_positions)

# 3. Optimality sweep (Category 3)
sweep = run_optimality_sweep(; n_values=[2, 8], mu_values=[5.0, 10.0], n_trials=50)
write_sweep(sweep; output_dir="output")
plot_sweep(sweep; output_dir="output")        # requires CairoMakie

# 4. Speed test
speed = run_speed_test(; n_locs_range=[100, 1000, 10000])
write_speed(speed; output_dir="output")
plot_speed(speed; output_dir="output")        # requires CairoMakie

# 5. Direct chain access
result = run_collapsed_chain(locs;
    n_iterations=10000, burn_in=2000,
    accumulators=AbstractAccumulator[EmitterCountHist(), PosteriorImage(pixel_size=0.001)]
)
```

### Accumulators

Accumulators collect statistics from the chain without storing full samples:

- `EmitterCountHist` - Histogram of K per iteration
- `PosteriorImage` - Rao-Blackwellized posterior image (Gaussian blobs per cluster)
- `NNDistHist` - Nearest-neighbor distance histogram between emitter positions
- `PartitionSamples` - Stores thinned assignment vectors for MAP-N estimation
- `PSMAccumulator` - Posterior similarity matrix (co-assignment frequencies) for Dahl/PSM/VI estimators

### MAP-N Estimation

Multiple estimation methods available, all using stored assignment samples:

- `estimate_mapn_collapsed(samples, locs)` — histogram-mode K + Hungarian matching for label switching. Median positions + posterior covariances.
- `estimate_dahl(samples, locs, psm)` — Dahl consensus partition (sample closest to PSM). Preferred default.
- `estimate_mapn_psm(samples, locs, psm)` — PSM thresholding with Hungarian refinement.
- `estimate_vi_greedy(samples, locs, psm)` — Variational inference greedy approximation.

PSM-based methods require `PSMAccumulator` in the accumulator list to build the co-assignment matrix.

**Fallback:** `extract_emitters(state, locs)` — final chain state only (single sample).

### ClusterStats

Immutable sufficient statistics for a cluster. All operations O(1):
- `add_loc(cs, loc)` / `remove_loc(cs, loc)` — exact inverses
- `log_marginal_likelihood(cs, log_area)` — positions integrated out
- `log_predictive(cs, loc, log_area)` — predictive for Gibbs allocation
- `posterior_mean(cs)` / `posterior_cov(cs)` — posterior position estimates

### Chain Archive

Opt-in via `archive_path` kwarg. Binary format per partition:
- Header (64 bytes) + per-sample: Int16[n_locs] assignments + Float64 μ + Float64 shape
- Post-hoc analysis: `compute_from_archive(AccType, archive, partition_id, locs)`

### Partitioned BaGoL (Large Datasets)
The main `run_bagol` always partitions data using precision-weighted DBSCAN.
Boundary emitters are deduplicated via Hungarian matching after merge.

**Count Prior (from Fazel et al. 2022):** Uses P(N|K) = Gamma(N; K*shape, μ/shape) where N is total localizations.

### Diagnostics Infrastructure

Sampler correctness validation tools in `src/diagnostics/`. See `src/diagnostics/DIAGNOSTICS.md` for full math and interpretation guide.

- **Target densities** (`target.jl`): `AbstractTargetDensity` with implementations `DecoupledTarget`, `DMFlatTarget`, `DirectNegBinFlatTarget`. Decouple diagnostics from specific posterior formulations.
- **Finite-state validation** (`finite_state.jl`): `enumerate_canonical_partitions`, `exact_posterior`, `run_kernel_invariance_test`. Brute-force correctness proofs for small N.
- **Detailed balance** (`detailed_balance.jl`): `check_detailed_balance` verifies DB holds for specific transitions.
- **Mixing diagnostics** (`mixing.jl`): `effective_sample_size`, `split_gelman_rubin`, `indicator_ess`, `run_mixing_test`. Chain efficiency assessment (not correctness).

## Dependencies

Key external packages:
- `SMLMData` - Core SMLM types (`Emitter2DFit`, `BasicSMLD`, `IdealCamera`)
- `Hungarian` - Optimal assignment for MAP-N and boundary deduplication
- `NearestNeighbors` - KDTree for spatial clustering
- `Mmap` - Memory-mapped chain archive (stdlib)

Examples use `CairoMakie` and `SMLMRender` for visualization.

## Reference Implementation (MATLAB)

This package is a Julia reimplementation of BaGoL. The current MATLAB implementation is in [LidkeLab/smite](https://github.com/LidkeLab/smite) at `+smi/@BaGoL`, integrated into the SMITE toolbox with shared SMF/SMD data structures.

**Paper:** Fazel et al., "High-Precision Estimation of Emitter Positions using Bayesian Grouping of Localizations", *Nature Communications* 13, 7152 (2022). [doi:10.1038/s41467-022-34894-2](https://doi.org/10.1038/s41467-022-34894-2)

## Sampler Research Protocol

Multi-session research framework for improving the collapsed Gibbs sampler. Adapted from MillenniumPrize framework.

### Key Files

- `dev/STATUS.md` — Current research state. Read at start of session, update at end.
- `dev/KNOWLEDGE_BASE.md` — Dead ends and working techniques. Check before proposing new approaches.
- `docs/math_reference.md` — Authoritative mathematical reference for the sampler. Must be updated each research round.
- `dev/sampler_reference.md` — Compact sampler reference kept in sync with docs/math_reference.md.

### Session Protocol

**"Run a round" / "next round" / "sampler round"** = execute the full protocol below.

1. **Start:** Read `dev/STATUS.md` + relevant KNOWLEDGE_BASE.md entries
2. **Plan:** Draft the round plan (what to try, why, expected outcome).
3. **Codex review:** Run `/ask-codex` with the plan + relevant STATUS.md context. Incorporate feedback before proceeding.
4. **Validate first:** Run diagnostics before making changes. Let numerics lead.
5. **Work:** Make changes, run tests, analyze results.
6. **End (non-negotiable):** Update `dev/STATUS.md` (Recent Activity, thread status). Update `dev/KNOWLEDGE_BASE.md` if dead end found or technique proven.

### Validation-First Principle

Run `brute_force_enumeration.jl` and `detailed_balance_check.jl` before AND after sampler changes. The diagnostic verdict is ground truth. Do not record analytical claims without numerical backing.

```bash
# Quick validation (~2 min)
julia --project=dev dev/detailed_balance_check.jl

# Full validation (~30 min)
julia --project=dev dev/brute_force_enumeration.jl

# Practical validation (~20 min)
julia --threads=auto --project=dev dev/smlmsim_highdensity.jl
```

## Git Workflow

Finding parent branch:
```bash
git show-branch | head -10
```

The `rjmcmc` branch preserves the legacy RJMCMC sampler implementation.

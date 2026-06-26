```@meta
CurrentModule = SMLMBaGoL
```

# API Reference

Every exported type and function, grouped by topic.

```@index
```

## Main entry points

```@docs
run_bagol
BaGoLConfig
BaGoLDiagnostics
BaGoLResult
```

## Uncertainty correction

```@docs
estimate_se_adjust
apply_se_adjust
```

## Collapsed chain & state

```@docs
run_collapsed_chain
initialize_from_assignments
CollapsedState
CollapsedChainResult
ClusterStats
```

## MAP-N estimation

```@docs
estimate_dahl
estimate_mapn_collapsed
estimate_mapn_psm
estimate_vi_greedy
estimate_mapn_overlap
```

## Accumulators

```@docs
AbstractAccumulator
EmitterCountHist
PosteriorImage
NNDistHist
PartitionSamples
PSMAccumulator
```

## Partitioning

```@docs
Partition
partition_locs
```

## Spatial, allocation & count priors

```@docs
AbstractSpatialModel
FlatSpatial
LocmixSpatial
spatial_ml
spatial_pred
AbstractAllocationModel
DMAllocation
DecoupledAllocation
CategoricalAllocation
UniformSpatialPrior
count_model_map_k
```

## Posterior image & chain archive

```@docs
save_posterior_png
BaGoLArchive
```

## Reports

```@docs
compute_report
write_report
match_positions
compute_fov
plot_report
render_report
```

## Simulation

```@docs
SimulationResult
nmer_positions
nmer_grid_positions
nmer_random_positions
simulate_localizations
simulate_nmer
simulate_nmer_grid
make_camera
print_simulation_summary
```

## Optimality & speed tests

```@docs
run_optimality_sweep
run_speed_test
write_sweep
write_speed
plot_sweep
plot_speed
plot_se_adjust
```

---

# Advanced & diagnostics

The rest of this page covers experimental features and the sampler-validation toolkit —
useful for development and correctness checking, not needed for everyday grouping.

## Multi-cue grouping (experimental)

Group on additional conjugate features (spectral, lifetime, …) alongside position. The API
is in place but there is no end-user workflow page yet.

```@docs
MultiClusterStats
FeatureSet
run_multicue_chain
build_multicue_precisions
extract_multicue
gaussian_contribution
```

## Diagnostics: target densities

```@docs
AbstractTargetDensity
DecoupledTarget
DecoupledLocmixTarget
DMFlatTarget
DMLocmixTarget
DirectNegBinFlatTarget
DirectNegBinLocmixTarget
FazelFlatTarget
log_target
evaluate_target
```

## Diagnostics: finite-state validation

```@docs
enumerate_canonical_partitions
enumerate_labeled_partitions
canonicalize
exact_posterior
run_kernel_invariance_test
run_labeled_invariance_test
```

## Diagnostics: detailed balance

```@docs
DetailedBalanceResult
check_detailed_balance
```

## Diagnostics: mixing

```@docs
ChainDiagnosticAccumulator
effective_sample_size
autocorrelation
split_gelman_rubin
indicator_ess
run_mixing_test
```

# SMLMBaGoL — API Overview

AI-parseable public API reference for **SMLMBaGoL.jl** (Bayesian Grouping of
Localizations). This is a compact companion to the Documenter site (`docs/src/api.md`);
it lists the exported surface with real signatures and defaults. Units: **positions and
uncertainties are in micrometers (μm)** throughout.

Package: reimplementation of BaGoL (Fazel et al., *Nat. Commun.* 13:7152, 2022). Groups
repeated blinks of one emitter into single emitters via a collapsed Gibbs sampler
(RJMCMC split/merge + birth/death), with MAP-N position estimation.

---

## 1. Main entry point

```julia
run_bagol(smld::SMLMData.SMLD; kwargs...) -> (result_smld::BasicSMLD, diagnostics::BaGoLDiagnostics)
run_bagol(locs::Vector{<:SMLMData.Emitter2DFit}; camera, kwargs...)   # also accepts raw locs
run_bagol(smld, cfg::BaGoLConfig)                                      # config-struct form (integrated pipeline)
```

Runs the full partitioned pipeline (precision-weighted DBSCAN → per-partition collapsed
Gibbs → boundary dedup → MAP-N). Returns the grouped emitters as an SMLD plus diagnostics.

Key keyword arguments (defaults shown):

```julia
# Count model
μ = nothing                 # mean locs/emitter; nothing = start from prior mean (10) then learn
shape = 2.0                 # NegBin shape α (1=exponential/dSTORM, >1=peaked/DNA-PAINT)
learn_distribution = true   # true | false | :mu | :shape
gamma = nothing             # DM concentration; nothing = use shape

# MCMC
n_iterations = 4000
burn_in = 2000
sync_interval = 100         # iterations between global hierarchical μ/shape/ρ updates
allocation_model = :dm      # :dm (Dirichlet-Multinomial) | :decoupled | :categorical
spatial_model = :locmix     # :locmix (localization mixture) | :flat
k_prior = :auto             # :auto | :poisson (only valid with :flat) | :none
n_restricted_scans = 5      # Jain-Neal restricted Gibbs scans for split/merge
n_bd_substeps = 5           # birth/death substeps per BD selection

# Per-emitter linear motion (opt-in)
motion = :none              # :none | :linear (θ(t)=μ+v·δ, velocity integrated out)
motion_sigma = 0.002        # per-axis end-to-end drift prior SD (μm); needs per-loc frame

# Partitioning
partition_sigma = 3.0       # DBSCAN threshold in σ units (Inf = no partitioning)
min_partition_size = 0
max_partition_size = 1000   # METIS-split partitions larger than this
skip_partition_size = typemax(Int)
overlap = :auto             # boundary overlap width (:auto or Float64 μm)

# Uncertainty correction (standalone runs only; leave 0.0 in the integrated pipeline)
se_adjust = 0.0             # 0.0 | :auto (run estimate_se_adjust finder) | τ (scalar/tuple/vector, μm) in quadrature σ²+τ²
force_se_adjust = false     # apply even if SMLD already σ-corrected
keep_se_finder = false      # retain finder diagnostics for plot_se_adjust

# Output
posterior_pixel_size = 0.002  # Rao-Blackwellized posterior image (0.0 = disable)
posterior_xlim = nothing; posterior_ylim = nothing
archive_path = nothing        # mmap chain archive
progress_file = nothing
verbose = true
# Hyperpriors (rarely changed): μ_prior_shape/scale, shape_prior_shape/scale, ρ_prior_shape/rate
```

`BaGoLConfig` — `Base.@kwdef struct` (serializable to TOML) mirroring the `run_bagol`
kwargs for the integrated SMLMAnalysis pipeline. Note its default `se_adjust = :auto`
(vs `0.0` for the bare `run_bagol` function). Fields `bridge_ratio` / `min_split_size`
exist on the struct but are not currently forwarded by `run_bagol`.

`BaGoLDiagnostics` fields: `n_emitters`, `posterior_k`, `acceptance_rates`, `final_μ`,
`final_shape`, `n_partitions`, `posterior_image`, `se_adjust`, `convergence_trace`,
`motion` (velocities + mean/std when `motion=:linear`), `se_finder` (when
`keep_se_finder=true`).

---

## 2. Uncertainty correction (localization-error finder)

```julia
estimate_se_adjust(smld::SMLMData.SMLD; kwargs...) -> τ̂          # isotropic over-merge descent (slow)
apply_se_adjust(smld, τ) -> SMLMData.SMLD                        # σ² → σ²+τ² in quadrature (self-guards double-apply)
```

Independent of the sampler. `run_bagol(...; se_adjust=:auto)` runs the finder internally.

---

## 3. MAP-N position estimation

All operate on stored assignment samples from the chain.

```julia
estimate_mapn_collapsed(samples, locs)         # histogram-mode K + Hungarian label matching
estimate_dahl(samples, locs, psm)              # Dahl consensus partition (preferred default)
estimate_mapn_psm(samples, locs, psm)          # PSM thresholding + Hungarian refinement
estimate_vi_greedy(samples, locs, psm)         # variational greedy approximation
estimate_mapn_overlap(...)                      # overlap-based matching
```

PSM-based methods need a `PSMAccumulator` in the accumulator list. Fallback single-sample:
`extract_emitters(state, locs)` (unexported).

---

## 4. Low-level chain

```julia
run_collapsed_chain(locs; n_iterations, burn_in, accumulators::Vector{AbstractAccumulator}, kwargs...) -> CollapsedChainResult
initialize_from_assignments(...)
```

Types: `CollapsedState`, `CollapsedChainResult`, `ClusterStats` (immutable sufficient
statistics: `add_loc`/`remove_loc`, `log_marginal_likelihood`, `log_predictive`,
`posterior_mean`/`posterior_cov`).

Multi-cue variant: `run_multicue_chain`, `build_multicue_precisions`, `extract_multicue`,
`MultiClusterStats`, `FeatureSet`, `gaussian_contribution`.

---

## 5. Accumulators

Collect chain statistics without storing full samples (`AbstractAccumulator`):

```julia
EmitterCountHist()                                   # histogram of K per iteration
PosteriorImage(; pixel_size, ...)                    # Rao-Blackwellized posterior image
NNDistHist(; max_dist=0.1, n_bins=100)               # NN distances between emitter posteriors
PartitionSamples(...)                                # thinned assignment vectors (for MAP-N)
PSMAccumulator(...)                                  # posterior similarity matrix (co-assignment)
```

---

## 6. Reports & output

```julia
compute_report(result_smld, diagnostics; true_positions=nothing, locs_smld=nothing) -> report
write_report(report; output_dir="output")
match_positions(estimated, truth; ...)               # Hungarian matching to ground truth
compute_fov(smld; margin=nothing)                    # field-of-view bounds
save_posterior_png(filename, post::NamedTuple; ...)
```

Extension stubs (activate when the weak dep is loaded):

```julia
plot_report(report; output_dir)      # requires CairoMakie   (BaGoLMakieExt)
plot_se_adjust(...)                  # requires CairoMakie
render_report(locs_smld, result_smld; output_dir, ...)   # requires SMLMRender (BaGoLRenderExt)
```

---

## 7. Simulation

```julia
nmer_positions(n, diameter; ...)              # n points on a circle
nmer_grid_positions(; ...)                    # grid of n-mers
nmer_random_positions(; ...)
make_camera(field_size, pixel_size) -> IdealCamera
simulate_localizations(...) -> SimulationResult
simulate_nmer(; ...) -> SimulationResult      # single n-mer test scene
simulate_nmer_grid(; ...) -> SimulationResult
print_simulation_summary(sim)
```

`SimulationResult` carries `.smld` (localizations), `.true_positions`, and camera.

---

## 8. Partitioning & archive

```julia
partition_locs(locs, ...)                     # precision-weighted DBSCAN (advanced)
Partition                                     # spatial cluster: locs, original_indices, boundary flags
UniformSpatialPrior(locs; padding=0.05)
BaGoLArchive                                  # mmap binary chain archive (opt-in via archive_path)
compute_from_archive(AccType, archive, partition_id, locs)   # post-hoc accumulator over archived samples
```

---

## 9. Optimality / benchmarking

```julia
run_optimality_sweep(; n_values, mu_values, n_trials, ...) -> sweep
run_speed_test(; n_locs_range, ...) -> speed
count_model_map_k(n_locs, mu, shape)          # Q-PAINT count-only MAP K baseline
write_sweep(sweep; output_dir);  write_speed(speed; output_dir)
plot_sweep(...); plot_speed(...)              # require CairoMakie
```

---

## 10. Sampler diagnostics (validation)

Correctness/mixing tools (see `docs/math_reference.md` and `src/diagnostics/`):

- Target densities: `AbstractTargetDensity`, `DecoupledTarget`, `DMFlatTarget`,
  `DirectNegBinFlatTarget`, `FazelFlatTarget`, … `log_target`, `evaluate_target`.
- Spatial/allocation models: `FlatSpatial`, `LocmixSpatial`, `spatial_ml`, `spatial_pred`;
  `DMAllocation`, `DecoupledAllocation`, `CategoricalAllocation`.
- Finite-state: `enumerate_canonical_partitions`, `exact_posterior`,
  `run_kernel_invariance_test`.
- Detailed balance: `check_detailed_balance`, `DetailedBalanceResult`.
- Mixing: `effective_sample_size`, `autocorrelation`, `split_gelman_rubin`,
  `indicator_ess`, `run_mixing_test`, `ChainDiagnosticAccumulator`.

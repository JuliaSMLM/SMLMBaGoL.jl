# Feature-Model Dispatch Architecture

Status: **design locked, implementation in progress** on branch `feature-model-dispatch`.
Reviewed by Codex (2026-06-13); math + "refactor not rewrite" verdict validated.

## Motivation

BaGoL groups localizations into emitters. Today it groups on **2D position only**. The goal:
group on an arbitrary set of **features** (cues) — 2D/3D position, spectral wavelength
(λ ± σ_λ), fluorescence lifetime, or any future per-localization quantity with an
uncertainty — where adding a new feature is "define a type + ~5 methods," not a rewrite.
**Position must not be privileged.**

## Thesis

BaGoL is a **collapsed conjugate grouping sampler**. An emitter has a latent feature vector
θ = (θ₁,…,θ_M). Each localization provides noisy observations of some subset of features.
The sampler infers cluster membership with the latent features **integrated out analytically**.
This is standard collapsed-conjugate DP-mixture machinery (Neal 2000; Jain–Neal split-merge —
already used here). Generalizing from "position" to "features" is a change to the
sufficient-statistic core, **not** to the sampler.

## Key decisions

1. **Feature-symmetric core.** Position is just `GaussianFeature{2}` on `(x,y)`. Spectral is
   `GaussianFeature{1}` on `(λ,)`. 3D is `GaussianFeature{3}`. No feature is special in the core.
2. **Block-diagonal across features.** A cluster's sufficient statistics = a tuple of
   *independent* per-feature conjugate blocks. Total log-marginal / log-predictive = **sum**
   over blocks (conditional independence of features given the shared emitter). Each block
   keeps full D×D *within-feature* covariance; no cross-feature coupling. Correlated Gaussian
   features may later be merged into one joint higher-D block (closure property); the loc
   bundle carries covariance so this is non-breaking.
3. **Generic representation (option B).** The engine works on a precomputed per-loc
   *contribution bundle* (tuple of per-feature contributions), built once by boundary
   converters from SMLMData emitters / NamedTuples / future types. The engine never reads
   `x`/`y`/`λ`. Generalizes today's `LocPrecision`.
4. **Refactor, not rewrite.** Verified by audit: the MCMC engine touches geometry only through
   the seam `{add_loc, remove_loc, feature_logpred (→spatial_pred), feature_logml (→spatial_ml),
   area, n, empty}`. Engine, allocation priors (DM/categorical/decoupled), NegBin count model,
   partition (DBSCAN, spatial), hierarchical updates, and the chain archive are reused.

## Interface — the "add a cue" surface

A feature type `F <: AbstractFeature` (stateless config; isbits or singleton) implements:

- `loc_contribution(f, loc) -> C` — precompute one loc's contribution (isbits). Setup-time
  only (not in the hot loop), so field/symbol accessors are fine here.
- `empty_block(f) -> B`; `accumulate(b, f, c) -> b'`; `deaccumulate(b, f, c) -> b'` — O(1)
  exact-inverse natural-parameter updates.
- `feature_logml(b, f) -> Float64` — collapsed marginal (latent feature integrated out under
  f's prior).
- `feature_logpred(b, f, c) -> Float64` — predictive for Gibbs
  (`= feature_logml(accumulate(b,f,c),f) - feature_logml(b,f)`).
- `feature_summary(b, f) -> NamedTuple` — posterior summary (e.g. `(mean=, cov=)`) for output.

Each block tracks its own observed count `n_obs` (a loc may **lack** a feature → not
accumulated to that block). The cluster's **membership** count `n` is separate and is what the
DM allocation prior and NegBin count model use.

Priors are **per-feature**: `FlatPrior(domain)` (proper — contributes an *intentional*
−log(volume) Occam term) or `LocmixPrior(...)` (data-driven; default). Improper flat priors
are disallowed (they make K-changing Bayes factors undefined).

The **emitter-count prior** is a *separate* policy object `KPriorPolicy` — e.g.
`PoissonField(ρ, A)` over the spatial domain, or `NoKPrior` — **not** a feature method. The
Poisson(ρA) prior stays spatial (it is an emitter-process density over the field, not a
product of feature volumes).

## Type design

```julia
struct ClusterStats{BT<:Tuple}
    blocks::BT      # one isbits block per feature
    n::Int32        # membership count (DM allocation + count model)
end
empty_stats(features::Tuple) = ClusterStats(map(empty_block, features), Int32(0))

add_loc(cs, contribs)       = ClusterStats(map(accumulate, cs.blocks, FEATURES, contribs), cs.n + Int32(1))
total_logml(cs, feats)      = mapreduce(feature_logml, +, cs.blocks, feats)        # == spatial_ml
total_logpred(cs, feats, c) = mapreduce((b,f,ci)->feature_logpred(b,f,ci), +, cs.blocks, feats, c)  # == spatial_pred
```

**`CollapsedState` must be parameterized over the concrete block-tuple type `BT` and
contribution type `CT`** so `clusters::Vector{ClusterStats{BT}}` and `_loc_precs::Vector{CT}`
stay **concrete** — this is the single thing that preserves the zero-allocation hot loop.
The no-arg `ClusterStats()` becomes `empty_stats(state)` / `zero(eltype(state.clusters))`.
Use a heterogeneous **tuple** of concrete feature models/blocks; never `Vector{AbstractFeature}`,
abstract fields, or `Function` accessors in the sampler. Prefer tuple recursion / inlined
`mapreduce`; reach for `@generated` only if `@code_warntype` proves inference failed.

## Reused unchanged

`collapsed_moves.jl` (the entire kernel), DM/categorical/decoupled allocation, NegBin count
model, `partition.jl` (spatial DBSCAN), hierarchical μ/shape updates, archive.

## Refactored

- `cluster_stats.jl` → feature framework + `GaussianFeature{D}` block.
- `types.jl` → parameterize `CollapsedState` over `BT`/`CT`.
- `collapsed_moves.jl` → ~8 `ClusterStats()` sites → `empty_stats`; the seam fns
  `spatial_ml`/`spatial_pred` keep their call sites but become folds over feature blocks.
- output seam → `feature_summary` + `make_emitter(summary, n, id)` dispatched on the feature
  set; replaces hardcoded `Emitter2DFit` at `collapsed_sampler.jl:379`, `rjmcmc.jl`, `mapn.jl`,
  `partitioned.jl`.
- `posterior_mean`/`posterior_cov` consumers (`mapn.jl`, `accumulators.jl`) → `feature_summary`.
- partition + MAP-N + reports + render stay **spatial** (read the position feature's summary).

## Math (Codex-validated)

For cluster c, with features independent given the shared latent θ and priors that factor:

    p(D_c) = ∫ ∏_i ∏_f p(y_if | θ_f) · ∏_f p_f(θ_f) dθ = ∏_f p_f(D_cf)

⟹ log-marginal and log-predictive are **sums** over features. DM allocation and NegBin count
terms depend only on K and membership counts → **no new bookkeeping**. Split/merge/birth/death
proposal densities need **no extra RJ Jacobians**: the feature parameters are collapsed out,
not part of the Markov state. Feature priors must be **proper**.

## Limits

- Arbitrary covariance is analytic only **within** a conjugate family. Cross-*family*
  correlation (Gaussian position × Gamma lifetime) breaks the closed-form collapse →
  transform to a common family, approximate, or don't collapse that feature.
- `locmix` as a dense grid is 2D-only (saddle-point approx at the posterior mean). 1D spectral
  locmix is cheap; 3D locmix needs exact-O(N) or a tree approximation, not a dense grid.
- Open target-definition questions for later: missing-not-at-random cues, censored/truncated
  features, assignment-dependent feature availability, learned per-feature hyperparameters,
  empirical locmix priors rebuilt during sampling.

## Implementation sequence + validation gates

1. **DONE — Behavior-preserving refactor to a dimension-parametric `ClusterStats{D,L}`**
   (the single-feature 2D core). GATE met: 296/296 tests + brute-force 4/4 +
   detailed-balance bias 0.0; `@inferred`/`isbits` zero-alloc hot loop.
2. **DONE — `GaussianFeature{3}` (3D position).** `feature_dim`/`_loc_precision`/`posterior_*`
   for D=3; dimension-derived engine empties (`empty_cluster` / `zero(eltype(clusters))`);
   `make_emitter` output seam (2D→Emitter2DFit, 3D→Emitter3DFit). GATE met: 307/307 tests
   (incl. 3D core + z-separation round-trip) + detailed balance holds + brute-force 2D green.
   Follow-ups: 3D-locmix, full `run_bagol`-3D (MAP-N/dedup/posterior-image), 3D volume flat prior.
3. **Add spectral `GaussianFeature{1}`** + missing-cue (`n_obs`) handling; tuple-of-blocks
   multi-cue; a joint multi-cue grouping test.

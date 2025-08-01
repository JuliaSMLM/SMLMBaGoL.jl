- using statements must only go in the main SMLMBaGoL.jl file. 
- all `using` or `import` commands should be in main module file only, not in other files. 

- You can ask me to find the parent branch using any of these approaches:

  Direct commands:
  - "What branch was this created from?"
  - "Find the parent branch"
  - "What's the source branch for this feature branch?"

  More specific Git requests:
  - "Run git show-branch to find the parent"
  - "Use git merge-base to find where this branched from"
  - "Check git log --graph to see the branch point"

  The most reliable approach:
  git show-branch | head -10
  This shows the branching relationship clearly.

  Or ask me to:
  - "Run the right git commands to find which branch this was created from"
  - "Check the git history to find the actual parent branch"

  The key is I should look at the first commit on the feature branch (9be0e2e) and see what it was based on, rather than just looking at merge-base with main. The git branch --contains <commit> command I used was the
  right approach to see that commit 8d6563f exists on negative-binomial-overdispersion.

- use multiple threads when running examples

# Code Architecture Overview

## Key Type Hierarchy
- `AbstractLocalization` - Base type for localization data
- `AbstractEmitter` (from SMLMData) - Base type for emitters
- `AbstractPrior` - Base type for priors
  - `AbstractSpatialPrior` → `UniformSpatialPrior`
  - `AbstractCountPrior` → `HierarchicalNegBinomialPrior`
- `AbstractRJMCMCMove` - Base type for MCMC moves (Birth, Death, Move, Allocate)
- `AbstractChainState` → `BaGoLState`

## Main Entry Points
- `run_bagol()` - Primary analysis function
- `simulate_static_smlm()` - SMLMSim integration for realistic simulations
- `estimate_mapn()` - Extract MAP-N (Maximum A Posteriori Number) emitter positions
- `diagnose_chains()` - Assess MCMC chain quality

## Common Workflows
1. **Basic Analysis**: localizations → `run_bagol()` → `estimate_mapn()` → results
2. **With Visualization**: Add `gen_sr_image()` and `sr_circles()` for uncertainty plots
3. **Hierarchical Bayes**: Enable with `hierarchical_interval` parameter in `run_bagol()`
4. **Parallel Processing**: Use `partition_radius` for spatial partitioning with threading

## Performance Tips
- Enable threading with `julia --threads=auto` for parallel partition processing
- Use spatial partitioning for large datasets (>10k localizations)
- Pre-compile with small test run before large analyses
- Hierarchical updates improve convergence but add computational cost

## Current Development Focus
- Improving Negative Binomial prior fitting for better handling of under-dispersed data
- Enhanced filtering of spurious low-count emitters
- Better κ (overdispersion) parameter initialization strategies

## Consistency Likelihood - New Development

### Problem Identified
The standard Gaussian likelihood systematically over-segments because it penalizes ALL deviation, including statistically expected variation. This is a fundamental issue, not a prior or parameter problem.

### Solution Implemented
Created `consistency_log_likelihood()` that penalizes variance mismatch using KL divergence:
```julia
# In addition to standard likelihood, adds penalty:
Penalty = α * n/2 * (var - log(var) - 1)
```

Key features:
- Penalizes both overfitting (var < 1) and underfitting (var > 1)
- Parameter α controls penalty strength (α ≈ 20 works well)
- `adaptive_consistency_likelihood()` auto-adjusts α based on K/N ratio

### Status
- ✅ Implemented in `/src/likelihood/consistency_likelihood.jl`
- ✅ Demonstrated to prevent over-segmentation with sufficient α
- ⏳ TODO: Integrate into RJMCMC algorithm
- ⏳ TODO: Test on real examples (nmer_demo, smlmsim_demo)
- ⏳ TODO: Optimize α parameter selection
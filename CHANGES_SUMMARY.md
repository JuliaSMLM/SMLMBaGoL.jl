# Summary of Mathematical Corrections

## Branch: fix/gibbs-sampling-correctness

### 1. Fixed Move Step (src/moves/move.jl)
**Issue**: The Move step was not implementing proper Gibbs sampling. It was using an ad-hoc noise term (`avg_sigma_x * 0.5`) instead of sampling from the posterior distribution.

**Fix**: Implemented correct Gibbs sampling following the mathematical reference (equations 291-301):
- Calculate precision-weighted mean: `x_mean = Σ(x_i/σ_i²) / Σ(1/σ_i²)`
- Calculate posterior variance: `x_variance = 1 / Σ(1/σ_i²)`
- Sample from posterior: `x_new ~ N(x_mean, x_variance)`

### 2. Changed Allocate Move to Full Sweep (src/moves/allocate.jl)
**Issue**: The Allocate move only reallocated a single random localization, which is valid but inefficient.

**Fix**: Changed to perform a full Gibbs sweep of all localizations:
- Iterate through all localizations
- For each, compute likelihood-based probabilities
- Sample new allocation from categorical distribution
- This matches the implementation in the refactor-rjmcmc branch

**Acceptance Ratio**: For full Gibbs sweep, the proposal probabilities cancel out, so only the likelihood ratio is needed.

### 3. Implemented Dirichlet-Multinomial Model
**Implementation**: Added full Dirichlet-multinomial model for handling overdispersion in allocation counts:

**New Files**:
- `src/utils/dirichlet_multinomial.jl`: Core Dirichlet-multinomial functions

**Modified Files**:
- `src/core/priors.jl`: Added `λ_prior` field to `CompoundPrior`
- `src/moves/birth_death.jl`: Updated acceptance ratios to include Dirichlet-multinomial likelihood

**Key Features**:
- Handles overdispersion (allocation variance higher than expected from multinomial)
- Useful when emitters have varying "attractiveness" beyond spatial likelihood
- Concentration parameters estimated using method of moments from λ prior
- Log-space calculations for numerical stability

**Mathematical Basis**: The Dirichlet-multinomial distribution is a compound distribution where:
- Multinomial probabilities p ~ Dirichlet(α)
- Counts ~ Multinomial(n, p)

This allows for extra variance in allocation counts beyond what would be expected from a simple multinomial.

## Testing
Verified the Move step now correctly samples from the posterior distribution with a test showing empirical statistics match theoretical values.
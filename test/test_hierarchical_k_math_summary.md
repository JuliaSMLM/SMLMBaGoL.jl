# Hierarchical K Mathematical Tests Summary

This test suite validates the mathematical correctness of estimating μ and κ parameters from empirical count data, as done in the hierarchical Bayesian analysis.

## Test Coverage

### 1. **Known Distribution Recovery - Method of Moments**
- Tests that MoM correctly recovers parameters from data generated with known μ and κ
- Verifies the mathematical formula: κ = μ²/(v-μ)
- Tests convergence as sample size increases (50, 100, 500, 1000 samples)
- Results: μ estimates within 10%, κ estimates within 30% for large samples

### 2. **Edge Cases - Method of Moments**
- **Underdispersed (Poisson-like) data**: Correctly returns large κ (100.0 cap)
- **Extreme overdispersion** (κ=0.5): Captures the pattern with small κ estimates
- **Empty data**: Returns default values (1.0, 1.0)

### 3. **Maximum Likelihood Estimation**
- Verifies MLE achieves higher likelihood than MoM
- Tests parameter recovery: μ within 15%, κ within 35%
- Confirms MLE optimization improves on MoM initialization

### 4. **Mathematical Consistency - Moments**
- Validates the relationship: Var[X] = μ + μ²/κ
- Tests consistency triangle: empirical ≈ estimated ≈ theoretical moments
- Empirical vs estimated: <1% mean error, <5% variance error

### 5. **Parameter Space Exploration**
Tests across different parameter regimes:
- Small μ (5.0), high overdispersion (κ=1.0)
- Large μ (100.0), moderate overdispersion (κ=50.0)
- Moderate μ (20.0), extreme overdispersion (κ=0.5)
- Near-Poisson case (μ=50.0, κ=200.0)

### 6. **Empirical Chain State Simulation**
Simulates realistic MCMC scenarios:
- Multiple emitters (50) with hierarchical structure
- Varying observation counts per emitter (5-20)
- Tests both pooled and aggregated count estimation
- Recovery of global parameters: μ within 20%, κ within 40%

## Key Mathematical Validations

1. **MoM Formula Correctness**: κ = μ²/(v-μ) correctly implemented
2. **Likelihood Improvement**: MLE ≥ MoM likelihood
3. **Moment Relationships**: Theoretical moments match empirical data
4. **Parameter Transformations**: (μ,κ) ↔ (r,p) conversions work correctly
5. **Edge Case Handling**: Proper behavior for extreme parameter values

## Usage in Hierarchical Updates

These tests validate that the fundamental parameter estimation from count data is mathematically sound. This forms the basis for:
- Initial parameter estimation in hierarchical models
- Updates during MCMC sampling
- Diagnostic assessment of model fit

The tests use Distributions.jl for ground truth and the codebase's lower-level functions (`fit_negbinomial_mom`, `fit_negbinomial_mle`) to ensure mathematical correctness without requiring full MCMC machinery.
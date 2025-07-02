# τ² Parameter: Technical Summary

## What is τ²?

τ² (tau-squared) represents **additional systematic localization uncertainty** beyond the reported per-localization uncertainties (σx, σy). It accounts for systematic errors in SMLM data such as:

- Stage drift and thermal effects
- PSF calibration errors  
- Algorithm fitting limitations
- Environmental vibrations

## Mathematical Model

### Likelihood with τ²
Each localization follows:
```
loc ~ N(emitter_position, [σx² + τ²    0     ])
                          [0        σy² + τ²]
```

### Hierarchical Prior
```
τ² ~ InverseGamma(a_τ, b_τ)

where:
a_τ = 2.0                    # Weakly informative shape
b_τ = 2 × initial_τ²         # Scale based on data
initial_τ² = (0.1 × median_σ)²  # 10% of median localization precision
```

## Implementation

### Automatic Initialization
```julia
# Data-driven initial estimate
median_σ = median([sqrt(loc.σx² + loc.σy²) for loc in localizations])
initial_τ² = (0.1 * median_σ)²

# Hyperprior parameters
τ²_hyperprior = (2.0, initial_τ² * 2.0)
```

### Gibbs Sampling Update
```julia
# Collect precision-weighted residuals from all chains
sum_squared_residuals = Σ [(x_residual² + y_residual²) / (σ² + τ²)]
total_count = 2 × n_allocated_localizations

# Posterior update (conjugate)
a_posterior = a_τ + total_count/2  
b_posterior = b_τ + sum_squared_residuals/2
τ²_new ~ InverseGamma(a_posterior, b_posterior)
```

## Practical Usage

### Default Behavior
- **Automatically enabled** in all hierarchical analyses
- **No user parameters** required
- **Adapts to data** quality and experimental conditions

### Interpretation
```
τ² = 100 nm² → additional 10 nm systematic uncertainty
τ² = 10,000 nm² → additional 100 nm systematic uncertainty  
```

### Monitoring Evolution
The hierarchical evolution plot shows τ² learning:
```
Initial: 0.5 nm² (conservative estimate)
Final: 50,000 nm² (learned from data)
```

## Benefits

1. **Improved Accuracy**: More realistic uncertainty estimates
2. **Better Model Fit**: Accounts for unmodeled systematic effects  
3. **Quality Assessment**: Large τ² indicates systematic issues
4. **Principled Uncertainty**: Proper Bayesian propagation

## When τ² is Important

- **High-precision experiments**: Where systematic errors dominate
- **Long-duration acquisitions**: Subject to drift and environmental effects
- **Multi-condition studies**: Comparing data quality across experiments
- **Method development**: Understanding systematic limitations

## Advanced Notes

### Convergence
τ² convergence indicates:
- Sufficient data to estimate systematic uncertainty
- Stable experimental conditions
- Appropriate model specification

### Large τ² Values
If τ² >> σ²:
- Consider experimental improvements
- Check for drift correction
- Validate PSF calibration
- Review fitting algorithms

### Integration with Analysis
τ² affects:
- All likelihood calculations
- Position refinement moves
- Uncertainty quantification
- Model selection criteria
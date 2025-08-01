"""
Consistency-based likelihood that penalizes both over- and under-fitting.

The key insight: Standard Gaussian likelihood rewards making residuals 
smaller than expected. This modified likelihood penalizes deviations 
from the expected variance in BOTH directions.
"""

# Standard log-likelihood for comparison
function standard_log_likelihood(state::BaGoLState)
    ll = 0.0
    for i in 1:length(state.localizations)
        if state.allocations[i] > 0
            emitter = state.emitters[state.allocations[i]]
            loc = state.localizations[i]
            latent = state.latent_positions[i]
            ll += log_likelihood(emitter, loc, state.τ²)
        end
    end
    return ll
end

"""
    consistency_log_likelihood(state::BaGoLState; α=1.0)

Modified log-likelihood that penalizes variance mismatch.

When residuals have variance less than expected (overfitting), 
this adds a penalty. The parameter α controls the penalty strength.

Based on the insight that residuals should have variance ≈ σ² + τ²,
not be minimized to zero.
"""
function consistency_log_likelihood(state::BaGoLState; α=1.0)
    # First compute standard likelihood and collect residuals
    ll = 0.0
    residuals_x = Float64[]
    residuals_y = Float64[]
    expected_variances_x = Float64[]
    expected_variances_y = Float64[]
    
    for i in 1:length(state.localizations)
        if state.allocations[i] > 0
            emitter = state.emitters[state.allocations[i]]
            loc = state.localizations[i]
            
            # Standard likelihood contribution
            ll += log_likelihood(emitter, loc, state.τ²)
            
            # Collect normalized residuals
            σx² = loc.σx^2 + state.τ²
            σy² = loc.σy^2 + state.τ²
            
            push!(residuals_x, (loc.x - emitter.x) / sqrt(σx²))
            push!(residuals_y, (loc.y - emitter.y) / sqrt(σy²))
            push!(expected_variances_x, σx²)
            push!(expected_variances_y, σy²)
        end
    end
    
    # Check variance consistency
    n = length(residuals_x)
    if n > 1
        # Empirical variance of normalized residuals (should be ≈ 1)
        var_x = var(residuals_x)
        var_y = var(residuals_y)
        
        # Penalty based on Kullback-Leibler divergence between
        # empirical and expected variance
        # KL(empirical || expected) = (v_emp/v_exp - log(v_emp/v_exp) - 1) * n/2
        penalty_x = α * n/2 * (var_x - log(var_x) - 1)
        penalty_y = α * n/2 * (var_y - log(var_y) - 1)
        
        # This penalty is 0 when var = 1, positive otherwise
        # It penalizes BOTH under-dispersion (var < 1) and over-dispersion (var > 1)
        ll -= (penalty_x + penalty_y)
    end
    
    return ll
end

"""
    adaptive_consistency_likelihood(state::BaGoLState)

An even better version that adapts the penalty based on the number of emitters.
With more emitters, we expect better fit, so we increase the penalty for overfitting.
"""
function adaptive_consistency_likelihood(state::BaGoLState)
    k = length(state.emitters)
    n = length(state.localizations)
    
    # Adaptive penalty weight: increases with k/n ratio
    # This counteracts the natural advantage of more parameters
    α = 1.0 + 2.0 * k / n
    
    return consistency_log_likelihood(state; α=α)
end
"""
    Initialization methods for hierarchical Negative Binomial parameters
    
    Provides data-driven initialization for μ and κ parameters to improve
    MCMC convergence and avoid poor local modes in slice sampling.
"""

"""
    fit_negbinomial_mom(counts::Vector{Int})
    
Method of moments estimation for Negative Binomial parameters.
Fast but potentially less accurate than MLE.

Returns: (μ, κ) estimates
"""
function fit_negbinomial_mom(counts::Vector{Int})
    # Handle edge cases
    if isempty(counts)
        return (1.0, 1.0)
    end
    
    # Calculate moments
    m = mean(counts)
    v = var(counts)
    
    # Handle underdispersed case (variance ≤ mean)
    if v <= m || m ≈ 0
        return (m, 100.0)  # Large κ = low overdispersion
    end
    
    # Method of moments estimators
    # From Var = μ + μ²/κ, solve for κ
    κ = m^2 / (v - m)
    
    # Bound κ to reasonable range
    κ = clamp(κ, 0.1, 1000.0)
    
    return (m, κ)
end

"""
    fit_negbinomial_mle(counts::Vector{Int}; init_method=:mom, maxiter=1000)
    
Maximum likelihood estimation for Negative Binomial parameters.
More accurate than MoM but slower.

Returns: (μ, κ) estimates
"""
function fit_negbinomial_mle(counts::Vector{Int}; init_method=:mom, maxiter=1000)
    # Handle edge cases
    if isempty(counts)
        return (1.0, 1.0)
    end
    
    n = length(counts)
    sum_counts = sum(counts)
    
    # Get initial values
    if init_method == :mom
        μ_init, κ_init = fit_negbinomial_mom(counts)
    else
        μ_init = mean(counts)
        κ_init = 1.0
    end
    
    # Log-likelihood function (negative for minimization)
    function neg_loglik(params)
        μ, log_κ = params
        κ = exp(log_κ)  # Ensure κ > 0
        
        if μ <= 0 || κ <= 0
            return Inf
        end
        
        # Convert to r, p parameterization for Distributions.jl
        r = κ
        p = κ / (κ + μ)
        
        # Check validity
        if !(0 < p < 1) || r <= 0
            return Inf
        end
        
        # Calculate log-likelihood
        ll = 0.0
        try
            nb_dist = NegativeBinomial(r, p)
            for count in counts
                ll += logpdf(nb_dist, count)
            end
        catch
            return Inf
        end
        
        return -ll
    end
    
    # Optimize
    result = optimize(neg_loglik, [μ_init, log(κ_init)], 
                     Optim.Options(iterations=maxiter, show_trace=false))
    
    if !Optim.converged(result)
        # Fall back to MoM if MLE fails
        return fit_negbinomial_mom(counts)
    end
    
    μ_opt, log_κ_opt = Optim.minimizer(result)
    κ_opt = exp(log_κ_opt)
    
    # Bound to reasonable range
    μ_opt = clamp(μ_opt, 0.1, 10000.0)
    κ_opt = clamp(κ_opt, 0.1, 1000.0)
    
    return (μ_opt, κ_opt)
end

"""
    initialize_hierarchical_prior(counts::Vector{Int}; 
                                 method=:auto,
                                 τ²_init=1e-6,
                                 hyperpriors=nothing)
    
Initialize a HierarchicalNegBinomialPrior from empirical count data.

Arguments:
- counts: Vector of emitter counts
- method: :auto (default), :mle, or :mom
- τ²_init: Initial value for additional variance (default 1e-6)
- hyperpriors: Optional tuple of (μ_hyperprior, κ_hyperprior, τ²_hyperprior)

Returns: HierarchicalNegBinomialPrior
"""
function initialize_hierarchical_prior(counts::Vector{Int}; 
                                     method=:auto,
                                     τ²_init=1e-6,
                                     hyperpriors=nothing)
    # Determine method
    if method == :auto
        # Use MLE for smaller datasets, MoM for larger ones
        method = length(counts) < 1000 ? :mle : :mom
    end
    
    # Fit parameters
    if method == :mle
        μ, κ = fit_negbinomial_mle(counts)
    else
        μ, κ = fit_negbinomial_mom(counts)
    end
    
    # Set hyperpriors if not provided
    if isnothing(hyperpriors)
        # Weakly informative hyperpriors
        μ_hyperprior = (1.0, 0.01)  # Gamma(1, 0.01) - mean 100, var 10000
        κ_hyperprior = (1.0, 0.1)   # Gamma(1, 0.1) - mean 10, var 100
        τ²_hyperprior = (2.0, 2e-6)  # InverseGamma(2, 2e-6) - mean 2e-6
    else
        μ_hyperprior, κ_hyperprior, τ²_hyperprior = hyperpriors
    end
    
    return HierarchicalNegBinomialPrior(
        μ, κ, τ²_init,
        μ_hyperprior,
        κ_hyperprior,
        τ²_hyperprior
    )
end

"""
    fit_negbinomial_prior(counts::Vector{Int}; method=:mle, kwargs...)
    
Convenience function that fits and returns a HierarchicalNegBinomialPrior.
Alias for initialize_hierarchical_prior with clearer name.
"""
function fit_negbinomial_prior(counts::Vector{Int}; method=:mle, kwargs...)
    return initialize_hierarchical_prior(counts; method=method, kwargs...)
end
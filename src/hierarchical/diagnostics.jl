"""
    Diagnostic functions for assessing Negative Binomial prior fit quality
    
    Provides quantitative metrics and statistical tests to evaluate how well
    the fitted Negative Binomial distribution matches empirical data.
"""

"""
    FitStatistics
    
Container for goodness-of-fit statistics.
"""
struct FitStatistics
    chi_squared::Float64
    chi_squared_pvalue::Float64
    ks_statistic::Float64
    ks_pvalue::Float64
    mean_abs_error::Float64
    max_abs_error::Float64
    aic::Float64
    bic::Float64
    empirical_mean::Float64
    empirical_var::Float64
    fitted_mean::Float64
    fitted_var::Float64
end

"""
    assess_negbinomial_fit(empirical_counts::Vector{Int}, prior::HierarchicalNegBinomialPrior)
    
Compute comprehensive goodness-of-fit statistics comparing empirical counts
to the fitted Negative Binomial distribution.

Returns: FitStatistics struct with various metrics
"""
function assess_negbinomial_fit(empirical_counts::Vector{Int}, prior::HierarchicalNegBinomialPrior)
    μ, κ = prior.μ, prior.κ
    n = length(empirical_counts)
    
    # Convert to NegativeBinomial parameterization
    r = κ
    p = κ / (κ + μ)
    nb_dist = NegativeBinomial(r, p)
    
    # Empirical statistics
    emp_mean = mean(empirical_counts)
    emp_var = var(empirical_counts)
    
    # Fitted statistics
    fitted_mean = μ
    fitted_var = μ + μ^2/κ
    
    # Chi-squared test
    chi_sq, chi_sq_p = compute_chi_squared_test(empirical_counts, nb_dist)
    
    # Kolmogorov-Smirnov test (using continuous approximation)
    ks_stat, ks_p = compute_ks_test(empirical_counts, nb_dist)
    
    # Mean absolute error in PMF
    mae, max_ae = compute_pmf_errors(empirical_counts, nb_dist)
    
    # Information criteria
    log_lik = sum(logpdf(nb_dist, x) for x in empirical_counts)
    k_params = 2  # μ and κ
    aic = -2 * log_lik + 2 * k_params
    bic = -2 * log_lik + k_params * log(n)
    
    return FitStatistics(
        chi_sq, chi_sq_p,
        ks_stat, ks_p,
        mae, max_ae,
        aic, bic,
        emp_mean, emp_var,
        fitted_mean, fitted_var
    )
end

"""
    compute_chi_squared_test(counts::Vector{Int}, dist::NegativeBinomial)
    
Perform chi-squared goodness-of-fit test.
Groups bins to ensure expected count ≥ 5 per bin.
"""
function compute_chi_squared_test(counts::Vector{Int}, dist::NegativeBinomial)
    # Create histogram
    max_count = maximum(counts)
    observed = zeros(max_count + 1)
    for c in counts
        observed[c + 1] += 1
    end
    
    n = length(counts)
    expected = [n * pdf(dist, k) for k in 0:max_count]
    
    # Group bins to ensure expected ≥ 5
    grouped_obs = Float64[]
    grouped_exp = Float64[]
    current_obs = 0.0
    current_exp = 0.0
    
    for i in 1:length(observed)
        current_obs += observed[i]
        current_exp += expected[i]
        
        if current_exp >= 5.0 || i == length(observed)
            push!(grouped_obs, current_obs)
            push!(grouped_exp, current_exp)
            current_obs = 0.0
            current_exp = 0.0
        end
    end
    
    # Compute chi-squared statistic
    chi_sq = 0.0
    for i in 1:length(grouped_obs)
        if grouped_exp[i] > 0
            chi_sq += (grouped_obs[i] - grouped_exp[i])^2 / grouped_exp[i]
        end
    end
    
    # Degrees of freedom: bins - 1 - parameters
    df = max(1, length(grouped_obs) - 1 - 2)
    p_value = 1 - cdf(Chisq(df), chi_sq)
    
    return chi_sq, p_value
end

"""
    compute_ks_test(counts::Vector{Int}, dist::NegativeBinomial)
    
Perform Kolmogorov-Smirnov test using continuous approximation.
"""
function compute_ks_test(counts::Vector{Int}, dist::NegativeBinomial)
    n = length(counts)
    sorted_counts = sort(counts)
    
    # Compute empirical CDF
    ks_stat = 0.0
    for i in 1:n
        # Empirical CDF just before and at this point
        F_emp_before = (i - 1) / n
        F_emp_at = i / n
        
        # Theoretical CDF
        F_theory = cdf(dist, sorted_counts[i])
        
        # Maximum difference
        ks_stat = max(ks_stat, abs(F_emp_before - F_theory))
        ks_stat = max(ks_stat, abs(F_emp_at - F_theory))
    end
    
    # Approximate p-value using asymptotic distribution
    # This is an approximation; exact computation requires special functions
    lambda = ks_stat * sqrt(n)
    p_value = 2 * exp(-2 * lambda^2)
    
    return ks_stat, p_value
end

"""
    compute_pmf_errors(counts::Vector{Int}, dist::NegativeBinomial)
    
Compute mean and max absolute error between empirical and fitted PMFs.
"""
function compute_pmf_errors(counts::Vector{Int}, dist::NegativeBinomial)
    # Create empirical PMF
    max_count = maximum(counts)
    empirical_pmf = zeros(max_count + 1)
    for c in counts
        empirical_pmf[c + 1] += 1
    end
    empirical_pmf ./= length(counts)
    
    # Compute errors
    mae = 0.0
    max_ae = 0.0
    
    for k in 0:max_count
        fitted_p = pdf(dist, k)
        empirical_p = empirical_pmf[k + 1]
        
        ae = abs(fitted_p - empirical_p)
        mae += ae
        max_ae = max(max_ae, ae)
    end
    
    mae /= (max_count + 1)
    
    return mae, max_ae
end

"""
    diagnose_poor_fit(stats::FitStatistics)
    
Provide diagnostic messages about why a fit might be poor.
"""
function diagnose_poor_fit(stats::FitStatistics)
    messages = String[]
    
    # Check if chi-squared test fails
    if stats.chi_squared_pvalue < 0.05
        push!(messages, "Chi-squared test rejects fit (p=$(round(stats.chi_squared_pvalue, digits=3)))")
    end
    
    # Check if KS test fails
    if stats.ks_pvalue < 0.05
        push!(messages, "KS test rejects fit (p=$(round(stats.ks_pvalue, digits=3)))")
    end
    
    # Check mean matching
    mean_error = abs(stats.empirical_mean - stats.fitted_mean) / stats.empirical_mean
    if mean_error > 0.1
        push!(messages, "Mean mismatch: empirical=$(round(stats.empirical_mean, digits=2)), fitted=$(round(stats.fitted_mean, digits=2))")
    end
    
    # Check variance matching
    var_error = abs(stats.empirical_var - stats.fitted_var) / stats.empirical_var
    if var_error > 0.2
        push!(messages, "Variance mismatch: empirical=$(round(stats.empirical_var, digits=2)), fitted=$(round(stats.fitted_var, digits=2))")
        
        if stats.empirical_var < stats.fitted_var
            push!(messages, "  → Try decreasing κ (more overdispersion)")
        else
            push!(messages, "  → Try increasing κ (less overdispersion)")
        end
    end
    
    # Check for zero-inflation
    zero_fraction = sum(stats.empirical_counts .== 0) / length(stats.empirical_counts)
    expected_zeros = pdf(NegativeBinomial(stats.fitted_mean, stats.fitted_var), 0)
    if zero_fraction > expected_zeros * 1.5
        push!(messages, "Possible zero-inflation detected")
    end
    
    return messages
end

"""
    print_fit_summary(stats::FitStatistics)
    
Print a formatted summary of fit statistics.
"""
function print_fit_summary(stats::FitStatistics)
    println("\n=== Negative Binomial Fit Summary ===")
    println("\nMoment Comparison:")
    println("  Empirical: mean=$(round(stats.empirical_mean, digits=2)), var=$(round(stats.empirical_var, digits=2))")
    println("  Fitted:    mean=$(round(stats.fitted_mean, digits=2)), var=$(round(stats.fitted_var, digits=2))")
    
    println("\nGoodness-of-Fit Tests:")
    println("  Chi-squared: χ²=$(round(stats.chi_squared, digits=2)), p=$(round(stats.chi_squared_pvalue, digits=3))")
    println("  KS test:     D=$(round(stats.ks_statistic, digits=3)), p=$(round(stats.ks_pvalue, digits=3))")
    
    println("\nPMF Errors:")
    println("  Mean absolute error: $(round(stats.mean_abs_error, digits=4))")
    println("  Max absolute error:  $(round(stats.max_abs_error, digits=4))")
    
    println("\nInformation Criteria:")
    println("  AIC: $(round(stats.aic, digits=2))")
    println("  BIC: $(round(stats.bic, digits=2))")
    
    # Diagnose issues if fit is poor
    if stats.chi_squared_pvalue < 0.05 || stats.ks_pvalue < 0.05
        println("\n⚠️  Potential Issues:")
        messages = diagnose_poor_fit(stats)
        for msg in messages
            println("  - $msg")
        end
    else
        println("\n✓ Fit appears adequate")
    end
end
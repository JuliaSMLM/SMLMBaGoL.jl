# Formal tests for hierarchical K parameter estimation
# These tests verify mathematical correctness and should pass in CI/CD

using Test
using SMLMBaGoL
using Distributions
using Statistics
using Random
using HypothesisTests

@testset "Hierarchical K Parameter Estimation" begin
    Random.seed!(42)
    
    @testset "Parameter Estimation from Empirical Data" begin
        @testset "Known Distribution Recovery - Method of Moments" begin
            # Test case 1: Standard overdispersed case
            μ_true = 20.0
            κ_true = 5.0
            
            # Convert to r,p parameterization for Distributions.jl
            r = κ_true
            p = κ_true / (κ_true + μ_true)
            nb_dist = NegativeBinomial(r, p)
            
            # Generate various sample sizes to test consistency
            for n_samples in [50, 100, 500, 1000]
                counts = rand(nb_dist, n_samples)
                
                # Use the codebase's MoM function
                μ_est, κ_est = SMLMBaGoL.fit_negbinomial_mom(counts)
                
                # Verify the mathematical formula: κ = μ²/(v-μ)
                empirical_mean = mean(counts)
                empirical_var = var(counts)
                κ_manual = empirical_mean^2 / (empirical_var - empirical_mean)
                
                @test μ_est ≈ empirical_mean
                @test κ_est ≈ κ_manual rtol=0.01
                
                # Check convergence as sample size increases
                if n_samples >= 500
                    @test abs(μ_est - μ_true) / μ_true < 0.1  # Within 10%
                    @test abs(κ_est - κ_true) / κ_true < 0.3  # Within 30% (κ harder to estimate)
                end
            end
        end
        
        @testset "Edge Cases - Method of Moments" begin
            # Test underdispersed data (variance ≤ mean)
            # This should return large κ (approaching Poisson)
            λ = 10.0
            poisson_counts = rand(Poisson(λ), 100)
            μ_est, κ_est = SMLMBaGoL.fit_negbinomial_mom(poisson_counts)
            
            @test μ_est ≈ mean(poisson_counts) rtol=0.01
            @test κ_est == 100.0  # Should hit the upper clamp for Poisson-like data
            
            # Test extreme overdispersion
            μ_extreme = 10.0
            κ_extreme = 0.5  # Very small κ = high overdispersion
            r_extreme = κ_extreme
            p_extreme = κ_extreme / (κ_extreme + μ_extreme)
            nb_extreme = NegativeBinomial(r_extreme, p_extreme)
            
            counts_extreme = rand(nb_extreme, 200)
            μ_est_extreme, κ_est_extreme = SMLMBaGoL.fit_negbinomial_mom(counts_extreme)
            
            # With extreme overdispersion, estimates should at least capture the pattern
            @test μ_est_extreme > 0
            @test κ_est_extreme < 2.0  # Should be small
            
            # Test empty data
            μ_empty, κ_empty = SMLMBaGoL.fit_negbinomial_mom(Int[])
            @test μ_empty == 1.0 && κ_empty == 1.0  # Default values
        end
        
        @testset "Maximum Likelihood Estimation" begin
            # Test MLE on well-behaved distribution
            μ_true = 30.0
            κ_true = 10.0
            r = κ_true
            p = κ_true / (κ_true + μ_true)
            nb_dist = NegativeBinomial(r, p)
            
            # Generate sufficient data for MLE
            counts = rand(nb_dist, 300)
            
            # Test MLE with MoM initialization
            μ_mle, κ_mle = SMLMBaGoL.fit_negbinomial_mle(counts; init_method=:mom)
            
            # MLE should generally outperform MoM
            μ_mom, κ_mom = SMLMBaGoL.fit_negbinomial_mom(counts)
            
            # Compute log-likelihoods to verify MLE is better
            function negbinom_loglik(counts, μ, κ)
                r = κ
                p = κ / (κ + μ)
                nb = NegativeBinomial(r, p)
                return sum(logpdf(nb, c) for c in counts)
            end
            
            ll_mle = negbinom_loglik(counts, μ_mle, κ_mle)
            ll_mom = negbinom_loglik(counts, μ_mom, κ_mom)
            
            @test ll_mle >= ll_mom - 1e-6  # MLE should have higher likelihood (allowing small numerical error)
            
            # Check reasonable parameter recovery
            @test abs(μ_mle - μ_true) / μ_true < 0.15  # Within 15%
            @test abs(κ_mle - κ_true) / κ_true < 0.35  # Within 35%
        end
        
        @testset "Mathematical Consistency - Moments" begin
            # Verify that estimated parameters produce correct theoretical moments
            μ_test = 25.0
            κ_test = 8.0
            
            # Theoretical moments for negative binomial
            theoretical_mean = μ_test
            theoretical_var = μ_test + μ_test^2 / κ_test
            
            # Generate data and estimate parameters
            r = κ_test
            p = κ_test / (κ_test + μ_test)
            nb_dist = NegativeBinomial(r, p)
            counts = rand(nb_dist, 1000)
            
            μ_est, κ_est = SMLMBaGoL.fit_negbinomial_mle(counts)
            
            # Compute theoretical moments from estimated parameters
            estimated_mean = μ_est
            estimated_var = μ_est + μ_est^2 / κ_est
            
            # Empirical moments
            empirical_mean = mean(counts)
            empirical_var = var(counts)
            
            # Check consistency triangle: empirical ≈ estimated ≈ theoretical
            @test abs(estimated_mean - empirical_mean) / empirical_mean < 0.01
            @test abs(estimated_var - empirical_var) / empirical_var < 0.05
            @test abs(theoretical_mean - empirical_mean) / theoretical_mean < 0.1
            @test abs(theoretical_var - empirical_var) / theoretical_var < 0.2
        end
        
        @testset "Parameter Space Exploration" begin
            # Test estimation across different regions of parameter space
            test_cases = [
                (μ=5.0, κ=1.0, desc="Small μ, high overdispersion"),
                (μ=100.0, κ=50.0, desc="Large μ, moderate overdispersion"),
                (μ=20.0, κ=0.5, desc="Moderate μ, extreme overdispersion"),
                (μ=50.0, κ=200.0, desc="Near-Poisson case")
            ]
            
            for (μ_true, κ_true, desc) in test_cases
                @testset "$desc" begin
                    r = κ_true
                    p = κ_true / (κ_true + μ_true)
                    
                    # Skip if parameters would create invalid distribution
                    if r <= 0 || p <= 0 || p >= 1
                        @test_skip "Invalid parameters"
                        continue
                    end
                    
                    nb_dist = NegativeBinomial(r, p)
                    counts = rand(nb_dist, 200)
                    
                    # Test both estimation methods
                    μ_mom, κ_mom = SMLMBaGoL.fit_negbinomial_mom(counts)
                    μ_mle, κ_mle = SMLMBaGoL.fit_negbinomial_mle(counts)
                    
                    # Basic sanity checks
                    @test μ_mom > 0
                    @test κ_mom > 0
                    @test μ_mle > 0
                    @test κ_mle > 0
                    
                    # Check order of magnitude is correct
                    @test log10(μ_mom / μ_true) < 1.0  # Within order of magnitude
                    @test log10(κ_mom / κ_true) < 1.5  # κ allowed more variation
                end
            end
        end
        
        @testset "Empirical Chain State Simulation" begin
            # Simulate what happens in actual MCMC chain analysis
            # Multiple emitters with varying count distributions
            
            n_emitters = 50
            true_μ_global = 35.0
            true_κ_global = 7.0
            
            # Simulate hierarchical structure: each emitter has slightly different μ
            emitter_means = rand(Normal(true_μ_global, sqrt(true_μ_global * 0.1)), n_emitters)
            emitter_means = max.(emitter_means, 1.0)  # Ensure positive
            
            # Generate counts for each emitter
            all_counts = Int[]
            for μ_i in emitter_means
                r_i = true_κ_global
                p_i = true_κ_global / (true_κ_global + μ_i)
                nb_i = NegativeBinomial(r_i, p_i)
                
                # Each emitter contributes different number of observations
                n_obs = rand(5:20)
                append!(all_counts, rand(nb_i, n_obs))
            end
            
            # Estimate global parameters from pooled data
            μ_pooled, κ_pooled = SMLMBaGoL.fit_negbinomial_mle(all_counts)
            
            # Should recover approximate global parameters
            @test abs(μ_pooled - true_μ_global) / true_μ_global < 0.2
            @test abs(κ_pooled - true_κ_global) / true_κ_global < 0.4
            
            # Test with aggregated counts (as might happen in chain state)
            emitter_totals = Int[]
            for μ_i in emitter_means
                r_i = true_κ_global
                p_i = true_κ_global / (true_κ_global + μ_i)
                nb_i = NegativeBinomial(r_i, p_i)
                push!(emitter_totals, rand(nb_i))
            end
            
            μ_agg, κ_agg = SMLMBaGoL.fit_negbinomial_mle(emitter_totals)
            
            # Aggregated estimation should also be reasonable
            @test μ_agg > 0
            @test κ_agg > 0
            @test abs(μ_agg - true_μ_global) / true_μ_global < 0.3
        end
    end
    
    @testset "Goodness-of-Fit Tests" begin
        @testset "KS Test for Parameter Estimation" begin
            # Generate data from known distribution
            μ_true = 40.0
            κ_true = 6.0
            r = κ_true
            p = κ_true / (κ_true + μ_true)
            true_dist = NegativeBinomial(r, p)
            
            # Test with different sample sizes
            for n_samples in [100, 500, 1000]
                counts = rand(true_dist, n_samples)
                
                # Fit using both methods
                μ_mom, κ_mom = SMLMBaGoL.fit_negbinomial_mom(counts)
                μ_mle, κ_mle = SMLMBaGoL.fit_negbinomial_mle(counts)
                
                # Create fitted distributions
                r_mom = κ_mom
                p_mom = κ_mom / (κ_mom + μ_mom)
                fitted_dist_mom = NegativeBinomial(r_mom, p_mom)
                
                r_mle = κ_mle
                p_mle = κ_mle / (κ_mle + μ_mle)
                fitted_dist_mle = NegativeBinomial(r_mle, p_mle)
                
                # Perform KS tests
                # Generate theoretical samples for comparison
                theoretical_mom = rand(fitted_dist_mom, n_samples * 10)
                theoretical_mle = rand(fitted_dist_mle, n_samples * 10)
                
                ks_test_mom = ApproximateTwoSampleKSTest(counts, theoretical_mom)
                ks_test_mle = ApproximateTwoSampleKSTest(counts, theoretical_mle)
                
                # With good fits, we shouldn't reject the null hypothesis
                @test pvalue(ks_test_mom) > 0.01  # Don't reject at 1% level
                @test pvalue(ks_test_mle) > 0.01
                
                # MLE should generally have better p-value (but not always due to randomness)
                # Just check that both methods produce reasonable fits
                @test pvalue(ks_test_mle) > 0.01 || pvalue(ks_test_mom) > 0.01
            end
        end
        
        @testset "Chi-squared Goodness-of-Fit" begin
            # Test with well-specified model
            μ_true = 25.0
            κ_true = 10.0
            r = κ_true
            p = κ_true / (κ_true + μ_true)
            true_dist = NegativeBinomial(r, p)
            
            counts = rand(true_dist, 500)
            
            # Fit parameters
            μ_fit, κ_fit = SMLMBaGoL.fit_negbinomial_mle(counts)
            r_fit = κ_fit
            p_fit = κ_fit / (κ_fit + μ_fit)
            fitted_dist = NegativeBinomial(r_fit, p_fit)
            
            # Create bins for chi-squared test
            max_count = maximum(counts)
            bin_edges = 0:5:max_count+5
            n_bins = length(bin_edges) - 1
            
            # Compute observed frequencies
            observed = zeros(n_bins)
            for c in counts
                bin = findfirst(i -> bin_edges[i] <= c < bin_edges[i+1], 1:n_bins)
                if !isnothing(bin)
                    observed[bin] += 1
                end
            end
            
            # Compute expected frequencies
            expected = zeros(n_bins)
            n_total = length(counts)
            for i in 1:n_bins
                lower = Int(bin_edges[i])
                upper = Int(bin_edges[i+1]) - 1
                prob = sum(pdf(fitted_dist, k) for k in lower:upper)
                expected[i] = n_total * prob
            end
            
            # Perform chi-squared test (only on bins with expected >= 5)
            valid_bins = expected .>= 5
            if sum(valid_bins) >= 3  # Need at least 3 bins
                chi_sq_stat = sum((observed[valid_bins] .- expected[valid_bins]).^2 ./ expected[valid_bins])
                df = sum(valid_bins) - 3  # -1 for constraint, -2 for estimated parameters
                
                if df > 0
                    chi_sq_dist = Chisq(df)
                    p_value = 1 - cdf(chi_sq_dist, chi_sq_stat)
                    
                    # Should not reject for well-specified model
                    @test p_value > 0.05
                end
            end
        end
        
        @testset "Quantile-based Goodness-of-Fit" begin
            # Test using probability integral transform
            μ_true = 30.0
            κ_true = 8.0
            r = κ_true
            p = κ_true / (κ_true + μ_true)
            true_dist = NegativeBinomial(r, p)
            
            counts = rand(true_dist, 300)
            
            # Fit parameters
            μ_fit, κ_fit = SMLMBaGoL.fit_negbinomial_mle(counts)
            r_fit = κ_fit
            p_fit = κ_fit / (κ_fit + μ_fit)
            fitted_dist = NegativeBinomial(r_fit, p_fit)
            
            # Compute empirical CDF values
            u_values = [cdf(fitted_dist, c) for c in counts]
            
            # Under correct model, u_values should be uniform on [0,1]
            # Test using KS test for uniformity
            uniform_test = ApproximateOneSampleKSTest(u_values, Uniform(0, 1))
            
            @test pvalue(uniform_test) > 0.05
            
            # Also check moments of transformed values
            @test abs(mean(u_values) - 0.5) < 0.1
            @test abs(var(u_values) - 1/12) < 0.05  # Var of Uniform(0,1) = 1/12
        end
        
        @testset "Poor Fit Detection with KS Test" begin
            # Generate overdispersed data
            μ_true = 50.0
            κ_true = 2.0  # High overdispersion
            r_true = κ_true
            p_true = κ_true / (κ_true + μ_true)
            true_dist = NegativeBinomial(r_true, p_true)
            
            counts = rand(true_dist, 200)
            
            # Fit with wrong model (force low overdispersion)
            μ_wrong = mean(counts)
            κ_wrong = 50.0  # Much too high (low overdispersion)
            r_wrong = κ_wrong
            p_wrong = κ_wrong / (κ_wrong + μ_wrong)
            wrong_dist = NegativeBinomial(r_wrong, p_wrong)
            
            # Generate theoretical samples from wrong distribution
            theoretical_wrong = rand(wrong_dist, 2000)
            
            # KS test should detect the difference
            ks_test = ApproximateTwoSampleKSTest(counts, theoretical_wrong)
            
            @test pvalue(ks_test) < 0.05  # Should reject bad fit
            
            # Now test with correct fit
            μ_fit, κ_fit = SMLMBaGoL.fit_negbinomial_mle(counts)
            r_fit = κ_fit
            p_fit = κ_fit / (κ_fit + μ_fit)
            fitted_dist = NegativeBinomial(r_fit, p_fit)
            
            theoretical_fit = rand(fitted_dist, 2000)
            ks_test_good = ApproximateTwoSampleKSTest(counts, theoretical_fit)
            
            @test pvalue(ks_test_good) > 0.01  # Should not reject good fit
            @test pvalue(ks_test_good) > pvalue(ks_test)  # Good fit should have higher p-value
        end
        
        @testset "Moment-based Goodness-of-Fit" begin
            # Test that fitted distributions match empirical moments
            test_cases = [
                (μ=20.0, κ=5.0, n=500),
                (μ=50.0, κ=15.0, n=500),
                (μ=10.0, κ=1.0, n=500)  # High overdispersion
            ]
            
            for (μ_true, κ_true, n_samples) in test_cases
                r = κ_true
                p = κ_true / (κ_true + μ_true)
                true_dist = NegativeBinomial(r, p)
                
                counts = rand(true_dist, n_samples)
                
                # Fit parameters
                μ_fit, κ_fit = SMLMBaGoL.fit_negbinomial_mle(counts)
                
                # Compare moments
                empirical_mean = mean(counts)
                empirical_var = var(counts)
                empirical_cv = sqrt(empirical_var) / empirical_mean  # Coefficient of variation
                
                fitted_mean = μ_fit
                fitted_var = μ_fit + μ_fit^2 / κ_fit
                fitted_cv = sqrt(fitted_var) / fitted_mean
                
                # Moments should match closely
                @test abs(fitted_mean - empirical_mean) / empirical_mean < 0.02
                @test abs(fitted_var - empirical_var) / empirical_var < 0.1
                @test abs(fitted_cv - empirical_cv) / empirical_cv < 0.1
                
                # Also check third moment (skewness-related)
                empirical_m3 = mean((counts .- empirical_mean).^3)
                fitted_m3 = μ_fit + 3*μ_fit^2/κ_fit + 2*μ_fit^3/κ_fit^2
                
                if abs(empirical_m3) > 1  # Only test if third moment is substantial
                    @test abs(fitted_m3 - empirical_m3) / abs(empirical_m3) < 0.5  # Third moment harder to estimate
                end
            end
        end
    end
end
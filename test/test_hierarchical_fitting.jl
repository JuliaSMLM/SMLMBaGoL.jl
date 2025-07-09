using Test
using SMLMBaGoL
using Distributions
using Statistics
using Random

@testset "Hierarchical Negative Binomial Fitting" begin
    Random.seed!(42)
    
    @testset "Method of Moments Fitting" begin
        # Test with known distribution
        μ_true = 50.0
        κ_true = 5.0
        r = κ_true
        p = κ_true / (κ_true + μ_true)
        nb_dist = NegativeBinomial(r, p)
        
        # Generate data
        counts = rand(nb_dist, 100)
        
        # Fit using MoM
        μ_fit, κ_fit = SMLMBaGoL.fit_negbinomial_mom(counts)
        
        # Check reasonable fit
        @test abs(μ_fit - μ_true) / μ_true < 0.3  # Within 30%
        @test abs(κ_fit - κ_true) / κ_true < 0.5  # Within 50% (κ is harder)
    end
    
    @testset "Maximum Likelihood Fitting" begin
        # Test with known distribution
        μ_true = 30.0
        κ_true = 10.0
        r = κ_true
        p = κ_true / (κ_true + μ_true)
        nb_dist = NegativeBinomial(r, p)
        
        # Generate data
        counts = rand(nb_dist, 200)
        
        # Fit using MLE
        μ_fit, κ_fit = SMLMBaGoL.fit_negbinomial_mle(counts)
        
        # MLE should be more accurate
        @test abs(μ_fit - μ_true) / μ_true < 0.2  # Within 20%
        @test abs(κ_fit - κ_true) / κ_true < 0.4  # Within 40%
    end
    
    @testset "Initialize from Data" begin
        # Generate overdispersed count data
        μ_true = 40.0
        κ_true = 2.0  # High overdispersion
        r = κ_true
        p = κ_true / (κ_true + μ_true)
        nb_dist = NegativeBinomial(r, p)
        
        counts = rand(nb_dist, 100)
        
        # Initialize prior
        prior = initialize_hierarchical_prior(counts; method=:mle)
        
        @test prior isa HierarchicalNegBinomialPrior
        @test prior.μ > 0
        @test prior.κ > 0
        @test prior.τ² > 0
        
        # Check reasonable initialization
        @test abs(prior.μ - mean(counts)) / mean(counts) < 0.3
    end
    
    @testset "Adaptive Slice Sampling" begin
        # Test that adaptive sampling works
        counts = [10, 15, 20, 25, 30, 35, 40, 45, 50]
        μ = 30.0
        hyperprior = (1.0, 0.1)
        κ_current = 5.0
        
        # Run adaptive slice sampling
        κ_new = update_kappa_slice_adaptive(counts, μ, hyperprior, κ_current; n_steps=10)
        
        @test κ_new > 0
        @test κ_new != κ_current  # Should have moved
    end
    
    @testset "Fit Assessment" begin
        # Generate well-fitted data
        μ = 25.0
        κ = 8.0
        r = κ
        p = κ / (κ + μ)
        nb_dist = NegativeBinomial(r, p)
        counts = rand(nb_dist, 500)
        
        # Create prior with true parameters
        prior = HierarchicalNegBinomialPrior(
            μ, κ, 1e-6,
            (1.0, 0.01),
            (1.0, 0.1),
            (2.0, 2e-6)
        )
        
        # Assess fit
        stats = assess_negbinomial_fit(counts, prior)
        
        @test stats isa FitStatistics
        @test stats.chi_squared_pvalue > 0.05  # Should not reject
        @test abs(stats.empirical_mean - stats.fitted_mean) / stats.fitted_mean < 0.1
        @test abs(stats.empirical_var - stats.fitted_var) / stats.fitted_var < 0.2
    end
    
    @testset "Poor Fit Detection" begin
        # Generate data with different distribution than fitted
        μ_true = 50.0
        κ_true = 2.0  # High overdispersion
        r = κ_true
        p = κ_true / (κ_true + μ_true)
        nb_dist = NegativeBinomial(r, p)
        counts = rand(nb_dist, 200)
        
        # Create prior with wrong parameters
        prior_wrong = HierarchicalNegBinomialPrior(
            50.0, 20.0, 1e-6,  # κ too high (low overdispersion)
            (1.0, 0.01),
            (1.0, 0.1),
            (2.0, 2e-6)
        )
        
        # Assess fit
        stats = assess_negbinomial_fit(counts, prior_wrong)
        
        # Should detect poor fit
        @test stats.chi_squared_pvalue < 0.5 || abs(stats.empirical_var - stats.fitted_var) / stats.fitted_var > 0.5
    end
end
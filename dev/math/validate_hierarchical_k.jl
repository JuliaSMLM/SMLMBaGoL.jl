#=
# Hierarchical K Parameter Mathematical Validation
# 
# This script validates the mathematical correctness of estimating μ and κ parameters
# from empirical count data, as done in the hierarchical Bayesian analysis.
#
# Outputs:
# - Parameter recovery accuracy plots
# - Goodness-of-fit visualizations
# - KS test p-value distributions
# - Method comparison (MoM vs MLE)
=#

using SMLMBaGoL
using Distributions
using Statistics
using Random
using HypothesisTests
using CairoMakie
using Printf

# Set random seed for reproducibility
Random.seed!(42)

# Create output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

# ===== Mathematical Validation Functions =====

function plot_parameter_recovery(; n_simulations=100)
    """Test parameter recovery across different true values"""
    
    println("\n=== Parameter Recovery Analysis ===\n")
    
    # Test cases
    test_cases = [
        (μ=10.0, κ=2.0, label="High overdispersion"),
        (μ=30.0, κ=10.0, label="Moderate overdispersion"),
        (μ=50.0, κ=50.0, label="Low overdispersion")
    ]
    
    fig = Figure(size=(1200, 800))
    
    for (idx, (μ_true, κ_true, label)) in enumerate(test_cases)
        println("Testing $label: μ=$μ_true, κ=$κ_true")
        
        μ_mom_estimates = Float64[]
        κ_mom_estimates = Float64[]
        μ_mle_estimates = Float64[]
        κ_mle_estimates = Float64[]
        
        for _ in 1:n_simulations
            # Generate data
            r = κ_true
            p = κ_true / (κ_true + μ_true)
            nb_dist = NegativeBinomial(r, p)
            counts = rand(nb_dist, 200)
            
            # Estimate parameters
            μ_mom, κ_mom = SMLMBaGoL.fit_negbinomial_mom(counts)
            μ_mle, κ_mle = SMLMBaGoL.fit_negbinomial_mle(counts)
            
            push!(μ_mom_estimates, μ_mom)
            push!(κ_mom_estimates, κ_mom)
            push!(μ_mle_estimates, μ_mle)
            push!(κ_mle_estimates, κ_mle)
        end
        
        # Plot μ estimates
        ax_mu = Axis(fig[idx, 1], 
                     title="$label: μ=$μ_true",
                     ylabel="μ estimate",
                     xticks=([1, 2], ["MoM", "MLE"]))
        
        violin!(ax_mu, [1], μ_mom_estimates, color=(:blue, 0.6), width=0.8)
        violin!(ax_mu, [2], μ_mle_estimates, color=(:red, 0.6), width=0.8)
        hlines!(ax_mu, [μ_true], color=:black, linestyle=:dash, linewidth=2)
        
        # Plot κ estimates
        ax_kappa = Axis(fig[idx, 2], 
                        title="$label: κ=$κ_true",
                        ylabel="κ estimate",
                        xticks=([1, 2], ["MoM", "MLE"]))
        
        violin!(ax_kappa, [1], κ_mom_estimates, color=(:blue, 0.6), width=0.8)
        violin!(ax_kappa, [2], κ_mle_estimates, color=(:red, 0.6), width=0.8)
        hlines!(ax_kappa, [κ_true], color=:black, linestyle=:dash, linewidth=2)
        
        # Print statistics
        println("  μ MoM: mean=$(round(mean(μ_mom_estimates), digits=2)), " *
                "std=$(round(std(μ_mom_estimates), digits=2))")
        println("  μ MLE: mean=$(round(mean(μ_mle_estimates), digits=2)), " *
                "std=$(round(std(μ_mle_estimates), digits=2))")
        println("  κ MoM: mean=$(round(mean(κ_mom_estimates), digits=2)), " *
                "std=$(round(std(κ_mom_estimates), digits=2))")
        println("  κ MLE: mean=$(round(mean(κ_mle_estimates), digits=2)), " *
                "std=$(round(std(κ_mle_estimates), digits=2))")
        println()
    end
    
    # Add legend
    Legend(fig[0, :], [LineElement(color=:black, linestyle=:dash)], ["True value"], 
           tellwidth=false, tellheight=true, orientation=:horizontal)
    
    return fig
end

function analyze_goodness_of_fit()
    """Analyze goodness-of-fit using multiple methods"""
    
    println("\n=== Goodness-of-Fit Analysis ===\n")
    
    # Generate data with known parameters
    μ_true = 35.0
    κ_true = 7.0
    r = κ_true
    p = κ_true / (κ_true + μ_true)
    true_dist = NegativeBinomial(r, p)
    
    counts = rand(true_dist, 500)
    
    # Fit parameters
    μ_mom, κ_mom = SMLMBaGoL.fit_negbinomial_mom(counts)
    μ_mle, κ_mle = SMLMBaGoL.fit_negbinomial_mle(counts)
    
    println("True parameters: μ=$μ_true, κ=$κ_true")
    println("MoM estimates: μ=$(round(μ_mom, digits=2)), κ=$(round(κ_mom, digits=2))")
    println("MLE estimates: μ=$(round(μ_mle, digits=2)), κ=$(round(κ_mle, digits=2))")
    
    # Create fitted distributions
    r_mom = κ_mom
    p_mom = κ_mom / (κ_mom + μ_mom)
    fitted_mom = NegativeBinomial(r_mom, p_mom)
    
    r_mle = κ_mle
    p_mle = κ_mle / (κ_mle + μ_mle)
    fitted_mle = NegativeBinomial(r_mle, p_mle)
    
    # Create visualization
    fig = Figure(size=(1200, 1000))
    
    # 1. Histogram comparison
    ax1 = Axis(fig[1, 1], title="Distribution Comparison", 
               xlabel="Count", ylabel="Probability")
    
    max_count = maximum(counts)
    hist!(ax1, counts, normalization=:pdf, bins=0:2:max_count+2, 
          color=(:gray, 0.6), label="Empirical")
    
    x_range = 0:max_count
    lines!(ax1, x_range, [pdf(true_dist, x) for x in x_range], 
           linewidth=3, label="True", color=:green)
    lines!(ax1, x_range, [pdf(fitted_mom, x) for x in x_range], 
           linewidth=2, label="MoM fit", color=:blue, linestyle=:dash)
    lines!(ax1, x_range, [pdf(fitted_mle, x) for x in x_range], 
           linewidth=2, label="MLE fit", color=:red, linestyle=:dot)
    axislegend(ax1, position=:rt)
    
    # 2. Q-Q plot
    ax2 = Axis(fig[1, 2], title="Q-Q Plot",
               xlabel="Theoretical quantiles", ylabel="Empirical quantiles")
    
    theoretical_quantiles = quantile.(true_dist, (1:length(counts))/(length(counts)+1))
    empirical_quantiles = sort(counts)
    
    scatter!(ax2, theoretical_quantiles, empirical_quantiles, 
             color=(:blue, 0.5), markersize=4, label="Empirical vs True")
    lines!(ax2, [0, maximum(theoretical_quantiles)], [0, maximum(theoretical_quantiles)],
           color=:red, linestyle=:dash, label="y=x")
    axislegend(ax2, position=:lt)
    
    # 3. CDF comparison
    ax3 = Axis(fig[2, 1], title="CDF Comparison",
               xlabel="Count", ylabel="CDF")
    
    count_range = 0:maximum(counts)
    lines!(ax3, count_range, [cdf(true_dist, x) for x in count_range], 
           linewidth=3, label="True CDF", color=:green)
    lines!(ax3, count_range, [cdf(fitted_mom, x) for x in count_range], 
           linewidth=2, label="MoM CDF", color=:blue, linestyle=:dash)
    lines!(ax3, count_range, [cdf(fitted_mle, x) for x in count_range], 
           linewidth=2, label="MLE CDF", color=:red, linestyle=:dot)
    
    # Add empirical CDF as step function
    sorted_counts = sort(counts)
    ecdf_vals = (1:length(counts)) / length(counts)
    
    # Plot empirical CDF as steps
    for i in 1:length(sorted_counts)
        if i == 1
            lines!(ax3, [0, sorted_counts[i]], [0, ecdf_vals[i]], 
                   color=:black, linewidth=1, label="Empirical CDF")
        else
            lines!(ax3, [sorted_counts[i-1], sorted_counts[i]], 
                   [ecdf_vals[i-1], ecdf_vals[i]], 
                   color=:black, linewidth=1)
        end
    end
    axislegend(ax3, position=:rb)
    
    # 4. Residual analysis
    ax4 = Axis(fig[2, 2], title="Residual Analysis (MLE)",
               xlabel="Bin center", ylabel="Standardized residual")
    
    # Compute standardized residuals for binned data
    bin_edges = 0:5:max_count+5
    n_bins = length(bin_edges) - 1
    
    observed = zeros(n_bins)
    for c in counts
        bin = findfirst(i -> bin_edges[i] <= c < bin_edges[i+1], 1:n_bins)
        if !isnothing(bin)
            observed[bin] += 1
        end
    end
    
    # Expected frequencies
    expected_mle = zeros(n_bins)
    for i in 1:n_bins
        lower = Int(bin_edges[i])
        upper = Int(bin_edges[i+1]) - 1
        prob = sum(pdf(fitted_mle, k) for k in lower:upper)
        expected_mle[i] = length(counts) * prob
    end
    
    # Standardized residuals
    valid_bins = expected_mle .> 5
    if sum(valid_bins) > 0
        residuals = (observed[valid_bins] .- expected_mle[valid_bins]) ./ 
                    sqrt.(expected_mle[valid_bins])
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in 1:n_bins][valid_bins]
        
        scatter!(ax4, bin_centers, residuals, markersize=6)
        hlines!(ax4, [0], color=:red, linestyle=:dash)
        hlines!(ax4, [-2, 2], color=:gray, linestyle=:dot, alpha=0.5)
    end
    
    # Perform formal tests
    println("\nFormal goodness-of-fit tests:")
    
    # KS tests
    theoretical_mom_sample = rand(fitted_mom, 5000)
    theoretical_mle_sample = rand(fitted_mle, 5000)
    
    ks_mom = ApproximateTwoSampleKSTest(counts, theoretical_mom_sample)
    ks_mle = ApproximateTwoSampleKSTest(counts, theoretical_mle_sample)
    
    println("  KS test p-value (MoM): $(round(pvalue(ks_mom), digits=4))")
    println("  KS test p-value (MLE): $(round(pvalue(ks_mle), digits=4))")
    
    return fig
end

function explore_ks_test_power()
    """Explore the power of KS test to detect misspecification"""
    
    println("\n=== KS Test Power Analysis ===\n")
    
    # True parameters
    μ_true = 30.0
    κ_true = 5.0
    
    # Test different levels of misspecification
    κ_multipliers = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0]
    n_simulations = 100
    
    p_values = Dict()
    
    for mult in κ_multipliers
        κ_test = κ_true * mult
        p_vals = Float64[]
        
        for _ in 1:n_simulations
            # Generate data from true distribution
            r_true = κ_true
            p_true = κ_true / (κ_true + μ_true)
            true_dist = NegativeBinomial(r_true, p_true)
            counts = rand(true_dist, 200)
            
            # Test against misspecified distribution
            r_test = κ_test
            p_test = κ_test / (κ_test + μ_true)
            test_dist = NegativeBinomial(r_test, p_test)
            theoretical = rand(test_dist, 2000)
            
            ks_test = ApproximateTwoSampleKSTest(counts, theoretical)
            push!(p_vals, pvalue(ks_test))
        end
        
        p_values[mult] = p_vals
    end
    
    # Plot results
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], 
              xlabel="κ multiplier", 
              ylabel="Rejection rate",
              title="KS Test Power: Detection of κ Misspecification")
    
    multipliers = sort(collect(keys(p_values)))
    rejection_rates = [mean(p_values[m] .< 0.05) for m in multipliers]
    
    lines!(ax, multipliers, rejection_rates, linewidth=2, color=:blue)
    scatter!(ax, multipliers, rejection_rates, markersize=8, color=:blue)
    hlines!(ax, [0.05], color=:red, linestyle=:dash, label="α=0.05")
    vlines!(ax, [1.0], color=:green, linestyle=:dash, label="True κ")
    
    ylims!(ax, 0, 1)
    axislegend(ax, position=:rt)
    
    println("KS test rejection rates for different κ misspecifications:")
    for (mult, rate) in zip(multipliers, rejection_rates)
        println("  κ = $(round(κ_true * mult, digits=1)) " *
                "($(mult)x true): $(round(rate*100, digits=1))%")
    end
    
    return fig
end

function run_formal_tests()
    """Run the formal test suite and report results"""
    
    println("\n=== Running Formal Test Suite ===\n")
    
    try
        # Change to the test directory and run tests
        test_result = run(pipeline(`julia --project=../.. -e "using Pkg; Pkg.test()"`, 
                                  stdout=devnull, stderr=devnull))
        
        if test_result.exitcode == 0
            println("✅ All formal tests PASSED")
            return true
        else
            println("❌ Some formal tests FAILED (exit code: $(test_result.exitcode))")
            return false
        end
    catch e
        println("❌ Error running formal tests: $e")
        return false
    end
end

# ===== Run Analyses =====

println("Running Hierarchical K Mathematical Validation")
println("=" ^ 50)

# 1. Parameter recovery analysis
println("Generating parameter recovery plots...")
fig1 = plot_parameter_recovery(n_simulations=100)
save(joinpath(output_dir, "parameter_recovery_analysis.png"), fig1)
println("✅ Saved: parameter_recovery_analysis.png")

# 2. Goodness-of-fit analysis
println("\nGenerating goodness-of-fit analysis...")
fig2 = analyze_goodness_of_fit()
save(joinpath(output_dir, "goodness_of_fit_analysis.png"), fig2)
println("✅ Saved: goodness_of_fit_analysis.png")

# 3. KS test power analysis
println("\nGenerating KS test power analysis...")
fig3 = explore_ks_test_power()
save(joinpath(output_dir, "ks_test_power_analysis.png"), fig3)
println("✅ Saved: ks_test_power_analysis.png")

# 4. Run formal test suite
println("\nRunning formal test suite...")
formal_tests_passed = run_formal_tests()

println("\n" * "="^50)
println("📊 VALIDATION SUMMARY")
println("="^50)
println("Generated plots:")
println("  • Parameter recovery analysis")
println("  • Goodness-of-fit analysis")  
println("  • KS test power analysis")
println("\nFormal tests: $(formal_tests_passed ? "✅ PASSED" : "❌ FAILED")")
println("\n📁 All outputs saved to: $(output_dir)")
println("\n🔍 Check generated plots for visual validation of mathematical concepts.")
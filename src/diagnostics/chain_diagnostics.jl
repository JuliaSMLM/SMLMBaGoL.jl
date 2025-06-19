# Main interface for SMLMBaGoL chain diagnostics

include("burn_in_assessment.jl")
include("chain_length_assessment.jl")

"""
    diagnose_chains(chains::Union{RJMCMCChain, Vector{RJMCMCChain}}; 
                   pixel_size::Real = 0.005,
                   correlation_threshold::Float64 = 0.95,
                   verbose::Bool = true)

Comprehensive diagnostic analysis for SMLMBaGoL RJMCMC chains.

Performs two-stage analysis:
1. **Burn-in Assessment**: Determines adequate burn-in via K distribution stability
2. **Chain Length Assessment**: Evaluates mixing via posterior histogram correlation

# Arguments
- `chains`: Single chain or vector of chains (from spatial partitions)
- `pixel_size`: Pixel size for posterior histogram generation (μm)
- `correlation_threshold`: Minimum correlation for chain length convergence
- `verbose`: Print detailed diagnostic report

# Returns
Named tuple with complete diagnostic results and recommendations.

# Example
```julia
# After running BaGoL analysis
result = run_bagol(localizations, n_iterations=10000, burn_in=2000)

# Diagnose chains
diagnosis = diagnose_chains(result)

# Check recommendations
println(diagnosis.summary)
```
"""
function diagnose_chains(chains::Union{RJMCMCChain, Vector{<:RJMCMCChain}}; 
                        pixel_size::Real = 0.005,
                        correlation_threshold::Float64 = 0.95,
                        verbose::Bool = true)
    
    # Normalize input to vector
    chain_vector = isa(chains, Vector) ? chains : [chains]
    n_chains = length(chain_vector)
    
    # Stage 1: Burn-in Assessment
    verbose && println("=== SMLMBaGoL Chain Diagnostics ===")
    verbose && println("Analyzing $(n_chains) chain$(n_chains > 1 ? "s" : "") from $(n_chains > 1 ? "spatial partitions" : "single partition")...")
    verbose && println()
    
    verbose && println("Stage 1: Burn-in Assessment (K distribution stability)")
    
    # Call appropriate version based on number of chains
    if n_chains == 1
        burn_in_result = assess_burn_in(chain_vector[1])  # Single chain version
    else
        burn_in_result = assess_burn_in(chain_vector)     # Multi-chain version
    end
    recommended_burn_in = burn_in_result.burn_in_length
    
    if verbose
        println("  Method: Sliding window KS test on emitter count distributions")
        println("  Result: $(burn_in_result.status)")
        println("  Recommended burn-in: $(recommended_burn_in) iterations")
        
        if n_chains > 1 && haskey(burn_in_result, :per_chain_results)
            println("  Per-partition burn-in needs:")
            for (i, result) in enumerate(burn_in_result.per_chain_results)
                println("    Partition $i: $(result.burn_in_length) iterations")
            end
        end
        println()
    end
    
    # Stage 2: Chain Length Assessment
    verbose && println("Stage 2: Chain Length Assessment (posterior correlation)")
    
    # Call appropriate version based on number of chains
    if n_chains == 1
        chain_length_result = assess_chain_length(chain_vector[1], recommended_burn_in; 
                                                 pixel_size=pixel_size, 
                                                 correlation_threshold=correlation_threshold)
    else
        chain_length_result = assess_chain_length(chain_vector, recommended_burn_in; 
                                                 pixel_size=pixel_size, 
                                                 correlation_threshold=correlation_threshold)
    end
    
    if verbose
        println("  Method: Quartile posterior histogram correlation analysis")
        
        if n_chains == 1
            result = chain_length_result
            println("  Quartile correlations:")
            corr_matrix = result.quartile_correlations
            
            # Print correlation matrix in readable format
            quartile_pairs = [
                ("Q1-Q2", corr_matrix[1,2]), ("Q1-Q3", corr_matrix[1,3]), ("Q1-Q4", corr_matrix[1,4]),
                ("Q2-Q3", corr_matrix[2,3]), ("Q2-Q4", corr_matrix[2,4]), ("Q3-Q4", corr_matrix[3,4])
            ]
            
            for (i, (pair, corr)) in enumerate(quartile_pairs)
                print("    $pair: $(round(corr, digits=3))")
                if i % 3 == 0
                    println()
                else
                    print("  ")
                end
            end
            if length(quartile_pairs) % 3 != 0
                println()
            end
            
            println("  Status: $(result.recommendation)")
        else
            println("  Multi-partition analysis:")
            println("  $(chain_length_result.overall_recommendation)")
            
            # Show per-partition summary
            println("  Per-partition correlations (min/mean/max):")
            for (i, result) in enumerate(chain_length_result.per_chain_results)
                corr_min = round(result.min_correlation, digits=3)
                corr_mean = round(result.mean_correlation, digits=3)
                corr_max = round(result.max_correlation, digits=3)
                status_symbol = result.converged ? "✓" : "⚠"
                println("    Partition $i: $corr_min/$corr_mean/$corr_max $status_symbol")
            end
        end
        println()
    end
    
    # Overall Summary and Recommendations
    overall_converged = if n_chains == 1
        chain_length_result.converged
    else
        chain_length_result.all_converged
    end
    
    # Generate summary report
    summary_lines = String[]
    push!(summary_lines, "=== DIAGNOSTIC SUMMARY ===")
    push!(summary_lines, "Burn-in recommendation: $(recommended_burn_in) iterations")
    
    if overall_converged
        push!(summary_lines, "Chain mixing: ✓ CONVERGED - chains are well-mixed")
        push!(summary_lines, "Action: Analysis results are reliable for interpretation")
    else
        if n_chains == 1
            push!(summary_lines, "Chain mixing: ⚠ NEEDS MORE SAMPLES")
            push!(summary_lines, "$(chain_length_result.recommendation)")
        else
            push!(summary_lines, "Chain mixing: ⚠ SOME PARTITIONS NEED MORE SAMPLES")
            push!(summary_lines, "$(chain_length_result.overall_recommendation)")
        end
        push!(summary_lines, "Action: Increase n_iterations and re-run analysis")
    end
    
    summary = join(summary_lines, "\n")
    
    if verbose
        println(summary)
        println()
    end
    
    # Return comprehensive results
    return (
        # Stage 1 results
        burn_in_assessment = burn_in_result,
        recommended_burn_in = recommended_burn_in,
        
        # Stage 2 results  
        chain_length_assessment = chain_length_result,
        converged = overall_converged,
        
        # Summary
        summary = summary,
        n_chains = n_chains,
        
        # Analysis parameters used
        analysis_params = (
            pixel_size = pixel_size,
            correlation_threshold = correlation_threshold
        )
    )
end

"""
    quick_diagnose(chains::Union{RJMCMCChain, Vector{RJMCMCChain}})

Quick chain diagnostic with minimal output - just the essential verdict.
"""
function quick_diagnose(chains::Union{RJMCMCChain, Vector{RJMCMCChain}})
    result = diagnose_chains(chains, verbose=false)
    
    println("Quick Diagnostic:")
    println("  Recommended burn-in: $(result.recommended_burn_in) iterations")
    
    if result.converged
        println("  Chain status: ✓ CONVERGED")
        println("  → Results are reliable for analysis")
    else
        println("  Chain status: ⚠ NEEDS MORE SAMPLES")
        println("  → Increase iterations and re-run")
    end
    
    return result
end
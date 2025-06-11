using Statistics
using HypothesisTests

"""
    assess_burn_in(chain::RJMCMCChain; window_size::Int = 1000, alpha::Float64 = 0.05)

Assess burn-in period by comparing K (emitter count) distributions in sliding windows.

# Method
- Use sliding windows of `window_size` iterations
- Compare consecutive windows with Kolmogorov-Smirnov test
- Burn-in complete when p-value > `alpha` (distributions statistically similar)

# Returns
- `burn_in_length`: Recommended burn-in length (iterations)
- `ks_p_values`: P-values from KS tests for each window comparison
- `stable_from`: First iteration where distribution becomes stable
"""
function assess_burn_in(chain::RJMCMCChain; window_size::Int = 1000, alpha::Float64 = 0.05)
    n_samples = length(chain.samples)
    
    # Need at least 2 windows to compare
    if n_samples < 2 * window_size
        return (
            burn_in_length = min(div(n_samples, 4), 1000),  # Conservative fallback
            ks_p_values = Float64[],
            stable_from = nothing,
            status = "Insufficient samples for burn-in assessment (need ≥$(2*window_size))"
        )
    end
    
    # Extract emitter counts for all samples
    emitter_counts = [length(sample.emitters) for sample in chain.samples]
    
    # Sliding window comparison
    ks_p_values = Float64[]
    stable_from = nothing
    
    max_start = n_samples - 2 * window_size + 1
    
    for start_idx in 1:window_size:max_start
        # Define two consecutive windows
        window1_end = start_idx + window_size - 1
        window2_start = window1_end + 1
        window2_end = min(window2_start + window_size - 1, n_samples)
        
        # Extract K values for each window
        k1 = emitter_counts[start_idx:window1_end]
        k2 = emitter_counts[window2_start:window2_end]
        
        # Kolmogorov-Smirnov test for distribution similarity
        ks_test = ApproximateTwoSampleKSTest(k1, k2)
        p_value = pvalue(ks_test)
        push!(ks_p_values, p_value)
        
        # Check if this is the first stable comparison
        if p_value > alpha && stable_from === nothing
            stable_from = window2_start
        end
    end
    
    # Determine burn-in recommendation
    if stable_from !== nothing
        burn_in_length = stable_from
        status = "K distribution stabilized"
    else
        # If never stable, recommend conservative burn-in
        burn_in_length = min(div(n_samples, 3), 5000)
        status = "K distribution never stabilized - using conservative estimate"
    end
    
    return (
        burn_in_length = burn_in_length,
        ks_p_values = ks_p_values,
        stable_from = stable_from,
        status = status
    )
end

"""
    assess_burn_in(chains::Vector{RJMCMCChain}; kwargs...)

Assess burn-in for multiple chains (from spatial partitions).
Returns worst-case (longest) burn-in recommendation across all chains.
"""
function assess_burn_in(chains::Vector{RJMCMCChain}; kwargs...)
    if isempty(chains)
        error("No chains provided for burn-in assessment")
    end
    
    # Analyze each chain individually
    results = [assess_burn_in(chain; kwargs...) for chain in chains]
    
    # Find worst case (longest burn-in needed)
    worst_idx = argmax([r.burn_in_length for r in results])
    worst_result = results[worst_idx]
    
    return (
        burn_in_length = worst_result.burn_in_length,
        per_chain_results = results,
        worst_chain_index = worst_idx,
        status = "Multi-chain analysis: worst case from partition $worst_idx"
    )
end
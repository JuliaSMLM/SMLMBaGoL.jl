#!/usr/bin/env julia

"""
Debug the allocation counting issue - are we including emitters with 0 or very few allocations?
"""

using Pkg; Pkg.activate("examples")
using SMLMBaGoL
using Statistics
using Random

Random.seed!(123)

println("=== Debugging Allocation Counting Issue ===\n")

# Generate simple test case
println("1. Generate synthetic data...")
localizations, spatial_prior, count_prior = simulate_n_mer(
    n=6,
    diameter=0.100,
    photons=1000,
    localizations_per_emitter_mean=20,
    localizations_per_emitter_variance=5,
    tau=0.001
)

println("   ✓ Created $(length(localizations)) localizations for 6 emitters")

# Run very short analysis to get allocation patterns
println("\n2. Running minimal BaGoL analysis...")
chains_result = run_bagol(localizations; 
    n_iterations=1000,
    burn_in=100,
    enable_hierarchical=false,  # Disable hierarchical to see raw patterns
    partition_data=false
)

chains = isa(chains_result, Vector) ? chains_result : [chains_result]
chain = chains[1]

println("   ✓ Completed $(length(chain.samples)) samples")

# Analyze allocation patterns in detail
println("\n3. Detailed allocation pattern analysis...")

# Look at several samples
for i in [1, length(chain.samples)÷2, length(chain.samples)]
    sample = chain.samples[i]
    emitter_counts = SMLMBaGoL.count_allocations(sample)
    
    println("   Sample $i:")
    println("     Total emitters: $(length(sample.emitters))")
    println("     Emitter counts: $emitter_counts")
    println("     Emitters with 0 allocations: $(count(x -> x == 0, emitter_counts))")
    println("     Emitters with 1-2 allocations: $(count(x -> 1 ≤ x ≤ 2, emitter_counts))")
    println("     Emitters with >5 allocations: $(count(x -> x > 5, emitter_counts))")
    
    # Check if allocation counts sum to total localizations
    total_allocated = sum(emitter_counts)
    println("     Total allocations: $total_allocated (should be $(length(localizations)))")
    
    if total_allocated != length(localizations)
        println("     ⚠ WARNING: Allocation mismatch!")
    end
end

# Test what happens if we filter out low-count emitters
println("\n4. Testing filtered allocation counting...")

function collect_emitter_counts_filtered(chains::Vector{<:RJMCMCChain}; min_count::Int = 3)
    """Only count emitters with at least min_count allocations"""
    all_counts = Int[]
    for chain in chains
        counts = SMLMBaGoL.count_allocations(chain.current_state)
        # Filter out emitters with too few allocations
        filtered_counts = filter(c -> c >= min_count, counts)
        append!(all_counts, filtered_counts)
    end
    return all_counts
end

# Compare different filtering approaches
for min_count in [1, 2, 3, 5]
    filtered_counts = collect_emitter_counts_filtered(chains; min_count=min_count)
    
    if !isempty(filtered_counts)
        emp_mean = mean(filtered_counts)
        emp_var = var(filtered_counts)
        ratio = emp_var / emp_mean
        
        # Estimate κ assuming this is the correct data
        expected_κ = emp_var > emp_mean ? emp_mean^2 / (emp_var - emp_mean) : Inf
        
        println("   Min count ≥ $min_count: $(length(filtered_counts)) emitters")
        println("     Mean: $(round(emp_mean, digits=1)), Var: $(round(emp_var, digits=1)), Ratio: $(round(ratio, digits=2))")
        println("     Expected κ: $(isfinite(expected_κ) ? round(expected_κ, digits=1) : "∞ (under-dispersed)")")
    else
        println("   Min count ≥ $min_count: No emitters remain")
    end
end

# Test what happens with a more realistic SMLM scenario
println("\n5. Testing with higher localization density...")

localizations2, spatial_prior2, count_prior2 = simulate_n_mer(
    n=6,
    diameter=0.100,
    photons=1000,
    localizations_per_emitter_mean=50,  # Higher density
    localizations_per_emitter_variance=10,  # Lower relative variance
    tau=0.001
)

chains_result2 = run_bagol(localizations2; 
    n_iterations=1000,
    burn_in=100,
    enable_hierarchical=false,
    partition_data=false
)

chains2 = isa(chains_result2, Vector) ? chains_result2 : [chains_result2]
sample2 = chains2[1].samples[end]
emitter_counts2 = SMLMBaGoL.count_allocations(sample2)

println("   Higher density scenario:")
println("     Total localizations: $(length(localizations2))")
println("     Total emitters: $(length(sample2.emitters))")
println("     Emitter counts: $emitter_counts2")
println("     Zero allocation emitters: $(count(x -> x == 0, emitter_counts2))")

# Filter and analyze
filtered_counts2 = filter(c -> c >= 3, emitter_counts2)
if !isempty(filtered_counts2)
    emp_mean2 = mean(filtered_counts2)
    emp_var2 = var(filtered_counts2)
    ratio2 = emp_var2 / emp_mean2
    expected_κ2 = emp_var2 > emp_mean2 ? emp_mean2^2 / (emp_var2 - emp_mean2) : Inf
    
    println("     Filtered (≥3): Mean=$(round(emp_mean2, digits=1)), Var=$(round(emp_var2, digits=1)), κ=$(isfinite(expected_κ2) ? round(expected_κ2, digits=1) : "∞")")
end

println("\n=== Key Insights ===")
println("1. Low-allocation emitters (0-2 locs) may artificially inflate variance")
println("2. RJMCMC naturally creates many emitters with few allocations") 
println("3. Need to be careful about what counts to include in hierarchical fitting")
println("4. May need minimum threshold or different data collection strategy")
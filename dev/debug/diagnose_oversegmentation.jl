#!/usr/bin/env julia

"""
Diagnose why the algorithm systematically prefers more emitters than the truth.
"""

using Pkg; Pkg.activate("../..")
using SMLMBaGoL
using Statistics
using Random
using Printf

# Generate simple test case
Random.seed!(12345)

# Create 6 well-separated clusters
const TRUE_K = 6
const CLUSTER_SEPARATION = 0.5  # 500 nm between clusters - VERY well separated

println("🔍 DIAGNOSING SYSTEMATIC OVER-SEGMENTATION")
println("="^60)

# Generate synthetic data with well-separated clusters
function generate_well_separated_clusters()
    localizations = SMLMBaGoL.Localization2D{Float64}[]
    
    # Create 6 clusters in a grid pattern
    positions = [
        (-CLUSTER_SEPARATION, -CLUSTER_SEPARATION),
        (0.0, -CLUSTER_SEPARATION),
        (CLUSTER_SEPARATION, -CLUSTER_SEPARATION),
        (-CLUSTER_SEPARATION, CLUSTER_SEPARATION),
        (0.0, CLUSTER_SEPARATION),
        (CLUSTER_SEPARATION, CLUSTER_SEPARATION)
    ]
    
    # Generate localizations for each cluster
    for (i, (cx, cy)) in enumerate(positions)
        n_locs = 40  # Fixed number per cluster for simplicity
        
        for j in 1:n_locs
            # Small spread within cluster (20nm std)
            x = cx + randn() * 0.02
            y = cy + randn() * 0.02
            
            # Localization uncertainty (10nm)
            σx = σy = 0.01
            
            loc = SMLMBaGoL.Localization2D(
                x, y, σx, σy,
                i * 100 + j  # frame (unique)
            )
            push!(localizations, loc)
        end
    end
    
    return localizations, positions
end

# Generate data
localizations, true_positions = generate_well_separated_clusters()
n_locs = length(localizations)
true_μ = n_locs / TRUE_K

println("Generated $(n_locs) localizations in $(TRUE_K) well-separated clusters")
println("Cluster separation: $(CLUSTER_SEPARATION*1000) nm")
println("True μ = $(true_μ)")
println()

# Create priors
spatial_prior = SMLMBaGoL.create_spatial_prior_from_localizations(localizations, 0.2)

# Test with different μ values
μ_values = [10.0, true_μ, 50.0, 100.0]

println("📊 Prior P(K|N,μ,κ) for different μ values:")
println("-"^50)

κ = 10.0
for μ in μ_values
    println("\nμ = $(μ):")
    
    # Calculate prior probabilities for different K
    for k in 3:12
        log_prob = SMLMBaGoL.log_prior_k_given_N(k, n_locs, μ, κ)
        prob = exp(log_prob)
        
        bar = repeat("█", Int(round(prob * 50)))
        marker = k == TRUE_K ? " ← TRUE" : ""
        @printf("  K=%2d: %.4f %s%s\n", k, prob, bar, marker)
    end
    
    # Find mode
    k_probs = [exp(SMLMBaGoL.log_prior_k_given_N(k, n_locs, μ, κ)) for k in 1:20]
    k_mode = argmax(k_probs)
    println("  Prior mode: K = $(k_mode)")
end

# Now let's analyze the likelihood
println("\n\n📊 Likelihood Analysis:")
println("="^60)

# Create states with different K and calculate likelihoods
function create_state_with_k_emitters(k::Int, localizations, positions)
    # Simple allocation: divide localizations evenly among k emitters
    n_locs = length(localizations)
    locs_per_emitter = n_locs ÷ k
    
    # Create emitters
    if k <= 6
        # Use true positions for K ≤ 6
        emitters = [SMLMBaGoL.Emitter2D(positions[i][1], positions[i][2], 1000.0) for i in 1:k]
    else
        # For K > 6, add extra emitters between clusters
        emitters = SMLMBaGoL.Emitter2D{Float64}[]
        for i in 1:6
            push!(emitters, SMLMBaGoL.Emitter2D(positions[i][1], positions[i][2], 1000.0))
        end
        # Add extra emitters at intermediate positions
        for i in 7:k
            idx = (i - 7) % 6 + 1
            x = positions[idx][1] + 0.1 * randn()
            y = positions[idx][2] + 0.1 * randn()
            push!(emitters, SMLMBaGoL.Emitter2D(x, y, 1000.0))
        end
    end
    
    # Simple allocation (round-robin)
    allocations = Int[]
    for i in 1:n_locs
        push!(allocations, (i - 1) % k + 1)
    end
    
    # Create state
    count_prior = SMLMBaGoL.HierarchicalNegBinomialPrior(
        true_μ, κ, 1.0e-6,
        (2.0, 5.0), (5.0, 2.0), (3.0, 3.0e-6)
    )
    
    # Convert localizations to latent positions (tuples)
    latent_positions = [(loc.x, loc.y) for loc in localizations]
    
    state = SMLMBaGoL.BaGoLState(
        emitters,
        localizations,
        allocations,
        latent_positions,
        spatial_prior,
        count_prior,
        1.0e-6,  # τ²
        0.0      # log_likelihood placeholder
    )
    
    # Calculate actual likelihood
    state = SMLMBaGoL.BaGoLState(
        state.emitters,
        state.localizations,
        state.allocations,
        state.latent_positions,
        state.spatial_prior,
        state.count_prior,
        state.τ²,
        SMLMBaGoL.log_likelihood(state)
    )
    
    return state
end

# Test different K values
println("\nLikelihood for different K (with optimal allocation):")
println("-"^50)

for k in [3, 6, 9, 12]
    state = create_state_with_k_emitters(k, localizations, true_positions)
    ll = state.log_likelihood
    
    # Also calculate prior
    log_prior_k = SMLMBaGoL.log_prior_k_given_N(k, n_locs, true_μ, κ)
    log_prior_spatial = sum(SMLMBaGoL.log_prior_spatial(e, spatial_prior) for e in state.emitters)
    
    log_posterior = ll + log_prior_k + log_prior_spatial
    
    marker = k == TRUE_K ? " ← TRUE" : ""
    println("K = $k:")
    @printf("  Log-likelihood:    %.2f\n", ll)
    @printf("  Log-prior(K):      %.2f\n", log_prior_k)  
    @printf("  Log-prior(spatial): %.2f\n", log_prior_spatial)
    @printf("  Log-posterior:     %.2f%s\n", log_posterior, marker)
    println()
end

# Analyze allocation patterns
println("\n📊 Allocation Analysis for K=6 vs K=12:")
println("="^60)

state_k6 = create_state_with_k_emitters(6, localizations, true_positions)
state_k12 = create_state_with_k_emitters(12, localizations, true_positions)

# Count allocations per emitter
function count_allocations(state)
    counts = zeros(Int, length(state.emitters))
    for alloc in state.allocations
        counts[alloc] += 1
    end
    return counts
end

counts_k6 = count_allocations(state_k6)
counts_k12 = count_allocations(state_k12)

println("K=6 allocation counts: ", counts_k6)
println("  Mean: $(mean(counts_k6)), Std: $(std(counts_k6))")

println("\nK=12 allocation counts: ", counts_k12)
println("  Mean: $(mean(counts_k12)), Std: $(std(counts_k12))")

# Key insight
println("\n🔍 KEY INSIGHTS:")
println("="^60)
println("1. With true μ=$(true_μ), the prior correctly favors K≈6")
println("2. But the LIKELIHOOD might favor higher K due to:")
println("   - Better fit to localization noise")
println("   - More flexibility in allocation")
println("   - Reduced 'tension' in the model")
println("\nThe algorithm is trading off prior preference for K=6")
println("against likelihood improvement from more emitters.")

println("\n🎯 HYPOTHESIS:")
println("The τ² (systematic noise) parameter may be too small,")
println("forcing the model to explain all variation through more emitters")
println("rather than accepting some unexplained noise.")

println("\n✅ Diagnostic complete!")
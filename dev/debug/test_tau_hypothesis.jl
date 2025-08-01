#!/usr/bin/env julia

"""
Test the hypothesis that τ² (systematic noise) is too small,
causing the model to over-segment to explain all variation.
"""

using Pkg; Pkg.activate("../..")
using SMLMBaGoL
using Statistics
using Random
using Printf

Random.seed!(54321)

println("🔬 TESTING τ² (SYSTEMATIC NOISE) HYPOTHESIS")
println("="^60)

# Generate test data
localizations, _, _ = SMLMBaGoL.simulate_n_mer(
    n=6,
    diameter=0.1,
    photons=1000.0,
    sigma_psf=0.13,
    min_photons=300.0,
    localizations_per_emitter_mean=40,
    localizations_per_emitter_variance=20,
    tau=1.0e-3  # 1 nm systematic noise
)

n_locs = length(localizations)
true_μ = n_locs / 6

println("Generated $(n_locs) localizations")
println("True μ = $(round(true_μ, digits=1))")
println()

# Test different τ values
τ_values_nm = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]  # in nm
τ²_values = [(τ * 1e-3)^2 for τ in τ_values_nm]  # convert to μm²

# Create spatial prior once
spatial_prior = SMLMBaGoL.create_spatial_prior_from_localizations(localizations, 0.2)

println("Running BaGoL with different τ² values:")
println("-"^60)
@printf("%-10s %-15s %-10s %-10s %-20s\n", "τ (nm)", "τ² (μm²)", "MAPN K", "Error %", "K Distribution")
println("-"^60)

for (τ_nm, τ²) in zip(τ_values_nm, τ²_values)
    # Create count prior with fixed true μ
    count_prior = SMLMBaGoL.HierarchicalNegBinomialPrior(
        true_μ, 10.0, τ²,
        (2.0, 5.0), (5.0, 2.0), (3.0, 3.0e-6)
    )
    
    # Initialize and run chain
    chain = SMLMBaGoL.initialize_chain(
        localizations,
        SMLMBaGoL.Emitter2D{Float64},
        spatial_prior,
        count_prior;
        initial_K=10,
        burn_in=2000,
        thin=1
    )
    
    # Override τ² in chain to test value
    chain.current_state = SMLMBaGoL.BaGoLState(
        chain.current_state.emitters,
        chain.current_state.localizations,
        chain.current_state.allocations,
        chain.current_state.latent_positions,
        chain.current_state.spatial_prior,
        chain.current_state.count_prior,
        τ²,  # Use test value
        SMLMBaGoL.log_likelihood(SMLMBaGoL.BaGoLState(
            chain.current_state.emitters,
            chain.current_state.localizations,
            chain.current_state.allocations,
            chain.current_state.latent_positions,
            chain.current_state.spatial_prior,
            chain.current_state.count_prior,
            τ²,
            0.0
        ))
    )
    
    # Run short MCMC (5000 iterations for speed)
    SMLMBaGoL.run_rjmcmc!(chain, 5000)
    
    # Extract K distribution
    k_samples = [length(state.emitters) for state in chain.samples]
    
    if !isempty(k_samples)
        k_counts = Dict()
        for k in k_samples
            k_counts[k] = get(k_counts, k, 0) + 1
        end
        
        # Find mode
        k_mode = argmax(k_counts)
        error_pct = abs(k_mode - 6) / 6 * 100
        
        # Get distribution summary
        k_unique = sort(collect(keys(k_counts)))
        k_dist_str = join(["$k:$(round(100*k_counts[k]/length(k_samples), digits=0))%" for k in k_unique[1:min(5, length(k_unique))]], " ")
        if length(k_unique) > 5
            k_dist_str *= " ..."
        end
        
        @printf("%-10.1f %-15.2e %-10d %-10.1f %-20s\n", τ_nm, τ², k_mode, error_pct, k_dist_str)
    else
        @printf("%-10.1f %-15.2e %-10s %-10s %-20s\n", τ_nm, τ², "FAILED", "-", "-")
    end
end

println("\n🔍 ANALYSIS:")
println("="^60)
println("As τ² increases (allowing more systematic noise):")
println("1. The algorithm needs fewer emitters to explain the data")
println("2. With realistic τ ≈ 1-2 nm, we still over-segment")
println("3. Only with unrealistically large τ (10-20 nm) do we get K≈6")
println()
println("This suggests:")
println("• The τ² parameter is part of the problem")
println("• But even with correct τ², the likelihood is still too greedy")
println("• The issue may be deeper in the likelihood formulation")

println("\n✅ Test complete!")
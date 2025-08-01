#!/usr/bin/env julia

"""
Demonstrate consistency likelihood with stronger penalty to show the effect.
"""

using Pkg
cd(@__DIR__)
Pkg.activate("../..")
using SMLMBaGoL
using Statistics
using Random
using Printf

Random.seed!(42)

println("🚀 STRONGER CONSISTENCY LIKELIHOOD DEMONSTRATION")
println("="^70)

# Generate more extreme over-segmentation scenario
n_locs = 200
true_x = 0.0
true_y = 0.0
σ_loc = 0.01  # 10 nm
τ = 0.001     # 1 nm

# Generate localizations
localizations = SMLMBaGoL.Localization2D{Float64}[]
for i in 1:n_locs
    x = true_x + σ_loc * randn()
    y = true_y + σ_loc * randn()
    push!(localizations, SMLMBaGoL.Localization2D(x, y, σ_loc, σ_loc, i))
end

println("Generated $(n_locs) localizations from 1 emitter")
println("Testing K = 1, 2, 4, 8 emitters")

# Setup
spatial_prior = SMLMBaGoL.UniformSpatialPrior(-0.1, 0.1, -0.1, 0.1)
count_prior = SMLMBaGoL.HierarchicalNegBinomialPrior(
    40.0, 10.0, τ^2,
    (2.0, 5.0), (5.0, 2.0), (3.0, 3.0e-6)
)

# Test different K values
K_values = [1, 2, 4, 8]
α_values = [0.0, 1.0, 5.0, 10.0, 20.0]

println("\n📊 LIKELIHOOD COMPARISON:")
println("-"^100)
@printf("%-3s", "K")
for α in α_values
    @printf(" %15s", α == 0.0 ? "Standard" : "Consistency α=$α")
end
println("\n" * "-"^100)

results = Dict()

for K in K_values
    @printf("%-3d", K)
    
    # Split data into K groups
    locs_per_group = n_locs ÷ K
    emitters = SMLMBaGoL.Emitter2D{Float64}[]
    allocations = Int[]
    latent_positions = Tuple{Float64, Float64}[]
    
    for k in 1:K
        start_idx = (k-1) * locs_per_group + 1
        end_idx = k == K ? n_locs : k * locs_per_group
        group_locs = localizations[start_idx:end_idx]
        
        # Fit emitter to this group
        mean_x = mean(loc.x for loc in group_locs)
        mean_y = mean(loc.y for loc in group_locs)
        push!(emitters, SMLMBaGoL.Emitter2D(mean_x, mean_y, 1000.0))
        
        # Allocations
        for i in start_idx:end_idx
            push!(allocations, k)
            push!(latent_positions, (mean_x, mean_y))
        end
    end
    
    # Create state
    state = SMLMBaGoL.BaGoLState(
        emitters,
        localizations,
        allocations,
        latent_positions,
        spatial_prior,
        count_prior,
        τ^2,
        0.0
    )
    
    # Compute likelihoods with different α
    for α in α_values
        if α == 0.0
            ll = SMLMBaGoL.standard_log_likelihood(state)
        else
            ll = SMLMBaGoL.consistency_log_likelihood(state; α=α)
        end
        results[(K, α)] = ll
        @printf(" %15.2f", ll)
    end
    
    # Also compute normalized residual variance
    residuals_x = Float64[]
    residuals_y = Float64[]
    for i in 1:n_locs
        emitter = emitters[allocations[i]]
        loc = localizations[i]
        push!(residuals_x, (loc.x - emitter.x) / sqrt(loc.σx^2 + τ^2))
        push!(residuals_y, (loc.y - emitter.y) / sqrt(loc.σy^2 + τ^2))
    end
    var_mean = mean([var(residuals_x), var(residuals_y)])
    @printf("  (var=%.3f)", var_mean)
    
    println()
end

# Show improvements relative to K=1
println("\n⚖️  LIKELIHOOD IMPROVEMENTS (relative to K=1):")
println("-"^100)
@printf("%-3s", "K")
for α in α_values
    @printf(" %15s", α == 0.0 ? "Standard" : "α=$α")
end
println("\n" * "-"^100)

for K in K_values[2:end]
    @printf("%-3d", K)
    for α in α_values
        improvement = results[(K, α)] - results[(1, α)]
        @printf(" %15.2f", improvement)
    end
    println()
end

println("\n💡 ANALYSIS:")
println("-"^70)
println("• Standard likelihood (α=0) always improves with more K")
println("• As α increases, the penalty for variance mismatch grows")
println("• With large enough α, the consistency likelihood prefers K=1")
println("• Normalized variance < 1.0 indicates overfitting")

# Find optimal α
println("\n🎯 FINDING OPTIMAL α:")
println("-"^70)
println("Looking for α where K=1 is preferred over K=2...")

α_test = 0.0:0.5:10.0
for α in α_test
    ll_1 = α == 0.0 ? results[(1, 0.0)] : SMLMBaGoL.consistency_log_likelihood(
        SMLMBaGoL.BaGoLState(
            [SMLMBaGoL.Emitter2D(true_x, true_y, 1000.0)],
            localizations,
            fill(1, n_locs),
            [(true_x, true_y) for _ in 1:n_locs],
            spatial_prior,
            count_prior,
            τ^2,
            0.0
        ); α=α
    )
    
    ll_2 = α == 0.0 ? results[(2, 0.0)] : (
        haskey(results, (2, α)) ? results[(2, α)] : 
        SMLMBaGoL.consistency_log_likelihood(
            SMLMBaGoL.BaGoLState(
                [emitters[1], emitters[2]],  # Use pre-computed from K=2
                localizations,
                vcat(fill(1, 100), fill(2, 100)),
                vcat([(emitters[1].x, emitters[1].y) for _ in 1:100],
                     [(emitters[2].x, emitters[2].y) for _ in 1:100]),
                spatial_prior,
                count_prior,
                τ^2,
                0.0
            ); α=α
        )
    )
    
    if ll_1 > ll_2
        println("α = $α: K=1 preferred! (ΔLL = $(round(ll_1 - ll_2, digits=2)))")
        break
    elseif α == last(α_test)
        println("Need α > $(last(α_test)) to prefer K=1")
    end
end

println("\n✅ Demonstration complete!")
println("\nKey insight: The consistency likelihood can prevent over-segmentation")
println("but requires tuning the penalty strength α for the specific problem.")
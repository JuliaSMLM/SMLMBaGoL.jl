#!/usr/bin/env julia

"""
Diagnose the circular dependency between K and μ in the prior.
"""

using Pkg; Pkg.activate("../..")
using SMLMBaGoL
using Distributions
using CairoMakie

# Scenario: 6-mer with ~227 localizations
N = 227  # Total localizations
true_k = 6
true_μ = N / true_k  # ≈ 38

# Range of μ values to test
μ_values = 5:5:100
k_values = 1:20

# Create figure
fig = Figure(size=(1200, 800))

# Plot 1: Prior on K for different μ values
ax1 = Axis(fig[1, 1], 
    title="Prior on K for different μ values (N=$N)",
    xlabel="Number of emitters (K)",
    ylabel="Log probability",
    limits=(0, 21, nothing, nothing))

κ = 1.0  # After our fix

for (i, μ) in enumerate([10, 20, 30, 40, 50])
    log_probs = [SMLMBaGoL.log_prior_k_given_N(k, N, Float64(μ), κ) for k in k_values]
    # Normalize for visualization
    probs = exp.(log_probs .- maximum(log_probs))
    
    lines!(ax1, k_values, log_probs, 
        label="μ=$μ (expects k≈$(round(N/μ, digits=1)))",
        linewidth=2)
end

vlines!(ax1, [true_k], color=:red, linestyle=:dash, linewidth=2)
text!(ax1, true_k + 0.5, -20, text="True K=6", color=:red)
axislegend(ax1, position=:rt)

# Plot 2: Expected K vs μ
ax2 = Axis(fig[1, 2],
    title="Expected K = N/μ relationship",
    xlabel="μ (mean locs per emitter)",
    ylabel="Expected K",
    limits=(0, 105, nothing, nothing))

expected_k = N ./ μ_values
lines!(ax2, μ_values, expected_k, linewidth=3, color=:blue)
hlines!(ax2, [true_k], color=:red, linestyle=:dash, linewidth=2)
vlines!(ax2, [true_μ], color=:red, linestyle=:dash, linewidth=2)
scatter!(ax2, [true_μ], [true_k], color=:red, markersize=15)
text!(ax2, true_μ + 2, true_k + 0.5, text="True point", color=:red)

# Add problematic point
scatter!(ax2, [12.6], [N/12.6], color=:orange, markersize=15)
text!(ax2, 12.6 + 2, N/12.6 - 1, text="Algorithm\nconverges here", color=:orange)

# Plot 3: Show the feedback loop
ax3 = Axis(fig[2, 1:2],
    title="Feedback Loop Visualization",
    xlabel="Iteration",
    ylabel="Parameter value",
    limits=(0, 10, nothing, nothing))

# Simulate the feedback loop
iterations = 0:9
μ_evolution = [50, 30, 20, 15, 12, 11, 11, 12, 12, 12.6]
k_evolution = N ./ μ_evolution

lines!(ax3, iterations, μ_evolution, label="μ", linewidth=3, color=:blue)
lines!(ax3, iterations, k_evolution, label="Expected K = N/μ", linewidth=3, color=:orange)
hlines!(ax3, [true_μ], color=:red, linestyle=:dash, alpha=0.5)
hlines!(ax3, [true_k], color=:red, linestyle=:dash, alpha=0.5)
axislegend(ax3, position=:rt)

# Skip text annotation for now due to API issues

save("k_prior_diagnostic.png", fig)
println("Diagnostic plot saved to k_prior_diagnostic.png")

# Now let's calculate actual prior probabilities for our scenario
println("\nPrior probabilities for K given N=$N:")
println("=====================================")

for μ in [12.6, 20.0, 30.0, true_μ]
    println("\nWith μ=$μ (expects $(round(N/μ, digits=1)) emitters):")
    
    # Calculate probabilities for K=4 to K=12
    log_probs = [SMLMBaGoL.log_prior_k_given_N(k, N, Float64(μ), κ) for k in 4:12]
    probs = exp.(log_probs)
    probs ./= sum(probs)  # Normalize
    
    for (k, p) in zip(4:12, probs)
        bar = "█" ^ Int(round(p * 50))
        marker = k == true_k ? " <-- TRUE" : ""
        println("  K=$k: $(round(p*100, digits=1))% $bar$marker")
    end
end

println("\n🔍 KEY INSIGHT:")
println("The prior strongly favors K ≈ N/μ. When μ converges to ~12.6,")
println("the prior expects ~18 emitters, making K=6 extremely unlikely!")
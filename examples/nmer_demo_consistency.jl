#!/usr/bin/env julia

"""
N-mer Performance Study with Consistency Likelihood

This demo compares the standard likelihood with the new consistency likelihood
for recovering known n-mer structures.
"""

using Pkg; Pkg.activate("examples")
using SMLMBaGoL
using Statistics
using CairoMakie
using Random
using Dates

# Simple mode function
function mode(x)
    counts = Dict()
    for val in x
        counts[val] = get(counts, val, 0) + 1
    end
    return argmax(counts)
end

#=============================================================================
USER PARAMETERS
=============================================================================#

# Core n-mer parameters
const N_EMITTERS = 6                    # Number of emitters in n-mer
const DIAMETER = 0.100                  # n-mer diameter in μm (50 nm)

# Photophysics parameters  
const PHOTONS_MEAN = 1000              # Mean photons per localization
const LOCS_PER_EMITTER = 50            # Mean localizations per emitter
const LOCS_VARIANCE = LOCS_PER_EMITTER # Variance in localizations per emitter

# Noise parameters
const PSF_WIDTH = 0.13                 # PSF sigma in μm (130 nm)
const PSF_WIDTH_MIN = 0.10             # Minimum PSF width (100 nm) 
const PHOTON_THRESHOLD = 300           # Minimum photons to keep localization
const TAU_SYSTEMATIC = 0.002           # Systematic noise τ in μm (2 nm)

# RJMCMC parameters
const N_ITERATIONS = 10_000             # Total RJMCMC iterations
const BURN_IN = 2_000                  # Burn-in iterations
const THIN = 1                         # Store every nth sample
const ENABLE_HIERARCHICAL = true       # Use hierarchical updates
const HIERARCHICAL_INTERVAL = 2000     # Update hyperparameters every N iterations

# Random seed
Random.seed!(42)

println("="^70)
println("🔬 N-MER PERFORMANCE STUDY WITH CONSISTENCY LIKELIHOOD")
println("="^70)
println("Configuration:")
println("  • $(N_EMITTERS)-mer with diameter $(DIAMETER*1000) nm")
println("  • ~$(LOCS_PER_EMITTER) localizations per emitter")
println("  • PSF width: $(PSF_WIDTH*1000) nm")
println("  • Systematic noise τ: $(TAU_SYSTEMATIC*1000) nm")
println("  • RJMCMC: $(N_ITERATIONS) iterations ($(BURN_IN) burn-in)")
println("  • Threading: $(Threads.nthreads()) threads available")
println()

#=============================================================================
DATA GENERATION
=============================================================================#

println("📊 GENERATING SYNTHETIC DATA")
println("-"^70)

# Generate n-mer data
localizations, true_emitters, true_allocations = simulate_n_mer(
    n=N_EMITTERS,
    diameter=DIAMETER,
    photons=PHOTONS_MEAN,
    sigma_psf=PSF_WIDTH,
    min_photons=PHOTON_THRESHOLD,
    localizations_per_emitter_mean=LOCS_PER_EMITTER,
    localizations_per_emitter_variance=LOCS_VARIANCE,
    tau=TAU_SYSTEMATIC
)

N_LOCS = length(localizations)
println("  ✓ Generated $(N_LOCS) localizations from $(N_EMITTERS) emitters")
println("  ✓ True μ = $(round(N_LOCS/N_EMITTERS, digits=1)) locs/emitter")

#=============================================================================
RJMCMC ANALYSIS - Standard Likelihood
=============================================================================#

println("\n🔄 RUNNING RJMCMC WITH STANDARD LIKELIHOOD")
println("-"^70)

t_start_std = time()
chains_std = run_bagol(
    localizations;
    n_iterations=N_ITERATIONS,
    burn_in=BURN_IN,
    thin=THIN,
    tau_mean=TAU_SYSTEMATIC,
    enable_hierarchical=ENABLE_HIERARCHICAL,
    hierarchical_interval=HIERARCHICAL_INTERVAL,
    partition_data=false,  # Single chain for comparison
    likelihood_config=StandardLikelihood()  # Default
)
t_elapsed_std = time() - t_start_std
println("  ✓ Completed in $(round(t_elapsed_std, digits=1))s")

# Extract K distribution
k_samples_std = [length(state.emitters) for state in chains_std.samples]
k_mode_std = mode(k_samples_std)
k_mean_std = mean(k_samples_std)

println("\n📊 Standard Likelihood Results:")
println("  • Mode K = $(k_mode_std) (true: $(N_EMITTERS))")
println("  • Mean K = $(round(k_mean_std, digits=1))")
println("  • Error = $(round(abs(k_mode_std - N_EMITTERS)/N_EMITTERS * 100))%")

#=============================================================================
RJMCMC ANALYSIS - Consistency Likelihood
=============================================================================#

println("\n🔄 RUNNING RJMCMC WITH CONSISTENCY LIKELIHOOD (α=10)")
println("-"^70)

t_start_cons = time()
chains_cons = run_bagol(
    localizations;
    n_iterations=N_ITERATIONS,
    burn_in=BURN_IN,
    thin=THIN,
    tau_mean=TAU_SYSTEMATIC,
    enable_hierarchical=ENABLE_HIERARCHICAL,
    hierarchical_interval=HIERARCHICAL_INTERVAL,
    partition_data=false,  # Single chain for comparison
    likelihood_config=ConsistencyLikelihood(10.0)  # α=10
)
t_elapsed_cons = time() - t_start_cons
println("  ✓ Completed in $(round(t_elapsed_cons, digits=1))s")

# Extract K distribution
k_samples_cons = [length(state.emitters) for state in chains_cons.samples]
k_mode_cons = mode(k_samples_cons)
k_mean_cons = mean(k_samples_cons)

println("\n📊 Consistency Likelihood Results:")
println("  • Mode K = $(k_mode_cons) (true: $(N_EMITTERS))")
println("  • Mean K = $(round(k_mean_cons, digits=1))")
println("  • Error = $(round(abs(k_mode_cons - N_EMITTERS)/N_EMITTERS * 100))%")

#=============================================================================
COMPARISON VISUALIZATION
=============================================================================#

println("\n📈 CREATING COMPARISON PLOTS")
println("-"^70)

fig = Figure(size=(1200, 800))

# K distribution comparison
ax1 = Axis(fig[1, 1], title="Emitter Count Distribution", 
           xlabel="Number of Emitters (K)", ylabel="Frequency")

# Standard likelihood
k_counts_std = Dict()
for k in k_samples_std
    k_counts_std[k] = get(k_counts_std, k, 0) + 1
end
k_vals_std = sort(collect(keys(k_counts_std)))
k_freqs_std = [k_counts_std[k] for k in k_vals_std]

barplot!(ax1, k_vals_std, k_freqs_std, alpha=0.7, label="Standard", color=:steelblue)

# Consistency likelihood
k_counts_cons = Dict()
for k in k_samples_cons
    k_counts_cons[k] = get(k_counts_cons, k, 0) + 1
end
k_vals_cons = sort(collect(keys(k_counts_cons)))
k_freqs_cons = [k_counts_cons[k] for k in k_vals_cons]

barplot!(ax1, k_vals_cons .+ 0.2, k_freqs_cons, alpha=0.7, label="Consistency", color=:coral)

# True value
vlines!(ax1, [N_EMITTERS], color=:black, linestyle=:dash, linewidth=2, label="True K")

axislegend(ax1)

# Convergence comparison
ax2 = Axis(fig[1, 2], title="K Evolution", 
           xlabel="Iteration", ylabel="Number of Emitters")

# Sample every 100 iterations for visualization
sample_interval = 100
iters = sample_interval:sample_interval:length(k_samples_std)

lines!(ax2, iters, k_samples_std[1:sample_interval:end], 
       alpha=0.7, label="Standard", color=:steelblue)
lines!(ax2, iters, k_samples_cons[1:sample_interval:end], 
       alpha=0.7, label="Consistency", color=:coral)
hlines!(ax2, [N_EMITTERS], color=:black, linestyle=:dash, linewidth=2, label="True K")

axislegend(ax2)

# MAPN estimates comparison
mapn_std = estimate_mapn([chains_std])
mapn_cons = estimate_mapn([chains_cons])

ax3 = Axis(fig[2, 1:2], title="MAPN Emitter Positions", 
           xlabel="X (μm)", ylabel="Y (μm)", aspect=DataAspect())

# Skip position plot for now - just show the key results
text!(ax3, 0, 0, text="Results:\nStandard: K=$(k_mode_std) (error: $(round(abs(k_mode_std - N_EMITTERS)/N_EMITTERS * 100))%)\nConsistency: K=$(k_mode_cons) (error: $(round(abs(k_mode_cons - N_EMITTERS)/N_EMITTERS * 100))%)",
      align=(:center, :center), fontsize=16)

# axislegend(ax3, position=:lt)  # Skip legend for text plot

save("nmer_likelihood_comparison.png", fig, px_per_unit=2)
println("  ✓ Saved comparison plot: nmer_likelihood_comparison.png")

#=============================================================================
SUMMARY
=============================================================================#

println("\n"*"="^70)
println("📊 PERFORMANCE SUMMARY")
println("="^70)
println("Standard Likelihood:")
println("  • K = $(k_mode_std) ($(round(abs(k_mode_std - N_EMITTERS)/N_EMITTERS * 100))% error)")
println("  • Runtime: $(round(t_elapsed_std, digits=1))s")
println()
println("Consistency Likelihood (α=10):")
println("  • K = $(k_mode_cons) ($(round(abs(k_mode_cons - N_EMITTERS)/N_EMITTERS * 100))% error)")
println("  • Runtime: $(round(t_elapsed_cons, digits=1))s")
println()
println("Improvement: $(round((abs(k_mode_std - N_EMITTERS) - abs(k_mode_cons - N_EMITTERS))/N_EMITTERS * 100))% better accuracy")
println("="^70)
println()

# Circular helper function for plotting
function circle!(ax, center, radius; n_points=50, kwargs...)
    θ = range(0, 2π, length=n_points)
    x = center[1] .+ radius * cos.(θ)
    y = center[2] .+ radius * sin.(θ)
    lines!(ax, x, y; kwargs...)
end
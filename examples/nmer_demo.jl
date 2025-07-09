#!/usr/bin/env julia

"""
N-mer Performance Study: Analyzing BaGoL's ability to recover known n-mer structures

This demo specifically focuses on:
1. Recovery of correct emitter count (K)
2. Evolution of hyperparameters (μ, κ, τ²) 
3. Position accuracy of recovered emitters
4. Mixing and convergence diagnostics

USAGE:
  julia --threads=16 --project=. examples/nmer_demo.jl   # Recommended: 16 threads
  julia --threads=auto --project=. examples/nmer_demo.jl # Use all available threads
  julia --project=. examples/nmer_demo.jl                # Single-threaded mode

The demo is fully configurable and generates comprehensive performance analysis.
"""

using Pkg; Pkg.activate("examples")
using SMLMBaGoL
using Statistics
using CairoMakie
using Random
using Dates

# Simple mode function since StatsBase is not available
function mode(x)
    counts = Dict()
    for val in x
        counts[val] = get(counts, val, 0) + 1
    end
    return argmax(counts)
end

#=============================================================================
USER PARAMETERS - Configure your n-mer study here
=============================================================================#

# Core n-mer parameters
const N_EMITTERS = 6                    # Number of emitters in n-mer
const DIAMETER = 0.100                  # n-mer diameter in μm (50 nm)

# Photophysics parameters  
const PHOTONS_MEAN = 1000              # Mean photons per localization
const LOCS_PER_EMITTER = 50            # Mean localizations per emitter
const LOCS_VARIANCE = LOCS_PER_EMITTER                # Variance in localizations per emitter

# Noise parameters
const PSF_WIDTH = 0.13                 # PSF sigma in μm (130 nm)
const MIN_PHOTONS = 300                # Minimum photon threshold
const TAU = 0.001                      # Systematic noise in μm (20 nm)

# Analysis parameters
const N_ITERATIONS = 50000             # RJMCMC iterations
const BURN_IN = 5000                   # Burn-in period
const HIERARCHICAL_INTERVAL = 1000     # Update hyperparameters every N iterations

# Visualization parameters
const SAVE_PLOTS = true                # Generate PNG files
const PIXEL_SIZE = 0.002               # μm per pixel for SR images

#=============================================================================
SETUP & SIMULATION
=============================================================================#

# Create output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("=== N-mer Performance Study ===\n")
println("Configuration:")
println("• N-mer pattern: $(N_EMITTERS)-mer with $(DIAMETER*1000) nm diameter")
println("• Mean photons per localization: $PHOTONS_MEAN")
println("• Expected localizations per emitter: $LOCS_PER_EMITTER ± $LOCS_VARIANCE")
println("• PSF width: $(PSF_WIDTH*1000) nm")
println("• Minimum photon threshold: $MIN_PHOTONS")
println("• Systematic noise (tau): $(TAU*1000) nm")
println("• RJMCMC iterations: $N_ITERATIONS (burn-in: $BURN_IN)")
println("• Hierarchical updates: enabled (interval: $HIERARCHICAL_INTERVAL)")
println("• Threading: $(Threads.nthreads()) threads available")
println("• Image pixel size: $(PIXEL_SIZE*1000) nm")
println()

#=============================================================================
1. Generate Synthetic N-mer Data
=============================================================================#

println("1. Generating synthetic n-mer data...")

# Generate synthetic n-mer data
localizations, spatial_prior, count_prior = simulate_n_mer(
    n=N_EMITTERS,
    diameter=DIAMETER,
    photons=PHOTONS_MEAN,
    sigma_psf=PSF_WIDTH,
    min_photons=MIN_PHOTONS,
    localizations_per_emitter_mean=LOCS_PER_EMITTER,
    localizations_per_emitter_variance=LOCS_VARIANCE,
    tau=TAU
)

println("   ✓ Created $(N_EMITTERS)-mer with $(length(localizations)) localizations")
println("   ✓ Localization density: $(round(length(localizations)/N_EMITTERS, digits=1)) locs/emitter")

# Generate ground truth positions for comparison
true_positions = SMLMBaGoL.generate_circular_positions(N_EMITTERS, DIAMETER/2, 0.0, 0.0)
println("   ✓ Ground truth positions generated")

#=============================================================================
2. Run BaGoL Analysis
=============================================================================#

println("\n2. Running BaGoL analysis...")

# Run BaGoL analysis with hierarchical updates
chains_result = run_bagol(localizations; 
    n_iterations=N_ITERATIONS,
    burn_in=BURN_IN,
    enable_hierarchical=true,
    hierarchical_interval=HIERARCHICAL_INTERVAL,
    tau_mean=TAU,
    partition_data=false  # Disable partitioning for small n-mer analysis
)

# Ensure chains is always a vector for consistent handling
chains = isa(chains_result, Vector) ? chains_result : [chains_result]

println("   ✓ Analysis completed with $(length(chains)) partition(s)")

# Extract MAPN results
mapn_results = estimate_mapn(chains)
println("   ✓ MAPN estimate: $(length(mapn_results)) emitters")

#=============================================================================
3. Extract K Trajectory and Analysis
=============================================================================#

function extract_k_trajectory(chain::RJMCMCChain)
    """Extract array of K values for each sample"""
    return [length(sample.emitters) for sample in chain.samples]
end

# Extract K trajectory (use first chain since partitioning is disabled)
k_trajectory = extract_k_trajectory(chains[1])
println("   ✓ Extracted K trajectory ($(length(k_trajectory)) samples)")

#=============================================================================
4. Plotting Functions
=============================================================================#

function plot_k_evolution(k_trajectory::Vector{Int}, true_k::Int, burn_in::Int;
                         filename::Union{Nothing,String} = nothing)
    """Plot evolution of emitter count K over iterations"""
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], 
              xlabel="Iteration", 
              ylabel="Number of emitters (K)",
              title="Evolution of Emitter Count")
    
    # Plot trajectory
    iterations = 1:length(k_trajectory)
    lines!(ax, iterations, k_trajectory, color=:black, alpha=0.3)
    
    # Add true value line
    hlines!(ax, [true_k], color=:red, linewidth=2, linestyle=:dash, 
            label="True K = $true_k")
    
    # Add burn-in indicator
    vlines!(ax, [burn_in], color=:gray, linewidth=2, linestyle=:dot,
            label="Burn-in")
    
    # Add running mean post-burn-in
    if length(k_trajectory) > burn_in
        post_burnin = k_trajectory[burn_in:end]
        running_mean = cumsum(post_burnin) ./ (1:length(post_burnin))
        lines!(ax, burn_in:length(k_trajectory), running_mean, 
               color=:blue, linewidth=2, label="Running mean")
    end
    
    axislegend(ax, position=:rt)
    
    if !isnothing(filename)
        save(filename, fig)
    end
    return fig
end

function plot_k_posterior(k_trajectory::Vector{Int}, true_k::Int, burn_in::Int;
                         filename::Union{Nothing,String} = nothing)
    """Plot posterior distribution of K"""
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], 
              xlabel="Number of emitters (K)", 
              ylabel="Frequency",
              title="Posterior Distribution of K")
    
    # Get post-burn-in samples
    k_post_burnin = k_trajectory[burn_in:end]
    
    # Plot histogram
    hist!(ax, k_post_burnin, bins=minimum(k_post_burnin)-0.5:maximum(k_post_burnin)+0.5,
          color=(:blue, 0.6))
    
    # Add true value line
    vlines!(ax, [true_k], color=:red, linewidth=3, linestyle=:dash,
            label="True K = $true_k")
    
    # Add mode (MAPN)
    k_mode = mode(k_post_burnin)
    vlines!(ax, [k_mode], color=:green, linewidth=3,
            label="Mode (MAPN) = $k_mode")
    
    axislegend(ax, position=:rt)
    
    if !isnothing(filename)
        save(filename, fig)
    end
    return fig
end


#=============================================================================
5. Generate Visualizations
=============================================================================#

println("\n3. Generating visualizations...")

if SAVE_PLOTS
    # K evolution plot
    plot_k_evolution(k_trajectory, N_EMITTERS, BURN_IN,
                    filename=joinpath(output_dir, "nmer_$(N_EMITTERS)_k_evolution.png"))
    println("   ✓ K evolution plot saved")
    
    # K posterior distribution
    plot_k_posterior(k_trajectory, N_EMITTERS, BURN_IN,
                    filename=joinpath(output_dir, "nmer_$(N_EMITTERS)_k_posterior.png"))
    println("   ✓ K posterior distribution saved")
    
    # Ground truth comparison using circle plots
    sr_circles_combined(localizations, mapn_results,
                       true_positions=true_positions,
                       true_color=:blue,
                       true_markersize=20,
                       true_marker=:xcross,
                       filename=joinpath(output_dir, "nmer_$(N_EMITTERS)_ground_truth.png"))
    println("   ✓ Ground truth comparison saved")
    
    # Generate super-resolution images using existing functions
    try
        # Localizations SR image
        gen_sr_image(localizations, 
                    filename=joinpath(output_dir, "nmer_$(N_EMITTERS)_localizations_sr.png"),
                    pixel_size=PIXEL_SIZE)
        println("   ✓ Localizations SR image saved")
        
        # MAPN emitters SR image
        if !isempty(mapn_results)
            gen_sr_image(mapn_results, 
                        filename=joinpath(output_dir, "nmer_$(N_EMITTERS)_mapn_sr.png"),
                        pixel_size=PIXEL_SIZE)
            println("   ✓ MAPN emitters SR image saved")
        end
        
        # Uncertainty circles plot (standard)
        if !isempty(mapn_results)
            sr_circles_combined(localizations, mapn_results,
                              filename=joinpath(output_dir, "nmer_$(N_EMITTERS)_uncertainty.png"))
            println("   ✓ Uncertainty circles plot saved")
        end
    catch e
        println("   ⚠ Some SR visualizations failed: $e")
    end
end

#=============================================================================
6. Use Existing Analysis Functions
=============================================================================#

# Hyperparameter evolution
if !isempty(chains[1].hierarchical_history)
    try
        plot_hierarchical_evolution(chains,
            filename=joinpath(output_dir, "nmer_$(N_EMITTERS)_hyperprior_evolution.png"))
        println("   ✓ Hierarchical evolution plot saved")
        
        # Convergence analysis
        conv_result = analyze_hierarchical_convergence(chains)
        println("\n4. Convergence Analysis:")
        println("   • $(conv_result.message)")
        println("   • Final μ: $(round(conv_result.final_μ, digits=2)) (true: $LOCS_PER_EMITTER)")
        println("   • Final τ: $(round(sqrt(conv_result.final_τ²)*1000, digits=1)) nm (true: $(TAU*1000) nm)")
        
        # NEW: Enhanced histogram with fit statistics
        plot_emitter_count_histogram(chains,
            filename=joinpath(output_dir, "nmer_$(N_EMITTERS)_count_histogram_enhanced.png"),
            true_mean=LOCS_PER_EMITTER,
            show_fit_stats=true)
        println("   ✓ Enhanced count histogram with fit statistics saved")
        
        # NEW: Comprehensive diagnostic plot
        plot_negbinomial_diagnostic(chains,
            filename=joinpath(output_dir, "nmer_$(N_EMITTERS)_negbinomial_diagnostic.png"))
        println("   ✓ Negative Binomial diagnostic plot saved")
        
        # NEW: Print fit summary
        if isa(chains[1].current_state.count_prior, HierarchicalNegBinomialPrior)
            all_counts = SMLMBaGoL.collect_emitter_counts(chains)
            stats = assess_negbinomial_fit(all_counts, chains[1].current_state.count_prior)
            print_fit_summary(stats)
        end
        
    catch e
        println("   ⚠ Hierarchical analysis failed: $e")
    end
end

# Generate movie of chain evolution
if !isempty(chains)
    try
        # Select the chain with most samples
        chain_for_movie = chains[1]
        for c in chains
            if length(c.samples) > length(chain_for_movie.samples)
                chain_for_movie = c
            end
        end
        
        if !isempty(chain_for_movie.samples)
            movie_filename = joinpath(output_dir, "nmer_$(N_EMITTERS)_chain_evolution.mp4")
            generate_chain_movie(
                chain_for_movie,
                movie_filename;
                fps=10,
                figsize=(1200, 800),
                sample_range=max(1, length(chain_for_movie.samples)-999):length(chain_for_movie.samples)  # Last 1000 samples
            )
            println("   ✓ Chain evolution movie saved: $(basename(movie_filename))")
        end
    catch e
        println("   ⚠ Movie generation failed: $e")
    end
end

# Chain diagnostics
try
    diagnosis = diagnose_chains(chains[1])
    println("\n5. Chain Diagnostics:")
    println("   • $(diagnosis.summary)")
catch e
    println("\n5. Chain Diagnostics:")
    println("   ⚠ Diagnostics failed: $e")
end

#=============================================================================
7. Performance Summary
=============================================================================#

println("\n6. Performance Summary:")

# K recovery analysis
k_post_burnin = k_trajectory[BURN_IN:end]
k_mode = mode(k_post_burnin)
k_accuracy = abs(k_mode - N_EMITTERS) / N_EMITTERS * 100
k_std = std(k_post_burnin)

println("   • K recovery:")
println("     - True K: $N_EMITTERS")
println("     - Mode (MAPN): $k_mode")
println("     - Error: $(round(k_accuracy, digits=1))%")
println("     - Posterior std: $(round(k_std, digits=2))")

# Posterior statistics
k_unique = unique(k_post_burnin)
k_frequencies = [count(==(k), k_post_burnin) for k in k_unique]
k_probabilities = k_frequencies ./ length(k_post_burnin)

println("   • K posterior distribution:")
for (k, prob) in zip(k_unique, k_probabilities)
    println("     - K=$k: $(round(prob*100, digits=1))%")
end

# Position accuracy (if correct number recovered)
if length(mapn_results) == N_EMITTERS
    println("   • Position accuracy:")
    println("     - Recovered correct number of emitters: ✓")
    
    # Simple position matching (closest pairs)
    mapn_positions = [(em.x, em.y) for em in mapn_results]
    
    # Calculate distances between all pairs and find minimum assignment
    distances = [sqrt((mp[1] - tp[1])^2 + (mp[2] - tp[2])^2) for mp in mapn_positions, tp in true_positions]
    
    # Greedy matching (not optimal but simple)
    matched_distances = Float64[]
    used_true = Set{Int}()
    used_mapn = Set{Int}()
    
    while length(matched_distances) < min(length(mapn_positions), length(true_positions))
        min_dist = Inf
        min_i, min_j = 0, 0
        
        for i in 1:length(mapn_positions)
            if i in used_mapn continue end
            for j in 1:length(true_positions)
                if j in used_true continue end
                if distances[i, j] < min_dist
                    min_dist = distances[i, j]
                    min_i, min_j = i, j
                end
            end
        end
        
        if min_i > 0 && min_j > 0
            push!(matched_distances, min_dist)
            push!(used_mapn, min_i)
            push!(used_true, min_j)
        else
            break
        end
    end
    
    if !isempty(matched_distances)
        mean_error = mean(matched_distances) * 1000  # Convert to nm
        rms_error = sqrt(mean(matched_distances.^2)) * 1000  # Convert to nm
        println("     - Mean position error: $(round(mean_error, digits=1)) nm")
        println("     - RMS position error: $(round(rms_error, digits=1)) nm")
    end
else
    println("   • Position accuracy:")
    println("     - Recovered $(length(mapn_results)) emitters (expected $N_EMITTERS): ✗")
end

# Timing and efficiency
total_locs = length(localizations)
locs_per_second = total_locs * N_ITERATIONS / 60  # Rough estimate
println("   • Computational efficiency:")
println("     - Total localizations: $total_locs")
println("     - RJMCMC iterations: $N_ITERATIONS")
println("     - Threads used: $(Threads.nthreads())")

#=============================================================================
8. Save Summary
=============================================================================#

if SAVE_PLOTS
    summary_file = joinpath(output_dir, "nmer_$(N_EMITTERS)_summary.txt")
    open(summary_file, "w") do io
        println(io, "N-mer Performance Summary")
        println(io, "========================")
        println(io, "Date: $(Dates.now())")
        println(io, "")
        println(io, "Configuration:")
        println(io, "• N-mer pattern: $(N_EMITTERS)-mer with $(DIAMETER*1000) nm diameter")
        println(io, "• Mean photons: $PHOTONS_MEAN")
        println(io, "• Localizations per emitter: $LOCS_PER_EMITTER ± $LOCS_VARIANCE")
        println(io, "• PSF width: $(PSF_WIDTH*1000) nm")
        println(io, "• Systematic noise: $(TAU*1000) nm")
        println(io, "• RJMCMC iterations: $N_ITERATIONS (burn-in: $BURN_IN)")
        println(io, "")
        println(io, "Results:")
        println(io, "• Total localizations: $(length(localizations))")
        println(io, "• True emitters: $N_EMITTERS")
        println(io, "• Recovered (MAPN): $(length(mapn_results))")
        println(io, "• K mode: $k_mode")
        println(io, "• K recovery error: $(round(k_accuracy, digits=1))%")
        println(io, "• K posterior std: $(round(k_std, digits=2))")
        println(io, "")
        println(io, "K Posterior Distribution:")
        for (k, prob) in zip(k_unique, k_probabilities)
            println(io, "• K=$k: $(round(prob*100, digits=1))%")
        end
        
        if length(mapn_results) == N_EMITTERS && !isempty(matched_distances)
            mean_error = mean(matched_distances) * 1000
            rms_error = sqrt(mean(matched_distances.^2)) * 1000
            println(io, "")
            println(io, "Position Accuracy:")
            println(io, "• Mean error: $(round(mean_error, digits=1)) nm")
            println(io, "• RMS error: $(round(rms_error, digits=1)) nm")
        end
    end
    println("   ✓ Summary saved to $(summary_file)")
end

println("\n=== N-mer Performance Study Complete ===")
println("Output files saved to: $output_dir")
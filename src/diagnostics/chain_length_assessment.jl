using Statistics

"""
    assess_chain_length(chain::RJMCMCChain, burn_in::Int; 
                       pixel_size::Real = 0.005, 
                       correlation_threshold::Float64 = 0.95)

Assess chain length adequacy via quartile-based posterior histogram correlation.

# Method
- Remove burn-in samples
- Split remaining chain into 4 quartiles  
- Generate posterior position histogram for each quartile
- Compute pairwise pixel correlations between quartiles
- Well-mixed chain: all correlations > threshold
- Still converging: progressive correlation increase needed

# Arguments
- `chain`: RJMCMC chain to analyze
- `burn_in`: Number of burn-in samples to discard
- `pixel_size`: Pixel size for posterior histogram generation
- `correlation_threshold`: Minimum correlation for convergence (default 0.95)

# Returns
Named tuple with:
- `quartile_correlations`: Matrix of pairwise correlations
- `min_correlation`: Minimum correlation across all pairs
- `max_correlation`: Maximum correlation across all pairs
- `converged`: Whether chain meets correlation threshold
- `recommendation`: Text recommendation for user
"""
function assess_chain_length(chain::RJMCMCChain, burn_in::Int; 
                            pixel_size::Real = 0.005, 
                            correlation_threshold::Float64 = 0.95)
    
    # Remove burn-in samples
    post_burnin_samples = chain.samples[(burn_in+1):end]
    n_post_burnin = length(post_burnin_samples)
    
    if n_post_burnin < 400  # Need at least 100 samples per quartile
        return (
            quartile_correlations = zeros(4, 4),
            min_correlation = 0.0,
            max_correlation = 0.0,
            converged = false,
            recommendation = "Insufficient post-burn-in samples (need ≥400, have $n_post_burnin)",
            status = "insufficient_samples"
        )
    end
    
    # Split into quartiles
    quartile_size = div(n_post_burnin, 4)
    quartiles = [
        post_burnin_samples[1:quartile_size],
        post_burnin_samples[(quartile_size+1):(2*quartile_size)],
        post_burnin_samples[(2*quartile_size+1):(3*quartile_size)],
        post_burnin_samples[(3*quartile_size+1):end]
    ]
    
    # Calculate bounds for consistent image sizing
    # Use all post-burn-in samples to define bounds
    all_coords = extract_all_emitter_coordinates(post_burnin_samples)
    if isempty(all_coords)
        return (
            quartile_correlations = zeros(4, 4),
            min_correlation = 0.0,
            max_correlation = 0.0,
            converged = false,
            recommendation = "No emitters found in post-burn-in samples",
            status = "no_emitters"
        )
    end
    
    x_coords, y_coords = all_coords
    margin = 0.02  # 20 nm margin
    x_min, x_max = extrema(x_coords) .+ (-margin, margin)
    y_min, y_max = extrema(y_coords) .+ (-margin, margin)
    bounds = (x_min, x_max, y_min, y_max)
    
    # Calculate image size
    width = ceil(Int, (x_max - x_min) / pixel_size)
    height = ceil(Int, (y_max - y_min) / pixel_size)
    image_size = (height, width)
    
    # Generate posterior histogram for each quartile
    quartile_images = Matrix{Float64}[]
    
    for quartile_samples in quartiles
        image = zeros(Float64, image_size...)
        
        # Count emitter positions in this quartile
        for sample in quartile_samples
            for emitter in sample.emitters
                # Convert to pixel coordinates (1-indexed)
                px = round(Int, (emitter.x - x_min) / pixel_size) + 1
                py = round(Int, (emitter.y - y_min) / pixel_size) + 1
                
                # Add count if within bounds
                if 1 ≤ px ≤ width && 1 ≤ py ≤ height
                    image[py, px] += 1.0
                end
            end
        end
        
        push!(quartile_images, image)
    end
    
    # Compute pairwise correlations between quartiles
    correlations = zeros(4, 4)
    
    for i in 1:4
        for j in 1:4
            if i == j
                correlations[i, j] = 1.0
            else
                # Flatten images for correlation calculation
                img1_flat = vec(quartile_images[i])
                img2_flat = vec(quartile_images[j])
                
                # Compute Pearson correlation
                if all(img1_flat .== 0) || all(img2_flat .== 0)
                    correlations[i, j] = 0.0  # No correlation if either image empty
                else
                    correlations[i, j] = cor(img1_flat, img2_flat)
                end
            end
        end
    end
    
    # Extract off-diagonal correlations for analysis
    off_diagonal_corrs = Float64[]
    for i in 1:4
        for j in (i+1):4
            push!(off_diagonal_corrs, correlations[i, j])
        end
    end
    
    min_correlation = minimum(off_diagonal_corrs)
    max_correlation = maximum(off_diagonal_corrs)
    mean_correlation = mean(off_diagonal_corrs)
    
    # Convergence assessment
    converged = min_correlation ≥ correlation_threshold
    
    # Generate recommendation
    if converged
        recommendation = "✓ Well-mixed chain (min correlation: $(round(min_correlation, digits=3)))"
        status = "converged"
    elseif mean_correlation > 0.8
        recommendation = "⚠ Still converging - recommend 1.5-2x more samples (min correlation: $(round(min_correlation, digits=3)))"
        status = "nearly_converged"
    else
        recommendation = "✗ Poor mixing - recommend doubling chain length (min correlation: $(round(min_correlation, digits=3)))"
        status = "poor_mixing"
    end
    
    return (
        quartile_correlations = correlations,
        min_correlation = min_correlation,
        max_correlation = max_correlation,
        mean_correlation = mean_correlation,
        converged = converged,
        recommendation = recommendation,
        status = status,
        n_post_burnin = n_post_burnin,
        quartile_sizes = [length(q) for q in quartiles]
    )
end

"""
    assess_chain_length(chains::Vector{RJMCMCChain}, burn_in::Int; kwargs...)

Assess chain length for multiple chains (from spatial partitions).
"""
function assess_chain_length(chains::Vector{RJMCMCChain}, burn_in::Int; kwargs...)
    if isempty(chains)
        error("No chains provided for chain length assessment")
    end
    
    # Analyze each chain individually
    results = [assess_chain_length(chain, burn_in; kwargs...) for chain in chains]
    
    # Summary across all chains
    all_min_corrs = [r.min_correlation for r in results]
    worst_chain_idx = argmin(all_min_corrs)
    worst_min_corr = all_min_corrs[worst_chain_idx]
    
    # Overall convergence status
    all_converged = all(r.converged for r in results)
    
    overall_recommendation = if all_converged
        "✓ All $(length(chains)) partitions well-mixed"
    else
        n_unconverged = sum(.!r.converged for r in results)
        "⚠ $n_unconverged/$(length(chains)) partitions need more samples (worst min correlation: $(round(worst_min_corr, digits=3)))"
    end
    
    return (
        per_chain_results = results,
        worst_chain_index = worst_chain_idx,
        worst_min_correlation = worst_min_corr,
        all_converged = all_converged,
        overall_recommendation = overall_recommendation
    )
end

"""
    extract_all_emitter_coordinates(samples::Vector{BaGoLState}) -> (x_coords, y_coords)

Extract all emitter coordinates from a vector of samples for bounds calculation.
"""
function extract_all_emitter_coordinates(samples::Vector{S}) where {S <: AbstractChainState}
    x_coords = Float64[]
    y_coords = Float64[]
    
    for sample in samples
        for emitter in sample.emitters
            push!(x_coords, emitter.x)
            push!(y_coords, emitter.y)
        end
    end
    
    return (x_coords, y_coords)
end
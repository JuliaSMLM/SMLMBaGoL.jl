function simulate_n_mer(; 
                        n::Int = 8,
                        diameter::Real = 0.2,  # microns
                        photons::Real = 1000,
                        sigma_psf::Real = 0.13,  # microns
                        min_photons::Real = 100,
                        localizations_per_emitter_mean::Real = 6.0,      # mean localizations per emitter
                        localizations_per_emitter_variance::Real = 6.0,  # variance in localizations per emitter
                        center_x::Real = 0.0,
                        center_y::Real = 0.0,
                        prior_K_mean::Real = 5.0,      # Mean of Gamma prior for emitter count
                        prior_K_variance::Real = 5.0,  # Variance of Gamma prior for emitter count  
                        rng::AbstractRNG = Random.GLOBAL_RNG)
    
    # Generate circular n-mer positions
    emitter_positions = generate_circular_positions(n, diameter/2, center_x, center_y)
    
    # Generate localizations for each emitter
    all_localizations = Localization2D{Float64}[]
    frame_counter = 1
    
    # Convert mean/variance to Gamma parameters for localizations per emitter
    loc_alpha = localizations_per_emitter_mean^2 / localizations_per_emitter_variance
    loc_beta = localizations_per_emitter_mean / localizations_per_emitter_variance
    
    for (emitter_idx, (ex, ey)) in enumerate(emitter_positions)
        # Sample number of localizations for this emitter from Gamma distribution
        n_locs_float = rand(rng, Gamma(loc_alpha, 1/loc_beta))  # Note: Distributions.jl uses scale parameterization
        n_locs = max(1, round(Int, n_locs_float))  # Ensure at least 1 localization
        
        for _ in 1:n_locs
            # Sample photon count from exponential distribution
            sampled_photons = rand(rng, Exponential(photons))
            
            # Skip if below threshold
            sampled_photons < min_photons && continue
            
            # Calculate localization precision (Cramér-Rao bound approximation)
            sigma_loc = sigma_psf / sqrt(sampled_photons)
            
            # Add localization noise
            loc_x = ex + sigma_loc * randn(rng)
            loc_y = ey + sigma_loc * randn(rng)
            
            # Create localization with sequential frame numbering
            push!(all_localizations, Localization2D{Float64}(
                loc_x, loc_y, sigma_loc, sigma_loc, frame_counter
            ))
            frame_counter += 1
        end
    end
    
    # Convert mean/variance to Gamma parameters
    prior_K_alpha = prior_K_mean^2 / prior_K_variance  # shape parameter
    prior_K_beta = prior_K_mean / prior_K_variance      # rate parameter
    
    return all_localizations, create_prior_from_params(all_localizations, prior_K_alpha, prior_K_beta)
end

function generate_circular_positions(n::Int, radius::Real, center_x::Real, center_y::Real)
    positions = Vector{Tuple{Float64, Float64}}(undef, n)
    
    for i in 1:n
        angle = 2π * (i - 1) / n
        x = center_x + radius * cos(angle)
        y = center_y + radius * sin(angle)
        positions[i] = (x, y)
    end
    
    return positions
end


function create_prior_from_params(localizations::Vector{<:AbstractLocalization}, 
                                alpha::Real, beta::Real)
    # Create spatial prior from localization bounds
    spatial_prior = create_spatial_prior_from_localizations(localizations, 0.2)
    
    # Create gamma prior for emitter count
    K_prior = GammaPrior(alpha, beta)
    
    return CompoundPrior(spatial_prior, K_prior)
end

function simulate_n_mer_with_prior(prior::AbstractPrior; kwargs...)
    # Extract spatial bounds from prior if CompoundPrior
    if isa(prior, CompoundPrior)
        spatial_prior = prior.spatial_prior
        K_prior = prior.K_prior
        
        # Sample center position from spatial prior
        center_x, center_y = sample_spatial_prior(spatial_prior)
        
        # Extract prior parameters for consistency
        if isa(K_prior, GammaPrior)
            # Convert back to mean/variance for user interface
            prior_mean = K_prior.α / K_prior.β
            prior_variance = K_prior.α / K_prior.β^2
            locs, _ = simulate_n_mer(; center_x=center_x, center_y=center_y, 
                                   prior_K_mean=prior_mean, prior_K_variance=prior_variance, kwargs...)
            return locs, prior
        else
            locs, sim_prior = simulate_n_mer(; center_x=center_x, center_y=center_y, kwargs...)
            return locs, sim_prior
        end
    else
        # Fallback to default center
        return simulate_n_mer(; kwargs...)
    end
end
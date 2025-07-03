"""
Birth and Death moves with correct q_birth using Distributions.jl

These moves add or remove emitters from the model using the proper birth proposal
distribution as a mixture of Gaussians centered at localizations.
"""


"""
Create birth proposal distribution from localizations.
Pre-computes the mixture distribution for efficient sampling and density evaluation.
"""
function create_birth_proposal(localizations::Vector{<:AbstractLocalization})
    N = length(localizations)
    
    # Create component distributions
    components = Vector{MultivariateNormal{Float64}}(undef, N)
    
    for (i, loc) in enumerate(localizations)
        # Mean vector
        μ = [loc.x, loc.y]
        
        # Covariance matrix (diagonal with σx², σy²)
        Σ = [loc.σx^2  0.0
             0.0       loc.σy^2]
        
        components[i] = MultivariateNormal(μ, Σ)
    end
    
    # Equal weights for all components (1/N each)
    weights = fill(1.0/N, N)
    
    # Create mixture model
    mixture = MixtureModel(components, weights)
    
    return BirthProposalDistribution{Float64}(mixture, localizations)
end

# Convenience methods for delegation to underlying mixture
Base.rand(rng::AbstractRNG, d::BirthProposalDistribution) = rand(rng, d.mixture)
Base.rand(d::BirthProposalDistribution) = rand(d.mixture)
Distributions.pdf(d::BirthProposalDistribution, x::AbstractVector) = pdf(d.mixture, x)
Distributions.logpdf(d::BirthProposalDistribution, x::AbstractVector) = logpdf(d.mixture, x)

function propose_move(::Type{Birth}, state::BaGoLState{E,L,T}, chain::RJMCMCChain, rng=Random.GLOBAL_RNG) where {E,L,T}
    # Step 1: Sample position from cached q_birth using Distributions.jl
    position = rand(rng, chain.birth_proposal)  # Returns [x, y]
    x_new, y_new = position[1], position[2]
    
    # Create new emitter at sampled position
    photons = 1000.0  # Default photon count
    new_emitter = E(x_new, y_new, photons)
    
    # Step 2: Sample number of localizations m
    # m ~ 1 + NegativeBinomial(κ, κ/(κ+μ))
    κ = get_concentration_parameter(state.count_prior)
    μ = state.count_prior.μ
    p = κ / (κ + μ)
    
    # Use Distributions.jl NegativeBinomial
    m = 1 + rand(rng, NegativeBinomial(κ, p))
    
    # Step 3: Calculate allocation probabilities
    n_locs = length(state.localizations)
    log_probs = Vector{Float64}(undef, n_locs)
    
    for (i, loc) in enumerate(state.localizations)
        # P(allocate loc i to new emitter) ∝ L(loc_i | new_emitter)
        log_probs[i] = log_likelihood(new_emitter, loc, state.τ²)
    end
    
    # Convert to probabilities
    max_log = maximum(log_probs)
    probs = exp.(log_probs .- max_log)
    probs ./= sum(probs)
    
    # Sample m localizations without replacement
    m_actual = min(m, n_locs)  # Can't allocate more than we have
    selected_indices = StatsBase.sample(rng, 1:n_locs, StatsBase.Weights(probs), m_actual, replace=false)
    
    # Step 4: Create new state
    new_emitters = [state.emitters; new_emitter]
    new_allocations = copy(state.allocations)
    
    # Reallocate selected localizations to new emitter
    new_emitter_idx = length(new_emitters)
    for idx in selected_indices
        new_allocations[idx] = new_emitter_idx
    end
    
    # Initialize latent positions (keep existing ones, sample new ones for reallocated)
    new_latent_positions = copy(state.latent_positions)

    # Sample latent positions for newly allocated localizations
    for idx in selected_indices
        loc = state.localizations[idx]
        
        # Sample latent position given new emitter and observation
        prec_x = 1/state.τ² + 1/loc.σx^2
        prec_y = 1/state.τ² + 1/loc.σy^2
        post_mean_x = (new_emitter.x/state.τ² + loc.x/loc.σx^2) / prec_x
        post_mean_y = (new_emitter.y/state.τ² + loc.y/loc.σy^2) / prec_y
        
        latent_x = post_mean_x + randn(rng) / sqrt(prec_x)
        latent_y = post_mean_y + randn(rng) / sqrt(prec_y)
        
        new_latent_positions[idx] = (latent_x, latent_y)
    end
    
    new_state_temp = BaGoLState(new_emitters, state.localizations, new_allocations,
                               new_latent_positions, state.spatial_prior, state.count_prior, state.τ², state.log_likelihood)
    
    new_likelihood = log_likelihood(new_state_temp)
    
    return BaGoLState(new_emitters, state.localizations, new_allocations,
                     new_latent_positions, state.spatial_prior, state.count_prior, state.τ², new_likelihood)
end

function propose_move(::Type{Death}, state::BaGoLState{E,L,T}, chain::RJMCMCChain, rng=Random.GLOBAL_RNG) where {E,L,T}
    length(state.emitters) == 0 && return nothing
    
    idx = rand(rng, 1:length(state.emitters))
    
    # Before removing, find localizations allocated to this emitter
    allocated_to_removed = [i for i in eachindex(state.allocations) 
                           if state.allocations[i] == idx]
    
    # Remove the emitter
    new_emitters = [state.emitters[i] for i in 1:length(state.emitters) if i != idx]
    new_allocations = reallocate_from_removed(state.allocations, idx, rng)
    
    # If there are remaining emitters and some localizations need reallocation
    if !isempty(new_emitters) && !isempty(allocated_to_removed)
        # Keep track of updated latent positions
        new_latent_positions = copy(state.latent_positions)
        
        # For reallocated localizations, update their latent positions
        for loc_idx in allocated_to_removed
            new_emitter_idx = new_allocations[loc_idx]
            if new_emitter_idx > 0
                loc = state.localizations[loc_idx]
                emitter = new_emitters[new_emitter_idx]
                
                # Sample new latent position given new emitter assignment
                prec_x = 1/state.τ² + 1/loc.σx^2
                prec_y = 1/state.τ² + 1/loc.σy^2
                post_mean_x = (emitter.x/state.τ² + loc.x/loc.σx^2) / prec_x
                post_mean_y = (emitter.y/state.τ² + loc.y/loc.σy^2) / prec_y
                
                latent_x = post_mean_x + randn(rng) / sqrt(prec_x)
                latent_y = post_mean_y + randn(rng) / sqrt(prec_y)
                
                new_latent_positions[loc_idx] = (latent_x, latent_y)
            end
        end
        
        # Track which emitters received reallocated localizations
        affected_emitters = Set{Int}()
        for loc_idx in allocated_to_removed
            new_emitter_idx = new_allocations[loc_idx]
            if 1 ≤ new_emitter_idx ≤ length(new_emitters)
                push!(affected_emitters, new_emitter_idx)
            end
        end
        
        # Update positions only for emitters that received new localizations
        for emitter_idx in affected_emitters
            # Get all latent positions for this emitter
            latent_positions_for_emitter = [new_latent_positions[i] 
                                           for i in eachindex(state.localizations) 
                                           if new_allocations[i] == emitter_idx]
            
            if !isempty(latent_positions_for_emitter)
                # Calculate mean of latent positions
                x_mean = mean(pos[1] for pos in latent_positions_for_emitter)
                y_mean = mean(pos[2] for pos in latent_positions_for_emitter)
                
                # Sample from posterior
                n_j = length(latent_positions_for_emitter)
                x_new = x_mean + randn(rng) * sqrt(state.τ² / n_j)
                y_new = y_mean + randn(rng) * sqrt(state.τ² / n_j)
                
                # Update emitter position
                old_emitter = new_emitters[emitter_idx]
                new_emitters[emitter_idx] = E(x_new, y_new, old_emitter.photons)
            end
        end
    else
        new_latent_positions = state.latent_positions
    end
    
    # Create final state with updated likelihood
    new_likelihood = log_likelihood(BaGoLState(new_emitters, state.localizations, 
                                              new_allocations, new_latent_positions, state.spatial_prior, 
                                              state.count_prior, state.τ², T(0.0)))
    
    return BaGoLState(new_emitters, state.localizations, new_allocations, 
                     new_latent_positions, state.spatial_prior, state.count_prior, state.τ², new_likelihood)
end

# Update acceptance ratio calculations to use the cached distribution
function log_acceptance_ratio(::Type{Birth}, current::BaGoLState, proposed::BaGoLState, chain::RJMCMCChain)
    @assert length(proposed.emitters) == length(current.emitters) + 1
    
    # Find the new emitter (last one in proposed state)
    new_emitter = proposed.emitters[end]
    new_position = [new_emitter.x, new_emitter.y]
    
    # Log density under birth proposal
    log_q_birth = logpdf(chain.birth_proposal, new_position)
    
    # Log probability of death move selecting this emitter
    log_q_death = -log(length(proposed.emitters))  # Uniform selection
    
    # Prior ratio for new emitter position
    log_prior_ratio = log_prior_spatial(new_emitter, current.spatial_prior)
    
    # Likelihood ratio
    log_likelihood_ratio = proposed.log_likelihood - current.log_likelihood
    
    # Note: allocation probabilities cancel in forward/reverse moves
    
    return log_prior_ratio + log_likelihood_ratio + log_q_death - log_q_birth
end

function log_acceptance_ratio(::Type{Death}, current::BaGoLState, proposed::BaGoLState, chain::RJMCMCChain)
    # Find removed emitter by comparing current and proposed
    removed_idx = 0
    for i in 1:length(current.emitters)
        if i > length(proposed.emitters) || 
           current.emitters[i].x != proposed.emitters[i].x ||
           current.emitters[i].y != proposed.emitters[i].y
            removed_idx = i
            break
        end
    end
    
    if removed_idx == 0
        error("Could not identify removed emitter")
    end
    
    removed_emitter = current.emitters[removed_idx]
    removed_position = [removed_emitter.x, removed_emitter.y]
    
    # Log density of removed position under birth proposal  
    log_q_birth = logpdf(chain.birth_proposal, removed_position)
    
    # Log probability of death selecting this emitter
    log_q_death = -log(length(current.emitters))
    
    # Prior ratio (negative because we're removing)
    log_prior_ratio = -log_prior_spatial(removed_emitter, current.spatial_prior)
    
    # Likelihood ratio
    log_likelihood_ratio = proposed.log_likelihood - current.log_likelihood
    
    return log_prior_ratio + log_likelihood_ratio + log_q_birth - log_q_death
end

# Fallback methods that don't use chain (for other parts of code that might call directly)
function log_acceptance_ratio(::Type{Birth}, current::BaGoLState, proposed::BaGoLState)
    # Create temporary birth proposal for evaluation
    birth_proposal = create_birth_proposal(current.localizations)
    
    @assert length(proposed.emitters) == length(current.emitters) + 1
    
    # Find the new emitter (last one in proposed state)
    new_emitter = proposed.emitters[end]
    new_position = [new_emitter.x, new_emitter.y]
    
    # Log density under birth proposal
    log_q_birth = logpdf(birth_proposal, new_position)
    
    # Log probability of death move selecting this emitter
    log_q_death = -log(length(proposed.emitters))  # Uniform selection
    
    # Prior ratio for new emitter position
    log_prior_ratio = log_prior_spatial(new_emitter, current.spatial_prior)
    
    # Likelihood ratio
    log_likelihood_ratio = proposed.log_likelihood - current.log_likelihood
    
    return log_prior_ratio + log_likelihood_ratio + log_q_death - log_q_birth
end

function log_acceptance_ratio(::Type{Death}, current::BaGoLState, proposed::BaGoLState)
    -log_acceptance_ratio(Birth, proposed, current)  # Mathematical inverse
end
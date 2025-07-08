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
    
    # Step 2: Create provisional state with new emitter added
    new_emitters = [state.emitters; new_emitter]
    new_allocations = copy(state.allocations)
    new_latent_positions = copy(state.latent_positions)
    
    # Create intermediate state for allocate and move operations
    intermediate_state = BaGoLState(
        new_emitters,
        state.localizations,
        new_allocations,
        new_latent_positions,
        state.spatial_prior,
        state.count_prior,
        state.τ²,
        state.log_likelihood
    )
    
    # Step 3: Perform full allocate move (Gibbs sweep with Pólya weights)
    # This will naturally allocate some localizations to the new emitter
    allocated_state = propose_move(Allocate, intermediate_state, chain, rng)
    
    # If allocate move failed, return nothing
    if isnothing(allocated_state)
        return nothing
    end
    
    # Step 4: Perform move operation to update all emitter positions
    # This updates positions based on the new allocations
    final_state = propose_move(Move, allocated_state, chain, rng)
    
    # If move operation failed, return nothing
    if isnothing(final_state)
        return nothing
    end
    
    # The likelihood has already been calculated in the move operation
    return final_state
end

function propose_move(::Type{Death}, state::BaGoLState{E,L,T}, chain::RJMCMCChain, rng=Random.GLOBAL_RNG) where {E,L,T}
    length(state.emitters) == 0 && return nothing
    
    # Step 1: Select emitter to remove
    idx = rand(rng, 1:length(state.emitters))
    
    # IMPORTANT: Store the removed emitter position for acceptance ratio calculation
    removed_emitter = state.emitters[idx]
    
    # Step 2: Create provisional state with emitter removed
    new_emitters = [state.emitters[i] for i in 1:length(state.emitters) if i != idx]
    
    # Adjust allocations: shift indices down for emitters after the removed one
    new_allocations = copy(state.allocations)
    for i in eachindex(new_allocations)
        if new_allocations[i] == idx
            # This localization was allocated to the removed emitter
            # Will be handled by the allocate move
            new_allocations[i] = 0  # Temporarily unallocated
        elseif new_allocations[i] > idx
            # Shift down indices for emitters after the removed one
            new_allocations[i] -= 1
        end
    end
    
    # Create intermediate state
    intermediate_state = BaGoLState(
        new_emitters,
        state.localizations,
        new_allocations,
        copy(state.latent_positions),
        state.spatial_prior,
        state.count_prior,
        state.τ²,
        state.log_likelihood
    )
    
    # Step 3: Perform full allocate move (Gibbs sweep with Pólya weights)
    # This will reallocate the orphaned localizations to remaining emitters
    allocated_state = propose_move(Allocate, intermediate_state, chain, rng)
    
    # If allocate move failed or no emitters left, return the allocated state
    if isnothing(allocated_state) || isempty(allocated_state.emitters)
        return allocated_state
    end
    
    # Step 4: Perform move operation to update all emitter positions
    # This updates positions based on the new allocations
    final_state = propose_move(Move, allocated_state, chain, rng)
    
    # Store the removed emitter in a way that the acceptance ratio can access it
    # We'll add it as metadata to the chain
    chain.last_removed_emitter = removed_emitter
    
    # The likelihood has already been calculated in the move operation
    return final_state
end

# Update acceptance ratio calculations to use the cached distribution
function log_acceptance_ratio(::Type{Birth}, current::BaGoLState, proposed::BaGoLState, chain::RJMCMCChain)
    # Handle case where birth move didn't actually add an emitter (due to allocate removing it)
    if length(proposed.emitters) == length(current.emitters)
        # No change in emitter count - reject this move
        return -Inf
    end
    
    # In rare cases, allocate might create or remove additional emitters
    if length(proposed.emitters) != length(current.emitters) + 1
        # Birth didn't result in exactly one new emitter - handle gracefully
        # This can happen if allocate removes empty emitters
        # For now, use a simple heuristic based on the change
        delta_k = length(proposed.emitters) - length(current.emitters)
        if delta_k <= 0
            return -Inf  # Reject if no net increase
        end
        # Otherwise proceed with the calculation using the actual change
    end
    
    # For birth moves with allocate+move, we need to consider:
    # 1. The originally sampled position (which may have moved)
    # 2. Allocate and Move are Gibbs moves (always accepted)
    
    # Since we can't track the original position through allocate+move,
    # we use the final position as an approximation
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
    
    # Prior on k given N total localizations
    N = length(current.localizations)
    k_current = length(current.emitters)
    k_proposed = length(proposed.emitters)
    
    # Extract hyperparameters from count prior
    μ = chain.count_prior.μ
    κ = chain.count_prior.κ
    
    log_prior_k_ratio = log_prior_k_given_N(k_proposed, N, μ, κ) - 
                        log_prior_k_given_N(k_current, N, μ, κ)
    
    return log_prior_ratio + log_likelihood_ratio + log_q_death - log_q_birth + log_prior_k_ratio
end

function log_acceptance_ratio(::Type{Death}, current::BaGoLState, proposed::BaGoLState, chain::RJMCMCChain)
    # Handle edge cases where death move might not actually remove an emitter
    if length(proposed.emitters) == length(current.emitters)
        # No change in emitter count - reject this move
        return -Inf
    end
    
    # Handle case where allocate removed additional emitters
    if length(proposed.emitters) < length(current.emitters) - 1
        # More than one emitter removed - this is OK but we need to handle it
        # Use the tracked removed emitter for the original death
        if isnothing(chain.last_removed_emitter)
            error("No removed emitter tracked for death move acceptance ratio")
        end
        
        removed_emitter = chain.last_removed_emitter
        removed_position = [removed_emitter.x, removed_emitter.y]
        
        # Calculate acceptance ratio for removing one emitter
        log_q_birth = logpdf(chain.birth_proposal, removed_position)
        log_q_death = -log(length(current.emitters))
        log_prior_ratio = -log_prior_spatial(removed_emitter, current.spatial_prior)
        log_likelihood_ratio = proposed.log_likelihood - current.log_likelihood
        
        N = length(current.localizations)
        k_current = length(current.emitters)
        k_proposed = length(proposed.emitters)
        μ = chain.count_prior.μ
        κ = chain.count_prior.κ
        log_prior_k_ratio = log_prior_k_given_N(k_proposed, N, μ, κ) - 
                            log_prior_k_given_N(k_current, N, μ, κ)
        
        chain.last_removed_emitter = nothing
        return log_prior_ratio + log_likelihood_ratio + log_q_birth - log_q_death + log_prior_k_ratio
    end
    
    @assert length(proposed.emitters) == length(current.emitters) - 1
    
    # Use the tracked removed emitter
    if isnothing(chain.last_removed_emitter)
        error("No removed emitter tracked for death move acceptance ratio")
    end
    
    removed_emitter = chain.last_removed_emitter
    removed_position = [removed_emitter.x, removed_emitter.y]
    
    # Log density of removed position under birth proposal
    # This is the position BEFORE allocate+move operations
    log_q_birth = logpdf(chain.birth_proposal, removed_position)
    
    # Log probability of death selecting this emitter (uniform)
    log_q_death = -log(length(current.emitters))
    
    # Prior ratio (negative because we're removing)
    log_prior_ratio = -log_prior_spatial(removed_emitter, current.spatial_prior)
    
    # Likelihood ratio
    log_likelihood_ratio = proposed.log_likelihood - current.log_likelihood
    
    # Prior on k given N total localizations
    N = length(current.localizations)
    k_current = length(current.emitters)
    k_proposed = length(proposed.emitters)
    
    # Extract hyperparameters from count prior
    μ = chain.count_prior.μ
    κ = chain.count_prior.κ
    
    log_prior_k_ratio = log_prior_k_given_N(k_proposed, N, μ, κ) - 
                        log_prior_k_given_N(k_current, N, μ, κ)
    
    # Clear the tracked emitter after use
    chain.last_removed_emitter = nothing
    
    return log_prior_ratio + log_likelihood_ratio + log_q_birth - log_q_death + log_prior_k_ratio
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
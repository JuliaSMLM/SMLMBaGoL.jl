# RJMCMC move proposals for BaGoL

"""
Propose birth of a new emitter at the position of a random localization.
This informed proposal gives the new emitter a good starting position.
"""
function propose_birth!(
    chain::RJMCMCChain,
    locs::Vector{<:SMLMData.AbstractEmitter},
    spatial_prior::UniformSpatialPrior
)
    state = chain.current_state
    config = chain.config
    k = length(state.emitters)
    n_locs = length(locs)

    if n_locs == 0
        return false
    end

    # Sample position at a random localization (informed proposal)
    loc_idx = rand(1:n_locs)
    loc = locs[loc_idx]
    x, y = loc.x, loc.y

    new_emitter = Emitter(x, y)

    # Log prior ratio: P(K+1)/P(K)
    log_prior_ratio = log_prior_k(k + 1, config.λ_K) - log_prior_k(k, config.λ_K)

    # Proposal ratio for informed birth/death
    # Birth: pick random localization (1/n_locs)
    # Death: pick random emitter (1/(k+1))
    log_proposal_ratio = log(n_locs) - log(k + 1)

    # Count prior for new emitter with 0 localizations (will get allocations via allocate)
    log_count_prior = log_prior_count(0, chain.μ, config.α)

    log_acceptance = log_prior_ratio + log_proposal_ratio + log_count_prior

    if log(rand()) < log_acceptance
        push!(state.emitters, new_emitter)
        return true
    end
    return false
end

"""
Propose death of an existing emitter. Returns true if accepted.
"""
function propose_death!(
    chain::RJMCMCChain,
    locs::Vector{<:SMLMData.AbstractEmitter},
    spatial_prior::UniformSpatialPrior
)
    state = chain.current_state
    config = chain.config
    k = length(state.emitters)

    if k == 0
        return false
    end

    # Pick random emitter to kill
    idx = rand(1:k)
    emitter = state.emitters[idx]
    n_allocated = length(emitter.allocated)

    # Log prior ratio: P(K-1)/P(K)
    log_prior_ratio = log_prior_k(k - 1, config.λ_K) - log_prior_k(k, config.λ_K)

    # Proposal ratio: birth prob / death prob
    log_proposal_ratio = log(k) - log(area(spatial_prior))

    # Lose count prior for this emitter
    log_count_prior = -log_prior_count(n_allocated, chain.μ, config.α)

    # Lose likelihood for allocated localizations
    log_ll_loss = -log_likelihood_emitter(locs, emitter, config.τ)

    log_acceptance = log_prior_ratio + log_proposal_ratio + log_count_prior + log_ll_loss

    if log(rand()) < log_acceptance
        # Unallocate the localizations (will be reallocated in allocate step)
        deleteat!(state.emitters, idx)
        return true
    end
    return false
end

"""
Propose moving an emitter position. Returns true if accepted.
"""
function propose_move!(
    chain::RJMCMCChain,
    locs::Vector{<:SMLMData.AbstractEmitter},
    spatial_prior::UniformSpatialPrior
)
    state = chain.current_state
    config = chain.config

    if isempty(state.emitters)
        return false
    end

    # Pick random emitter
    idx = rand(1:length(state.emitters))
    emitter = state.emitters[idx]

    # Propose new position with Gaussian perturbation
    x_new = emitter.x + randn() * config.move_σ
    y_new = emitter.y + randn() * config.move_σ

    # Check spatial prior
    if log_spatial_prior(x_new, y_new, spatial_prior) == -Inf
        return false
    end

    # Compute likelihood change
    old_ll = log_likelihood_emitter(locs, emitter, config.τ)

    # Temporarily move emitter
    x_old, y_old = emitter.x, emitter.y
    emitter.x, emitter.y = x_new, y_new

    new_ll = log_likelihood_emitter(locs, emitter, config.τ)

    log_acceptance = new_ll - old_ll

    if log(rand()) < log_acceptance
        return true
    else
        # Revert
        emitter.x, emitter.y = x_old, y_old
        return false
    end
end

"""
Propose reallocating a localization to a different emitter. Returns true if accepted.
"""
function propose_allocate!(
    chain::RJMCMCChain,
    locs::Vector{<:SMLMData.AbstractEmitter},
    spatial_prior::UniformSpatialPrior
)
    state = chain.current_state
    config = chain.config
    n_locs = length(locs)
    k = length(state.emitters)

    if k == 0
        return false
    end

    # Pick random localization
    loc_idx = rand(1:n_locs)
    loc = locs[loc_idx]

    # Find current assignment
    current_emitter_idx = 0
    for (i, emitter) in enumerate(state.emitters)
        if loc_idx in emitter.allocated
            current_emitter_idx = i
            break
        end
    end

    # If not currently allocated, assign to nearest emitter (always accept)
    if current_emitter_idx == 0
        # Find nearest emitter
        best_dist = Inf
        best_idx = 1
        for (j, emitter) in enumerate(state.emitters)
            dist = (loc.x - emitter.x)^2 + (loc.y - emitter.y)^2
            if dist < best_dist
                best_dist = dist
                best_idx = j
            end
        end
        push!(state.emitters[best_idx].allocated, loc_idx)
        return true
    end

    # Pick new emitter (could be same)
    new_emitter_idx = rand(1:k)

    if new_emitter_idx == current_emitter_idx
        return false
    end

    new_emitter = state.emitters[new_emitter_idx]
    current_emitter = state.emitters[current_emitter_idx]

    # Compute likelihood change
    old_ll = log_likelihood_single(loc, current_emitter, config.τ)
    new_ll = log_likelihood_single(loc, new_emitter, config.τ)

    # Compute count prior change
    n_old_before = length(current_emitter.allocated)
    n_new_before = length(new_emitter.allocated)

    log_count_old = log_prior_count(n_old_before - 1, chain.μ, config.α) +
                    log_prior_count(n_new_before + 1, chain.μ, config.α) -
                    log_prior_count(n_old_before, chain.μ, config.α) -
                    log_prior_count(n_new_before, chain.μ, config.α)

    log_acceptance = (new_ll - old_ll) + log_count_old

    if log(rand()) < log_acceptance
        # Remove from old emitter
        filter!(x -> x != loc_idx, current_emitter.allocated)
        # Add to new emitter
        push!(new_emitter.allocated, loc_idx)
        return true
    end
    return false
end

"""
Initialize allocations: assign each localization to nearest emitter.
"""
function initialize_allocations!(
    state::BaGoLState,
    locs::Vector{<:SMLMData.AbstractEmitter}
)
    # Clear existing allocations
    for emitter in state.emitters
        empty!(emitter.allocated)
    end

    if isempty(state.emitters)
        return
    end

    # Assign each localization to nearest emitter
    for (i, loc) in enumerate(locs)
        best_dist = Inf
        best_idx = 1
        for (j, emitter) in enumerate(state.emitters)
            dist = (loc.x - emitter.x)^2 + (loc.y - emitter.y)^2
            if dist < best_dist
                best_dist = dist
                best_idx = j
            end
        end
        push!(state.emitters[best_idx].allocated, i)
    end
end

"""
Update emitter positions to be centroid of allocated localizations (MLE).
"""
function update_emitter_positions!(
    state::BaGoLState,
    locs::Vector{<:SMLMData.AbstractEmitter},
    τ::Float64
)
    for emitter in state.emitters
        if isempty(emitter.allocated)
            continue
        end

        # Weighted centroid using precision (1/variance)
        sum_wx = 0.0
        sum_wy = 0.0
        sum_w = 0.0

        for idx in emitter.allocated
            loc = locs[idx]
            var_x = loc.σ_x^2 + τ^2
            var_y = loc.σ_y^2 + τ^2
            w = 1 / sqrt(var_x * var_y)
            sum_wx += w * loc.x
            sum_wy += w * loc.y
            sum_w += w
        end

        if sum_w > 0
            emitter.x = sum_wx / sum_w
            emitter.y = sum_wy / sum_w
        end
    end
end

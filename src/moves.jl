# RJMCMC move proposals for BaGoL

# ============================================================================
# Proposal distribution: mixture of Gaussians from localizations
# ============================================================================

"""
Sample position from mixture of Gaussians centered at localizations.
q(x,y) = (1/N) Σᵢ N((x,y) | (xᵢ,yᵢ), σᵢ² + τ²)
"""
function sample_from_mixture(locs::Vector{<:SMLMData.AbstractEmitter}, τ::Float64)
    i = rand(1:length(locs))
    loc = locs[i]
    σ_x = sqrt(loc.σ_x^2 + τ^2)
    σ_y = sqrt(loc.σ_y^2 + τ^2)
    x = loc.x + randn() * σ_x
    y = loc.y + randn() * σ_y
    return x, y
end

"""
Log density of mixture of Gaussians at (x, y).
"""
function log_mixture_density(x::Float64, y::Float64,
                            locs::Vector{<:SMLMData.AbstractEmitter}, τ::Float64)
    n = length(locs)
    if n == 0
        return -Inf
    end

    log_components = Vector{Float64}(undef, n)
    for (i, loc) in enumerate(locs)
        var_x = loc.σ_x^2 + τ^2
        var_y = loc.σ_y^2 + τ^2
        log_components[i] = -0.5 * log(2π * var_x) - 0.5 * log(2π * var_y) -
                            0.5 * (x - loc.x)^2 / var_x - 0.5 * (y - loc.y)^2 / var_y
    end

    # logsumexp for numerical stability
    max_lc = maximum(log_components)
    return max_lc + log(sum(exp.(log_components .- max_lc))) - log(n)
end

# ============================================================================
# Allocation helpers
# ============================================================================

"""
Assign each localization to nearest emitter.
"""
function allocate_to_nearest!(state::BaGoLState, locs::Vector{<:SMLMData.AbstractEmitter})
    for e in state.emitters
        empty!(e.allocated)
    end

    if isempty(state.emitters)
        return
    end

    for (i, loc) in enumerate(locs)
        best_j = 1
        best_d = Inf
        for (j, e) in enumerate(state.emitters)
            d = (loc.x - e.x)^2 + (loc.y - e.y)^2
            if d < best_d
                best_d = d
                best_j = j
            end
        end
        push!(state.emitters[best_j].allocated, i)
    end
end

"""
Remove emitters with zero allocations.
"""
function remove_empty_emitters!(state::BaGoLState)
    filter!(e -> !isempty(e.allocated), state.emitters)
end

# ============================================================================
# Birth move
# ============================================================================

"""
Compute total log posterior: K prior + count priors + likelihood.
"""
function compute_log_posterior(
    state::BaGoLState,
    locs::Vector{<:SMLMData.AbstractEmitter},
    μ::Float64,
    config::RJMCMCConfig
)
    k = length(state.emitters)

    # K prior
    log_post = log_prior_k(k, config.λ_K)

    # Count priors and likelihood for each emitter
    for emitter in state.emitters
        n_alloc = length(emitter.allocated)
        log_post += log_prior_count(n_alloc, μ, config.α)
        log_post += log_likelihood_emitter(locs, emitter, config.τ)
    end

    return log_post
end

"""
Birth: sample from mixture q(x,y), add emitter, allocate, cleanup.
Computes full posterior ratio for acceptance.
"""
function propose_birth!(
    chain::RJMCMCChain,
    locs::Vector{<:SMLMData.AbstractEmitter},
    spatial_prior::UniformSpatialPrior
)
    state = chain.current_state
    config = chain.config
    k = length(state.emitters)
    n = length(locs)

    if n == 0
        return false
    end

    # Save old state
    old_emitters = deepcopy(state.emitters)
    old_log_post = compute_log_posterior(state, locs, chain.μ, config)

    # Sample position from mixture
    x, y = sample_from_mixture(locs, config.τ)

    # Check bounds
    if x < spatial_prior.x_min || x > spatial_prior.x_max ||
       y < spatial_prior.y_min || y > spatial_prior.y_max
        return false
    end

    log_q = log_mixture_density(x, y, locs, config.τ)

    # Add emitter, allocate, update positions, cleanup
    push!(state.emitters, Emitter(x, y))
    allocate_to_nearest!(state, locs)
    update_emitter_positions!(state, locs, config.τ)
    remove_empty_emitters!(state)

    k_new = length(state.emitters)

    # If new emitter was removed (got 0 allocations), birth failed
    if k_new <= k
        state.emitters = old_emitters
        return false
    end

    # Compute new posterior
    new_log_post = compute_log_posterior(state, locs, chain.μ, config)

    # Posterior ratio
    log_post_ratio = new_log_post - old_log_post

    # Proposal ratio: forward q(x,y), reverse 1/k_new
    log_proposal_ratio = -log(k_new) - log_q

    log_acceptance = log_post_ratio + log_proposal_ratio

    if log(rand()) < log_acceptance
        return true
    else
        state.emitters = old_emitters
        return false
    end
end

# ============================================================================
# Death move
# ============================================================================

"""
Death: pick emitter uniformly, use q(x,y) for detailed balance, reallocate, cleanup.
Computes full posterior ratio for acceptance.
Never removes the last emitter (K >= 1 always).
"""
function propose_death!(
    chain::RJMCMCChain,
    locs::Vector{<:SMLMData.AbstractEmitter},
    spatial_prior::UniformSpatialPrior
)
    state = chain.current_state
    config = chain.config
    k = length(state.emitters)

    # Never remove last emitter
    if k <= 1
        return false
    end

    # Save old state
    old_emitters = deepcopy(state.emitters)
    old_log_post = compute_log_posterior(state, locs, chain.μ, config)

    # Pick emitter to kill
    idx = rand(1:k)
    emitter = state.emitters[idx]

    # Mixture density at emitter position for detailed balance
    log_q = log_mixture_density(emitter.x, emitter.y, locs, config.τ)

    # Remove emitter, reallocate, update positions, cleanup
    deleteat!(state.emitters, idx)
    allocate_to_nearest!(state, locs)
    update_emitter_positions!(state, locs, config.τ)
    remove_empty_emitters!(state)

    k_new = length(state.emitters)

    # Compute new posterior
    new_log_post = compute_log_posterior(state, locs, chain.μ, config)

    # Posterior ratio
    log_post_ratio = new_log_post - old_log_post

    # Proposal ratio: forward 1/k, reverse q(x,y)
    log_proposal_ratio = log(k) + log_q

    log_acceptance = log_post_ratio + log_proposal_ratio

    if log(rand()) < log_acceptance
        return true
    else
        state.emitters = old_emitters
        return false
    end
end

# ============================================================================
# Allocate move
# ============================================================================

"""
Allocate: Gibbs sample one localization's assignment.
"""
function propose_allocate!(
    chain::RJMCMCChain,
    locs::Vector{<:SMLMData.AbstractEmitter},
    spatial_prior::UniformSpatialPrior
)
    state = chain.current_state
    config = chain.config
    k = length(state.emitters)
    n = length(locs)

    if k == 0 || n == 0
        return false
    end

    # Pick random localization
    loc_idx = rand(1:n)
    loc = locs[loc_idx]

    # Find current owner
    current_j = 0
    for (j, e) in enumerate(state.emitters)
        if loc_idx in e.allocated
            current_j = j
            break
        end
    end

    # Compute log probability for each emitter (likelihood + count prior change)
    log_probs = Vector{Float64}(undef, k)
    for (j, e) in enumerate(state.emitters)
        log_ll = log_likelihood_single(loc, e, config.τ)

        # Count prior contribution
        n_j = length(e.allocated)
        if j == current_j
            # Already has this loc, count stays same
            log_count = 0.0
        else
            # Would gain this loc
            log_count = log_prior_count(n_j + 1, chain.μ, config.α) -
                       log_prior_count(n_j, chain.μ, config.α)
        end

        log_probs[j] = log_ll + log_count
    end

    # Sample from softmax
    max_lp = maximum(log_probs)
    probs = exp.(log_probs .- max_lp)
    probs ./= sum(probs)

    new_j = 1
    u = rand()
    cumsum = 0.0
    for (j, p) in enumerate(probs)
        cumsum += p
        if u < cumsum
            new_j = j
            break
        end
    end

    if new_j == current_j
        return false
    end

    # Move allocation
    if current_j > 0
        filter!(x -> x != loc_idx, state.emitters[current_j].allocated)
    end
    push!(state.emitters[new_j].allocated, loc_idx)

    # Cleanup empty emitters
    remove_empty_emitters!(state)

    return true
end

# ============================================================================
# Move (position perturbation)
# ============================================================================

"""
Move: Gaussian perturbation of emitter position.
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

    idx = rand(1:length(state.emitters))
    emitter = state.emitters[idx]

    # Propose new position
    x_new = emitter.x + randn() * config.move_σ
    y_new = emitter.y + randn() * config.move_σ

    # Check bounds
    if x_new < spatial_prior.x_min || x_new > spatial_prior.x_max ||
       y_new < spatial_prior.y_min || y_new > spatial_prior.y_max
        return false
    end

    # Likelihood ratio
    old_ll = log_likelihood_emitter(locs, emitter, config.τ)
    x_old, y_old = emitter.x, emitter.y
    emitter.x, emitter.y = x_new, y_new
    new_ll = log_likelihood_emitter(locs, emitter, config.τ)

    if log(rand()) < new_ll - old_ll
        return true
    else
        emitter.x, emitter.y = x_old, y_old
        return false
    end
end

# ============================================================================
# Initialization
# ============================================================================

"""
Initialize allocations: assign each localization to nearest emitter.
"""
function initialize_allocations!(
    state::BaGoLState,
    locs::Vector{<:SMLMData.AbstractEmitter}
)
    allocate_to_nearest!(state, locs)
end

"""
Update emitter positions to weighted centroids of allocated localizations.
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

        sum_wx, sum_wy, sum_w = 0.0, 0.0, 0.0
        for idx in emitter.allocated
            loc = locs[idx]
            var_x = loc.σ_x^2 + τ^2
            var_y = loc.σ_y^2 + τ^2
            w = 1.0 / sqrt(var_x * var_y)
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

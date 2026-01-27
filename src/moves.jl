# RJMCMC move proposals for BaGoL

# ============================================================================
# Proposal distribution: mixture of Gaussians from localizations
# ============================================================================

"""
Sample position from mixture of Gaussians centered at localizations.
q(x,y) = (1/N) Σᵢ N((x,y) | (xᵢ,yᵢ), σᵢ²)
"""
function sample_from_mixture(locs::Vector{<:SMLMData.AbstractEmitter})
    i = rand(1:length(locs))
    loc = locs[i]
    x = loc.x + randn() * loc.σ_x
    y = loc.y + randn() * loc.σ_y
    return x, y
end

"""
Log density of mixture of Gaussians at (x, y).
Optimized single-pass with reduced operations.
"""
function log_mixture_density(x::Real, y::Real,
                            locs::Vector{E}) where E<:SMLMData.AbstractEmitter
    n = length(locs)
    n == 0 && return -Inf

    # First element
    @inbounds loc = locs[1]
    σx, σy = loc.σ_x, loc.σ_y
    dx, dy = x - loc.x, y - loc.y
    # log(2π) ≈ 1.8378770664093453
    max_lc = -(log(σx) + log(σy) + 0.5*dx*dx/(σx*σx) + 0.5*dy*dy/(σy*σy) + 1.8378770664093453)
    sum_exp = 1.0

    @inbounds for i in 2:n
        loc = locs[i]
        σx, σy = loc.σ_x, loc.σ_y
        dx, dy = x - loc.x, y - loc.y
        lc = -(log(σx) + log(σy) + 0.5*dx*dx/(σx*σx) + 0.5*dy*dy/(σy*σy) + 1.8378770664093453)

        if lc > max_lc
            sum_exp = sum_exp * exp(max_lc - lc) + 1.0
            max_lc = lc
        else
            sum_exp += exp(lc - max_lc)
        end
    end

    return max_lc + log(sum_exp) - log(n)
end

# ============================================================================
# Gibbs position sampling
# ============================================================================

"""
Sample emitter position from posterior given allocated localizations.

Given allocations, the posterior for emitter position is Gaussian:
- Posterior precision: Λ_post = Σ_i Λ_i
- Posterior mean: μ_post = Λ_post⁻¹ · (Σ_i Λ_i · μ_i)

Returns (x_new, y_new) sampled from N(μ_post, Σ_post).
"""
function sample_position_gibbs(
    emitter::Emitter{T},
    locs::Vector{<:SMLMData.AbstractEmitter}
) where T
    if isempty(emitter.allocated)
        return emitter.x, emitter.y
    end

    # Sum precision matrices and precision-weighted positions
    Λ_xx, Λ_xy, Λ_yy = 0.0, 0.0, 0.0
    η_x, η_y = 0.0, 0.0

    for idx in emitter.allocated
        loc = locs[idx]
        var_x = loc.σ_x^2
        var_y = loc.σ_y^2
        cov_xy = get_cov_xy(loc)
        det = var_x * var_y - cov_xy^2

        if det <= 0
            # Fallback for degenerate covariance
            det = var_x * var_y
            cov_xy = 0.0
        end

        # Precision matrix elements: Λ = Σ⁻¹
        λ_xx = var_y / det
        λ_yy = var_x / det
        λ_xy = -cov_xy / det

        Λ_xx += λ_xx
        Λ_xy += λ_xy
        Λ_yy += λ_yy

        η_x += λ_xx * loc.x + λ_xy * loc.y
        η_y += λ_xy * loc.x + λ_yy * loc.y
    end

    # Invert posterior precision to get covariance
    det_post = Λ_xx * Λ_yy - Λ_xy^2
    if det_post <= 0
        # Fallback: return current position
        return emitter.x, emitter.y
    end

    Σ_xx = Λ_yy / det_post
    Σ_yy = Λ_xx / det_post
    Σ_xy = -Λ_xy / det_post

    # Posterior mean: μ = Σ · η
    μ_x = Σ_xx * η_x + Σ_xy * η_y
    μ_y = Σ_xy * η_x + Σ_yy * η_y

    # Sample from N(μ, Σ) using Cholesky decomposition
    # Σ = L · Lᵀ where L is lower triangular
    L_xx = sqrt(Σ_xx)
    L_yx = Σ_xy / L_xx
    L_yy_sq = Σ_yy - L_yx^2
    L_yy = L_yy_sq > 0 ? sqrt(L_yy_sq) : 0.0

    z1, z2 = randn(), randn()
    x_new = μ_x + L_xx * z1
    y_new = μ_y + L_yx * z1 + L_yy * z2

    return T(x_new), T(y_new)
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
    chain::RJMCMCChain
)
    k = length(state.emitters)
    config = chain.config

    # K prior
    log_post = log_prior_k(k, config.λ_K)

    # Count priors and likelihood for each emitter
    for emitter in state.emitters
        n_alloc = length(emitter.allocated)
        log_post += log_prior_count(n_alloc, chain.μ, chain.α)
        log_post += log_likelihood_emitter(locs, emitter)
    end

    return log_post
end

"""
Birth: sample from mixture q(x,y), add emitter, allocate, cleanup.
Computes full posterior ratio for acceptance.
"""
function propose_birth!(
    chain::RJMCMCChain{T},
    locs::Vector{<:SMLMData.AbstractEmitter},
    spatial_prior::UniformSpatialPrior
) where T
    state = chain.current_state
    config = chain.config
    k = length(state.emitters)
    n = length(locs)

    if n == 0
        return false
    end

    # Save old state
    old_emitters = deepcopy(state.emitters)
    old_log_post = compute_log_posterior(state, locs, chain)

    # Sample position from mixture
    x, y = sample_from_mixture(locs)

    # Check bounds
    if x < spatial_prior.x_min || x > spatial_prior.x_max ||
       y < spatial_prior.y_min || y > spatial_prior.y_max
        return false
    end

    log_q = log_mixture_density(x, y, locs)

    # Add emitter, allocate, update positions, cleanup
    push!(state.emitters, Emitter(T(x), T(y)))
    allocate_to_nearest!(state, locs)
    update_emitter_positions!(state, locs)
    remove_empty_emitters!(state)

    k_new = length(state.emitters)

    # If new emitter was removed (got 0 allocations), birth failed
    if k_new <= k
        state.emitters = old_emitters
        return false
    end

    # Compute new posterior
    new_log_post = compute_log_posterior(state, locs, chain)

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
    old_log_post = compute_log_posterior(state, locs, chain)

    # Pick emitter to kill
    idx = rand(1:k)
    emitter = state.emitters[idx]

    # Mixture density at emitter position for detailed balance
    log_q = log_mixture_density(emitter.x, emitter.y, locs)

    # Remove emitter, reallocate, update positions, cleanup
    deleteat!(state.emitters, idx)
    allocate_to_nearest!(state, locs)
    update_emitter_positions!(state, locs)
    remove_empty_emitters!(state)

    k_new = length(state.emitters)

    # Compute new posterior
    new_log_post = compute_log_posterior(state, locs, chain)

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
        log_ll = log_likelihood_single(loc, e)

        # Count prior contribution
        n_j = length(e.allocated)
        if j == current_j
            # Already has this loc, count stays same
            log_count = 0.0
        else
            # Would gain this loc
            log_count = log_prior_count(n_j + 1, chain.μ, chain.α) -
                       log_prior_count(n_j, chain.μ, chain.α)
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

    # Gibbs sample positions for affected emitters
    # Update the emitter that gained a localization
    state.emitters[new_j].x, state.emitters[new_j].y =
        sample_position_gibbs(state.emitters[new_j], locs)

    # Update the emitter that lost a localization (if it still has some)
    if current_j > 0 && !isempty(state.emitters[current_j].allocated)
        state.emitters[current_j].x, state.emitters[current_j].y =
            sample_position_gibbs(state.emitters[current_j], locs)
    end

    # Cleanup empty emitters
    remove_empty_emitters!(state)

    return true
end

# ============================================================================
# Move (Gibbs position update)
# ============================================================================

"""
Move: Gibbs sample emitter position from posterior given allocations.
"""
function propose_move!(
    chain::RJMCMCChain{T},
    locs::Vector{<:SMLMData.AbstractEmitter},
    spatial_prior::UniformSpatialPrior
) where T
    state = chain.current_state

    if isempty(state.emitters)
        return false
    end

    idx = rand(1:length(state.emitters))
    emitter = state.emitters[idx]

    if isempty(emitter.allocated)
        return false
    end

    # Gibbs sample new position
    x_new, y_new = sample_position_gibbs(emitter, locs)

    # Check bounds - reject if outside (truncated posterior)
    if x_new < spatial_prior.x_min || x_new > spatial_prior.x_max ||
       y_new < spatial_prior.y_min || y_new > spatial_prior.y_max
        return false
    end

    emitter.x, emitter.y = x_new, y_new
    return true
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
Update emitter positions by Gibbs sampling from posterior given allocations.
"""
function update_emitter_positions!(
    state::BaGoLState{T},
    locs::Vector{<:SMLMData.AbstractEmitter}
) where T
    for emitter in state.emitters
        if !isempty(emitter.allocated)
            emitter.x, emitter.y = sample_position_gibbs(emitter, locs)
        end
    end
end

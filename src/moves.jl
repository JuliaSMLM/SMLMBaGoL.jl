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
Assign each localization to nearest emitter (deterministic).
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
Gibbs sample all localization assignments based on likelihood.
Each loc assigned to emitter j with P ∝ exp(log_likelihood(loc, emitter_j)).
"""
function allocate_gibbs!(state::BaGoLState, locs::Vector{<:SMLMData.AbstractEmitter})
    for e in state.emitters
        empty!(e.allocated)
    end

    k = length(state.emitters)
    if k == 0
        return
    end

    log_probs = Vector{Float64}(undef, k)

    for (i, loc) in enumerate(locs)
        # Compute log likelihood for each emitter
        for (j, e) in enumerate(state.emitters)
            log_probs[j] = log_likelihood_single(loc, e)
        end

        # Sample from softmax
        max_lp = maximum(log_probs)
        probs = exp.(log_probs .- max_lp)
        probs ./= sum(probs)

        # Sample assignment
        u = rand()
        cumsum = 0.0
        assigned_j = k  # default to last
        for (j, p) in enumerate(probs)
            cumsum += p
            if u < cumsum
                assigned_j = j
                break
            end
        end

        push!(state.emitters[assigned_j].allocated, i)
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
Compute total log posterior: K prior + spatial prior + marginal count prior + likelihood.

Uses the marginal count prior P(N | K) = Gamma(N; K*shape, μ/shape) from
Fazel et al. (2022), which correctly accounts for the constraint that
individual counts must sum to N.

The spatial prior P(θ|K) = (1/A)^K for K emitters uniformly distributed over
area A. This term is essential for proper detailed balance with uniform
birth/death proposals - the A terms cancel in the acceptance ratio.
"""
function compute_log_posterior(
    state::BaGoLState,
    locs::Vector{<:SMLMData.AbstractEmitter},
    chain::RJMCMCChain,
    spatial_prior::UniformSpatialPrior
)
    k = length(state.emitters)
    config = chain.config
    N = length(locs)

    # K prior
    log_post = log_prior_k(k, config.λ_K)

    # Spatial prior: P(θ|K) = (1/A)^K
    # This is required for proper detailed balance with uniform birth/death
    log_post += k * (-log(area(spatial_prior)))

    # Marginal count prior: P(N | K, μ, shape)
    log_post += log_prior_total_count(N, k, chain.μ, chain.shape)

    # Likelihood for each emitter
    for emitter in state.emitters
        log_post += log_likelihood_emitter(locs, emitter)
    end

    return log_post
end

"""
Compute log posterior WITHOUT spatial prior term.

Used by split/merge moves which operate in allocation space rather than
position space. The spatial prior (1/A)^K would create bias since split/merge
don't sample positions from the spatial distribution.
"""
function compute_log_posterior_no_spatial(
    state::BaGoLState,
    locs::Vector{<:SMLMData.AbstractEmitter},
    chain::RJMCMCChain
)
    k = length(state.emitters)
    config = chain.config
    N = length(locs)

    # K prior
    log_post = log_prior_k(k, config.λ_K)

    # Marginal count prior: P(N | K, μ, shape)
    log_post += log_prior_total_count(N, k, chain.μ, chain.shape)

    # Likelihood for each emitter
    for emitter in state.emitters
        log_post += log_likelihood_emitter(locs, emitter)
    end

    return log_post
end

"""
Birth: sample from mixture q(x,y), add emitter, reallocate.
Computes full posterior ratio for acceptance.

Note: We do NOT remove empty emitters here. With the marginal count prior
P(N|K), empty emitters are handled correctly - they contribute 0 to likelihood
and the count prior determines the appropriate K.
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
    old_log_post = compute_log_posterior(state, locs, chain, spatial_prior)

    # Sample position from mixture
    x, y = sample_from_mixture(locs)

    # Check bounds
    if x < spatial_prior.x_min || x > spatial_prior.x_max ||
       y < spatial_prior.y_min || y > spatial_prior.y_max
        return false
    end

    log_q = log_mixture_density(x, y, locs)

    # Add emitter and Gibbs reallocate all
    push!(state.emitters, Emitter(T(x), T(y)))
    allocate_gibbs!(state, locs)

    k_new = k + 1  # We added exactly one emitter

    # Compute new posterior
    new_log_post = compute_log_posterior(state, locs, chain, spatial_prior)

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
Death: pick emitter uniformly, use q(x,y) for detailed balance, reallocate.
Computes full posterior ratio for acceptance.
Never removes the last emitter (K >= 1 always).

Note: We do NOT remove empty emitters here. With the marginal count prior
P(N|K), empty emitters are handled correctly by the posterior.
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
    old_log_post = compute_log_posterior(state, locs, chain, spatial_prior)

    # Pick emitter to kill
    idx = rand(1:k)
    emitter = state.emitters[idx]

    # Mixture density at emitter position for detailed balance
    log_q = log_mixture_density(emitter.x, emitter.y, locs)

    # Remove emitter and Gibbs reallocate all
    deleteat!(state.emitters, idx)
    allocate_gibbs!(state, locs)

    k_new = k - 1  # We removed exactly one emitter

    # Compute new posterior
    new_log_post = compute_log_posterior(state, locs, chain, spatial_prior)

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
# Uniform birth/death moves (for testing mixing with flat spatial proposal)
# ============================================================================

"""
Birth with uniform spatial proposal over 2σ-extended localization bounds.
Uses flat q(x,y) = 1/area instead of mixture proposal.
"""
function propose_birth_uniform!(
    chain::RJMCMCChain{T},
    locs::Vector{<:SMLMData.AbstractEmitter},
    spatial_prior::UniformSpatialPrior
) where T
    state = chain.current_state
    k = length(state.emitters)
    n = length(locs)

    if n == 0
        return false
    end

    # Save old state
    old_emitters = deepcopy(state.emitters)
    old_log_post = compute_log_posterior(state, locs, chain, spatial_prior)

    # Sample position uniformly from spatial prior
    x, y = sample_position(spatial_prior)

    # Add emitter and Gibbs reallocate all
    push!(state.emitters, Emitter(T(x), T(y)))
    allocate_gibbs!(state, locs)

    k_new = k + 1

    # Compute new posterior
    new_log_post = compute_log_posterior(state, locs, chain, spatial_prior)

    # Posterior ratio
    log_post_ratio = new_log_post - old_log_post

    # Proposal ratio: forward uniform 1/area, reverse 1/k_new
    # q(reverse) / q(forward) = (1/k_new) / (1/area) = area/k_new
    log_area = log(area(spatial_prior))
    log_proposal_ratio = log_area - log(k_new)

    log_acceptance = log_post_ratio + log_proposal_ratio

    if log(rand()) < log_acceptance
        return true
    else
        state.emitters = old_emitters
        return false
    end
end

"""
Death with uniform spatial reverse proposal (conjugate to propose_birth_uniform!).
"""
function propose_death_uniform!(
    chain::RJMCMCChain,
    locs::Vector{<:SMLMData.AbstractEmitter},
    spatial_prior::UniformSpatialPrior
)
    state = chain.current_state
    k = length(state.emitters)

    if k <= 1
        return false
    end

    # Save old state
    old_emitters = deepcopy(state.emitters)
    old_log_post = compute_log_posterior(state, locs, chain, spatial_prior)

    # Pick emitter to kill uniformly
    idx = rand(1:k)

    # Remove emitter and Gibbs reallocate all
    deleteat!(state.emitters, idx)
    allocate_gibbs!(state, locs)

    k_new = k - 1

    # Compute new posterior
    new_log_post = compute_log_posterior(state, locs, chain, spatial_prior)

    # Posterior ratio
    log_post_ratio = new_log_post - old_log_post

    # Proposal ratio: forward 1/k, reverse uniform 1/area
    # q(reverse) / q(forward) = (1/area) / (1/k) = k/area
    log_area = log(area(spatial_prior))
    log_proposal_ratio = log(k) - log_area

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

With the marginal count prior P(N | K), individual allocations don't have
separate count priors - only the total N given K matters. Since K is fixed
during allocation, the count prior cancels and we sample based on likelihood only.
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

    # Compute log probability for each emitter (likelihood only)
    # With marginal count prior P(N|K), individual allocations don't matter
    log_probs = Vector{Float64}(undef, k)
    for (j, e) in enumerate(state.emitters)
        log_probs[j] = log_likelihood_single(loc, e)
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

    # Gibbs always "accepts" - even if sample equals current value
    if new_j == current_j
        return true  # Valid Gibbs sample, just happened to pick same assignment
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

    # Update the emitter that lost a localization
    if current_j > 0
        if !isempty(state.emitters[current_j].allocated)
            # Still has locs: Gibbs sample from remaining allocations
            state.emitters[current_j].x, state.emitters[current_j].y =
                sample_position_gibbs(state.emitters[current_j], locs)
        end
    end

    # Remove empty emitters - they shouldn't count toward K
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

# ============================================================================
# Split move (K → K+1)
# ============================================================================

"""
Split: randomly partition one emitter's allocations into two emitters.

This move operates in allocation space rather than position space, avoiding
the proposal density problem that plagues birth moves at tight clusters.

Algorithm:
1. Choose emitter j uniformly from K emitters
2. Randomly partition j's locs into two non-empty sets (coin flip each loc)
3. Create two emitters with the partitioned allocations
4. Gibbs sample positions for both from their allocations
5. Accept/reject with MH ratio

Proposal ratio: The allocation-dependent terms (0.5^n_j) are intentionally
omitted to avoid biasing the marginal distribution over K. This makes the
proposal ratio simply 2/(K+1), which gives unbiased K estimation.

The spatial prior term (1/A)^K is also excluded since split/merge operate
in allocation space, not position space.
"""
function propose_split!(
    chain::RJMCMCChain{T},
    locs::Vector{<:SMLMData.AbstractEmitter},
    spatial_prior::UniformSpatialPrior
) where T
    state = chain.current_state
    k = length(state.emitters)

    if k == 0
        return false
    end

    # Choose emitter to split uniformly
    idx = rand(1:k)
    emitter = state.emitters[idx]
    n_j = length(emitter.allocated)

    # Need at least 2 locs to split
    if n_j < 2
        return false
    end

    # Random partition via coin flips
    left_locs = Int[]
    right_locs = Int[]
    for loc_idx in emitter.allocated
        if rand() < 0.5
            push!(left_locs, loc_idx)
        else
            push!(right_locs, loc_idx)
        end
    end

    # Reject if either side empty (required for valid split)
    if isempty(left_locs) || isempty(right_locs)
        return false
    end

    # Save old state for rejection
    old_emitters = deepcopy(state.emitters)

    # Compute old posterior WITHOUT spatial prior (split/merge are allocation-based)
    old_log_post = compute_log_posterior_no_spatial(state, locs, chain)

    # Remove original emitter, add two new ones
    deleteat!(state.emitters, idx)

    # Left emitter with Gibbs-sampled position
    left_emitter = Emitter(emitter.x, emitter.y, left_locs)
    left_emitter.x, left_emitter.y = sample_position_gibbs(left_emitter, locs)
    push!(state.emitters, left_emitter)

    # Right emitter with Gibbs-sampled position
    right_emitter = Emitter(emitter.x, emitter.y, right_locs)
    right_emitter.x, right_emitter.y = sample_position_gibbs(right_emitter, locs)
    push!(state.emitters, right_emitter)

    k_new = k + 1

    # Compute new posterior WITHOUT spatial prior
    new_log_post = compute_log_posterior_no_spatial(state, locs, chain)
    log_post_ratio = new_log_post - old_log_post

    # Allocation-independent proposal ratio: 2/(K+1)
    # This avoids the bias from allocation-dependent terms
    log_proposal_ratio = log(2) - log(k_new)

    log_acceptance = log_post_ratio + log_proposal_ratio

    if log(rand()) < log_acceptance
        return true
    else
        state.emitters = old_emitters
        return false
    end
end

# ============================================================================
# Merge move (K → K-1)
# ============================================================================

"""
Merge: combine two emitters' allocations into one emitter.

This is the reverse of split, operating in allocation space.

Algorithm:
1. If K ≤ 1, reject (need at least 2 to merge)
2. Choose unordered pair (i, j) uniformly from C(K, 2) pairs
3. Combine allocations into single emitter
4. Gibbs sample position from combined allocations
5. Accept/reject with MH ratio

Proposal ratio: Symmetric with split, using K/2 to match split's 2/(K+1).
Spatial prior is excluded (see propose_split!).
"""
function propose_merge!(
    chain::RJMCMCChain{T},
    locs::Vector{<:SMLMData.AbstractEmitter},
    spatial_prior::UniformSpatialPrior
) where T
    state = chain.current_state
    k = length(state.emitters)

    # Need at least 2 emitters to merge
    if k <= 1
        return false
    end

    # Choose unordered pair uniformly
    # Sample i ∈ [1,k], then j ∈ [1,k-1] and shift if j >= i
    i = rand(1:k)
    j = rand(1:k-1)
    if j >= i
        j += 1
    end

    emitter_i = state.emitters[i]
    emitter_j = state.emitters[j]

    # Save old state for rejection
    old_emitters = deepcopy(state.emitters)
    old_log_post = compute_log_posterior_no_spatial(state, locs, chain)

    # Combine allocations
    combined_locs = vcat(emitter_i.allocated, emitter_j.allocated)

    # Remove both emitters (larger index first to preserve indices)
    if i > j
        deleteat!(state.emitters, i)
        deleteat!(state.emitters, j)
    else
        deleteat!(state.emitters, j)
        deleteat!(state.emitters, i)
    end

    # Create merged emitter with Gibbs-sampled position
    merged_emitter = Emitter(emitter_i.x, emitter_i.y, combined_locs)
    if !isempty(combined_locs)
        merged_emitter.x, merged_emitter.y = sample_position_gibbs(merged_emitter, locs)
    end
    push!(state.emitters, merged_emitter)

    k_new = k - 1

    # Compute new posterior WITHOUT spatial prior
    new_log_post = compute_log_posterior_no_spatial(state, locs, chain)
    log_post_ratio = new_log_post - old_log_post

    # Allocation-independent proposal ratio: K/2
    # This is symmetric with split's 2/(K+1): for K→K-1, ratio is K/2
    log_proposal_ratio = log(k) - log(2)

    log_acceptance = log_post_ratio + log_proposal_ratio

    if log(rand()) < log_acceptance
        return true
    else
        state.emitters = old_emitters
        return false
    end
end

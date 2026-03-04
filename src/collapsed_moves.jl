# Collapsed RJMCMC moves for BaGoL
#
# Three move types for the collapsed Gibbs sampler:
# 1. Allocation Gibbs sweep — full sweep reassigning each loc
# 2. Block birth — create new cluster from seed loc + nearby locs
# 3. Block death — dissolve a cluster, redistributing locs

# ============================================================================
# Helpers
# ============================================================================

"""
    _find_inactive_slot(state) -> Int

Find an inactive cluster slot, growing the arrays if needed.
Zero-allocation when an inactive slot exists.
"""
function _find_inactive_slot(state::CollapsedState)
    @inbounds for i in eachindex(state.active)
        if !state.active[i]
            return i
        end
    end
    # Grow arrays
    push!(state.clusters, ClusterStats())
    push!(state.active, false)
    return length(state.clusters)
end

"""
    _activate_cluster!(state, slot, cs)

Activate a cluster slot with given stats.
"""
function _activate_cluster!(state::CollapsedState, slot::Integer, cs::ClusterStats)
    state.clusters[slot] = cs
    state.active[slot] = true
    state.n_active += 1
end

"""
    _deactivate_cluster!(state, slot)

Deactivate a cluster slot.
"""
function _deactivate_cluster!(state::CollapsedState, slot::Integer)
    state.clusters[slot] = ClusterStats()
    state.active[slot] = false
    state.n_active -= 1
end

"""
    _collapsed_log_posterior(state, N, μ, shape, λ_K) -> Float64

Collapsed log posterior: priors + sum of marginal likelihoods.
"""
function _collapsed_log_posterior(state::CollapsedState, N::Int,
                                  μ::Float64, shape::Float64, λ_K::Float64)
    K = state.n_active
    lp = log_prior_k(K, λ_K) + log_prior_total_count(N, K, μ, shape)
    for (j, cs) in enumerate(state.clusters)
        state.active[j] || continue
        lp += log_marginal_likelihood(cs, state.log_area)
    end
    return lp
end

# ============================================================================
# Allocation Gibbs sweep
# ============================================================================

"""
    gibbs_allocation_sweep!(state, locs, μ, shape, λ_K)

Full Gibbs sweep: for each loc (random order), reassign to the cluster
with highest predictive probability, including the option of creating
a new cluster.

Uses precomputed LocPrecision data for zero-allocation inner loop.
"""
function gibbs_allocation_sweep!(state::CollapsedState,
                                  locs::Vector{<:SMLMData.AbstractEmitter},
                                  μ::Float64, shape::Float64, λ_K::Float64)
    N = length(locs)
    loc_precs = state._loc_precs

    # In-place Fisher-Yates shuffle of workspace permutation buffer
    perm = state._perm
    @inbounds for i in N:-1:2
        j = rand(1:i)
        perm[i], perm[j] = perm[j], perm[i]
    end

    active_slots = state._active_slots
    log_probs = state._log_probs

    @inbounds for loc_pos in 1:N
        loc_idx = perm[loc_pos]
        lp = loc_precs[loc_idx]
        old_cluster = state.assignments[loc_idx]

        # Remove loc from current cluster
        if old_cluster > 0 && state.active[old_cluster]
            state.clusters[old_cluster] = remove_loc(state.clusters[old_cluster], lp)

            # If cluster is now empty, deactivate it
            if state.clusters[old_cluster].n == 0
                _deactivate_cluster!(state, old_cluster)
            end
        end

        # Compute log predictive for each active cluster + new cluster
        K = state.n_active
        n_options = K + 1  # existing clusters + new cluster

        # Collect active cluster indices into workspace buffer
        idx = 0
        for j in eachindex(state.active)
            if state.active[j]
                idx += 1
                active_slots[idx] = j
            end
        end

        # Existing clusters
        for i in 1:K
            slot = active_slots[i]
            cs = state.clusters[slot]
            # Predictive × CRP-like weight (proportional to cluster size)
            log_probs[i] = log_predictive(cs, lp, state.log_area) + log(Float64(cs.n))
        end

        # New cluster option
        K_new = K + 1
        log_pk_ratio = log_prior_k(K_new, λ_K) - log_prior_k(K, λ_K)
        log_pn_ratio = log_prior_total_count(N, K_new, μ, shape) -
                       log_prior_total_count(N, max(K, 1), μ, shape)
        log_probs[n_options] = -state.log_area + log_pk_ratio + log_pn_ratio

        # In-place log-sum-exp normalization → probabilities in log_probs[1:n_options]
        max_lp = log_probs[1]
        for i in 2:n_options
            if log_probs[i] > max_lp
                max_lp = log_probs[i]
            end
        end
        total = 0.0
        for i in 1:n_options
            v = exp(log_probs[i] - max_lp)
            log_probs[i] = v
            total += v
        end
        inv_total = 1.0 / total

        # Sample from categorical (cumulative sum)
        u = rand()
        cumsum_p = 0.0
        chosen = n_options  # default to new cluster
        for i in 1:n_options
            cumsum_p += log_probs[i] * inv_total
            if u < cumsum_p
                chosen = i
                break
            end
        end

        if chosen <= K
            # Assign to existing cluster
            slot = active_slots[chosen]
            state.clusters[slot] = add_loc(state.clusters[slot], lp)
            state.assignments[loc_idx] = Int16(slot)
        else
            # Create new cluster
            slot = _find_inactive_slot(state)
            cs_new = add_loc(ClusterStats(), lp)
            _activate_cluster!(state, slot, cs_new)
            state.assignments[loc_idx] = Int16(slot)
        end
    end
end

# ============================================================================
# Block birth move
# ============================================================================

"""
    _save_rollback!(state)

Save current state into rollback buffers for MH rejection.
Zero-allocation: copies into pre-allocated buffers.
"""
function _save_rollback!(state::CollapsedState)
    n = length(state.clusters)
    # Resize rollback buffers if needed (rare — only when cluster array grew)
    if length(state._rollback_clusters) < n
        resize!(state._rollback_clusters, n)
        resize!(state._rollback_active, n)
    end
    copyto!(state._rollback_assignments, state.assignments)
    @inbounds for i in 1:n
        state._rollback_clusters[i] = state.clusters[i]
        state._rollback_active[i] = state.active[i]
    end
end

"""
    _restore_rollback!(state, old_n_active, old_len)

Restore state from rollback buffers.
"""
function _restore_rollback!(state::CollapsedState, old_n_active::Int, old_len::Int)
    copyto!(state.assignments, state._rollback_assignments)
    resize!(state.clusters, old_len)
    resize!(state.active, old_len)
    @inbounds for i in 1:old_len
        state.clusters[i] = state._rollback_clusters[i]
        state.active[i] = state._rollback_active[i]
    end
    state.n_active = old_n_active
end

"""
    propose_block_birth!(state, locs, μ, shape, λ_K) -> Bool

Block birth: pick a random seed loc, create new cluster, probabilistically
recruit nearby locs via predictive. MH accept/reject.
"""
function propose_block_birth!(state::CollapsedState,
                               locs::Vector{<:SMLMData.AbstractEmitter},
                               μ::Float64, shape::Float64, λ_K::Float64)
    N = length(locs)
    N == 0 && return false
    loc_precs = state._loc_precs

    # Save state for rollback (into pre-allocated buffers)
    old_len = length(state.clusters)
    old_n_active = state.n_active
    _save_rollback!(state)

    old_log_post = _collapsed_log_posterior(state, N, μ, shape, λ_K)

    # Pick random seed loc
    seed_idx = rand(1:N)
    seed_lp = loc_precs[seed_idx]
    old_cluster = state.assignments[seed_idx]

    # Remove seed from its current cluster
    if old_cluster > 0 && state.active[old_cluster]
        state.clusters[old_cluster] = remove_loc(state.clusters[old_cluster], seed_lp)
        if state.clusters[old_cluster].n == 0
            _deactivate_cluster!(state, old_cluster)
        end
    end

    # Create new cluster with seed
    new_slot = _find_inactive_slot(state)
    cs_new = add_loc(ClusterStats(), seed_lp)
    _activate_cluster!(state, new_slot, cs_new)
    state.assignments[seed_idx] = Int16(new_slot)

    # Recruit nearby locs (within 5σ of seed)
    seed_x = seed_lp.x
    seed_y = seed_lp.y
    radius = 5 * seed_lp.σ
    log_q_forward = 0.0

    @inbounds for i in 1:N
        i == seed_idx && continue
        lp_i = loc_precs[i]
        dx = lp_i.x - seed_x
        dy = lp_i.y - seed_y
        d = sqrt(dx^2 + dy^2)
        d > radius && continue

        # Compute predictive for new cluster vs current cluster
        current_slot = state.assignments[i]
        cs_current = (current_slot > 0 && state.active[current_slot]) ?
                     state.clusters[current_slot] : ClusterStats()
        cs_new_with = state.clusters[new_slot]

        log_p_new = log_predictive(cs_new_with, lp_i, state.log_area)
        log_p_old = log_predictive(cs_current, lp_i, state.log_area)

        # Probability of moving to new cluster
        max_lp = max(log_p_new, log_p_old)
        p_new = exp(log_p_new - max_lp)
        p_old = exp(log_p_old - max_lp)
        prob_move = p_new / (p_new + p_old)

        if rand() < prob_move
            # Move loc to new cluster
            if current_slot > 0 && state.active[current_slot]
                state.clusters[current_slot] = remove_loc(state.clusters[current_slot], lp_i)
                if state.clusters[current_slot].n == 0
                    _deactivate_cluster!(state, current_slot)
                end
            end
            state.clusters[new_slot] = add_loc(state.clusters[new_slot], lp_i)
            state.assignments[i] = Int16(new_slot)
            log_q_forward += log(prob_move)
        else
            log_q_forward += log(1 - prob_move)
        end
    end

    new_log_post = _collapsed_log_posterior(state, N, μ, shape, λ_K)

    # Compute reverse proposal probability (block death of new cluster)
    K_new = state.n_active
    log_q_reverse = -log(Float64(K_new))

    log_acceptance = (new_log_post - old_log_post) + (log_q_reverse - log_q_forward)

    if log(rand()) < log_acceptance
        return true
    else
        _restore_rollback!(state, old_n_active, old_len)
        return false
    end
end

# ============================================================================
# Block death move
# ============================================================================

"""
    propose_block_death!(state, locs, μ, shape, λ_K) -> Bool

Block death: pick a random active cluster (K > 1), redistribute its locs
to remaining clusters via predictive. MH accept/reject.
"""
function propose_block_death!(state::CollapsedState,
                               locs::Vector{<:SMLMData.AbstractEmitter},
                               μ::Float64, shape::Float64, λ_K::Float64)
    K = state.n_active
    K <= 1 && return false

    N = length(locs)
    loc_precs = state._loc_precs

    # Save state for rollback (into pre-allocated buffers)
    old_len = length(state.clusters)
    old_n_active = state.n_active
    _save_rollback!(state)

    old_log_post = _collapsed_log_posterior(state, N, μ, shape, λ_K)

    # Pick random active cluster to kill (use workspace to avoid findall)
    active_slots = state._active_slots
    n_active = 0
    @inbounds for j in eachindex(state.active)
        if state.active[j]
            n_active += 1
            active_slots[n_active] = j
        end
    end
    kill_slot = active_slots[rand(1:K)]

    # Forward proposal: choosing this cluster to kill (1/K)
    log_q_forward = -log(Float64(K))

    # Deactivate the cluster
    _deactivate_cluster!(state, kill_slot)

    # Collect remaining active slots into workspace
    K_remaining = state.n_active
    n_remaining = 0
    @inbounds for j in eachindex(state.active)
        if state.active[j]
            n_remaining += 1
            active_slots[n_remaining] = j
        end
    end

    # Redistribute locs in killed cluster via predictive
    log_q_redistribute = 0.0
    log_probs = state._log_probs

    @inbounds for loc_idx in 1:N
        state._rollback_assignments[loc_idx] == kill_slot || continue
        lp_i = loc_precs[loc_idx]

        # Compute predictive for each remaining cluster (in-place)
        for i in 1:n_remaining
            slot = active_slots[i]
            log_probs[i] = log_predictive(state.clusters[slot], lp_i, state.log_area) +
                           log(Float64(state.clusters[slot].n))
        end

        # In-place log-sum-exp → probabilities
        max_lp = log_probs[1]
        for i in 2:n_remaining
            if log_probs[i] > max_lp
                max_lp = log_probs[i]
            end
        end
        total = 0.0
        for i in 1:n_remaining
            v = exp(log_probs[i] - max_lp)
            log_probs[i] = v
            total += v
        end
        inv_total = 1.0 / total

        # Sample from categorical
        u = rand()
        cumsum_p = 0.0
        chosen = n_remaining
        for i in 1:n_remaining
            cumsum_p += log_probs[i] * inv_total
            if u < cumsum_p
                chosen = i
                break
            end
        end

        target_slot = active_slots[chosen]
        state.clusters[target_slot] = add_loc(state.clusters[target_slot], lp_i)
        state.assignments[loc_idx] = Int16(target_slot)
        log_q_redistribute += log(log_probs[chosen] * inv_total)
    end

    new_log_post = _collapsed_log_posterior(state, N, μ, shape, λ_K)

    log_acceptance = (new_log_post - old_log_post) + (log_q_forward - log_q_redistribute)

    if log(rand()) < log_acceptance
        return true
    else
        _restore_rollback!(state, old_n_active, old_len)
        return false
    end
end

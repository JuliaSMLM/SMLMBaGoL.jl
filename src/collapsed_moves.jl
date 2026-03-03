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
"""
function _find_inactive_slot(state::CollapsedState)
    idx = findfirst(.!state.active)
    if idx !== nothing
        return idx
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

This naturally handles birth/death: if a cluster empties it dies,
if a loc picks "new cluster" it births.
"""
function gibbs_allocation_sweep!(state::CollapsedState,
                                  locs::Vector{<:SMLMData.AbstractEmitter},
                                  μ::Float64, shape::Float64, λ_K::Float64)
    N = length(locs)
    perm = randperm(N)

    for loc_idx in perm
        loc = locs[loc_idx]
        old_cluster = state.assignments[loc_idx]

        # Remove loc from current cluster
        if old_cluster > 0 && state.active[old_cluster]
            state.clusters[old_cluster] = remove_loc(state.clusters[old_cluster], loc)

            # If cluster is now empty, deactivate it
            if state.clusters[old_cluster].n == 0
                _deactivate_cluster!(state, old_cluster)
            end
        end

        # Compute log predictive for each active cluster + new cluster
        K = state.n_active
        n_options = K + 1  # existing clusters + new cluster

        # Collect active cluster indices
        active_slots = Vector{Int}(undef, K)
        idx = 0
        for j in eachindex(state.active)
            if state.active[j]
                idx += 1
                active_slots[idx] = j
            end
        end

        log_probs = Vector{Float64}(undef, n_options)

        # Existing clusters
        for (i, slot) in enumerate(active_slots)
            cs = state.clusters[slot]
            # Predictive × CRP-like weight (proportional to cluster size)
            log_probs[i] = log_predictive(cs, loc, state.log_area) + log(Float64(cs.n))
        end

        # New cluster option
        # p(new) ∝ (1/A) × P(K+1)/P(K) × P(N|K+1)/P(N|K) × α
        # where α is an effective concentration parameter
        K_new = K + 1
        log_pk_ratio = log_prior_k(K_new, λ_K) - log_prior_k(K, λ_K)
        log_pn_ratio = log_prior_total_count(N, K_new, μ, shape) -
                       log_prior_total_count(N, max(K, 1), μ, shape)
        log_probs[n_options] = -state.log_area + log_pk_ratio + log_pn_ratio

        # Sample from categorical (log-sum-exp)
        max_lp = maximum(log_probs)
        probs = exp.(log_probs .- max_lp)
        total = sum(probs)
        probs ./= total

        # Sample
        u = rand()
        cumsum_p = 0.0
        chosen = n_options  # default to new cluster
        for i in eachindex(probs)
            cumsum_p += probs[i]
            if u < cumsum_p
                chosen = i
                break
            end
        end

        if chosen <= K
            # Assign to existing cluster
            slot = active_slots[chosen]
            state.clusters[slot] = add_loc(state.clusters[slot], loc)
            state.assignments[loc_idx] = Int16(slot)
        else
            # Create new cluster
            slot = _find_inactive_slot(state)
            cs_new = add_loc(ClusterStats(), loc)
            _activate_cluster!(state, slot, cs_new)
            state.assignments[loc_idx] = Int16(slot)
        end
    end
end

# ============================================================================
# Block birth move
# ============================================================================

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

    # Save state for rollback
    old_assignments = copy(state.assignments)
    old_clusters = copy(state.clusters)
    old_active = copy(state.active)
    old_n_active = state.n_active

    old_log_post = _collapsed_log_posterior(state, N, μ, shape, λ_K)

    # Pick random seed loc
    seed_idx = rand(1:N)
    seed_loc = locs[seed_idx]
    old_cluster = state.assignments[seed_idx]

    # Remove seed from its current cluster
    if old_cluster > 0 && state.active[old_cluster]
        state.clusters[old_cluster] = remove_loc(state.clusters[old_cluster], seed_loc)
        if state.clusters[old_cluster].n == 0
            _deactivate_cluster!(state, old_cluster)
        end
    end

    # Create new cluster with seed
    new_slot = _find_inactive_slot(state)
    cs_new = add_loc(ClusterStats(), seed_loc)
    _activate_cluster!(state, new_slot, cs_new)
    state.assignments[seed_idx] = Int16(new_slot)

    # Recruit nearby locs (within 5σ of seed)
    σ_seed = sqrt(seed_loc.σ_x^2 + seed_loc.σ_y^2)
    radius = 5 * σ_seed
    log_q_forward = 0.0

    for i in 1:N
        i == seed_idx && continue
        loc = locs[i]
        dx = loc.x - seed_loc.x
        dy = loc.y - seed_loc.y
        d = sqrt(dx^2 + dy^2)
        d > radius && continue

        # Compute predictive for new cluster vs current cluster
        current_slot = state.assignments[i]
        cs_current = (current_slot > 0 && state.active[current_slot]) ?
                     state.clusters[current_slot] : ClusterStats()
        cs_new_with = state.clusters[new_slot]

        log_p_new = log_predictive(cs_new_with, loc, state.log_area)
        log_p_old = log_predictive(cs_current, loc, state.log_area)

        # Probability of moving to new cluster
        max_lp = max(log_p_new, log_p_old)
        p_new = exp(log_p_new - max_lp)
        p_old = exp(log_p_old - max_lp)
        prob_move = p_new / (p_new + p_old)

        if rand() < prob_move
            # Move loc to new cluster
            if current_slot > 0 && state.active[current_slot]
                state.clusters[current_slot] = remove_loc(state.clusters[current_slot], loc)
                if state.clusters[current_slot].n == 0
                    _deactivate_cluster!(state, current_slot)
                end
            end
            state.clusters[new_slot] = add_loc(state.clusters[new_slot], loc)
            state.assignments[i] = Int16(new_slot)
            log_q_forward += log(prob_move)
        else
            log_q_forward += log(1 - prob_move)
        end
    end

    new_log_post = _collapsed_log_posterior(state, N, μ, shape, λ_K)

    # Compute reverse proposal probability (block death of new cluster)
    # Simplified: reverse is choosing this cluster to kill with prob 1/K_new
    K_new = state.n_active
    log_q_reverse = -log(Float64(K_new))

    log_acceptance = (new_log_post - old_log_post) + (log_q_reverse - log_q_forward)

    if log(rand()) < log_acceptance
        return true
    else
        # Rollback
        state.assignments .= old_assignments
        resize!(state.clusters, length(old_clusters))
        state.clusters .= old_clusters
        resize!(state.active, length(old_active))
        state.active .= old_active
        state.n_active = old_n_active
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

    # Save state for rollback
    old_assignments = copy(state.assignments)
    old_clusters = copy(state.clusters)
    old_active = copy(state.active)
    old_n_active = state.n_active

    old_log_post = _collapsed_log_posterior(state, N, μ, shape, λ_K)

    # Pick random active cluster to kill
    active_slots = findall(state.active)
    kill_slot = active_slots[rand(1:K)]

    # Forward proposal: choosing this cluster to kill (1/K)
    log_q_forward = -log(Float64(K))

    # Find locs in this cluster
    kill_locs = findall(a -> a == kill_slot, state.assignments)

    # Deactivate the cluster
    _deactivate_cluster!(state, kill_slot)

    # Redistribute locs via predictive to remaining clusters
    log_q_redistribute = 0.0
    remaining_slots = findall(state.active)

    for loc_idx in kill_locs
        loc = locs[loc_idx]

        # Compute predictive for each remaining cluster
        n_remaining = length(remaining_slots)
        log_probs = Vector{Float64}(undef, n_remaining)
        for (i, slot) in enumerate(remaining_slots)
            log_probs[i] = log_predictive(state.clusters[slot], loc, state.log_area) +
                           log(Float64(state.clusters[slot].n))
        end

        # Sample from categorical
        max_lp = maximum(log_probs)
        probs = exp.(log_probs .- max_lp)
        total = sum(probs)
        probs ./= total

        u = rand()
        cumsum_p = 0.0
        chosen = n_remaining
        for i in eachindex(probs)
            cumsum_p += probs[i]
            if u < cumsum_p
                chosen = i
                break
            end
        end

        target_slot = remaining_slots[chosen]
        state.clusters[target_slot] = add_loc(state.clusters[target_slot], loc)
        state.assignments[loc_idx] = Int16(target_slot)
        log_q_redistribute += log(probs[chosen])
    end

    new_log_post = _collapsed_log_posterior(state, N, μ, shape, λ_K)

    # Reverse proposal: block birth creating this cluster
    # Simplified symmetric approximation
    log_q_reverse = log_q_redistribute  # Approximate reverse probability

    log_acceptance = (new_log_post - old_log_post) + (log_q_forward - log_q_redistribute)

    if log(rand()) < log_acceptance
        return true
    else
        # Rollback
        state.assignments .= old_assignments
        resize!(state.clusters, length(old_clusters))
        state.clusters .= old_clusters
        resize!(state.active, length(old_active))
        state.active .= old_active
        state.n_active = old_n_active
        return false
    end
end

# Collapsed Gibbs sampler moves for BaGoL
#
# Three move types:
# 1. Allocation Gibbs sweep — reassign each loc among K clusters (K fixed)
# 2. Split — divide one cluster into two via restricted Gibbs scan
# 3. Merge — combine two clusters into one

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
    _log_count_posterior(K, N, shape, μ) -> Float64

Count-model posterior for K emitters (Fazel et al. 2022).

  P(K|N) ∝ P(N|K, μ, α) = NegBin(N; K×α, α/(α+μ))

No separate prior on K — the NegBin likelihood regularizes K through its
shape parameter Kα. Under the localization mixture prior there is no spatial
Poisson process, so no Poisson(λA) prior on K exists.

Uses fixed μ₀ (prior mean) rather than adaptive μ to prevent the μ-K
positive feedback loop where adaptive μ tracks K, making the count model
non-informative for K changes.
"""
function _log_count_posterior(K::Int, N::Int,
                              shape::Float64, μ::Float64)
    p = shape / (shape + μ)
    return logpdf(NegativeBinomial(K * shape, p), N)
end

# ============================================================================
# Allocation Gibbs sweep
# ============================================================================

"""
    gibbs_allocation_sweep!(state, locs, μ, shape)

Full Gibbs sweep: for each loc (random order), reassign among the K active
clusters with weight proportional to the collapsed spatial predictive.

At fixed K, the Gibbs conditional is purely spatial:
  P(z_i = k | rest) ∝ predictive(d_i | cluster_k)
Cluster sizes are determined by spatial evidence alone.

Sole occupants are skipped to maintain K during the sweep. Only split/merge
moves change K.

Uses precomputed LocPrecision data for zero-allocation inner loop.
"""
function gibbs_allocation_sweep!(state::CollapsedState,
                                  locs::Vector{<:SMLMData.AbstractEmitter},
                                  μ::Float64, shape::Float64)
    N = length(locs)
    loc_precs = state._loc_precs
    grid = state._locmix_grid
    K = state.n_active  # Fixed for this sweep

    # In-place Fisher-Yates shuffle of workspace permutation buffer
    perm = state._perm
    @inbounds for i in N:-1:2
        j = rand(1:i)
        perm[i], perm[j] = perm[j], perm[i]
    end

    active_slots = state._active_slots
    log_probs = state._log_probs

    # Collect active cluster indices once (K is fixed during sweep)
    idx = 0
    @inbounds for j in eachindex(state.active)
        if state.active[j]
            idx += 1
            active_slots[idx] = j
        end
    end

    @inbounds for loc_pos in 1:N
        loc_idx = perm[loc_pos]
        lp = loc_precs[loc_idx]
        old_cluster = state.assignments[loc_idx]

        # Skip sole occupants to maintain K during sweep
        if old_cluster > 0 && state.active[old_cluster] && state.clusters[old_cluster].n <= 1
            continue
        end

        # Remove loc from current cluster
        if old_cluster > 0 && state.active[old_cluster]
            state.clusters[old_cluster] = remove_loc(state.clusters[old_cluster], lp)
        end

        # MFM-weighted predictive for each active cluster:
        #   P(z_i = k | rest) ∝ (n_{k,-i} + γ) × predictive(d_i | cluster_k)
        # where γ = shape (NegBin shape = Dirichlet concentration)
        for i in 1:K
            slot = active_slots[i]
            cs = state.clusters[slot]
            log_probs[i] = log(Float64(cs.n) + shape) + log_predictive_locmix(cs, lp, grid)
        end

        # In-place log-sum-exp normalization → probabilities in log_probs[1:K]
        max_lp = log_probs[1]
        for i in 2:K
            if log_probs[i] > max_lp
                max_lp = log_probs[i]
            end
        end
        total = 0.0
        for i in 1:K
            v = exp(log_probs[i] - max_lp)
            log_probs[i] = v
            total += v
        end
        inv_total = 1.0 / total

        # Sample from categorical (cumulative sum)
        u = rand()
        cumsum_p = 0.0
        chosen = K  # default to last cluster
        for i in 1:K
            cumsum_p += log_probs[i] * inv_total
            if u < cumsum_p
                chosen = i
                break
            end
        end

        # Assign to chosen cluster
        slot = active_slots[chosen]
        state.clusters[slot] = add_loc(state.clusters[slot], lp)
        state.assignments[loc_idx] = Int16(slot)
    end
end

# ============================================================================
# Rollback helpers (used by split/merge MH moves)
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

# ============================================================================
# K±1 split/merge with matched proposals
#
# Standard MH on partitions (not RJMCMC — positions integrated out).
# Split and merge are exact reverses of each other:
#   split(C → A,B) ↔ merge(A,B → C)
#
# Acceptance ratio:
#   log α = [log π(z') - log π(z)] + [log q(z|z') - log q(z'|z)]
#
# where π = spatial_LML × count_model and q = selection × SMC_assignment.
# ============================================================================

"""
    _total_spatial_lml(state) -> Float64

Sum of log marginal likelihoods across all active clusters.
"""
function _total_spatial_lml(state::CollapsedState)
    grid = state._locmix_grid
    total = 0.0
    @inbounds for j in eachindex(state.active)
        if state.active[j]
            total += log_marginal_likelihood_locmix(state.clusters[j], grid)
        end
    end
    return total
end

"""
    propose_split_merge!(state, locs, μ, shape) -> (Bool, Symbol)

K±1 split/merge with matched SMC proposals and full MH acceptance.

Coin flip: split (K→K+1) or merge (K→K-1).

Acceptance ratio includes:
1. Spatial LML change: Σ log ML(z') - Σ log ML(z)
2. Count model change: log P(N|K') - log P(N|K)
3. Proposal ratio: log q(z|z') - log q(z'|z)
   where q = cluster_selection × SMC_assignment

At d=0 the spatial LML penalty and SMC proposal density cancel,
leaving only the count model — matching qPAINT exactly.
"""
function propose_split_merge!(state::CollapsedState,
                               locs::Vector{<:SMLMData.AbstractEmitter},
                               μ::Float64, shape::Float64)
    N = length(locs)
    N < 2 && return false, :split
    K = state.n_active

    # Coin flip: split or merge
    do_split = rand() < 0.5
    if do_split && K >= N
        return false, :split  # Can't have more clusters than locs
    end
    if !do_split && K <= 1
        return false, :merge  # Can't merge with only 1 cluster
    end

    # Save state for potential rollback
    _save_rollback!(state)
    old_K = K
    old_len = length(state.clusters)

    # Compute spatial LML before
    lml_before = _total_spatial_lml(state)

    move_type = do_split ? :split : :merge
    log_q_fwd = 0.0
    log_q_rev = 0.0
    n_parent = 0
    n_child_a = 0
    n_child_b = 0

    if do_split
        log_q_fwd, log_q_rev, n_parent, n_child_a, n_child_b = _smc_split!(state, locs, shape)
    else
        log_q_fwd, log_q_rev, n_parent, n_child_a, n_child_b = _smc_merge!(state, locs, shape)
    end

    K_new = state.n_active

    # No-op: split/merge returned without changing state
    if K_new == old_K
        _restore_rollback!(state, old_K, old_len)
        return false, move_type
    end

    # Compute spatial LML after
    lml_after = _total_spatial_lml(state)

    # Full MH acceptance with MFM partition prior
    Δ_spatial = lml_after - lml_before
    Δ_count = _log_count_posterior(K_new, N, shape, μ) -
              _log_count_posterior(old_K, N, shape, μ)
    Δ_proposal = log_q_rev - log_q_fwd

    # MFM partition prior ratio (γ = shape)
    if do_split
        Δ_partition = log_mfm_partition_ratio(n_child_a, n_child_b, n_parent,
                                               old_K, N, shape)
    else
        Δ_partition = -log_mfm_partition_ratio(n_child_a, n_child_b, n_parent,
                                                K_new, N, shape)
    end

    Δ = Δ_spatial + Δ_count + Δ_partition + Δ_proposal

    if Δ >= 0 || rand() < exp(Δ)
        return true, move_type
    else
        _restore_rollback!(state, old_K, old_len)
        return false, move_type
    end
end

"""
    _smc_split!(state, locs, shape) -> (log_q_fwd, log_q_rev, n_parent, n_child_a, n_child_b)

Split: pick a cluster uniformly, MFM-weighted SMC-assign its locs to two sub-clusters.
"""
function _smc_split!(state::CollapsedState,
                      locs::Vector{<:SMLMData.AbstractEmitter},
                      shape::Float64)
    loc_precs = state._loc_precs
    grid = state._locmix_grid
    N = length(locs)
    K = state.n_active
    γ = shape

    # Select cluster to split: uniform over K active clusters
    active_slots = Int[]
    @inbounds for j in eachindex(state.active)
        state.active[j] && push!(active_slots, j)
    end
    split_idx = rand(1:K)
    split_slot = active_slots[split_idx]
    n_members = Int(state.clusters[split_slot].n)
    n_members < 2 && return 0.0, 0.0, 0, 0, 0

    # Collect member indices (deterministic order for reversibility)
    member_indices = Int[]
    @inbounds for i in 1:N
        state.assignments[i] == split_slot && push!(member_indices, i)
    end
    m = length(member_indices)

    # New cluster slot
    new_slot = _find_inactive_slot(state)

    # Seed both sub-clusters
    cs_a = add_loc(ClusterStats(), loc_precs[member_indices[1]])
    cs_b = add_loc(ClusterStats(), loc_precs[member_indices[2]])
    state.assignments[member_indices[2]] = Int16(new_slot)

    # SMC sequential assignment
    log_q_assign = 0.0
    for idx in 3:m
        loc_idx = member_indices[idx]
        lp = loc_precs[loc_idx]

        log_pred_a = log_marginal_likelihood_locmix(add_loc(cs_a, lp), grid) -
                     log_marginal_likelihood_locmix(cs_a, grid)
        log_pred_b = log_marginal_likelihood_locmix(add_loc(cs_b, lp), grid) -
                     log_marginal_likelihood_locmix(cs_b, grid)

        # MFM weight: (n_k + γ) × predictive
        log_w_a = log(Float64(cs_a.n) + γ) + log_pred_a
        log_w_b = log(Float64(cs_b.n) + γ) + log_pred_b
        max_lw = max(log_w_a, log_w_b)
        prob_b = exp(log_w_b - max_lw) / (exp(log_w_a - max_lw) + exp(log_w_b - max_lw))
        prob_b = clamp(prob_b, 1e-300, 1.0 - 1e-300)

        if rand() < prob_b
            cs_b = add_loc(cs_b, lp)
            state.assignments[loc_idx] = Int16(new_slot)
            log_q_assign += log(prob_b)
        else
            cs_a = add_loc(cs_a, lp)
            log_q_assign += log(1.0 - prob_b)
        end
    end

    # Update state
    state.clusters[split_slot] = cs_a
    _activate_cluster!(state, new_slot, cs_b)

    n_a = Int(cs_a.n)
    n_b = Int(cs_b.n)

    # Forward proposal: select cluster (1/K) × SMC assignment
    log_q_fwd = -log(K) + log_q_assign

    # Reverse proposal: select pair to merge = uniform over (K+1 choose 2) pairs
    K_new = K + 1
    log_q_rev = -log(K_new * (K_new - 1) / 2)

    return log_q_fwd, log_q_rev, n_members, n_a, n_b
end

"""
    _smc_merge!(state, locs, shape) -> (log_q_fwd, log_q_rev, n_parent, n_child_a, n_child_b)

Merge: pick a pair of clusters uniformly, combine them.
"""
function _smc_merge!(state::CollapsedState,
                      locs::Vector{<:SMLMData.AbstractEmitter},
                      shape::Float64)
    loc_precs = state._loc_precs
    grid = state._locmix_grid
    N = length(locs)
    K = state.n_active
    γ = shape

    K <= 1 && return 0.0, 0.0, 0, 0, 0

    # Select pair to merge: uniform over K(K-1)/2 pairs
    active_slots = Int[]
    @inbounds for j in eachindex(state.active)
        state.active[j] && push!(active_slots, j)
    end

    # Pick two distinct active clusters uniformly
    idx1 = rand(1:K)
    idx2 = rand(1:K-1)
    idx2 >= idx1 && (idx2 += 1)
    slot_a = active_slots[idx1]  # survives
    slot_b = active_slots[idx2]  # gets absorbed

    # Collect all members in index order (must match split's deterministic ordering)
    all_members = Int[]
    @inbounds for i in 1:N
        if state.assignments[i] == slot_a || state.assignments[i] == slot_b
            push!(all_members, i)
        end
    end
    m = length(all_members)

    # Determine which slot is "B" (the one containing the second-lowest-index loc).
    # The split always seeds A with all_members[1], B with all_members[2].
    # So "B" is whichever slot contains all_members[2].
    b_slot = m >= 2 ? state.assignments[all_members[2]] : slot_b
    is_b = Set{Int}()
    @inbounds for i in all_members
        state.assignments[i] == b_slot && push!(is_b, i)
    end
    log_q_assign_rev = 0.0

    if m >= 2
        # Shuffle all_members the same way the reverse split would
        # (we use a deterministic order: A members first, then B)
        cs_ra = add_loc(ClusterStats(), loc_precs[all_members[1]])
        cs_rb = add_loc(ClusterStats(), loc_precs[all_members[2]])

        for idx in 3:m
            loc_idx = all_members[idx]
            lp = loc_precs[loc_idx]
            goes_to_b = loc_idx ∈ is_b

            log_pred_a = log_marginal_likelihood_locmix(add_loc(cs_ra, lp), grid) -
                         log_marginal_likelihood_locmix(cs_ra, grid)
            log_pred_b = log_marginal_likelihood_locmix(add_loc(cs_rb, lp), grid) -
                         log_marginal_likelihood_locmix(cs_rb, grid)

            # MFM-weighted proposal (must match forward split)
            log_w_a = log(Float64(cs_ra.n) + γ) + log_pred_a
            log_w_b = log(Float64(cs_rb.n) + γ) + log_pred_b
            max_lw = max(log_w_a, log_w_b)
            prob_b = exp(log_w_b - max_lw) / (exp(log_w_a - max_lw) + exp(log_w_b - max_lw))
            prob_b = clamp(prob_b, 1e-300, 1.0 - 1e-300)

            if goes_to_b
                cs_rb = add_loc(cs_rb, lp)
                log_q_assign_rev += log(prob_b)
            else
                cs_ra = add_loc(cs_ra, lp)
                log_q_assign_rev += log(1.0 - prob_b)
            end
        end
    end

    # Record child sizes before merge (for MFM partition prior)
    n_child_a_size = Int(state.clusters[slot_a].n)
    n_child_b_size = Int(state.clusters[slot_b].n)

    # Execute the merge
    a_slot = b_slot == slot_b ? slot_a : slot_b
    @inbounds for loc_idx in all_members
        if state.assignments[loc_idx] == b_slot
            lp = loc_precs[loc_idx]
            state.clusters[b_slot] = remove_loc(state.clusters[b_slot], lp)
            state.clusters[a_slot] = add_loc(state.clusters[a_slot], lp)
            state.assignments[loc_idx] = Int16(a_slot)
        end
    end
    _deactivate_cluster!(state, b_slot)

    n_parent_size = n_child_a_size + n_child_b_size

    # Forward proposal: select pair = uniform over K(K-1)/2
    log_q_fwd = -log(K * (K - 1) / 2)

    # Reverse proposal: select merged cluster (1/(K-1)) × SMC assignment
    K_new = K - 1
    log_q_rev = -log(K_new) + log_q_assign_rev

    return log_q_fwd, log_q_rev, n_parent_size, n_child_a_size, n_child_b_size
end


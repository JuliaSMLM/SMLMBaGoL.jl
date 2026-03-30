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

        # Compute allocation weight for each active cluster:
        #   P(z_i = k | rest) ∝ (n_{-i,k} + γ) × predictive(x_i | cluster_k)
        # The (n_{-i,k} + γ) term is the Dirichlet-Multinomial partition prior,
        # giving a "rich get richer" effect that compensates for the combinatorial
        # explosion of partitions at higher K.
        γ = shape
        for i in 1:K
            slot = active_slots[i]
            cs = state.clusters[slot]
            log_probs[i] = log(Float64(cs.n) + γ) + log_predictive_locmix(cs, lp, grid)
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
# Direct K sampling with count-model posterior
# ============================================================================

"""
    _total_spatial_lml(state) -> Float64

Sum of log marginal likelihoods across all active clusters.
Uses neighbor-filtered locmix for O(K×|A|) instead of O(K×N).
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
    _log_dm_partition(state, N, γ) -> Float64

Log Dirichlet-Multinomial partition prior for the current allocation.

P(z|K) = Γ(Kγ) / [Γ(γ)^K × Γ(N+Kγ)] × ∏_k Γ(n_k + γ)

This prior regularizes the allocation by penalizing the combinatorial
explosion of partitions at higher K.
"""
function _log_dm_partition(state::CollapsedState, N::Int, γ::Float64)
    K = state.n_active
    lp = loggamma(K * γ) - K * loggamma(γ) - loggamma(N + K * γ)
    @inbounds for j in eachindex(state.active)
        if state.active[j]
            lp += loggamma(Float64(state.clusters[j].n) + γ)
        end
    end
    return lp
end

# ============================================================================
# Sequential predictive allocation (computable proposal density)
# ============================================================================

"""
    _log_sequential_allocation(member_indices, is_in_b, loc_precs, grid, γ) -> Float64

Compute the log probability of the allocation `is_in_b` under sequential
predictive allocation of the members of a parent cluster.

Sequential allocation:
  - member[1] seeds sub-cluster A, member[2] seeds sub-cluster B
  - For j = 3..m: assign member[j] proportional to (n_sub + γ) × predictive

Returns the log probability of the given allocation (product of conditional
assignment probabilities for members 3..m).
"""
function _log_sequential_allocation(member_indices::Vector{Int},
                                     is_in_b::Vector{Bool},
                                     loc_precs::Vector{LocPrecision},
                                     grid::LocmixGrid,
                                     γ::Float64)
    m = length(member_indices)
    m < 3 && return 0.0  # Only seeds, no choices

    # Seed: member[1] → A, member[2] → B
    cs_a = add_loc(ClusterStats(), loc_precs[member_indices[1]])
    cs_b = add_loc(ClusterStats(), loc_precs[member_indices[2]])

    log_q = 0.0
    for idx in 3:m
        loc_idx = member_indices[idx]
        lp = loc_precs[loc_idx]

        # DM-weighted spatial predictive for each sub-cluster
        log_pred_a = log_marginal_likelihood_locmix(add_loc(cs_a, lp), grid) -
                     log_marginal_likelihood_locmix(cs_a, grid)
        log_pred_b = log_marginal_likelihood_locmix(add_loc(cs_b, lp), grid) -
                     log_marginal_likelihood_locmix(cs_b, grid)

        log_w_a = log(Float64(cs_a.n) + γ) + log_pred_a
        log_w_b = log(Float64(cs_b.n) + γ) + log_pred_b

        # Normalize
        max_lw = max(log_w_a, log_w_b)
        p_b = exp(log_w_b - max_lw) / (exp(log_w_a - max_lw) + exp(log_w_b - max_lw))
        p_b = clamp(p_b, 1e-300, 1.0 - 1e-300)

        if is_in_b[idx]
            cs_b = add_loc(cs_b, lp)
            log_q += log(p_b)
        else
            cs_a = add_loc(cs_a, lp)
            log_q += log(1.0 - p_b)
        end
    end
    return log_q
end

"""
    _do_sequential_split!(state, parent_slot, locs, γ) -> (new_slot, log_q, member_indices, is_in_b)

Split `parent_slot` using sequential predictive allocation.
Returns the new cluster slot, log proposal density, member indices, and
allocation vector for computing reverse proposal density.
"""
function _do_sequential_split!(state::CollapsedState, parent_slot::Int,
                                locs::Vector{<:SMLMData.AbstractEmitter},
                                γ::Float64)
    loc_precs = state._loc_precs
    grid = state._locmix_grid
    N = length(locs)

    # Collect member indices (sorted for canonical ordering)
    member_indices = Int[]
    @inbounds for loc_idx in 1:N
        if state.assignments[loc_idx] == parent_slot
            push!(member_indices, loc_idx)
        end
    end
    m = length(member_indices)

    # Need at least 2 members to split
    if m < 2
        return -1, -Inf, member_indices, Bool[]
    end

    # Allocate new slot
    new_slot = _find_inactive_slot(state)

    # Seeds: member[1] → A (stays), member[2] → B (new)
    is_in_b = falses(m)
    is_in_b[2] = true

    cs_a = add_loc(ClusterStats(), loc_precs[member_indices[1]])
    cs_b = add_loc(ClusterStats(), loc_precs[member_indices[2]])
    log_q = 0.0

    # Sequential allocation for remaining members
    for idx in 3:m
        loc_idx = member_indices[idx]
        lp = loc_precs[loc_idx]

        log_pred_a = log_marginal_likelihood_locmix(add_loc(cs_a, lp), grid) -
                     log_marginal_likelihood_locmix(cs_a, grid)
        log_pred_b = log_marginal_likelihood_locmix(add_loc(cs_b, lp), grid) -
                     log_marginal_likelihood_locmix(cs_b, grid)

        log_w_a = log(Float64(cs_a.n) + γ) + log_pred_a
        log_w_b = log(Float64(cs_b.n) + γ) + log_pred_b

        max_lw = max(log_w_a, log_w_b)
        p_b = exp(log_w_b - max_lw) / (exp(log_w_a - max_lw) + exp(log_w_b - max_lw))
        p_b = clamp(p_b, 1e-300, 1.0 - 1e-300)

        # Sample
        if rand() < p_b
            cs_b = add_loc(cs_b, lp)
            is_in_b[idx] = true
            log_q += log(p_b)
        else
            cs_a = add_loc(cs_a, lp)
            log_q += log(1.0 - p_b)
        end
    end

    # Apply the allocation to the state
    # Rebuild parent (A) from scratch, build new (B)
    state.clusters[parent_slot] = ClusterStats()
    new_cs = ClusterStats()
    @inbounds for idx in 1:m
        loc_idx = member_indices[idx]
        lp = loc_precs[loc_idx]
        if is_in_b[idx]
            new_cs = add_loc(new_cs, lp)
            state.assignments[loc_idx] = Int16(new_slot)
        else
            state.clusters[parent_slot] = add_loc(state.clusters[parent_slot], lp)
        end
    end
    _activate_cluster!(state, new_slot, new_cs)

    return new_slot, log_q, member_indices, is_in_b
end

"""
    propose_split_merge!(state, locs, μ, shape) -> (Bool, Symbol)

Proper RJMCMC split/merge with computable proposal densities.

1. Propose K_new from π_count(K) ∝ P(N|K)  (count model cancels in MH)
2. For ΔK = +1: sequential predictive split (computable q_split)
   For ΔK = -1: deterministic merge (q_merge = selection probability)
   For |ΔK| > 1: chain ±1 steps
3. Accept/reject with full MH ratio:

   log α = Δ_spatial + Δ_partition + Δ_proposal

where count model terms cancel between posterior and proposal.

Returns (accepted, move_type) where move_type is :split or :merge.
"""
function propose_split_merge!(state::CollapsedState,
                               locs::Vector{<:SMLMData.AbstractEmitter},
                               μ::Float64, shape::Float64)
    N = length(locs)
    N < 2 && return false, :split
    K = state.n_active

    # Compute count-model posterior for K = 1..K_max
    K_max = max(2 * K, min(N, 30))
    log_posts = state._log_probs  # reuse workspace
    if K_max > length(log_posts)
        K_max = length(log_posts)
    end
    @inbounds for k in 1:K_max
        log_posts[k] = _log_count_posterior(k, N, shape, μ)
    end

    # Sample from normalized distribution (in-place log-sum-exp)
    max_lp = log_posts[1]
    @inbounds for k in 2:K_max
        if log_posts[k] > max_lp
            max_lp = log_posts[k]
        end
    end
    total = 0.0
    @inbounds for k in 1:K_max
        v = exp(log_posts[k] - max_lp)
        log_posts[k] = v
        total += v
    end

    u = rand() * total
    K_new = K_max
    cumsum_p = 0.0
    @inbounds for k in 1:K_max
        cumsum_p += log_posts[k]
        if u < cumsum_p
            K_new = k
            break
        end
    end

    K_new == K && return false, :split

    # Save state for potential rollback
    _save_rollback!(state)
    old_n_active = K
    old_len = length(state.clusters)

    γ = Float64(shape)

    # Compute target density components BEFORE the move
    lml_before = _total_spatial_lml(state)
    dm_before = _log_dm_partition(state, N, γ)

    # Execute split(s) or merge(s) with computable proposal densities
    move_type = K_new > K ? :split : :merge
    log_q_fwd = 0.0  # log forward proposal density
    log_q_rev = 0.0  # log reverse proposal density

    if K_new > K
        # Chain of splits: K → K+1 → ... → K_new
        for step in 1:(K_new - K)
            K_cur = K + step - 1

            # Select cluster uniformly at random
            target_slot = 0
            count = 0
            r = rand(1:K_cur)
            @inbounds for j in eachindex(state.active)
                if state.active[j]
                    count += 1
                    if count == r
                        target_slot = j
                        break
                    end
                end
            end

            # Forward: select (1/K_cur) + sequential allocation
            log_q_fwd += -log(Float64(K_cur))

            _, log_q_alloc, member_indices, is_in_b = _do_sequential_split!(
                state, target_slot, locs, γ)

            if log_q_alloc == -Inf
                # Can't split (cluster too small)
                _restore_rollback!(state, old_n_active, old_len)
                return false, :split
            end
            log_q_fwd += log_q_alloc

            # Reverse: uniform pair selection from K_cur+1 clusters
            # The specific pair (parent_slot, new_slot) is 1 of C(K_cur+1, 2) pairs
            K_after = K_cur + 1
            log_q_rev += -log(Float64(K_after * (K_after - 1)) / 2.0)
        end
    else
        # Chain of merges: K → K-1 → ... → K_new
        for step in 1:(K - K_new)
            K_cur = K - step + 1

            # Select pair uniformly at random
            n_pairs = K_cur * (K_cur - 1) ÷ 2
            pair_idx = rand(1:n_pairs)
            log_q_fwd += -log(Float64(n_pairs))

            # Enumerate active slots to find the selected pair
            slot_a, slot_b = 0, 0
            pair_count = 0
            @inbounds for j1 in eachindex(state.active)
                state.active[j1] || continue
                for j2 in (j1+1):length(state.active)
                    state.active[j2] || continue
                    pair_count += 1
                    if pair_count == pair_idx
                        slot_a, slot_b = j1, j2
                        break
                    end
                end
                slot_b > 0 && break
            end

            # Collect all members of both clusters (sorted for canonical ordering)
            member_indices = Int[]
            is_in_b = Bool[]
            @inbounds for loc_idx in 1:N
                a = state.assignments[loc_idx]
                if a == slot_a
                    push!(member_indices, loc_idx)
                    push!(is_in_b, false)
                elseif a == slot_b
                    push!(member_indices, loc_idx)
                    push!(is_in_b, true)
                end
            end

            # Compute reverse sequential allocation density
            # (what would the split proposal density be for this allocation?)
            log_q_alloc_rev = _log_sequential_allocation(
                member_indices, is_in_b, state._loc_precs, state._locmix_grid, γ)

            # Reverse: select cluster from K_cur-1 + sequential allocation
            K_after = K_cur - 1
            log_q_rev += -log(Float64(K_after)) + log_q_alloc_rev

            # Execute the merge: move all locs from slot_b into slot_a
            @inbounds for loc_idx in 1:N
                if state.assignments[loc_idx] == slot_b
                    lp = state._loc_precs[loc_idx]
                    state.clusters[slot_b] = remove_loc(state.clusters[slot_b], lp)
                    state.clusters[slot_a] = add_loc(state.clusters[slot_a], lp)
                    state.assignments[loc_idx] = Int16(slot_a)
                end
            end
            _deactivate_cluster!(state, slot_b)
        end
    end

    # Compute target density components AFTER the move
    lml_after = _total_spatial_lml(state)
    dm_after = _log_dm_partition(state, N, γ)

    # Full MH acceptance ratio (count model cancels between posterior and proposal):
    #   log α = Δ_spatial + Δ_partition + Δ_proposal
    Δ_spatial = lml_after - lml_before
    Δ_partition = dm_after - dm_before
    Δ_proposal = log_q_rev - log_q_fwd
    log_α = Δ_spatial + Δ_partition + Δ_proposal

    if log_α >= 0 || rand() < exp(log_α)
        return true, move_type
    else
        _restore_rollback!(state, old_n_active, old_len)
        return false, move_type
    end
end


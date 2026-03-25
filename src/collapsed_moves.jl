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
    _log_count_posterior(K, N, shape, μ, λ_K) -> Float64

Count-model log posterior for K emitters.

  π(K) ∝ P(K) × P(N|K)

where P(N|K) = NegBin(N; K×α, α/(α+μ)) is the total count model.

The theoretical MAP-N accuracy is limited by the count model's variance:
  σ_N = √(K × μ(μ+α)/α)
For K=8, μ=20, α=5: σ_N ≈ 28, giving ~28% theoretical maximum accuracy.
This is a fundamental property of the count model, not a sampler limitation.

Uses fixed μ₀ (prior mean) rather than adaptive μ to prevent the μ-K
positive feedback loop where adaptive μ tracks K, making the count model
non-informative for K changes.
"""
function _log_count_posterior(K::Int, N::Int,
                              shape::Float64, μ::Float64, λ_K::Float64)
    lp = log_prior_k(K, λ_K)
    p = shape / (shape + μ)
    lp += logpdf(NegativeBinomial(K * shape, p), N)
    return lp
end

# ============================================================================
# Allocation Gibbs sweep
# ============================================================================

"""
    gibbs_allocation_sweep!(state, locs, μ, shape, λ_K)

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
                                  μ::Float64, shape::Float64, λ_K::Float64)
    N = length(locs)
    loc_precs = state._loc_precs
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

        # Compute spatial predictive for each active cluster
        if state.use_locmix_prior
            for i in 1:K
                slot = active_slots[i]
                cs = state.clusters[slot]
                log_probs[i] = log_predictive_locmix(cs, lp, loc_precs)
            end
        else
            for i in 1:K
                slot = active_slots[i]
                cs = state.clusters[slot]
                log_probs[i] = log_predictive(cs, lp, state.log_area)
            end
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
"""
function _total_spatial_lml(state::CollapsedState)
    total = 0.0
    if state.use_locmix_prior
        @inbounds for j in eachindex(state.active)
            if state.active[j]
                total += log_ml_locmix(state.clusters[j], state._loc_precs)
            end
        end
    else
        @inbounds for j in eachindex(state.active)
            if state.active[j]
                total += log_marginal_likelihood(state.clusters[j], state.log_area)
            end
        end
    end
    return total
end

"""
    propose_split_merge!(state, locs, μ, shape, λ_K) -> (Bool, Symbol)

K sampling from count-model posterior with spatial MH correction.

1. Propose K_new from π_count(K) ∝ P(K) × P(N|K)  (count-only posterior)
2. Execute heuristic split/merge to adjust allocation
3. Accept/reject with spatial fit improvement (area-invariant):

   log α = Δ_fit = [Σ log ML(new) - Σ log ML(old)] + ΔK × log(A)

The +ΔK×log(A) cancels the -log(A) per cluster in log_marginal_likelihood,
implementing the spatial Poisson process prior. This makes the acceptance:
- Neutral (~0) for co-located emitters (pure Q-PAINT behavior)
- Positive for splits that align with spatial structure (beats Q-PAINT)

Returns (accepted, move_type) where move_type is :split or :merge.
"""
function propose_split_merge!(state::CollapsedState,
                               locs::Vector{<:SMLMData.AbstractEmitter},
                               μ::Float64, shape::Float64, λ_K::Float64)
    N = length(locs)
    N < 2 && return false, :split
    K = state.n_active

    # Compute count-model posterior for K = 1..K_max
    K_max = max(2 * K, min(N, 30))
    log_posts = state._log_probs  # reuse workspace
    @inbounds for k in 1:K_max
        log_posts[k] = _log_count_posterior(k, N, shape, μ, λ_K)
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

    # Compute spatial LML before the move
    lml_before = _total_spatial_lml(state)

    # Execute heuristic split or merge
    move_type = K_new > K ? :split : :merge
    if K_new > K
        _add_clusters!(state, locs, K_new - K)
    else
        _remove_clusters!(state, locs, K - K_new)
    end

    # Mini Gibbs relaxation: let the allocation settle spatially before
    # evaluating the MH acceptance. Without this, the heuristic split/merge
    # creates a random allocation that has terrible spatial LML, causing
    # the MH to reject even correct K changes.
    for _ in 1:5
        gibbs_allocation_sweep!(state, locs, μ, shape, λ_K)
    end

    # Compute spatial LML after the move + relaxation
    lml_after = _total_spatial_lml(state)

    # Spatial fit improvement.
    # For uniform prior: cancel -log(A) per cluster by adding ΔK × log(A).
    # For locmix prior: no area term exists, Δ_fit is the raw ML difference.
    if state.use_locmix_prior
        Δ_fit = lml_after - lml_before
    else
        ΔK = state.n_active - old_n_active
        Δ_fit = (lml_after - lml_before) + ΔK * state.log_area
    end

    # MH acceptance on spatial correction (count terms already in proposal)
    if Δ_fit >= 0 || rand() < exp(Δ_fit)
        return true, move_type
    else
        # Reject: restore state
        _restore_rollback!(state, old_n_active, old_len)
        return false, move_type
    end
end

"""
    _add_clusters!(state, locs, n_add)

Add n_add clusters by repeatedly splitting the largest active cluster.
Each split takes half the locs (by random selection) into a new cluster.
"""
function _add_clusters!(state::CollapsedState,
                         locs::Vector{<:SMLMData.AbstractEmitter},
                         n_add::Int)
    loc_precs = state._loc_precs
    N = length(locs)

    for _ in 1:n_add
        # Find the largest active cluster
        best_slot = 0
        best_n = 0
        @inbounds for j in eachindex(state.active)
            if state.active[j] && state.clusters[j].n > best_n
                best_n = Int(state.clusters[j].n)
                best_slot = j
            end
        end
        best_n < 2 && break  # Can't split a cluster with fewer than 2 locs

        # Collect loc indices in this cluster
        new_slot = _find_inactive_slot(state)
        new_cs = ClusterStats()
        n_moved = 0
        target = best_n ÷ 2

        # Move approximately half the locs to the new cluster (random selection)
        @inbounds for loc_idx in 1:N
            state.assignments[loc_idx] == best_slot || continue
            if n_moved < target && rand() < 0.5
                lp = loc_precs[loc_idx]
                state.clusters[best_slot] = remove_loc(state.clusters[best_slot], lp)
                new_cs = add_loc(new_cs, lp)
                state.assignments[loc_idx] = Int16(new_slot)
                n_moved += 1
            end
        end

        # Ensure we moved at least 1 loc
        if n_moved == 0
            @inbounds for loc_idx in 1:N
                if state.assignments[loc_idx] == best_slot
                    lp = loc_precs[loc_idx]
                    state.clusters[best_slot] = remove_loc(state.clusters[best_slot], lp)
                    new_cs = add_loc(new_cs, lp)
                    state.assignments[loc_idx] = Int16(new_slot)
                    break
                end
            end
        end

        _activate_cluster!(state, new_slot, new_cs)
    end
end

"""
    _remove_clusters!(state, locs, n_remove)

Remove n_remove clusters by merging the smallest active cluster into its
nearest neighbor (by posterior mean distance).
"""
function _remove_clusters!(state::CollapsedState,
                            locs::Vector{<:SMLMData.AbstractEmitter},
                            n_remove::Int)
    loc_precs = state._loc_precs
    N = length(locs)

    for _ in 1:n_remove
        state.n_active <= 1 && break

        # Find the smallest active cluster
        small_slot = 0
        small_n = typemax(Int)
        @inbounds for j in eachindex(state.active)
            if state.active[j] && state.clusters[j].n < small_n
                small_n = Int(state.clusters[j].n)
                small_slot = j
            end
        end

        # Find the nearest other active cluster (by posterior mean)
        sx, sy = if small_n > 0
            posterior_mean(state.clusters[small_slot])
        else
            0.0, 0.0
        end

        merge_slot = 0
        min_dist = Inf
        @inbounds for j in eachindex(state.active)
            j == small_slot && continue
            state.active[j] || continue
            mx, my = posterior_mean(state.clusters[j])
            d = (mx - sx)^2 + (my - sy)^2
            if d < min_dist
                min_dist = d
                merge_slot = j
            end
        end

        # Move all locs from small_slot to merge_slot
        @inbounds for loc_idx in 1:N
            if state.assignments[loc_idx] == small_slot
                lp = loc_precs[loc_idx]
                state.clusters[small_slot] = remove_loc(state.clusters[small_slot], lp)
                state.clusters[merge_slot] = add_loc(state.clusters[merge_slot], lp)
                state.assignments[loc_idx] = Int16(merge_slot)
            end
        end
        _deactivate_cluster!(state, small_slot)
    end
end


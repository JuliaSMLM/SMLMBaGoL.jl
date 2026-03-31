# Collapsed Gibbs sampler moves for BaGoL
#
# Move types:
# 1. Allocation Gibbs sweep — reassign each loc among K clusters (K fixed)
# 2. Split — divide one cluster into two via restricted Gibbs scan
# 3. Merge — combine two clusters into one
# 4. Birth — detach a loc from its cluster as a new singleton (K+1)
# 5. Death — absorb a singleton into another cluster (K-1)

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
    log_area = state.log_area
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

        # Predictive-only proposal weights (MH-corrected for DM target):
        #   q(z_i = k) ∝ predictive(x_i | cluster_k)
        # MH acceptance handles (n_{-i,k} + γ) DM factor: α = (n_new+γ)/(n_old+γ)
        γ = shape
        for i in 1:K
            slot = active_slots[i]
            cs = state.clusters[slot]
            log_probs[i] = log_predictive(cs, lp, log_area)
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

        # MH correction: propose from predictive, accept with DM ratio
        proposed_slot = active_slots[chosen]
        final_slot = proposed_slot
        if old_cluster > 0 && state.active[old_cluster] && proposed_slot != old_cluster
            n_proposed = Float64(state.clusters[proposed_slot].n)
            n_old = Float64(state.clusters[old_cluster].n)
            α = min(1.0, (n_proposed + γ) / (n_old + γ))
            if rand() >= α
                final_slot = old_cluster
            end
        end
        state.clusters[final_slot] = add_loc(state.clusters[final_slot], lp)
        state.assignments[loc_idx] = Int16(final_slot)
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
Uses flat uniform spatial prior with log_area.
"""
function _total_spatial_lml(state::CollapsedState)
    log_area = state.log_area
    total = 0.0
    @inbounds for j in eachindex(state.active)
        if state.active[j]
            total += log_marginal_likelihood(state.clusters[j], log_area)
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
# Restricted Gibbs sweeps (Jain-Neal improvement for split/merge)
# ============================================================================

"""
    _restricted_gibbs_sweep!(is_in_b, cs_a, cs_b, member_indices, loc_precs, grid, γ, track_density)

One restricted MH-Gibbs sweep over non-seed members of a split cluster.
Seeds (positions 1,2) are fixed; members 3..m are proposed proportional
to predictive only, then MH-corrected with DM ratio α = (n_target+γ)/(n_current+γ).

If `track_density=true`, returns the log transition density including
stay-via-rejection terms. Used for the final scan in Jain-Neal proposals.

Modifies `is_in_b` in place and returns updated (cs_a, cs_b, log_q).
"""
function _restricted_gibbs_sweep!(is_in_b::Union{BitVector, Vector{Bool}},
                                   cs_a::ClusterStats, cs_b::ClusterStats,
                                   member_indices::Vector{Int},
                                   loc_precs::Vector{LocPrecision},
                                   log_area::Float64, γ::Float64,
                                   track_density::Bool)
    m = length(member_indices)
    log_q = 0.0

    for idx in 3:m
        loc_idx = member_indices[idx]
        lp = loc_precs[loc_idx]
        was_in_b = is_in_b[idx]

        # Remove from current sub-cluster
        if was_in_b
            cs_b = remove_loc(cs_b, lp)
        else
            cs_a = remove_loc(cs_a, lp)
        end

        # Predictive-only proposal weights
        log_pred_a = log_marginal_likelihood(add_loc(cs_a, lp), log_area) -
                     log_marginal_likelihood(cs_a, log_area)
        log_pred_b = log_marginal_likelihood(add_loc(cs_b, lp), log_area) -
                     log_marginal_likelihood(cs_b, log_area)

        max_lp = max(log_pred_a, log_pred_b)
        q_b = exp(log_pred_b - max_lp) / (exp(log_pred_a - max_lp) + exp(log_pred_b - max_lp))
        q_b = clamp(q_b, 1e-300, 1.0 - 1e-300)

        # DM counts for MH correction (after removal of loc i)
        n_a = Float64(cs_a.n)
        n_b = Float64(cs_b.n)

        # Sample from predictive-only proposal, MH-correct with DM ratio
        if was_in_b
            # Currently in B. α(B→A) = min(1, (n_a+γ)/(n_b+γ))
            α_to_a = min(1.0, (n_a + γ) / (n_b + γ))
            proposed_a = rand() >= q_b
            ends_in_b = !(proposed_a && rand() < α_to_a)
        else
            # Currently in A. α(A→B) = min(1, (n_b+γ)/(n_a+γ))
            α_to_b = min(1.0, (n_b + γ) / (n_a + γ))
            proposed_b = rand() < q_b
            ends_in_b = proposed_b && rand() < α_to_b
        end

        # Track transition density including stay-via-rejection
        if track_density
            if was_in_b
                α_to_a = min(1.0, (n_a + γ) / (n_b + γ))
                if ends_in_b
                    # Stayed in B: P = q_b + (1-q_b)×(1-α_to_a)
                    log_q += log(max(q_b + (1.0 - q_b) * (1.0 - α_to_a), 1e-300))
                else
                    # Moved to A: P = (1-q_b) × α_to_a
                    log_q += log(max((1.0 - q_b) * α_to_a, 1e-300))
                end
            else
                α_to_b = min(1.0, (n_b + γ) / (n_a + γ))
                if ends_in_b
                    # Moved to B: P = q_b × α_to_b
                    log_q += log(max(q_b * α_to_b, 1e-300))
                else
                    # Stayed in A: P = (1-q_b) + q_b×(1-α_to_b)
                    log_q += log(max((1.0 - q_b) + q_b * (1.0 - α_to_b), 1e-300))
                end
            end
        end

        # Update state
        is_in_b[idx] = ends_in_b
        if ends_in_b
            cs_b = add_loc(cs_b, lp)
        else
            cs_a = add_loc(cs_a, lp)
        end
    end

    return cs_a, cs_b, log_q
end

"""
    _restricted_gibbs_transition_density(start_is_in_b, target_is_in_b, cs_a, cs_b,
                                          member_indices, loc_precs, log_area, γ)

Compute the log transition density of one restricted MH-Gibbs sweep producing
`target_is_in_b` starting from state `(cs_a, cs_b, start_is_in_b)`.

Uses predictive-only proposals with MH correction. Transition density includes
stay-via-rejection terms: P(stay) = q_same + q_other × (1-α).

The "hybrid state" at step j has:
- Members 1..j-1: in their TARGET positions (already transitioned)
- Member j: removed (being processed)
- Members j+1..m: in their START positions (not yet processed)

Does not modify any inputs.
"""
function _restricted_gibbs_transition_density(start_is_in_b::Union{BitVector, Vector{Bool}},
                                               target_is_in_b::Union{BitVector, Vector{Bool}},
                                               cs_a::ClusterStats, cs_b::ClusterStats,
                                               member_indices::Vector{Int},
                                               loc_precs::Vector{LocPrecision},
                                               log_area::Float64, γ::Float64)
    m = length(member_indices)
    m < 3 && return 0.0

    log_q = 0.0

    for idx in 3:m
        loc_idx = member_indices[idx]
        lp = loc_precs[loc_idx]
        was_in_b = start_is_in_b[idx]

        # Remove from START position (hybrid state: earlier members already transitioned)
        if was_in_b
            cs_b = remove_loc(cs_b, lp)
        else
            cs_a = remove_loc(cs_a, lp)
        end

        # Predictive-only proposal weights
        log_pred_a = log_marginal_likelihood(add_loc(cs_a, lp), log_area) -
                     log_marginal_likelihood(cs_a, log_area)
        log_pred_b = log_marginal_likelihood(add_loc(cs_b, lp), log_area) -
                     log_marginal_likelihood(cs_b, log_area)

        max_lp = max(log_pred_a, log_pred_b)
        q_b = exp(log_pred_b - max_lp) / (exp(log_pred_a - max_lp) + exp(log_pred_b - max_lp))
        q_b = clamp(q_b, 1e-300, 1.0 - 1e-300)

        # DM counts for MH correction
        n_a = Float64(cs_a.n)
        n_b = Float64(cs_b.n)

        # MH-corrected transition density to target assignment
        if was_in_b
            α_to_a = min(1.0, (n_a + γ) / (n_b + γ))
            if target_is_in_b[idx]
                # Stayed in B: P = q_b + (1-q_b)×(1-α_to_a)
                log_q += log(max(q_b + (1.0 - q_b) * (1.0 - α_to_a), 1e-300))
            else
                # Moved to A: P = (1-q_b) × α_to_a
                log_q += log(max((1.0 - q_b) * α_to_a, 1e-300))
            end
        else
            α_to_b = min(1.0, (n_b + γ) / (n_a + γ))
            if target_is_in_b[idx]
                # Moved to B: P = q_b × α_to_b
                log_q += log(max(q_b * α_to_b, 1e-300))
            else
                # Stayed in A: P = (1-q_b) + q_b×(1-α_to_b)
                log_q += log(max((1.0 - q_b) + q_b * (1.0 - α_to_b), 1e-300))
            end
        end

        # Add to TARGET position (building hybrid state for next step)
        if target_is_in_b[idx]
            cs_b = add_loc(cs_b, lp)
        else
            cs_a = add_loc(cs_a, lp)
        end
    end

    return log_q
end

"""
    _sample_sequential_launch(member_indices, loc_precs, grid, γ)

Sample a launch allocation via sequential predictive-only allocation.
Seeds at positions 1→A, 2→B; remaining members allocated proportional
to predictive only (no DM weighting).

Returns (is_in_b, cs_a, cs_b). Does NOT compute density (it's not needed —
only the final restricted Gibbs scan density enters the MH ratio).
"""
function _sample_sequential_launch(member_indices::Vector{Int},
                                    loc_precs::Vector{LocPrecision},
                                    log_area::Float64, γ::Float64)
    m = length(member_indices)
    is_in_b = falses(m)
    is_in_b[2] = true

    cs_a = add_loc(ClusterStats(), loc_precs[member_indices[1]])
    cs_b = add_loc(ClusterStats(), loc_precs[member_indices[2]])

    for idx in 3:m
        loc_idx = member_indices[idx]
        lp = loc_precs[loc_idx]

        # Predictive-only allocation (no DM weighting)
        log_pred_a = log_marginal_likelihood(add_loc(cs_a, lp), log_area) -
                     log_marginal_likelihood(cs_a, log_area)
        log_pred_b = log_marginal_likelihood(add_loc(cs_b, lp), log_area) -
                     log_marginal_likelihood(cs_b, log_area)

        max_lp = max(log_pred_a, log_pred_b)
        p_b = exp(log_pred_b - max_lp) / (exp(log_pred_a - max_lp) + exp(log_pred_b - max_lp))
        p_b = clamp(p_b, 1e-300, 1.0 - 1e-300)

        if rand() < p_b
            cs_b = add_loc(cs_b, lp)
            is_in_b[idx] = true
        else
            cs_a = add_loc(cs_a, lp)
        end
    end

    return is_in_b, cs_a, cs_b
end

# ============================================================================
# Sequential predictive allocation (computable proposal density)
# ============================================================================

"""
    _log_sequential_allocation(member_indices, is_in_b, loc_precs, grid, γ) -> Float64

Compute the log probability of the allocation `is_in_b` under sequential
predictive-only allocation of the members of a parent cluster.

Sequential allocation:
  - member[1] seeds sub-cluster A, member[2] seeds sub-cluster B
  - For j = 3..m: assign member[j] proportional to predictive only (no DM)

Returns the log probability of the given allocation (product of conditional
assignment probabilities for members 3..m).
"""
function _log_sequential_allocation(member_indices::Vector{Int},
                                     is_in_b::Vector{Bool},
                                     loc_precs::Vector{LocPrecision},
                                     log_area::Float64,
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

        # Predictive-only allocation (no DM weighting)
        log_pred_a = log_marginal_likelihood(add_loc(cs_a, lp), log_area) -
                     log_marginal_likelihood(cs_a, log_area)
        log_pred_b = log_marginal_likelihood(add_loc(cs_b, lp), log_area) -
                     log_marginal_likelihood(cs_b, log_area)

        max_lp = max(log_pred_a, log_pred_b)
        p_b = exp(log_pred_b - max_lp) / (exp(log_pred_a - max_lp) + exp(log_pred_b - max_lp))
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
    _do_sequential_split!(state, parent_slot, locs, γ; n_restricted_scans=0)

Split `parent_slot` using sequential predictive allocation with random seeds,
optionally followed by restricted Gibbs scans (Jain-Neal improvement).

Two members are randomly selected as seeds (one for each sub-cluster),
ensuring all binary partitions are reachable. Remaining members are processed
in canonical (sorted by loc index) order for reproducible proposal density.

With `n_restricted_scans > 0`: the sequential allocation serves as a "launch
state," then `n_restricted_scans - 1` intermediate Gibbs sweeps improve the
allocation, and one final sweep produces the proposal with computable density.
Only the final sweep's density enters the MH ratio (intermediate sweeps are free).

Returns (new_slot, log_q, member_indices, is_in_b).
"""
function _do_sequential_split!(state::CollapsedState, parent_slot::Int,
                                locs::Vector{<:SMLMData.AbstractEmitter},
                                γ::Float64;
                                n_restricted_scans::Int = 5)
    loc_precs = state._loc_precs
    log_area = state.log_area
    N = length(locs)

    # Collect member indices
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

    # Randomly select two seed locs from cluster members.
    s1 = rand(1:m)
    member_indices[1], member_indices[s1] = member_indices[s1], member_indices[1]
    s2 = rand(2:m)
    member_indices[2], member_indices[s2] = member_indices[s2], member_indices[2]
    # Canonical ordering of remaining members for reproducible proposal density
    if m > 2
        sort!(@view member_indices[3:m])
    end

    # Allocate new slot
    new_slot = _find_inactive_slot(state)

    # Seeds: position 1 → A (stays), position 2 → B (new)
    is_in_b = falses(m)
    is_in_b[2] = true

    cs_a = add_loc(ClusterStats(), loc_precs[member_indices[1]])
    cs_b = add_loc(ClusterStats(), loc_precs[member_indices[2]])
    log_q = 0.0

    # Sequential allocation for remaining members (launch state)
    for idx in 3:m
        loc_idx = member_indices[idx]
        lp = loc_precs[loc_idx]

        # Predictive-only allocation (no DM weighting)
        log_pred_a = log_marginal_likelihood(add_loc(cs_a, lp), log_area) -
                     log_marginal_likelihood(cs_a, log_area)
        log_pred_b = log_marginal_likelihood(add_loc(cs_b, lp), log_area) -
                     log_marginal_likelihood(cs_b, log_area)

        max_lp = max(log_pred_a, log_pred_b)
        p_b = exp(log_pred_b - max_lp) / (exp(log_pred_a - max_lp) + exp(log_pred_b - max_lp))
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

    # Restricted Gibbs scans (Jain-Neal improvement)
    if n_restricted_scans > 0
        # Intermediate scans: improve allocation without tracking density
        for _ in 1:(n_restricted_scans - 1)
            cs_a, cs_b, _ = _restricted_gibbs_sweep!(is_in_b, cs_a, cs_b,
                member_indices, loc_precs, log_area, γ, false)
        end
        # Final scan: sample new allocation + track density
        # This density REPLACES the sequential allocation density
        cs_a, cs_b, log_q = _restricted_gibbs_sweep!(is_in_b, cs_a, cs_b,
            member_indices, loc_precs, log_area, γ, true)
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
    propose_split_merge!(state, locs, μ, shape, ρ; n_restricted_scans=0) -> (Bool, Symbol)

RJMCMC split/merge with |ΔK|=1 proposals and computable proposal densities.

1. Randomly choose split (K→K+1) or merge (K→K-1) with equal probability.
   Boundary: K=1 always split, K≥N always merge.
2. Execute one split or merge with Jain-Neal restricted Gibbs scans.
3. Accept/reject with full MH ratio:

   log α = Δ_spatial + Δ_partition + Δ_proposal + Δ_count + Δ_K_prior + Δ_move_type

The Poisson(ρA) K prior contributes Δ_K_prior = log_prior_k_poisson(K',ρ,A) - log_prior_k_poisson(K,ρ,A).
Combined with the flat spatial prior's -log(A) per cluster, the area factors cancel:
  split: Δ_K_prior = log(ρ) - log(K+1), independent of A.

Returns (accepted, move_type) where move_type is :split or :merge.
"""
function propose_split_merge!(state::CollapsedState,
                               locs::Vector{<:SMLMData.AbstractEmitter},
                               μ::Float64, shape::Float64, ρ::Float64;
                               n_restricted_scans::Int = 5)
    N = length(locs)
    N < 2 && return false, :split
    K = state.n_active
    A = exp(state.log_area)

    # Choose split or merge with boundary handling
    # b_K = P(proposing split at K), d_K = P(proposing merge at K)
    if K <= 1
        do_split = true
        b_K = 1.0
    elseif K >= N
        do_split = false
        b_K = 0.0
    else
        do_split = rand() < 0.5
        b_K = 0.5
    end
    d_K = 1.0 - b_K

    # Save state for potential rollback
    _save_rollback!(state)
    old_n_active = K
    old_len = length(state.clusters)

    γ = Float64(shape)

    # Compute target density components BEFORE the move
    lml_before = _total_spatial_lml(state)
    dm_before = _log_dm_partition(state, N, γ)

    log_q_fwd = 0.0  # log forward structural proposal density
    log_q_rev = 0.0  # log reverse structural proposal density

    if do_split
        K_new = K + 1
        move_type = :split

        # Select cluster uniformly at random
        target_slot = 0
        count = 0
        r = rand(1:K)
        @inbounds for j in eachindex(state.active)
            if state.active[j]
                count += 1
                if count == r
                    target_slot = j
                    break
                end
            end
        end

        # Forward: select (1/K) + split allocation
        log_q_fwd = -log(Float64(K))

        _, log_q_alloc, member_indices, is_in_b = _do_sequential_split!(
            state, target_slot, locs, γ;
            n_restricted_scans = n_restricted_scans)

        if log_q_alloc == -Inf
            # Can't split (cluster too small)
            _restore_rollback!(state, old_n_active, old_len)
            return false, :split
        end
        log_q_fwd += log_q_alloc

        # Reverse: uniform pair selection from K+1 clusters
        log_q_rev = -log(Float64(K_new * (K_new - 1)) / 2.0)

        # Birth/death rate correction: d_{K+1} / b_K
        d_K_new = K_new >= N ? 1.0 : 0.5
        Δ_move_type = log(d_K_new) - log(b_K)
    else
        K_new = K - 1
        move_type = :merge

        # Select pair uniformly at random
        n_pairs = K * (K - 1) ÷ 2
        pair_idx = rand(1:n_pairs)
        log_q_fwd = -log(Float64(n_pairs))

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

        # Collect all members of both clusters (sorted by loc index)
        member_indices = Int[]
        @inbounds for loc_idx in 1:N
            a = state.assignments[loc_idx]
            if a == slot_a || a == slot_b
                push!(member_indices, loc_idx)
            end
        end
        m_merge = length(member_indices)

        # Random seed selection matching the split's RJMCMC bijection.
        # Seed density 1/(m(m-1)) appears in both split and merge proposals
        # as auxiliary variables and cancels in the MH ratio.
        s1 = rand(1:m_merge)
        member_indices[1], member_indices[s1] = member_indices[s1], member_indices[1]
        s2 = rand(2:m_merge)
        member_indices[2], member_indices[s2] = member_indices[s2], member_indices[2]
        if m_merge > 2
            sort!(@view member_indices[3:m_merge])
        end

        # is_in_b relative to sub-cluster labels:
        # member[i] goes to sub-B iff in same original cluster as seed_2
        seed_b_cluster = state.assignments[member_indices[2]]
        is_in_b = [state.assignments[member_indices[i]] == seed_b_cluster for i in 1:m_merge]

        log_area = state.log_area

        # Compute reverse (split) allocation density
        if n_restricted_scans > 0
            # Jain-Neal: launch → intermediate scans → transition density to current
            # 1. Sample a launch state via sequential allocation from merged cluster
            launch_is_in_b, launch_cs_a, launch_cs_b = _sample_sequential_launch(
                member_indices, state._loc_precs, log_area, γ)

            # 2. Run intermediate restricted Gibbs scans on the launch state
            for _ in 1:(n_restricted_scans - 1)
                launch_cs_a, launch_cs_b, _ = _restricted_gibbs_sweep!(
                    launch_is_in_b, launch_cs_a, launch_cs_b,
                    member_indices, state._loc_precs, log_area, γ, false)
            end

            # 3. Compute transition density: one Gibbs sweep from intermediate → current
            log_q_alloc_rev = _restricted_gibbs_transition_density(
                launch_is_in_b, is_in_b,
                launch_cs_a, launch_cs_b,
                member_indices, state._loc_precs, log_area, γ)
        else
            # No restricted Gibbs: use sequential allocation density (Round 3 behavior)
            log_q_alloc_rev = _log_sequential_allocation(
                member_indices, is_in_b, state._loc_precs, log_area, γ)
        end

        # Reverse: select cluster + allocation density (seed density cancels)
        log_q_rev = -log(Float64(K_new)) + log_q_alloc_rev

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

        # Birth/death rate correction: b_{K-1} / d_K
        b_K_new = K_new <= 1 ? 1.0 : 0.5
        Δ_move_type = log(b_K_new) - log(d_K)
    end

    # Compute target density components AFTER the move
    lml_after = _total_spatial_lml(state)
    dm_after = _log_dm_partition(state, N, γ)

    # Full MH acceptance ratio:
    #   log α = Δ_spatial + Δ_partition + Δ_proposal + Δ_count + Δ_K_prior + Δ_move_type
    # Δ_K_prior: Poisson(ρA) prior on K. Combined with flat spatial -log(A) per cluster,
    # the area factors cancel: split contributes log(ρ) - log(K+1).
    Δ_spatial = lml_after - lml_before
    Δ_partition = dm_after - dm_before
    Δ_proposal = log_q_rev - log_q_fwd
    Δ_count = _log_count_posterior(K_new, N, shape, μ) -
              _log_count_posterior(K, N, shape, μ)
    Δ_K_prior = log_prior_k_poisson(K_new, ρ, A) -
                log_prior_k_poisson(K, ρ, A)
    log_α = Δ_spatial + Δ_partition + Δ_proposal + Δ_count + Δ_K_prior + Δ_move_type

    if log_α >= 0 || rand() < exp(log_α)
        return true, move_type
    else
        _restore_rollback!(state, old_n_active, old_len)
        return false, move_type
    end
end

# ============================================================================
# Birth/death moves (incremental K-mixing)
# ============================================================================

"""
    propose_birth_death!(state, locs, μ, shape, ρ) -> (Bool, Symbol)

Birth/death move for incremental K-mixing.

Birth (K → K+1): pick a random non-sole-occupant loc, detach it as a singleton.
Death (K → K-1): pick a random singleton, absorb it into a cluster via
DM-weighted predictive.

MH acceptance:
  log α = Δ_spatial + Δ_partition + Δ_proposal + Δ_count + Δ_K_prior

The Poisson(ρA) K prior contributes Δ_K_prior. Combined with flat spatial
prior's -log(A) per cluster, the area factors cancel.

Returns (accepted, move_type) where move_type is :birth or :death.
"""
function propose_birth_death!(state::CollapsedState,
                               locs::Vector{<:SMLMData.AbstractEmitter},
                               μ::Float64, shape::Float64, ρ::Float64)
    N = length(locs)
    K = state.n_active
    γ = Float64(shape)
    loc_precs = state._loc_precs
    log_area = state.log_area
    A = exp(log_area)

    # Count singletons (clusters with n=1)
    n_singletons = 0
    @inbounds for j in eachindex(state.active)
        if state.active[j] && state.clusters[j].n == 1
            n_singletons += 1
        end
    end
    n_eligible = N - n_singletons  # locs in clusters with n ≥ 2

    # Feasibility
    can_birth = n_eligible > 0 && K < N
    can_death = n_singletons > 0 && K > 1

    if !can_birth && !can_death
        return false, :birth
    end

    p_birth = can_birth && can_death ? 0.5 :
              can_birth              ? 1.0 : 0.0

    do_birth = rand() < p_birth

    _save_rollback!(state)
    old_n_active = K
    old_len = length(state.clusters)

    lml_before = _total_spatial_lml(state)
    dm_before = _log_dm_partition(state, N, γ)

    log_q_fwd = 0.0
    log_q_rev = 0.0
    K_new = do_birth ? K + 1 : K - 1
    move_type = do_birth ? :birth : :death

    if do_birth
        # === BIRTH: K → K+1 ===
        # Pick random non-sole-occupant loc
        target = rand(1:n_eligible)
        chosen_loc = 0
        cnt = 0
        @inbounds for loc_idx in 1:N
            c = Int(state.assignments[loc_idx])
            if c > 0 && state.active[c] && state.clusters[c].n > 1
                cnt += 1
                if cnt == target
                    chosen_loc = loc_idx
                    break
                end
            end
        end

        old_cluster = Int(state.assignments[chosen_loc])
        lp = loc_precs[chosen_loc]

        # Execute: detach loc as singleton
        state.clusters[old_cluster] = remove_loc(state.clusters[old_cluster], lp)
        new_slot = _find_inactive_slot(state)
        _activate_cluster!(state, new_slot, add_loc(ClusterStats(), lp))
        state.assignments[chosen_loc] = Int16(new_slot)

        # Forward density: p_birth × (1/N_eligible)
        log_q_fwd = log(p_birth) - log(Float64(n_eligible))

        # Reverse density: death from state x'
        # Count singletons in x'
        n_sing_x = n_singletons + 1  # new singleton
        if state.clusters[old_cluster].n == 1
            n_sing_x += 1  # old cluster also became singleton
        end

        # Death feasibility in x'
        can_d = n_sing_x > 0 && K_new > 1
        can_b = (N - n_sing_x) > 0 && K_new < N
        p_death_x = can_d && can_b ? 0.5 : can_d ? 1.0 : 0.0

        # DM-weighted predictive for absorbing loc i into each cluster (excl new singleton)
        log_probs = state._log_probs
        dest_count = 0
        log_w_dest = -Inf
        @inbounds for j in eachindex(state.active)
            if state.active[j] && j != new_slot
                dest_count += 1
                cs = state.clusters[j]
                log_probs[dest_count] = log(Float64(cs.n) + γ) +
                                         log_predictive(cs, lp, log_area)
                if j == old_cluster
                    log_w_dest = log_probs[dest_count]
                end
            end
        end

        # log-sum-exp
        max_lw = log_probs[1]
        @inbounds for i in 2:dest_count
            if log_probs[i] > max_lw; max_lw = log_probs[i]; end
        end
        lse = 0.0
        @inbounds for i in 1:dest_count
            lse += exp(log_probs[i] - max_lw)
        end
        log_sum = max_lw + log(lse)

        log_q_rev = log(p_death_x) - log(Float64(n_sing_x)) +
                    log_w_dest - log_sum

    else
        # === DEATH: K → K-1 ===
        # Pick random singleton
        target = rand(1:n_singletons)
        singleton_slot = 0
        cnt = 0
        @inbounds for j in eachindex(state.active)
            if state.active[j] && state.clusters[j].n == 1
                cnt += 1
                if cnt == target
                    singleton_slot = j
                    break
                end
            end
        end

        # Find the loc in the singleton
        chosen_loc = 0
        @inbounds for loc_idx in 1:N
            if state.assignments[loc_idx] == singleton_slot
                chosen_loc = loc_idx
                break
            end
        end

        lp = loc_precs[chosen_loc]

        # Predictive-only destination weights (DM enters via Δ_partition in MH)
        log_probs = state._log_probs
        dest_slots = state._active_slots  # reuse buffer
        dest_count = 0
        @inbounds for j in eachindex(state.active)
            if state.active[j] && j != singleton_slot
                dest_count += 1
                cs = state.clusters[j]
                log_probs[dest_count] = log_predictive(cs, lp, log_area)
                dest_slots[dest_count] = j
            end
        end

        if dest_count == 0
            _restore_rollback!(state, old_n_active, old_len)
            return false, :death
        end

        # log-sum-exp + sample
        max_lw = log_probs[1]
        @inbounds for i in 2:dest_count
            if log_probs[i] > max_lw; max_lw = log_probs[i]; end
        end
        total = 0.0
        @inbounds for i in 1:dest_count
            total += exp(log_probs[i] - max_lw)
        end
        log_sum = max_lw + log(total)
        inv_total = 1.0 / total

        u = rand()
        cumsum_p = 0.0
        chosen_dest = dest_count
        @inbounds for i in 1:dest_count
            cumsum_p += exp(log_probs[i] - max_lw) * inv_total
            if u < cumsum_p
                chosen_dest = i
                break
            end
        end
        dest_slot = dest_slots[chosen_dest]

        # Forward density: p_death × (1/n_singletons) × w(dest)/Σw
        log_q_fwd = log(1.0 - p_birth) - log(Float64(n_singletons)) +
                    log_probs[chosen_dest] - log_sum

        # Execute death
        state.clusters[singleton_slot] = remove_loc(state.clusters[singleton_slot], lp)
        _deactivate_cluster!(state, singleton_slot)
        state.clusters[dest_slot] = add_loc(state.clusters[dest_slot], lp)
        state.assignments[chosen_loc] = Int16(dest_slot)

        # Reverse density: birth from state x'
        n_sing_x = 0
        @inbounds for j in eachindex(state.active)
            if state.active[j] && state.clusters[j].n == 1
                n_sing_x += 1
            end
        end
        n_elig_x = N - n_sing_x

        can_b = n_elig_x > 0 && K_new < N
        can_d = n_sing_x > 0 && K_new > 1
        p_birth_x = can_b && can_d ? 0.5 : can_b ? 1.0 : 0.0

        log_q_rev = log(p_birth_x) - log(Float64(n_elig_x))
    end

    # Post-move target densities
    lml_after = _total_spatial_lml(state)
    dm_after = _log_dm_partition(state, N, γ)

    Δ_spatial = lml_after - lml_before
    Δ_partition = dm_after - dm_before
    Δ_proposal = log_q_rev - log_q_fwd
    Δ_count = _log_count_posterior(K_new, N, shape, μ) -
              _log_count_posterior(K, N, shape, μ)
    Δ_K_prior = log_prior_k_poisson(K_new, ρ, A) -
                log_prior_k_poisson(K, ρ, A)

    log_α = Δ_spatial + Δ_partition + Δ_proposal + Δ_count + Δ_K_prior

    if log_α >= 0 || rand() < exp(log_α)
        return true, move_type
    else
        _restore_rollback!(state, old_n_active, old_len)
        return false, move_type
    end
end


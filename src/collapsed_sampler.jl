# Collapsed Gibbs sampler for BaGoL
#
# Main sampler loop that coordinates allocation Gibbs sweeps,
# block birth/death moves, hierarchical updates, and accumulators.

# ============================================================================
# State initialization and main sampler
# ============================================================================

"""
    initialize_collapsed_state(locs; spatial_prior) -> CollapsedState

Initialize collapsed state: all locs in one cluster.
"""
function initialize_collapsed_state(locs::Vector{<:SMLMData.AbstractEmitter},
                                     spatial_prior::UniformSpatialPrior;
                                     flat_prior::Bool=false)
    N = length(locs)
    log_area = log(area(spatial_prior))

    # Start with one cluster containing all locs
    cs = ClusterStats()
    for loc in locs
        cs = add_loc(cs, loc)
    end

    assignments = fill(Int16(1), N)
    clusters = [cs]
    active = BitVector([true])
    n_active = 1

    # Precompute loc precisions (locs never change during chain)
    _loc_precs = precompute_loc_precisions(locs)

    # Build grid-based locmix prior (O(N × grid_size) once)
    _locmix_grid = build_locmix_grid(_loc_precs; flat=flat_prior)

    # Pre-allocate workspace buffers
    max_K = max(N, 16)  # Upper bound on cluster count
    _perm = collect(1:N)
    _active_slots = Vector{Int}(undef, max_K)
    _log_probs = Vector{Float64}(undef, max_K + 1)
    _rollback_assignments = similar(assignments)
    _rollback_clusters = similar(clusters)
    _rollback_active = similar(active)

    return CollapsedState(assignments, clusters, active, n_active, log_area,
                          _loc_precs, _locmix_grid,
                          _perm, _active_slots, _log_probs,
                          _rollback_assignments, _rollback_clusters, _rollback_active)
end

"""
    run_collapsed_chain(locs; kwargs...) -> CollapsedChainResult

Run the collapsed Gibbs sampler on a set of localizations.

# Arguments
- `locs`: Vector of localizations (Emitter2DFit or similar)
- `n_iterations=10000`: Total MCMC iterations
- `burn_in=2000`: Burn-in iterations before accumulating
- `shape=2.0`: Initial Gamma shape for count distribution
- `learn_shape=true`: Update shape during MCMC
- `hierarchical_interval=100`: Iterations between μ/shape MH updates
- `accumulators=AbstractAccumulator[]`: List of accumulators to update after burn-in
- `verbose=false`: Print progress
- `callback`: Optional callback `(iter, state, μ, shape) -> nothing`
- `callback_interval=1`: How often to call callback
# Move distribution
- 50%: Allocation Gibbs sweep (full sweep per iteration)
- 25%: Split (K → K+1, restricted Gibbs scan)
- 25%: Merge (K → K-1, uniform pair selection)
"""
function run_collapsed_chain(
    locs::Vector{<:SMLMData.AbstractEmitter};
    n_iterations::Int = 10000,
    burn_in::Int = 2000,
    shape::Float64 = 2.0,
    learn_shape::Bool = true,
    μ_prior_shape::Float64 = 2.0,
    μ_prior_scale::Float64 = 5.0,
    shape_prior_shape::Float64 = 2.0,
    shape_prior_scale::Float64 = 1.0,
    hierarchical_interval::Int = 100,
    accumulators::Vector{<:AbstractAccumulator} = AbstractAccumulator[],
    verbose::Bool = false,
    callback::Union{Function, Nothing} = nothing,
    callback_interval::Int = 1,
    flat_prior::Bool = false
)
    N = length(locs)
    if N == 0
        error("No localizations provided")
    end

    # Initialize
    spatial_prior = UniformSpatialPrior(locs)
    state = initialize_collapsed_state(locs, spatial_prior; flat_prior)

    μ = μ_prior_shape * μ_prior_scale  # Initial μ from prior mean
    current_shape = shape

    acceptance = Dict{Symbol, Tuple{Int, Int}}(
        :gibbs_sweep => (0, 0),
        :split => (0, 0),
        :merge => (0, 0),
    )

    config_nt = (
        μ_prior_shape = μ_prior_shape,
        μ_prior_scale = μ_prior_scale,
        shape_prior_shape = shape_prior_shape,
        shape_prior_scale = shape_prior_scale,
    )

    for iter in 1:n_iterations
        r = rand()

        if r < 0.50
            # Gibbs allocation sweep (always "accepts" — it's exact Gibbs)
            gibbs_allocation_sweep!(state, locs, μ, current_shape)
            prev = acceptance[:gibbs_sweep]
            acceptance[:gibbs_sweep] = (prev[1] + 1, prev[2] + 1)
        else
            # Split-merge — use current μ (no fixed μ₀ hack)
            accepted, move_type = propose_split_merge!(state, locs, μ, current_shape)
            prev = acceptance[move_type]
            acceptance[move_type] = (prev[1] + (accepted ? 1 : 0), prev[2] + 1)
        end

        # Hierarchical updates
        if iter % hierarchical_interval == 0
            μ = _update_mu_collapsed(state, μ, current_shape, config_nt)
            if learn_shape
                current_shape = _update_shape_collapsed(state, μ, current_shape, config_nt)
            end
        end

        # Update accumulators after burn-in
        if iter > burn_in
            for acc in accumulators
                accumulator_update!(acc, state, locs, μ, current_shape, iter)
            end
        end

        # Callback
        if callback !== nothing && iter % callback_interval == 0
            callback(iter, state, μ, current_shape)
        end

        # Progress
        if verbose && iter % 1000 == 0
            K = state.n_active
            shape_str = learn_shape ? ", shape=$(round(current_shape, digits=2))" : ""
            println("Iter $iter: K=$K, μ=$(round(μ, digits=2))$shape_str")
        end
    end

    if verbose
        println("\nAcceptance rates:")
        for (move, (acc, tot)) in acceptance
            rate = tot > 0 ? round(100 * acc / tot, digits=1) : 0.0
            println("  $move: $rate% ($acc/$tot)")
        end
    end

    # Collect accumulator results
    acc_results = [accumulator_result(acc) for acc in accumulators]

    return CollapsedChainResult(state, μ, current_shape, acc_results, acceptance, n_iterations)
end

"""
    run_collapsed_iterations!(state, locs, n, μ, shape, accumulators, burn_in, current_iter;
                              acceptance=nothing)

Run n iterations on an existing collapsed state. Used for synchronized partitioned execution.
Returns updated_current_iter. If `acceptance` dict is provided, accumulates (accepted, total) counts.
"""
function run_collapsed_iterations!(
    state::CollapsedState,
    locs::Vector{<:SMLMData.AbstractEmitter},
    n::Int,
    μ::Float64,
    shape::Float64,
    accumulators::Vector{<:AbstractAccumulator},
    burn_in::Int,
    current_iter::Int;
    acceptance::Union{Dict{Symbol, Tuple{Int, Int}}, Nothing}=nothing
)
    for _ in 1:n
        current_iter += 1

        r = rand()
        if r < 0.50
            gibbs_allocation_sweep!(state, locs, μ, shape)
            if acceptance !== nothing
                prev = acceptance[:gibbs_sweep]
                acceptance[:gibbs_sweep] = (prev[1] + 1, prev[2] + 1)
            end
        else
            accepted, move_type = propose_split_merge!(state, locs, μ, shape)
            if acceptance !== nothing
                prev = acceptance[move_type]
                acceptance[move_type] = (prev[1] + (accepted ? 1 : 0), prev[2] + 1)
            end
        end

        # Update accumulators after burn-in
        if current_iter > burn_in
            for acc in accumulators
                accumulator_update!(acc, state, locs, μ, shape, current_iter)
            end
        end
    end

    return current_iter
end

"""
    extract_emitters(state, locs) -> Vector{Emitter2DFit}

Extract emitter positions and uncertainties from current collapsed state
using posterior mean and covariance from ClusterStats.
"""
function extract_emitters(state::CollapsedState,
                           locs::Vector{<:SMLMData.AbstractEmitter})
    emitters = SMLMData.Emitter2DFit[]
    id = 0
    for (j, cs) in enumerate(state.clusters)
        state.active[j] || continue
        cs.n == 0 && continue
        id += 1

        mx, my = posterior_mean(cs)
        Σ_xx, Σ_xy, Σ_yy = posterior_cov(cs)

        σ_x = sqrt(max(Σ_xx, 0.0))
        σ_y = sqrt(max(Σ_yy, 0.0))

        push!(emitters, SMLMData.Emitter2DFit(
            mx, my, Float64(cs.n), 0.0,  # photons = n_locs in cluster
            σ_x, σ_y, Σ_xy,
            0.0, 0.0, 1, 1, 0, id
        ))
    end
    return emitters
end

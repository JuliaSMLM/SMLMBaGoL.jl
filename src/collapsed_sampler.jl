# Collapsed Gibbs sampler for BaGoL
#
# Main sampler loop that coordinates allocation Gibbs sweeps,
# block birth/death moves, hierarchical updates, and accumulators.

"""
    initialize_collapsed_state(locs; spatial_prior) -> CollapsedState

Initialize collapsed state: all locs in one cluster.
"""
function initialize_collapsed_state(locs::Vector{<:SMLMData.AbstractEmitter},
                                     spatial_prior::UniformSpatialPrior)
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

    return CollapsedState(assignments, clusters, active, n_active, log_area)
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
- `λ_K`: Prior mean for emitter count (default: N/5)
- `hierarchical_interval=100`: Iterations between μ/shape MH updates
- `accumulators=AbstractAccumulator[]`: List of accumulators to update after burn-in
- `verbose=false`: Print progress
- `callback`: Optional callback `(iter, state, μ, shape) -> nothing`
- `callback_interval=1`: How often to call callback

# Move distribution
- 70%: Allocation Gibbs sweep (full sweep per iteration)
- 15%: Block birth
- 15%: Block death
"""
function run_collapsed_chain(
    locs::Vector{<:SMLMData.AbstractEmitter};
    n_iterations::Int = 10000,
    burn_in::Int = 2000,
    shape::Float64 = 2.0,
    learn_shape::Bool = true,
    λ_K::Float64 = Float64(length(locs)) / 5.0,
    μ_prior_shape::Float64 = 2.0,
    μ_prior_scale::Float64 = 5.0,
    shape_prior_shape::Float64 = 2.0,
    shape_prior_scale::Float64 = 1.0,
    hierarchical_interval::Int = 100,
    accumulators::Vector{<:AbstractAccumulator} = AbstractAccumulator[],
    verbose::Bool = false,
    callback::Union{Function, Nothing} = nothing,
    callback_interval::Int = 1
)
    N = length(locs)
    if N == 0
        error("No localizations provided")
    end

    # Initialize
    spatial_prior = UniformSpatialPrior(locs)
    state = initialize_collapsed_state(locs, spatial_prior)

    μ = μ_prior_shape * μ_prior_scale  # Initial μ from prior mean
    current_shape = shape

    acceptance = Dict{Symbol, Tuple{Int, Int}}(
        :gibbs_sweep => (0, 0),
        :block_birth => (0, 0),
        :block_death => (0, 0)
    )

    config_nt = (
        μ_prior_shape = μ_prior_shape,
        μ_prior_scale = μ_prior_scale,
        shape_prior_shape = shape_prior_shape,
        shape_prior_scale = shape_prior_scale,
    )

    for iter in 1:n_iterations
        r = rand()

        if r < 0.70
            # Gibbs allocation sweep (always "accepts" — it's exact Gibbs)
            gibbs_allocation_sweep!(state, locs, μ, current_shape, λ_K)
            prev = acceptance[:gibbs_sweep]
            acceptance[:gibbs_sweep] = (prev[1] + 1, prev[2] + 1)
        elseif r < 0.85
            # Block birth
            accepted = propose_block_birth!(state, locs, μ, current_shape, λ_K)
            prev = acceptance[:block_birth]
            acceptance[:block_birth] = (prev[1] + (accepted ? 1 : 0), prev[2] + 1)
        else
            # Block death
            accepted = propose_block_death!(state, locs, μ, current_shape, λ_K)
            prev = acceptance[:block_death]
            acceptance[:block_death] = (prev[1] + (accepted ? 1 : 0), prev[2] + 1)
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
    run_collapsed_iterations!(state, locs, n, μ, shape, λ_K, accumulators, burn_in, current_iter)

Run n iterations on an existing collapsed state. Used for synchronized partitioned execution.
Returns (new_μ, new_shape, updated_current_iter).
"""
function run_collapsed_iterations!(
    state::CollapsedState,
    locs::Vector{<:SMLMData.AbstractEmitter},
    n::Int,
    μ::Float64,
    shape::Float64,
    λ_K::Float64,
    accumulators::Vector{<:AbstractAccumulator},
    burn_in::Int,
    current_iter::Int
)
    for _ in 1:n
        current_iter += 1

        r = rand()
        if r < 0.70
            gibbs_allocation_sweep!(state, locs, μ, shape, λ_K)
        elseif r < 0.85
            propose_block_birth!(state, locs, μ, shape, λ_K)
        else
            propose_block_death!(state, locs, μ, shape, λ_K)
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
            mx, my, 0.0, 0.0,
            σ_x, σ_y, Σ_xy,
            0.0, 0.0, 1, 1, 0, id
        ))
    end
    return emitters
end

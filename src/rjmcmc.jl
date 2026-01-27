# Main RJMCMC algorithm for BaGoL

"""
Perform one RJMCMC step (propose and accept/reject one move).

Returns (accepted::Bool, move_type::Symbol).
"""
function rjmcmc_step!(
    chain::RJMCMCChain,
    locs::Vector{<:SMLMData.AbstractEmitter},
    spatial_prior::UniformSpatialPrior
)
    # Randomly select move type with weighted probabilities
    # More allocate moves help equilibrate after birth/death
    r = rand()
    if r < 0.1
        move_type = :birth
    elseif r < 0.2
        move_type = :death
    elseif r < 0.4
        move_type = :move
    else
        move_type = :allocate  # 60% allocate
    end

    accepted = false
    if move_type == :birth
        accepted = propose_birth!(chain, locs, spatial_prior)
    elseif move_type == :death
        accepted = propose_death!(chain, locs, spatial_prior)
    elseif move_type == :move
        accepted = propose_move!(chain, locs, spatial_prior)
    elseif move_type == :allocate
        accepted = propose_allocate!(chain, locs, spatial_prior)
    end

    # Update acceptance statistics
    prev = chain.acceptance[move_type]
    chain.acceptance[move_type] = (prev[1] + (accepted ? 1 : 0), prev[2] + 1)

    chain.iteration += 1
    return accepted, move_type
end

"""
Record current state as a sample.
"""
function record_sample!(chain::RJMCMCChain)
    state = chain.current_state
    # Deep copy emitters
    emitters_copy = [Emitter(e.x, e.y, copy(e.allocated)) for e in state.emitters]
    sample = BaGoLSample(emitters_copy, state.log_posterior, chain.μ, chain.shape)
    push!(chain.samples, sample)
end

"""
    run_iterations!(chain, locs, spatial_prior, n; record_after_burn_in=true)

Run n iterations on an existing chain. Used for synchronized partitioned execution.
Does NOT do hierarchical updates (those are done globally between sync points).
"""
function run_iterations!(
    chain::RJMCMCChain,
    locs::Vector{<:SMLMData.AbstractEmitter},
    spatial_prior::UniformSpatialPrior,
    n::Int;
    record_after_burn_in::Bool = true
)
    config = chain.config
    for _ in 1:n
        _, _ = rjmcmc_step!(chain, locs, spatial_prior)

        # Record sample after burn-in
        if record_after_burn_in && chain.iteration > config.burn_in
            record_sample!(chain)
        end
    end
end

"""
Build BaGoLDiagnostics from chain.
"""
function build_diagnostics(chain::RJMCMCChain, posterior_k::Vector{Int}, n_emitters::Int; n_partitions::Int=1)
    acceptance_rates = Dict{Symbol, Float64}()
    for (move, (acc, tot)) in chain.acceptance
        acceptance_rates[move] = tot > 0 ? acc / tot : 0.0
    end
    return BaGoLDiagnostics(n_emitters, posterior_k, acceptance_rates, chain.μ, chain.shape, n_partitions)
end

"""
Build BaGoLDiagnostics from multiple chains (partitioned run).
"""
function build_diagnostics(chains::Vector{<:RJMCMCChain}, posterior_k::Vector{Int}, n_emitters::Int)
    # Aggregate acceptance rates across chains
    acceptance_totals = Dict{Symbol, Tuple{Int, Int}}()
    for chain in chains
        for (move, (acc, tot)) in chain.acceptance
            prev = get(acceptance_totals, move, (0, 0))
            acceptance_totals[move] = (prev[1] + acc, prev[2] + tot)
        end
    end
    acceptance_rates = Dict{Symbol, Float64}()
    for (move, (acc, tot)) in acceptance_totals
        acceptance_rates[move] = tot > 0 ? acc / tot : 0.0
    end
    # Use first chain for final μ, shape (all chains have same value after global updates)
    return BaGoLDiagnostics(n_emitters, posterior_k, acceptance_rates,
                            chains[1].μ, chains[1].shape, length(chains))
end

# ============================================================================
# Core chain-building function (internal)
# ============================================================================

"""
    run_bagol_chain(locs; kwargs...) -> RJMCMCChain

Internal function that runs MCMC and returns the chain.
For advanced users who need direct chain access.

# Callback Support
For animation or per-iteration diagnostics, use:
- `callback`: Function called each iteration with signature
  `callback(iter, move_type, accepted, state, μ, shape)` where state is the current BaGoLState
- `callback_interval`: How often to call callback (default 1 = every iteration)

Example:
```julia
records = []
chain = run_bagol_chain(locs;
    callback = (i, mt, acc, st, μ, shape) -> push!(records, (i, length(st.emitters), acc)),
    callback_interval = 10
)
```
"""
function run_bagol_chain(
    locs::Vector{<:SMLMData.AbstractEmitter};
    shape::Float64 = 2.0,
    learn_shape::Bool = true,
    λ_K::Float64 = Float64(length(locs)) / 5.0,
    n_iterations::Int = 10000,
    burn_in::Int = 2000,
    hierarchical_interval::Int = 100,
    μ_prior_shape::Float64 = 2.0,
    μ_prior_scale::Float64 = 5.0,
    shape_prior_shape::Float64 = 2.0,
    shape_prior_scale::Float64 = 1.0,
    verbose::Bool = true,
    callback::Union{Function, Nothing} = nothing,
    callback_interval::Int = 1
)
    config = RJMCMCConfig(;
        shape = shape,
        λ_K = λ_K,
        n_iterations = n_iterations,
        burn_in = burn_in,
        hierarchical_interval = hierarchical_interval,
        μ_prior_shape = μ_prior_shape,
        μ_prior_scale = μ_prior_scale,
        shape_prior_shape = shape_prior_shape,
        shape_prior_scale = shape_prior_scale
    )

    # Create spatial prior from data
    spatial_prior = UniformSpatialPrior(locs)

    # Initialize with one emitter at centroid (preserve coordinate type)
    T = typeof(locs[1].x)
    xs = [loc.x for loc in locs]
    ys = [loc.y for loc in locs]
    initial_emitter = Emitter(T(mean(xs)), T(mean(ys)))
    initial_state = BaGoLState([initial_emitter], 0.0)

    # Initialize allocations
    initialize_allocations!(initial_state, locs)
    update_emitter_positions!(initial_state, locs)

    # Create chain
    chain = RJMCMCChain(config, initial_state; shape_init=shape, learn_shape=learn_shape)

    # Run MCMC
    for i in 1:n_iterations
        accepted, move_type = rjmcmc_step!(chain, locs, spatial_prior)

        # Hierarchical update
        if i % hierarchical_interval == 0
            update_mu!(chain)
            if chain.learn_shape
                update_shape!(chain)
            end
        end

        # Record sample after burn-in
        if i > burn_in
            record_sample!(chain)
        end

        # Call user callback if provided
        if callback !== nothing && i % callback_interval == 0
            callback(i, move_type, accepted, chain.current_state, chain.μ, chain.shape)
        end

        # Progress
        if verbose && i % 1000 == 0
            k = length(chain.current_state.emitters)
            shape_str = chain.learn_shape ? ", shape=$(round(chain.shape, digits=2))" : ""
            println("Iteration $i: K=$k, μ=$(round(chain.μ, digits=2))$shape_str")
        end
    end

    if verbose
        println("\nAcceptance rates:")
        for (move, (acc, tot)) in chain.acceptance
            rate = tot > 0 ? round(100 * acc / tot, digits=1) : 0.0
            println("  $move: $rate% ($acc/$tot)")
        end
        if chain.learn_shape
            println("\nFinal shape = $(round(chain.shape, digits=2))")
        end
    end

    return chain
end

# ============================================================================
# Main API: run_bagol returns (BasicSMLD, BaGoLDiagnostics)
# ============================================================================

"""
    run_bagol(smld::SMLD; kwargs...) -> (BasicSMLD, BaGoLDiagnostics)

Run BaGoL analysis on an SMLD. Returns grouped emitters as BasicSMLD and diagnostics.

Uses precision-weighted DBSCAN to partition localizations, then runs parallel MCMC
on each partition with global hierarchical updates.

# Count Model
n_j ~ Gamma(shape, μ/shape) where:
- μ = mean locs per emitter
- shape = 1: exponential (dSTORM), shape > 1: peaked (DNA-PAINT)

# Partitioning Arguments
- `nsigma=3.0`: DBSCAN threshold in sigma units (Inf = no partitioning)
- `min_partition_size=0`: Minimum locs per partition (smaller clusters dropped as noise)
- `max_partition_size=1000`: Split partitions larger than this
- `skip_partition_size=typemax(Int)`: Skip partitions larger than this

# MCMC Arguments
- `sync_interval=500`: Iterations between global μ/shape updates
- `n_iterations=10000`: Total MCMC iterations
- `burn_in=2000`: Burn-in iterations before recording
- `shape=2.0`: Initial Gamma shape (1=exponential, higher=more peaked)
- `learn_shape=true`: Whether to update shape during MCMC
- `verbose=true`: Print progress

# Returns
- `BasicSMLD`: Grouped emitter positions with uncertainties
- `BaGoLDiagnostics`: n_emitters, posterior_k, acceptance_rates, final parameters
"""
function run_bagol(
    smld::SMLMData.SMLD;
    nsigma::Float64 = 3.0,
    min_partition_size::Int = 0,
    max_partition_size::Int = 1000,
    skip_partition_size::Int = typemax(Int),
    sync_interval::Int = 500,
    n_iterations::Int = 10000,
    burn_in::Int = 2000,
    shape::Float64 = 2.0,
    learn_shape::Bool = true,
    verbose::Bool = true,
    kwargs...
)
    locs = smld.emitters
    camera = smld.camera

    if verbose
        println("Partitioning $(length(locs)) localizations (nsigma=$nsigma)...")
    end

    # Partition the data using precision-weighted DBSCAN
    partitions, skipped = partition_locs(locs; nsigma, min_size=min_partition_size,
                                          max_size=max_partition_size,
                                          skip_size=skip_partition_size)

    if verbose
        println("  Created $(length(partitions)) partitions")
        if !isempty(skipped)
            n_skipped = sum(length(p.locs) for p in skipped)
            println("  Skipped $(length(skipped)) oversized clusters ($n_skipped locs)")
        end
    end

    if isempty(partitions)
        @warn "No valid partitions"
        empty_smld = SMLMData.BasicSMLD(SMLMData.Emitter2DFit[], camera, 1, 1)
        empty_diag = BaGoLDiagnostics(0, Int[], Dict{Symbol,Float64}(), 0.0, shape, 0)
        return empty_smld, empty_diag
    end

    # Initialize chains for each partition
    n_partitions = length(partitions)
    chains = Vector{RJMCMCChain}(undef, n_partitions)
    spatial_priors = Vector{UniformSpatialPrior}(undef, n_partitions)

    μ_prior_shape = get(kwargs, :μ_prior_shape, 2.0)
    μ_prior_scale = get(kwargs, :μ_prior_scale, 5.0)
    shape_prior_shape = get(kwargs, :shape_prior_shape, 2.0)
    shape_prior_scale = get(kwargs, :shape_prior_scale, 1.0)

    Threads.@threads for i in 1:n_partitions
        p_locs = partitions[i].locs
        config = RJMCMCConfig(;
            shape = shape,
            λ_K = get(kwargs, :λ_K, Float64(length(p_locs)) / 5.0),
            n_iterations = n_iterations,
            burn_in = burn_in,
            hierarchical_interval = sync_interval,
            μ_prior_shape = μ_prior_shape,
            μ_prior_scale = μ_prior_scale,
            shape_prior_shape = shape_prior_shape,
            shape_prior_scale = shape_prior_scale
        )

        spatial_priors[i] = UniformSpatialPrior(p_locs)

        T = typeof(p_locs[1].x)
        xs = [loc.x for loc in p_locs]
        ys = [loc.y for loc in p_locs]
        initial_emitter = Emitter(T(mean(xs)), T(mean(ys)))
        initial_state = BaGoLState([initial_emitter], 0.0)
        initialize_allocations!(initial_state, p_locs)
        update_emitter_positions!(initial_state, p_locs)

        chains[i] = RJMCMCChain(config, initial_state; shape_init=shape, learn_shape=learn_shape)
    end

    # Synchronized outer loop
    n_outer = div(n_iterations, sync_interval)
    for outer in 1:n_outer
        Threads.@threads for i in 1:n_partitions
            record = (outer * sync_interval) > burn_in
            run_iterations!(chains[i], partitions[i].locs, spatial_priors[i],
                           sync_interval; record_after_burn_in=record)
        end

        update_mu_global!(chains)
        if learn_shape
            update_shape_global!(chains)
        end

        if verbose && outer % max(1, n_outer ÷ 5) == 0
            total_K = sum(length(c.current_state.emitters) for c in chains)
            shape_str = learn_shape ? ", shape=$(round(chains[1].shape, digits=2))" : ""
            println("Sync $outer/$n_outer: total K=$total_K, μ=$(round(chains[1].μ, digits=2))$shape_str")
        end
    end

    # Run any remaining iterations
    remaining = n_iterations - n_outer * sync_interval
    if remaining > 0
        Threads.@threads for i in 1:n_partitions
            run_iterations!(chains[i], partitions[i].locs, spatial_priors[i],
                           remaining; record_after_burn_in=true)
        end
    end

    if verbose
        println("\nMerging partition results...")
    end

    # Compute boundary margin for merge
    sigmas = [mean_sigma(loc) for loc in locs]
    boundary_margin = 5.0 * median(sigmas)

    # Merge results from all partitions
    merged_emitters, posterior_k = merge_partition_results(chains, partitions, boundary_margin)
    diagnostics = build_diagnostics(chains, posterior_k, length(merged_emitters))
    result_smld = SMLMData.BasicSMLD(merged_emitters, camera, 1, 1)

    if verbose
        println("Result: $(length(merged_emitters)) emitters")
    end

    return result_smld, diagnostics
end

"""
    run_bagol(locs::Vector{<:AbstractEmitter}; camera, kwargs...) -> (BasicSMLD, BaGoLDiagnostics)

Run BaGoL on a vector of localizations. Requires camera argument.

See `run_bagol(smld::SMLD; ...)` for full documentation.
"""
function run_bagol(
    locs::Vector{<:SMLMData.AbstractEmitter};
    camera::SMLMData.AbstractCamera,
    kwargs...
)
    # Wrap in SMLD and dispatch
    smld = SMLMData.BasicSMLD(locs, camera, 1, 1)
    return run_bagol(smld; kwargs...)
end

# Keep partition dispatch for internal use
run_bagol(p::Partition; camera::SMLMData.AbstractCamera, kwargs...) =
    run_bagol(p.locs; camera=camera, kwargs...)

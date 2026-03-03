# Main RJMCMC algorithm for BaGoL

"""
Perform one RJMCMC step (propose and accept/reject one move).

Returns (accepted::Bool, move_type::Symbol).

Move distribution:
- split (10%): Primary dimension-changing (K → K+1), operates in allocation space
- merge (10%): Primary dimension-changing (K → K-1), operates in allocation space
- birth (5%): Backup dimension-changing, uses mixture proposal
- death (5%): Backup dimension-changing, uses mixture proposal
- move (20%): Gibbs position update
- allocate (50%): Gibbs allocation update

Split/merge are preferred for dimension changes because they avoid the
proposal density penalty that makes birth/death inefficient at tight clusters.
"""
function rjmcmc_step!(
    chain::RJMCMCChain,
    locs::Vector{<:SMLMData.AbstractEmitter},
    spatial_prior::UniformSpatialPrior
)
    # Randomly select move type with weighted probabilities
    r = rand()
    if r < 0.10
        move_type = :birth      # Dimension-changing (K → K+1)
    elseif r < 0.20
        move_type = :death      # Dimension-changing (K → K-1)
    elseif r < 0.40
        move_type = :move       # Position Gibbs
    else
        move_type = :allocate   # Allocation Gibbs
    end

    accepted = false
    if move_type == :split
        accepted = propose_split!(chain, locs, spatial_prior)
    elseif move_type == :merge
        accepted = propose_merge!(chain, locs, spatial_prior)
    elseif move_type == :birth
        accepted = propose_birth!(chain, locs, spatial_prior)
    elseif move_type == :birth_uniform
        accepted = propose_birth_uniform!(chain, locs, spatial_prior)
    elseif move_type == :death
        accepted = propose_death!(chain, locs, spatial_prior)
    elseif move_type == :death_uniform
        accepted = propose_death_uniform!(chain, locs, spatial_prior)
    elseif move_type == :move
        accepted = propose_move!(chain, locs, spatial_prior)
    elseif move_type == :allocate
        accepted = propose_allocate!(chain, locs, spatial_prior)
    end

    # Update acceptance statistics
    prev = get(chain.acceptance, move_type, (0, 0))
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
function build_diagnostics(chain::RJMCMCChain, posterior_k::Vector{Int}, n_emitters::Int;
                           n_partitions::Int=1, post_image=nothing)
    acceptance_rates = Dict{Symbol, Float64}()
    for (move, (acc, tot)) in chain.acceptance
        acceptance_rates[move] = tot > 0 ? acc / tot : 0.0
    end
    return BaGoLDiagnostics(n_emitters, posterior_k, acceptance_rates, chain.μ, chain.shape, n_partitions, post_image)
end

"""
Build BaGoLDiagnostics from multiple chains (partitioned run).
"""
function build_diagnostics(chains::Vector{<:RJMCMCChain}, posterior_k::Vector{Int}, n_emitters::Int;
                           post_image=nothing)
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
                            chains[1].μ, chains[1].shape, length(chains), post_image)
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

        # Record sample after burn-in
        if i > burn_in
            record_sample!(chain)
        end

        # Hierarchical MH updates using counts from recent samples
        if i % hierarchical_interval == 0
            update_mu!(chain)
            if chain.learn_shape
                update_shape!(chain)
            end
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

# Sampler Selection
- `sampler=:collapsed`: Use collapsed Gibbs sampler (default, recommended)
- `sampler=:rjmcmc`: Use legacy RJMCMC sampler

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
- `Vector{RJMCMCChain}` (only when `return_chains=true`, rjmcmc only): Partition chains
"""
function run_bagol(
    smld::SMLMData.SMLD;
    sampler::Symbol = :collapsed,
    nsigma::Float64 = 3.0,
    min_partition_size::Int = 0,
    max_partition_size::Int = 1000,
    skip_partition_size::Int = typemax(Int),
    sync_interval::Int = 500,
    n_iterations::Int = 10000,
    burn_in::Int = 2000,
    shape::Float64 = 2.0,
    learn_shape::Bool = true,
    posterior_pixel_size::Float64 = 0.0,
    posterior_xlim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
    posterior_ylim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
    return_chains::Bool = false,
    archive_path::Union{Nothing, String} = nothing,
    verbose::Bool = true,
    kwargs...
)
    if sampler == :collapsed
        return _run_bagol_collapsed(smld;
            nsigma, min_partition_size, max_partition_size, skip_partition_size,
            sync_interval, n_iterations, burn_in, shape, learn_shape,
            posterior_pixel_size, posterior_xlim, posterior_ylim,
            archive_path, verbose, kwargs...)
    elseif sampler == :rjmcmc
        return _run_bagol_rjmcmc(smld;
            nsigma, min_partition_size, max_partition_size, skip_partition_size,
            sync_interval, n_iterations, burn_in, shape, learn_shape,
            posterior_pixel_size, posterior_xlim, posterior_ylim,
            return_chains, verbose, kwargs...)
    else
        error("Unknown sampler :$sampler. Use :collapsed or :rjmcmc.")
    end
end

# ============================================================================
# Collapsed Gibbs sampler dispatch
# ============================================================================

function _run_bagol_collapsed(
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
    posterior_pixel_size::Float64 = 0.0,
    posterior_xlim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
    posterior_ylim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
    archive_path::Union{Nothing, String} = nothing,
    verbose::Bool = true,
    kwargs...
)
    locs = smld.emitters
    camera = smld.camera

    if verbose
        println("Partitioning $(length(locs)) localizations (nsigma=$nsigma, sampler=collapsed)...")
    end

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
        empty_diag = BaGoLDiagnostics(0, Int[], Dict{Symbol,Float64}(), 0.0, shape, 0, nothing)
        return empty_smld, empty_diag
    end

    n_partitions = length(partitions)

    # Hyperprior config
    μ_prior_shape = get(kwargs, :μ_prior_shape, 2.0)
    μ_prior_scale = get(kwargs, :μ_prior_scale, 5.0)
    shape_prior_shape = get(kwargs, :shape_prior_shape, 2.0)
    shape_prior_scale = get(kwargs, :shape_prior_scale, 1.0)
    config_nt = (μ_prior_shape=μ_prior_shape, μ_prior_scale=μ_prior_scale,
                 shape_prior_shape=shape_prior_shape, shape_prior_scale=shape_prior_scale)

    # Initialize collapsed states and accumulators per partition
    states = Vector{CollapsedState}(undef, n_partitions)
    partition_accumulators = Vector{Vector{AbstractAccumulator}}(undef, n_partitions)
    count_hists = Vector{EmitterCountHist}(undef, n_partitions)

    Threads.@threads for i in 1:n_partitions
        p_locs = partitions[i].locs
        spatial_prior = UniformSpatialPrior(p_locs)
        states[i] = initialize_collapsed_state(p_locs, spatial_prior)

        # Per-partition accumulators
        accs = AbstractAccumulator[]
        count_hist = EmitterCountHist()
        push!(accs, count_hist)
        count_hists[i] = count_hist

        if posterior_pixel_size > 0.0
            push!(accs, PosteriorImage(pixel_size=posterior_pixel_size,
                                        xlim=posterior_xlim, ylim=posterior_ylim))
        end

        partition_accumulators[i] = accs
    end

    # Global μ, shape (shared across partitions)
    μ = μ_prior_shape * μ_prior_scale
    current_shape = shape

    # Initialize archive if requested
    archive = nothing
    if archive_path !== nothing
        archive = BaGoLArchive(archive_path, n_partitions,
                               [length(p.locs) for p in partitions])
    end

    # Synchronized outer loop
    n_outer = div(n_iterations, sync_interval)
    iter_counters = zeros(Int, n_partitions)

    for outer in 1:n_outer
        Threads.@threads for i in 1:n_partitions
            λ_K_i = get(kwargs, :λ_K, Float64(length(partitions[i].locs)) / 5.0)
            iter_counters[i] = run_collapsed_iterations!(
                states[i], partitions[i].locs, sync_interval,
                μ, current_shape, λ_K_i,
                partition_accumulators[i], burn_in, iter_counters[i]
            )
        end

        # Write archive samples if past burn-in
        if archive !== nothing && (outer * sync_interval) > burn_in
            for i in 1:n_partitions
                write_sample!(archive, i, states[i], μ, current_shape)
            end
        end

        # Global hierarchical updates
        μ = _update_mu_collapsed_global!(states, μ, current_shape, config_nt)
        if learn_shape
            current_shape = _update_shape_collapsed_global!(states, μ, current_shape, config_nt)
        end

        if verbose && outer % max(1, n_outer ÷ 5) == 0
            total_K = sum(s.n_active for s in states)
            shape_str = learn_shape ? ", shape=$(round(current_shape, digits=2))" : ""
            println("Sync $outer/$n_outer: total K=$total_K, μ=$(round(μ, digits=2))$shape_str")
        end
    end

    # Run remaining iterations
    remaining = n_iterations - n_outer * sync_interval
    if remaining > 0
        Threads.@threads for i in 1:n_partitions
            λ_K_i = get(kwargs, :λ_K, Float64(length(partitions[i].locs)) / 5.0)
            iter_counters[i] = run_collapsed_iterations!(
                states[i], partitions[i].locs, remaining,
                μ, current_shape, λ_K_i,
                partition_accumulators[i], burn_in, iter_counters[i]
            )
        end
    end

    if verbose
        println("\nMerging partition results...")
    end

    # Extract emitters from each partition's final state
    all_emitters = SMLMData.Emitter2DFit[]
    partition_ids = Int[]
    is_near_boundary = Bool[]

    sigmas = [mean_sigma(loc) for loc in locs]
    boundary_margin = 5.0 * median(sigmas)

    for (pid, (partition, state)) in enumerate(zip(partitions, states))
        emitters = extract_emitters(state, partition.locs)
        for emitter in emitters
            push!(all_emitters, emitter)
            push!(partition_ids, pid)
            near_boundary = emitter_near_boundary(emitter, partition, boundary_margin)
            push!(is_near_boundary, near_boundary)
        end
    end

    # Deduplicate boundary emitters
    if !isempty(all_emitters)
        merged_emitters = deduplicate_boundary_emitters(
            all_emitters, partition_ids, is_near_boundary, boundary_margin
        )
    else
        merged_emitters = SMLMData.Emitter2DFit[]
    end

    # Merge count histograms for posterior_k
    merged_count_hist = EmitterCountHist()
    for ch in count_hists
        accumulator_merge!(merged_count_hist, ch)
    end
    posterior_k = accumulator_result(merged_count_hist)

    # Merge posterior images if requested
    post_img = nothing
    if posterior_pixel_size > 0.0
        merged_post = nothing
        for accs in partition_accumulators
            for acc in accs
                if acc isa PosteriorImage
                    if merged_post === nothing
                        merged_post = acc
                    else
                        accumulator_merge!(merged_post, acc)
                    end
                end
            end
        end
        if merged_post !== nothing
            pi_result = accumulator_result(merged_post)
            # Convert to integer counts for backward compat
            int_image = round.(Int, pi_result.image)
            post_img = (image=int_image, edges_x=pi_result.edges_x,
                       edges_y=pi_result.edges_y, pixel_size=pi_result.pixel_size)
        end
    end

    # Build acceptance rates from first partition (representative)
    acceptance_rates = Dict{Symbol, Float64}()

    # Build diagnostics
    diagnostics = BaGoLDiagnostics(
        length(merged_emitters), posterior_k, acceptance_rates,
        μ, current_shape, n_partitions, post_img
    )
    result_smld = SMLMData.BasicSMLD(merged_emitters, camera, 1, 1)

    if verbose
        println("Result: $(length(merged_emitters)) emitters")
    end

    # Close archive
    if archive !== nothing
        close(archive)
    end

    return result_smld, diagnostics
end

# ============================================================================
# Legacy RJMCMC dispatch
# ============================================================================

function _run_bagol_rjmcmc(
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
    posterior_pixel_size::Float64 = 0.0,
    posterior_xlim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
    posterior_ylim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
    return_chains::Bool = false,
    verbose::Bool = true,
    kwargs...
)
    locs = smld.emitters
    camera = smld.camera

    if verbose
        println("Partitioning $(length(locs)) localizations (nsigma=$nsigma, sampler=rjmcmc)...")
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
        empty_diag = BaGoLDiagnostics(0, Int[], Dict{Symbol,Float64}(), 0.0, shape, 0, nothing)
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

        # Global MH updates using counts from recent samples across all partitions
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

    # Compute posterior image if requested
    post_img = nothing
    if posterior_pixel_size > 0.0
        post_img = posterior_image(chains; pixel_size=posterior_pixel_size,
                                   xlim=posterior_xlim, ylim=posterior_ylim)
    end

    diagnostics = build_diagnostics(chains, posterior_k, length(merged_emitters); post_image=post_img)
    result_smld = SMLMData.BasicSMLD(merged_emitters, camera, 1, 1)

    if verbose
        println("Result: $(length(merged_emitters)) emitters")
    end

    if return_chains
        return result_smld, diagnostics, chains
    end
    return result_smld, diagnostics
end

"""
    run_bagol(locs::Vector{<:AbstractEmitter}; camera, kwargs...) -> (BasicSMLD, BaGoLDiagnostics)

Run BaGoL on a vector of localizations. Requires camera argument.

See `run_bagol(smld::SMLD; ...)` for full documentation.
Pass `sampler=:collapsed` (default) or `sampler=:rjmcmc` to select the sampler.
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

# Main RJMCMC algorithm for BaGoL

"""
Perform one RJMCMC step (propose and accept/reject one move).
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
    return accepted
end

"""
Record current state as a sample.
"""
function record_sample!(chain::RJMCMCChain)
    state = chain.current_state
    # Deep copy emitters
    emitters_copy = [Emitter(e.x, e.y, copy(e.allocated)) for e in state.emitters]
    sample = BaGoLSample(emitters_copy, state.log_posterior, chain.μ, chain.α)
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
        rjmcmc_step!(chain, locs, spatial_prior)

        # Record sample after burn-in
        if record_after_burn_in && chain.iteration > config.burn_in
            record_sample!(chain)
        end
    end
end

"""
Run BaGoL RJMCMC analysis on localizations.

# Arguments
- `locs`: Vector of localizations (must have x, y, σ_x, σ_y fields)
- `α`: Shape parameter for count distribution. Can be:
  - `Float64`: Fixed value (default: 2.0)
  - `:auto`: Estimate from frame statistics using Fano factor
- `learn_α`: Whether to update α during MCMC (default: false)
- `n_iterations`: Number of MCMC iterations (default: 10000)
- `burn_in`: Burn-in iterations before recording samples (default: 2000)
- `hierarchical_interval`: Iterations between μ/α updates (default: 100)

# Returns
- `RJMCMCChain` containing samples and diagnostics
"""
function run_bagol(
    locs::Vector{<:SMLMData.AbstractEmitter};
    α::Union{Float64, Symbol} = 2.0,
    learn_α::Bool = false,
    λ_K::Float64 = Float64(length(locs)) / 5.0,  # Rough estimate: ~5 locs per emitter
    n_iterations::Int = 10000,
    burn_in::Int = 2000,
    hierarchical_interval::Int = 100,
    move_σ::Float64 = 0.010,
    μ_prior_a::Float64 = 2.0,
    μ_prior_b::Float64 = 0.2,
    verbose::Bool = true
)
    # Determine initial α value
    if α === :auto
        α_init = estimate_alpha_from_frames(locs)
        if verbose
            println("Auto-estimated α = $(round(α_init, digits=2)) from frame statistics")
        end
    else
        α_init = α::Float64
    end

    config = RJMCMCConfig(;
        α = α_init,  # Store initial value in config
        λ_K = λ_K,
        n_iterations = n_iterations,
        burn_in = burn_in,
        hierarchical_interval = hierarchical_interval,
        move_σ = move_σ,
        μ_prior_a = μ_prior_a,
        μ_prior_b = μ_prior_b
    )

    # Create spatial prior from data
    spatial_prior = UniformSpatialPrior(locs)

    # Initialize with one emitter at centroid
    xs = [loc.x for loc in locs]
    ys = [loc.y for loc in locs]
    initial_emitter = Emitter(mean(xs), mean(ys))
    initial_state = BaGoLState([initial_emitter], 0.0)

    # Initialize allocations
    initialize_allocations!(initial_state, locs)
    update_emitter_positions!(initial_state, locs)

    # Create chain with α learning settings
    chain = RJMCMCChain(config, initial_state; α_init=α_init, learn_α=learn_α)

    # Run MCMC
    for i in 1:n_iterations
        rjmcmc_step!(chain, locs, spatial_prior)

        # Hierarchical update
        if i % hierarchical_interval == 0
            update_mu_gibbs!(chain)
            if chain.learn_α
                update_alpha!(chain, locs)
            end
        end

        # Record sample after burn-in
        if i > burn_in
            record_sample!(chain)
        end

        # Progress
        if verbose && i % 1000 == 0
            k = length(chain.current_state.emitters)
            α_str = chain.learn_α ? ", α=$(round(chain.α, digits=2))" : ""
            println("Iteration $i: K=$k, μ=$(round(chain.μ, digits=2))$α_str")
        end
    end

    if verbose
        println("\nAcceptance rates:")
        for (move, (acc, tot)) in chain.acceptance
            rate = tot > 0 ? round(100 * acc / tot, digits=1) : 0.0
            println("  $move: $rate% ($acc/$tot)")
        end
        if chain.learn_α
            println("\nFinal α = $(round(chain.α, digits=2))")
        end
    end

    return chain
end

# ============================================================================
# Dispatch methods for SMLD and Partition types
# ============================================================================

"""
    run_bagol(p::Partition; kwargs...)

Run BaGoL on a partition. Dispatches to core Vector method.
"""
run_bagol(p::Partition; kwargs...) = run_bagol(p.locs; kwargs...)

"""
    run_bagol(smld::SMLMData.SMLD; partition_threshold=500, kwargs...)

Run BaGoL on an SMLD. Auto-partitions large datasets and uses global hierarchical updates.

# Arguments
- `partition_threshold`: Use partitioning if n_locs > threshold (0 = never partition)
- `nsigma`: DBSCAN threshold in sigma units (default 4.0)
- `max_partition_size`: Target max locs per partition (default 1000)
- `sync_interval`: Iterations between global μ/α updates (default 500)
- Other kwargs passed to core `run_bagol`

# Returns
- `MAPNResult` with grouped emitter positions
"""
function run_bagol(
    smld::SMLMData.SMLD;
    partition_threshold::Int = 500,
    nsigma::Float64 = 4.0,
    min_partition_size::Int = 10,
    max_partition_size::Int = 1000,
    sync_interval::Int = 500,
    n_iterations::Int = 10000,
    burn_in::Int = 2000,
    α::Union{Float64, Symbol} = 2.0,
    learn_α::Bool = false,
    verbose::Bool = true,
    kwargs...
)
    locs = smld.emitters

    # Small dataset: run directly
    if length(locs) <= partition_threshold || partition_threshold == 0
        chain = run_bagol(locs; n_iterations, burn_in, α, learn_α, verbose, kwargs...)
        return estimate_mapn(chain)
    end

    # Large dataset: partition with synchronized global updates
    if verbose
        println("Partitioning $(length(locs)) localizations...")
    end

    # Estimate α globally if :auto
    if α === :auto
        α_init = estimate_alpha_from_frames(locs)
        if verbose
            println("Auto-estimated α = $(round(α_init, digits=2)) from frame statistics")
        end
    else
        α_init = α::Float64
    end

    # Partition the data
    partitions, skipped = partition_locs(locs; nsigma, min_size=min_partition_size,
                                          max_size=max_partition_size)

    if verbose
        println("  Created $(length(partitions)) partitions")
        if !isempty(skipped)
            n_skipped = sum(length(p.locs) for p in skipped)
            println("  Skipped $(length(skipped)) oversized clusters ($n_skipped locs)")
        end
    end

    if isempty(partitions)
        @warn "No valid partitions"
        return MAPNResult(0, Tuple{Float64,Float64}[], Tuple{Float64,Float64}[], Int[])
    end

    # Initialize chains for each partition
    n_partitions = length(partitions)
    chains = Vector{RJMCMCChain}(undef, n_partitions)
    spatial_priors = Vector{UniformSpatialPrior}(undef, n_partitions)

    μ_prior_a = get(kwargs, :μ_prior_a, 2.0)
    μ_prior_b = get(kwargs, :μ_prior_b, 0.2)
    λ_K_default = Float64(length(locs)) / 5.0 / n_partitions

    Threads.@threads for i in 1:n_partitions
        p_locs = partitions[i].locs
        config = RJMCMCConfig(;
            α = α_init,
            λ_K = get(kwargs, :λ_K, Float64(length(p_locs)) / 5.0),
            n_iterations = n_iterations,
            burn_in = burn_in,
            hierarchical_interval = sync_interval,  # Not used internally, but stored
            move_σ = get(kwargs, :move_σ, 0.010),
            μ_prior_a = μ_prior_a,
            μ_prior_b = μ_prior_b
        )

        spatial_priors[i] = UniformSpatialPrior(p_locs)

        xs = [loc.x for loc in p_locs]
        ys = [loc.y for loc in p_locs]
        initial_emitter = Emitter(mean(xs), mean(ys))
        initial_state = BaGoLState([initial_emitter], 0.0)
        initialize_allocations!(initial_state, p_locs)
        update_emitter_positions!(initial_state, p_locs)

        chains[i] = RJMCMCChain(config, initial_state; α_init=α_init, learn_α=learn_α)
    end

    # Synchronized outer loop
    n_outer = div(n_iterations, sync_interval)
    for outer in 1:n_outer
        # Run sync_interval iterations on each partition (parallel)
        Threads.@threads for i in 1:n_partitions
            record = (outer * sync_interval) > burn_in
            run_iterations!(chains[i], partitions[i].locs, spatial_priors[i],
                           sync_interval; record_after_burn_in=record)
        end

        # Global hierarchical updates
        μ_new = update_mu_global!(chains, μ_prior_a, μ_prior_b)
        if learn_α
            update_alpha_global!(chains)
        end

        if verbose && outer % max(1, n_outer ÷ 5) == 0
            total_K = sum(length(c.current_state.emitters) for c in chains)
            α_str = learn_α ? ", α=$(round(chains[1].α, digits=2))" : ""
            println("Sync $outer/$n_outer: total K=$total_K, μ=$(round(μ_new, digits=2))$α_str")
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

    # Merge results
    result = merge_partition_results(chains, partitions, boundary_margin)

    if verbose
        println("Final result: $(result.n_emitters) emitters")
    end

    return result
end

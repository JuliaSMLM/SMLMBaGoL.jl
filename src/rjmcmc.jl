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
    sample = BaGoLSample(emitters_copy, state.log_posterior, chain.μ)
    push!(chain.samples, sample)
end

"""
Run BaGoL RJMCMC analysis on localizations.

# Arguments
- `locs`: Vector of localizations (must have x, y, σ_x, σ_y fields)
- `τ`: Systematic uncertainty (REQUIRED) - typically 0.003-0.010 μm
- `α`: Shape parameter for count distribution (default: 2.0)
- `n_iterations`: Number of MCMC iterations (default: 10000)
- `burn_in`: Burn-in iterations before recording samples (default: 2000)
- `hierarchical_interval`: Iterations between μ updates (default: 100)

# Returns
- `RJMCMCChain` containing samples and diagnostics
"""
function run_bagol(
    locs::Vector{<:SMLMData.AbstractEmitter};
    τ::Float64,  # Required - no default
    α::Float64 = 2.0,
    λ_K::Float64 = Float64(length(locs)) / 5.0,  # Rough estimate: ~5 locs per emitter
    n_iterations::Int = 10000,
    burn_in::Int = 2000,
    hierarchical_interval::Int = 100,
    move_σ::Float64 = 0.010,
    μ_prior_a::Float64 = 2.0,
    μ_prior_b::Float64 = 0.2,
    verbose::Bool = true
)
    config = RJMCMCConfig(;
        τ = τ,
        α = α,
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
    update_emitter_positions!(initial_state, locs, τ)

    # Create chain
    chain = RJMCMCChain(config, initial_state)

    # Run MCMC
    for i in 1:n_iterations
        rjmcmc_step!(chain, locs, spatial_prior)

        # Hierarchical update
        if i % hierarchical_interval == 0
            update_mu_gibbs!(chain)
        end

        # Record sample after burn-in
        if i > burn_in
            record_sample!(chain)
        end

        # Progress
        if verbose && i % 1000 == 0
            k = length(chain.current_state.emitters)
            println("Iteration $i: K=$k, μ=$(round(chain.μ, digits=2))")
        end
    end

    if verbose
        println("\nAcceptance rates:")
        for (move, (acc, tot)) in chain.acceptance
            rate = tot > 0 ? round(100 * acc / tot, digits=1) : 0.0
            println("  $move: $rate% ($acc/$tot)")
        end
    end

    return chain
end

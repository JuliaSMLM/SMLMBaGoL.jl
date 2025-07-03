function rjmcmc_step!(chain::RJMCMCChain)
    # Select move type based on weights
    move_type = sample_move_type(chain.move_weights, chain.rng)
    
    # Propose new state - pass chain for birth/death moves to access birth_proposal
    proposed = propose_move(move_type, chain.current_state, chain, chain.rng)
    
    # Handle failed proposals
    proposed === nothing && return false
    
    # Accept/reject - pass chain for birth/death moves
    acceptance_prob = if move_type in [Birth, Death]
        # Use chain-aware acceptance calculation for birth/death
        log_ratio = log_acceptance_ratio(move_type, chain.current_state, proposed, chain)
        exp(min(0.0, log_ratio))
    else
        # Use standard acceptance for other moves
        accept_probability(move_type, chain.current_state, proposed)
    end
    
    if rand(chain.rng) < acceptance_prob
        chain.current_state = proposed
        return true
    else
        return false
    end
end

function sample_move_type(move_weights::Dict{Type{<:AbstractRJMCMCMove}, Float64}, rng=Random.GLOBAL_RNG)
    move_types = collect(keys(move_weights))
    weights = [move_weights[mt] for mt in move_types]
    
    # Normalize weights
    weights ./= sum(weights)
    
    # Sample proportional to weights
    cumulative = cumsum(weights)
    r = rand(rng)
    
    for i in eachindex(cumulative)
        if r <= cumulative[i]
            return move_types[i]
        end
    end
    
    return move_types[end]  # Fallback
end

function should_store_sample(chain::RJMCMCChain, iteration::Int)
    # Skip burn-in period
    iteration <= chain.burn_in && return false
    
    # Apply thinning
    return (iteration - chain.burn_in) % chain.thin == 0
end

function run_rjmcmc!(chain::RJMCMCChain, n_iterations::Int)
    acceptances = 0
    
    for iter in 1:n_iterations
        # Perform RJMCMC step
        accepted = rjmcmc_step!(chain)
        accepted && (acceptances += 1)
        
        # Store sample if needed
        if should_store_sample(chain, iter)
            push!(chain.samples, deepcopy(chain.current_state))
        end
        
        # Progress monitoring disabled for cleaner output
        # Uncomment for detailed progress:
        # if iter % 1000 == 0
        #     println("Iteration $iter, acceptance rate: $(round(acceptances/iter, digits=3))")
        # end
    end
    
    return acceptances / n_iterations
end

function initialize_chain(localizations::Vector{L}, 
                         EmitterType::Type{E},
                         spatial_prior::AbstractSpatialPrior,
                         count_prior::AbstractCountPrior;
                         initial_K::Int = max(1, length(localizations) ÷ 10),
                         burn_in::Int = 1000,
                         thin::Int = 1,
                         rng::AbstractRNG = Random.GLOBAL_RNG) where {E<:AbstractEmitter, L<:AbstractLocalization}
    
    # Initialize emitters randomly in spatial prior
    initial_emitters = Vector{E}(undef, initial_K)
    for i in 1:initial_K
        x, y = sample_spatial_prior(spatial_prior, rng)
        # Use default photon count for new emitters
        photons = 1000.0  # Default photon count
        initial_emitters[i] = E(x, y, photons)
    end
    
    # Initialize allocations randomly
    initial_allocations = rand(rng, 1:initial_K, length(localizations))
    
    # Get initial τ² from count prior
    initial_τ² = isa(count_prior, HierarchicalNegBinomialPrior) ? count_prior.τ² : 1e-6

    # Initialize latent positions by sampling from posterior given initial allocations
    latent_positions = Vector{Tuple{eltype(localizations[1].x), eltype(localizations[1].x)}}(undef, length(localizations))

    for (i, loc) in enumerate(localizations)
        emitter_idx = initial_allocations[i]
        if 1 ≤ emitter_idx ≤ length(initial_emitters)
            emitter = initial_emitters[emitter_idx]
            
            # Sample from posterior given emitter and observation
            prec_x = 1/initial_τ² + 1/loc.σx^2
            prec_y = 1/initial_τ² + 1/loc.σy^2
            post_mean_x = (emitter.x/initial_τ² + loc.x/loc.σx^2) / prec_x
            post_mean_y = (emitter.y/initial_τ² + loc.y/loc.σy^2) / prec_y
            
            latent_x = post_mean_x + randn(rng) / sqrt(prec_x)
            latent_y = post_mean_y + randn(rng) / sqrt(prec_y)
            
            latent_positions[i] = (latent_x, latent_y)
        else
            # Unallocated - keep at observed position
            latent_positions[i] = (loc.x, loc.y)
        end
    end
    
    # Create initial state
    temp_state = BaGoLState(initial_emitters, localizations, initial_allocations, 
                           latent_positions, spatial_prior, count_prior, initial_τ², 0.0)
    initial_likelihood = log_likelihood(temp_state)
    initial_state = BaGoLState(initial_emitters, localizations, initial_allocations, 
                              latent_positions, spatial_prior, count_prior, initial_τ², initial_likelihood)
    
    # Default move weights
    default_move_weights = Dict{Type{<:AbstractRJMCMCMove}, Float64}(
        Birth => 0.10,
        Death => 0.10, 
        Move => 0.30,         # Reduced from 0.40
        Allocate => 0.30,     # Reduced from 0.40
        UpdateLatent => 0.20  # NEW: 20% of moves update latent positions
    )
    
    # Create birth proposal distribution
    birth_proposal = create_birth_proposal(localizations)
    
    # Create chain with hierarchical history
    chain = RJMCMCChain(
        localizations,
        initial_state,
        spatial_prior,
        count_prior,
        default_move_weights,
        BaGoLState{E,L,eltype(initial_likelihood)}[],
        burn_in,
        thin,
        rng,
        HierarchicalUpdate{eltype(initial_likelihood)}[],  # Empty hierarchical history
        birth_proposal  # Cached birth proposal
    )
    
    # Record initial hierarchical parameters if applicable
    if isa(count_prior, HierarchicalNegBinomialPrior)
        push!(chain.hierarchical_history, HierarchicalUpdate(
            0, count_prior.μ, count_prior.κ, count_prior.τ², 0.0, 0
        ))
    end
    
    return chain
end

function run_bagol(localizations::Vector{L};
                  EmitterType::Type{E} = Emitter2D{Float64},
                  n_iterations::Int = 10000,
                  burn_in::Int = 2000,
                  thin::Int = 1,
                  initial_K::Int = max(1, length(localizations) ÷ 10),
                  partition_radius::Real = estimate_partitioning_radius(localizations),
                  partition_data::Bool = true,
                  enable_hierarchical::Bool = false,
                  hierarchical_interval::Int = 1000,
                  enable_threading::Bool = true,
                  existing_chains::Union{Vector{RJMCMCChain}, RJMCMCChain, Nothing} = nothing,
                  continuation_mode::Symbol = :extend,
                  rng::AbstractRNG = Random.GLOBAL_RNG) where {E<:AbstractEmitter, L<:AbstractLocalization}
    
    # Handle chain continuation or initialization
    chains = if existing_chains !== nothing
        handle_chain_continuation(existing_chains, localizations, continuation_mode,
                                EmitterType, partition_data, partition_radius, 
                                enable_hierarchical, burn_in, thin, rng)
    else
        # Initialize new chains
        initialize_chains_from_data(localizations, EmitterType, partition_data, 
                                  partition_radius, enable_hierarchical, burn_in, thin, rng)
    end
    
    print_partitioning_summary(chains)
    
    # Run nested RJMCMC with hierarchical epochs and threading
    total_acceptance = 0.0
    total_iterations_completed = 0
    
    # Calculate hierarchical epochs
    n_hierarchical_epochs = enable_hierarchical ? ceil(Int, n_iterations / hierarchical_interval) : 1
    iterations_per_epoch = enable_hierarchical ? hierarchical_interval : n_iterations
    
    for epoch in 1:n_hierarchical_epochs
        # Determine iterations for this epoch (last epoch might be smaller)
        remaining_iterations = n_iterations - total_iterations_completed
        current_epoch_iterations = min(iterations_per_epoch, remaining_iterations)
        
        # RJMCMC iterations within this hierarchical epoch
        for iter in 1:current_epoch_iterations
            global_iter = total_iterations_completed + iter
            
            # Thread-safe parallel processing of partitions
            if enable_threading
                # Parallel execution across partitions
                acceptance_results = Vector{Bool}(undef, length(chains))
                Threads.@threads for i in eachindex(chains)
                    acceptance_results[i] = rjmcmc_step!(chains[i])
                    
                    # Store samples (thread-safe since each chain is independent)
                    if should_store_sample(chains[i], global_iter)
                        push!(chains[i].samples, deepcopy(chains[i].current_state))
                    end
                end
                accepted_count = count(acceptance_results)
            else
                # Sequential execution (fallback/debugging)
                accepted_count = 0
                for chain in chains
                    if rjmcmc_step!(chain)
                        accepted_count += 1
                    end
                    
                    # Store samples
                    if should_store_sample(chain, global_iter)
                        push!(chain.samples, deepcopy(chain.current_state))
                    end
                end
            end
            
            total_acceptance += accepted_count / length(chains)
            
            # Progress reporting - disabled for cleaner output
            # Uncomment for detailed progress monitoring
            # if global_iter % 1000 == 0
            #     avg_acceptance = total_acceptance / global_iter
            #     threading_status = enable_threading ? " (threaded)" : " (sequential)"
            #     println("Iteration $global_iter$threading_status, average acceptance rate: $(round(avg_acceptance, digits=3))")
            # end
        end
        
        total_iterations_completed += current_epoch_iterations
        
        # Hierarchical updates at end of epoch (synchronization point)
        if enable_hierarchical && epoch < n_hierarchical_epochs && total_iterations_completed > burn_in
            # Silent hierarchical update - details available in update_hierarchical! if needed
            update_hierarchical!(chains, total_iterations_completed)
        end
        
        # Break if we've completed all requested iterations
        if total_iterations_completed >= n_iterations
            break
        end
    end
    
    avg_acceptance_rate = total_acceptance / total_iterations_completed
    total_samples = sum(length(chain.samples) for chain in chains)
    
    # Concise summary
    println("RJMCMC completed: $total_iterations_completed iterations, $total_samples samples, acceptance rate: $(round(avg_acceptance_rate, digits=3))")
    
    # Return single chain if only one partition, otherwise return all chains
    return length(chains) == 1 ? chains[1] : chains
end

function initialize_chains_from_data(localizations::Vector{L}, EmitterType::Type{E}, 
                                   partition_data::Bool, partition_radius::Real,
                                   enable_hierarchical::Bool, burn_in::Int, thin::Int,
                                   rng::AbstractRNG) where {E<:AbstractEmitter, L<:AbstractLocalization}
    # Partition data into spatial regions if requested
    if partition_data
        partitioned_localizations = partition_localizations(localizations; 
                                                           radius=partition_radius, 
                                                           min_partition_size=1)
    else
        partitioned_localizations = [localizations]  # Single partition
    end
    
    # Create chains for each partition
    chains = Vector{RJMCMCChain}()
    for (i, partition_locs) in enumerate(partitioned_localizations)
        # Always create the same type of prior
        spatial_prior, count_prior = create_default_prior(partition_locs)
        
        chain = initialize_chain(partition_locs, EmitterType, spatial_prior, count_prior;
                               initial_K=max(1, length(partition_locs) ÷ 10),
                               burn_in=burn_in, thin=thin, rng=rng)
        push!(chains, chain)
    end
    
    return chains
end

function handle_chain_continuation(existing_chains::Union{Vector{RJMCMCChain}, RJMCMCChain}, 
                                 localizations::Vector{L}, continuation_mode::Symbol,
                                 EmitterType::Type{E}, partition_data::Bool, partition_radius::Real,
                                 enable_hierarchical::Bool, burn_in::Int, thin::Int,
                                 rng::AbstractRNG) where {E<:AbstractEmitter, L<:AbstractLocalization}
    
    # Convert single chain to vector for uniform handling
    chains_vec = existing_chains isa Vector ? existing_chains : [existing_chains]
    
    if continuation_mode == :extend
        # Simply return existing chains - they will continue from current state
        # Continuing existing chains silently
        return chains_vec
        
    elseif continuation_mode == :new_chain
        # Create new chains that will be concatenated with existing ones
        # Creating new chains to concatenate
        new_chains = initialize_chains_from_data(localizations, EmitterType, partition_data,
                                                partition_radius, enable_hierarchical, burn_in, thin, rng)
        
        # Store reference to existing chains for later concatenation
        for (i, new_chain) in enumerate(new_chains)
            if i <= length(chains_vec)
                # Store existing chain in metadata for concatenation
                new_chain.samples = []  # Start fresh for new samples
                # We'll need to implement concatenation logic later
            end
        end
        return new_chains
        
    elseif continuation_mode == :replace
        # Initialize new chains from final states of existing chains
        println("Initializing new chains from final states of $(length(chains_vec)) existing chains...")
        
        # For now, create new chains normally - enhancement: use final states as initial states
        new_chains = initialize_chains_from_data(localizations, EmitterType, partition_data,
                                                partition_radius, enable_hierarchical, burn_in, thin, rng)
        
        # TODO: Initialize from final states of existing chains
        return new_chains
        
    else
        error("Unknown continuation_mode: $continuation_mode. Supported modes: :extend, :new_chain, :replace")
    end
end

function print_partitioning_summary(chains::Vector{RJMCMCChain})
    # Minimal output for cleaner console
    if length(chains) > 1
        total_locs = sum(length(chain.localizations) for chain in chains)
        println("Data partitioned: $(length(chains)) partitions, $total_locs total localizations")
    end
    # Detailed output available by uncommenting:
    # println("Partitioning summary:")
    # println("  Total partitions: $(length(chains))")
    # for (i, chain) in enumerate(chains)
    #     partition = chain.localizations
    #     if length(partition) > 0
    #         x_coords = [loc.x for loc in partition]
    #         y_coords = [loc.y for loc in partition]
    #         x_center = sum(x_coords) / length(x_coords)
    #         y_center = sum(y_coords) / length(y_coords)
    #         println("    Partition $i: $(length(partition)) localizations at ($(round(x_center, digits=2)), $(round(y_center, digits=2)))")
    #     end
    # end
end

"""
    create_default_prior(localizations)

Create default hierarchical priors for BaGoL analysis with automatic τ² initialization.

# τ² Prior System
The τ² parameter captures additional systematic localization uncertainty beyond 
the reported per-localization uncertainties (σx, σy). 

## Automatic Initialization Strategy
1. **Data-driven estimate**: initial_τ² = (0.1 × median_σ)²
   - Uses 10% of median localization precision as conservative starting point
   - Adapts to the experimental data quality automatically

2. **Hyperprior specification**: τ² ~ InverseGamma(3.0, 4×initial_τ²)
   - Shape a_τ = 3.0: Moderately informative, allows learning while preventing extremes
   - Scale b_τ = 4×initial_τ²: Reasonable regularization around estimate  
   - Prior mean = 2×initial_τ²: Centered around data-driven estimate
   - Prior mode = initial_τ²: Mode at initial estimate

## Physical Interpretation
- initial_τ² ≈ 0.01×σ²: Conservative estimate assuming small systematic effects
- Final τ² learned via MCMC: Can be orders of magnitude larger if data supports it
- Large τ² indicates significant systematic uncertainty (drift, calibration, etc.)

# Returns
- `(spatial_prior, count_prior)`: Tuple of priors for BaGoL analysis
"""
function create_default_prior(localizations::Vector{<:AbstractLocalization})
    spatial_prior = create_spatial_prior_from_localizations(localizations, 0.2)
    
    # Calculate initial τ² estimate from localization precisions
    if !isempty(localizations)
        # Use 10% of median uncertainty squared as initial guess
        median_σ = median([sqrt(loc.σx^2 + loc.σy^2) for loc in localizations])
        initial_τ² = (0.1 * median_σ)^2
    else
        initial_τ² = 1e-6  # 1 nm² default
    end
    
    # Always hierarchical with sensible defaults
    # Prior on μ: mean=10, variance=50 → Gamma(2, 0.2)
    # Prior on κ: mean=2, variance=4 → Gamma(1, 0.5)
    # Prior on τ²: InverseGamma(3, 4×initial_τ²) → moderately informative, proper M-H sampling
    count_prior = HierarchicalNegBinomialPrior(
        10.0, 2.0, initial_τ²,      # Initial μ=10, κ=2, τ²
        (2.0, 0.2),                  # μ hyperprior
        (1.0, 0.5),                  # κ hyperprior
        (3.0, initial_τ² * 4.0)      # τ² hyperprior: InverseGamma(3, 4*initial_τ²)
    )
    
    return spatial_prior, count_prior
end


"""
    run_bagol(smld::SMLMSim.BasicSMLD; kwargs...)

Run BaGoL analysis on SMLMSim BasicSMLD data.

This method automatically converts the SMLD structure to BaGoL Localization2D format
and dispatches to the main run_bagol implementation.

# Arguments
- `smld`: SMLMSim BasicSMLD structure containing emitters
- `kwargs...`: All arguments supported by the main run_bagol method

# Returns
- Vector of RJMCMCChain objects, same as main run_bagol method

# Example
```julia
# Create SMLMSim data
smld = simulate_static_smlm(density=0.1, σ_psf=0.13)

# Run BaGoL analysis directly
chains = run_bagol(smld; n_iterations=5000, burn_in=1000)

# Extract results
mapn_results = estimate_mapn(chains)
```
"""
function run_bagol(smld::SMLMSim.BasicSMLD; kwargs...)
    # Convert SMLD to BaGoL localization format
    localizations = smld_to_localizations(smld)
    
    # Silent conversion - SMLMSim emitters to BaGoL localizations
    # ($(length(smld.emitters)) emitters → $(length(localizations)) localizations)
    
    # Dispatch to main run_bagol implementation
    return run_bagol(localizations; kwargs...)
end
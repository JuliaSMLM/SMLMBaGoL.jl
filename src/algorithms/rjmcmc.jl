function rjmcmc_step!(chain::RJMCMCChain)
    # Select move type based on weights
    move_type = sample_move_type(chain.move_weights, chain.rng)
    
    # Propose new state
    proposed = propose_move(move_type, chain.current_state, chain.rng)
    
    # Handle failed proposals
    proposed === nothing && return false
    
    # Accept/reject
    if rand(chain.rng) < accept_probability(move_type, chain.current_state, proposed)
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
        
        # Optional: print progress
        if iter % 1000 == 0
            println("Iteration $iter, acceptance rate: $(round(acceptances/iter, digits=3))")
        end
    end
    
    return acceptances / n_iterations
end

function initialize_chain(localizations::Vector{L}, 
                         EmitterType::Type{E},
                         prior::AbstractPrior;
                         initial_K::Int = max(1, length(localizations) ÷ 10),
                         burn_in::Int = 1000,
                         thin::Int = 1,
                         rng::AbstractRNG = Random.GLOBAL_RNG) where {E<:AbstractEmitter, L<:AbstractLocalization}
    
    # Extract spatial prior for emitter initialization
    spatial_prior = isa(prior, CompoundPrior) ? prior.spatial_prior :
                   create_spatial_prior_from_localizations(localizations)
    
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
    
    # Create initial state
    temp_state = BaGoLState(initial_emitters, localizations, initial_allocations, prior, 0.0)
    initial_likelihood = log_likelihood(temp_state)
    initial_state = BaGoLState(initial_emitters, localizations, initial_allocations, prior, initial_likelihood)
    
    # Default move weights
    default_move_weights = Dict{Type{<:AbstractRJMCMCMove}, Float64}(
        Birth => 0.2,
        Death => 0.2, 
        Split => 0.15,
        Merge => 0.15,
        Move => 0.2,
        Allocate => 0.1
    )
    
    # Create chain with hierarchical history
    chain = RJMCMCChain(
        localizations,
        initial_state,
        prior,
        default_move_weights,
        BaGoLState{E,L,eltype(initial_likelihood)}[],
        burn_in,
        thin,
        rng,
        Tuple{Int,Float64,Float64}[]  # Empty hierarchical history
    )
    
    # Record initial hierarchical parameters if applicable
    if isa(prior, CompoundPrior) && isa(prior.K_prior, HierarchicalGammaPrior)
        push!(chain.hierarchical_history, (0, prior.K_prior.α, prior.K_prior.β))
    elseif isa(prior, HierarchicalGammaPrior)
        push!(chain.hierarchical_history, (0, prior.α, prior.β))
    end
    
    return chain
end

function run_bagol(localizations::Vector{L};
                  EmitterType::Type{E} = Emitter2D{Float64},
                  prior::AbstractPrior = create_default_prior(localizations),
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
            
            # Progress reporting
            if global_iter % 1000 == 0
                avg_acceptance = total_acceptance / global_iter
                threading_status = enable_threading ? " (threaded)" : " (sequential)"
                println("Iteration $global_iter$threading_status, average acceptance rate: $(round(avg_acceptance, digits=3))")
            end
        end
        
        total_iterations_completed += current_epoch_iterations
        
        # Hierarchical updates at end of epoch (synchronization point)
        if enable_hierarchical && epoch < n_hierarchical_epochs && total_iterations_completed > burn_in
            println("Hierarchical update at iteration $total_iterations_completed...")
            update_hierarchical!(chains, total_iterations_completed)
        end
        
        # Break if we've completed all requested iterations
        if total_iterations_completed >= n_iterations
            break
        end
    end
    
    avg_acceptance_rate = total_acceptance / total_iterations_completed
    total_samples = sum(length(chain.samples) for chain in chains)
    
    println("RJMCMC completed:")
    println("  Total iterations: $total_iterations_completed")
    if enable_hierarchical
        println("  Hierarchical epochs: $n_hierarchical_epochs (interval: $hierarchical_interval)")
    end
    println("  Partitions: $(length(chains))")
    println("  Threading: $(enable_threading ? "enabled" : "disabled")")
    if enable_threading && length(chains) > 1
        println("  Thread utilization: $(min(Threads.nthreads(), length(chains)))/$(Threads.nthreads()) threads")
    end
    println("  Burn-in: $burn_in")
    println("  Total samples collected: $total_samples")
    println("  Average acceptance rate: $(round(avg_acceptance_rate, digits=3))")
    println("  Data partitioning: $(partition_data ? "enabled" : "disabled (single partition)")")
    if existing_chains !== nothing
        println("  Chain continuation: $continuation_mode mode")
    end
    
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
        partition_prior = enable_hierarchical ? 
                         create_hierarchical_prior(partition_locs) : 
                         create_default_prior(partition_locs)
        
        chain = initialize_chain(partition_locs, EmitterType, partition_prior;
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
        println("Continuing $(length(chains_vec)) existing chains...")
        return chains_vec
        
    elseif continuation_mode == :new_chain
        # Create new chains that will be concatenated with existing ones
        println("Creating new chains to concatenate with $(length(chains_vec)) existing chains...")
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
    println("Partitioning summary:")
    println("  Total partitions: $(length(chains))")
    for (i, chain) in enumerate(chains)
        partition = chain.localizations
        if length(partition) > 0
            x_coords = [loc.x for loc in partition]
            y_coords = [loc.y for loc in partition]
            x_center = sum(x_coords) / length(x_coords)
            y_center = sum(y_coords) / length(y_coords)
            println("    Partition $i: $(length(partition)) localizations at ($(round(x_center, digits=2)), $(round(y_center, digits=2)))")
        end
    end
end

function create_default_prior(localizations::Vector{<:AbstractLocalization})
    spatial_prior = create_spatial_prior_from_localizations(localizations, 0.2)
    K_prior = GammaPrior(2.0, 1.0)
    return CompoundPrior(spatial_prior, K_prior)
end

function create_hierarchical_prior(localizations::Vector{<:AbstractLocalization})
    spatial_prior = create_spatial_prior_from_localizations(localizations, 0.2)
    # Start with better initial values and less restrictive hyperpriors
    # Initial mean = α * β = 2.0 * 5.0 = 10.0 (reasonable starting point)
    hierarchical_K_prior = HierarchicalGammaPrior(
        2.0, 5.0,          # Initial α, β (mean = 10.0)
        (0.5, 0.1),        # α hyperprior (a₀, b₀) - less restrictive
        (0.5, 0.1)         # β hyperprior (c₀, d₀) - less restrictive
    )
    return CompoundPrior(spatial_prior, hierarchical_K_prior)
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
    
    println("SMLMSim data conversion:")
    println("  Input: $(length(smld.emitters)) emitters from SMLMSim")
    println("  Output: $(length(localizations)) localizations for BaGoL")
    
    # Dispatch to main run_bagol implementation
    return run_bagol(localizations; kwargs...)
end
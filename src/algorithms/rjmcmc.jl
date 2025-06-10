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
        initial_emitters[i] = E(x, y, i)
    end
    
    # Initialize allocations randomly
    initial_allocations = rand(rng, 1:initial_K, length(localizations))
    
    # Create initial state
    temp_state = BaGoLState(initial_emitters, localizations, initial_allocations, prior, 0.0)
    initial_likelihood = log_likelihood(temp_state)
    initial_state = BaGoLState(initial_emitters, localizations, initial_allocations, prior, initial_likelihood)
    
    # Default move weights
    default_move_weights = Dict{Type{<:AbstractRJMCMCMove}, Float64}(
        Birth => 0.25,
        Death => 0.25, 
        Split => 0.25,
        Merge => 0.25
    )
    
    # Create chain
    return RJMCMCChain(
        localizations,
        initial_state,
        prior,
        default_move_weights,
        BaGoLState{E,L,eltype(initial_likelihood)}[],
        burn_in,
        thin,
        rng
    )
end

function run_bagol(localizations::Vector{L};
                  EmitterType::Type{E} = Emitter2D{Float64},
                  prior::AbstractPrior = create_default_prior(localizations),
                  n_iterations::Int = 10000,
                  burn_in::Int = 2000,
                  thin::Int = 1,
                  initial_K::Int = max(1, length(localizations) ÷ 10),
                  rng::AbstractRNG = Random.GLOBAL_RNG) where {E<:AbstractEmitter, L<:AbstractLocalization}
    
    # Initialize chain
    chain = initialize_chain(localizations, EmitterType, prior; 
                           initial_K=initial_K, burn_in=burn_in, thin=thin, rng=rng)
    
    # Run RJMCMC
    acceptance_rate = run_rjmcmc!(chain, n_iterations)
    
    println("RJMCMC completed:")
    println("  Total iterations: $n_iterations")
    println("  Burn-in: $burn_in")
    println("  Samples collected: $(length(chain.samples))")
    println("  Overall acceptance rate: $(round(acceptance_rate, digits=3))")
    
    return chain
end

function create_default_prior(localizations::Vector{<:AbstractLocalization})
    spatial_prior = create_spatial_prior_from_localizations(localizations, 0.2)
    K_prior = GammaPrior(2.0, 1.0)
    return CompoundPrior(spatial_prior, K_prior)
end
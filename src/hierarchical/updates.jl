function count_allocations(state::BaGoLState)
    # Count how many localizations are assigned to each emitter
    emitter_counts = zeros(Int, length(state.emitters))
    for alloc in state.allocations
        if 1 ≤ alloc ≤ length(state.emitters)
            emitter_counts[alloc] += 1
        end
    end
    return emitter_counts
end

function collect_emitter_counts(chains::Vector{RJMCMCChain})
    all_counts = Int[]
    for chain in chains
        for sample in chain.samples
            counts = count_allocations(sample)
            append!(all_counts, counts)
        end
    end
    return all_counts
end

function update_gamma_hyperparameters(counts::Vector{Int}, 
                                     current_α::Real, current_β::Real,
                                     α_prior::Tuple{Real,Real}, β_prior::Tuple{Real,Real})
    # Extract hyperprior parameters
    a₀, b₀ = α_prior  # for α
    c₀, d₀ = β_prior  # for β
    
    n = length(counts)
    sum_counts = sum(counts)
    sum_log_counts = sum(log(max(1, k)) for k in counts)  # Avoid log(0)
    
    # Method of moments estimates
    sample_mean = sum_counts / n
    sample_var = sum((k - sample_mean)^2 for k in counts) / (n - 1)
    
    if sample_var > 0 && sample_mean > 0
        # Method of moments for Gamma parameters
        α_mom = sample_mean^2 / sample_var
        β_mom = sample_mean / sample_var
        
        # Shrinkage toward prior (simple Bayesian update)
        shrinkage_weight = 0.1  # 10% weight to prior
        
        new_α = (1 - shrinkage_weight) * α_mom + shrinkage_weight * current_α
        new_β = (1 - shrinkage_weight) * β_mom + shrinkage_weight * current_β
        
        # Apply bounds based on hyperpriors
        new_α = max(a₀, min(new_α, a₀ + 10.0))  # Bounded update
        new_β = max(1.0 / d₀, min(new_β, 1.0 / c₀))  # Bounded update
    else
        # No update if insufficient variation
        new_α = current_α
        new_β = current_β
    end
    
    return new_α, new_β
end

function update_hierarchical!(chains::Vector{RJMCMCChain})
    # Only proceed if all chains have hierarchical priors
    hierarchical_priors = HierarchicalGammaPrior[]
    
    for chain in chains
        if isa(chain.prior, CompoundPrior) && isa(chain.prior.K_prior, HierarchicalGammaPrior)
            push!(hierarchical_priors, chain.prior.K_prior)
        elseif isa(chain.prior, HierarchicalGammaPrior)
            push!(hierarchical_priors, chain.prior)
        else
            # No hierarchical prior found, skip update
            return
        end
    end
    
    if isempty(hierarchical_priors)
        return
    end
    
    # Use first prior as template (they should be similar)
    template_prior = hierarchical_priors[1]
    
    # Collect counts from all chains
    all_counts = collect_emitter_counts(chains)
    
    if length(all_counts) < 10  # Need sufficient data
        return
    end
    
    # Update hyperparameters
    new_α, new_β = update_gamma_hyperparameters(
        all_counts,
        template_prior.α, template_prior.β,
        template_prior.α_prior, template_prior.β_prior
    )
    
    # Create new hierarchical prior
    new_hierarchical_prior = HierarchicalGammaPrior(
        new_α, new_β,
        template_prior.α_prior, template_prior.β_prior
    )
    
    # Update all chains with new prior
    for chain in chains
        if isa(chain.prior, CompoundPrior)
            # Update K_prior within CompoundPrior
            new_compound_prior = CompoundPrior(
                chain.prior.spatial_prior,
                GammaPrior(new_α, new_β)  # Convert to simple GammaPrior for efficiency
            )
            chain.prior = new_compound_prior
            
            # Update current state prior
            chain.current_state = BaGoLState(
                chain.current_state.emitters,
                chain.current_state.localizations,
                chain.current_state.allocations,
                new_compound_prior,
                chain.current_state.log_likelihood
            )
        else
            # Direct hierarchical prior update
            chain.prior = new_hierarchical_prior
            chain.current_state = BaGoLState(
                chain.current_state.emitters,
                chain.current_state.localizations,
                chain.current_state.allocations,
                new_hierarchical_prior,
                chain.current_state.log_likelihood
            )
        end
    end
    
    println("Hierarchical update: α = $(round(new_α, digits=3)), β = $(round(new_β, digits=3))")
end
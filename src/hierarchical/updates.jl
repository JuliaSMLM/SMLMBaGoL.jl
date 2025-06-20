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
        # For Gamma(α, β) with β as scale parameter: mean = α*β, var = α*β²
        # Therefore: α = mean²/var, β = var/mean
        α_mom = sample_mean^2 / sample_var
        β_mom = sample_var / sample_mean  # Fixed: was incorrectly inverted
        
        # Adaptive shrinkage: less weight on prior as we get more data
        # Start with 20% shrinkage, decay to 5% as sample size grows
        shrinkage_weight = max(0.05, 0.20 * exp(-n / 1000))
        
        new_α = (1 - shrinkage_weight) * α_mom + shrinkage_weight * current_α
        new_β = (1 - shrinkage_weight) * β_mom + shrinkage_weight * current_β
        
        # Apply much more relaxed bounds to allow convergence
        # Allow α to vary widely around the hyperprior mean
        new_α = max(0.1, min(new_α, 100.0))  # Very wide bounds
        new_β = max(0.1, min(new_β, 100.0))  # Very wide bounds
    else
        # No update if insufficient variation
        new_α = current_α
        new_β = current_β
    end
    
    return new_α, new_β
end

function update_hierarchical!(chains::Vector{RJMCMCChain}, current_iteration::Int = 0)
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
    
    # Update hyperparameters
    new_α, new_β = update_gamma_hyperparameters(
        all_counts,
        template_prior.α, template_prior.β,
        template_prior.α_prior, template_prior.β_prior
    )
    
    # Record the update in hierarchical history for all chains
    for chain in chains
        push!(chain.hierarchical_history, (current_iteration, new_α, new_β))
    end
    
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
                new_hierarchical_prior  # Keep as HierarchicalGammaPrior to allow future updates
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
    
    # Calculate mean for reporting
    old_mean = template_prior.α * template_prior.β
    new_mean = new_α * new_β
    empirical_mean = length(all_counts) > 0 ? Statistics.mean(all_counts) : 0.0
    
    # Debug output disabled - uncomment for troubleshooting
    # println("Hierarchical update at iteration $current_iteration:")
    # println("  Old: α = $(round(template_prior.α, digits=3)), β = $(round(template_prior.β, digits=3)), mean = $(round(old_mean, digits=2))")
    # println("  New: α = $(round(new_α, digits=3)), β = $(round(new_β, digits=3)), mean = $(round(new_mean, digits=2))")
    # println("  Empirical mean from $(length(all_counts)) counts: $(round(empirical_mean, digits=2))")
end
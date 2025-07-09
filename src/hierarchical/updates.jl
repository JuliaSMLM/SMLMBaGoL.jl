# Hierarchical updates module - imports already handled by main module

# Count allocations per emitter
function count_allocations(state::BaGoLState)
    emitter_counts = zeros(Int, length(state.emitters))
    for alloc in state.allocations
        if 1 ≤ alloc ≤ length(state.emitters)
            emitter_counts[alloc] += 1
        end
    end
    return emitter_counts
end

function collect_emitter_counts(chains::Vector{<:RJMCMCChain})
    all_counts = Int[]
    for chain in chains
        counts = count_allocations(chain.current_state)
        append!(all_counts, counts)
    end
    return all_counts
end

# Gibbs update for μ (following math spec Section 8)
function update_mu_gibbs(counts::Vector{Int}, κ::Real, hyperprior::Tuple{Real,Real})
    a₀, b₀ = hyperprior
    n = length(counts)
    sum_counts = sum(counts)
    
    # μ ~ Gamma(a₀ + Σn_j, 1/(b₀ + n×κ))
    shape = a₀ + sum_counts
    rate = b₀ + n * κ
    
    return rand(Gamma(shape, 1/rate))
end

"""
    update_tau_squared_gibbs(chains, hyperprior)

Conjugate update for τ² using latent positions.
This is now a simple InverseGamma update with O(N) cost instead of O(N²).

τ² ~ InverseGamma(a_τ + n/2, b_τ + sum_squared_distances/2)
"""
function update_tau_squared_gibbs(chains::Vector{<:RJMCMCChain}, hyperprior::Tuple{Real,Real})
    a_τ, b_τ = hyperprior
    
    # Compute sufficient statistics: sum of squared distances from emitter to latent position
    sum_sq_dist = 0.0
    n_allocated = 0
    
    for chain in chains
        state = chain.current_state
        
        for i in eachindex(state.allocations)
            emitter_idx = state.allocations[i]
            if 1 ≤ emitter_idx ≤ length(state.emitters)
                emitter = state.emitters[emitter_idx]
                latent = state.latent_positions[i]
                
                # Squared distance from emitter to latent position
                sum_sq_dist += (latent[1] - emitter.x)^2 + (latent[2] - emitter.y)^2
                n_allocated += 2  # x and y components
            end
        end
    end
    
    # Conjugate InverseGamma update
    posterior_shape = a_τ + n_allocated / 2.0
    posterior_scale = b_τ + sum_sq_dist / 2.0
    
    # Sample from posterior
    return rand(InverseGamma(posterior_shape, posterior_scale))
end

# Slice sampling for κ (following math spec Section 8)
function update_kappa_slice(counts::Vector{Int}, μ::Real, hyperprior::Tuple{Real,Real}, 
                           current_κ::Real; n_steps::Int=10)
    # Use adaptive version by default
    return update_kappa_slice_adaptive(counts, μ, hyperprior, current_κ; n_steps=n_steps)
end

# Adaptive slice sampling for κ with better exploration
function update_kappa_slice_adaptive(counts::Vector{Int}, μ::Real, hyperprior::Tuple{Real,Real}, 
                                   current_κ::Real; n_steps::Int=10, adapt_width::Bool=true)
    c₀, d₀ = hyperprior
    n = length(counts)
    sum_counts = sum(counts)
    
    # Log density for κ (equation from Section 8)
    function log_density(κ)
        if κ ≤ 0
            return -Inf
        end
        
        log_p = (c₀ - 1) * log(κ) - κ / d₀
        
        # Sum over emitters
        for n_j in counts
            log_p -= loggamma(κ)
            log_p += loggamma(n_j + κ)
        end
        
        log_p -= (sum_counts + n * κ) * log(κ + μ)
        
        return log_p
    end
    
    # Initialize adaptive width based on current κ scale
    width = adapt_width ? max(0.1, min(10.0, current_κ * 0.5)) : 2.0
    
    # Track acceptance for adaptation
    accepted = 0
    κ = current_κ
    
    for step in 1:n_steps
        κ_old = κ
        
        # Sample height
        log_y = log_density(κ) + log(rand())
        
        # Find slice interval with current width
        lower = max(0.1, κ - width * rand())
        upper = κ + width * rand()
        
        # Expand interval
        expand_steps = 0
        while lower > 0.1 && log_density(lower) > log_y && expand_steps < 10
            lower = max(0.1, lower - width)
            expand_steps += 1
        end
        expand_steps = 0
        while log_density(upper) > log_y && expand_steps < 10
            upper = upper + width
            expand_steps += 1
        end
        
        # Sample from slice with shrinkage
        shrink_steps = 0
        while shrink_steps < 100
            κ_new = lower + (upper - lower) * rand()
            if log_density(κ_new) > log_y
                κ = κ_new
                accepted += 1
                break
            else
                if κ_new < κ
                    lower = κ_new
                else
                    upper = κ_new
                end
                shrink_steps += 1
            end
        end
        
        # Adapt width based on acceptance
        if adapt_width && step % 5 == 0
            acceptance_rate = accepted / 5
            if acceptance_rate < 0.3
                width *= 0.8  # Shrink if rejecting too much
            elseif acceptance_rate > 0.7
                width *= 1.2  # Grow if accepting too much
            end
            width = clamp(width, 0.05, 20.0)
            accepted = 0
        end
    end
    
    return κ
end

# Multi-start slice sampling for better global exploration
function update_kappa_multistart(counts::Vector{Int}, μ::Real, hyperprior::Tuple{Real,Real}, 
                               current_κ::Real; n_starts::Int=3, n_steps::Int=5, 
                               diagnostic::Bool=false)
    # Try multiple starting points
    start_points = Float64[]
    
    # Always include current value
    push!(start_points, current_κ)
    
    # Add data-driven starting point
    m = mean(counts)
    v = var(counts)
    if v > m
        κ_mom = m^2 / (v - m)
        push!(start_points, clamp(κ_mom, 0.1, 1000.0))
    else
        # For Poisson-like data
        push!(start_points, 100.0)
    end
    
    # Add some dispersed points
    push!(start_points, 1.0)
    push!(start_points, 10.0)
    push!(start_points, 50.0)
    
    if diagnostic
        println("    Multi-start with points: $start_points")
    end
    
    # Run short chains from each start
    best_κ = current_κ
    best_log_p = -Inf
    
    c₀, d₀ = hyperprior
    n = length(counts)
    sum_counts = sum(counts)
    
    function log_density(κ)
        if κ ≤ 0
            return -Inf
        end
        
        log_p = (c₀ - 1) * log(κ) - κ / d₀
        
        for n_j in counts
            log_p -= loggamma(κ)
            log_p += loggamma(n_j + κ)
        end
        
        log_p -= (sum_counts + n * κ) * log(κ + μ)
        
        return log_p
    end
    
    for start_κ in start_points[1:min(n_starts, length(start_points))]
        # Run short adaptive chain
        κ_candidate = update_kappa_slice_adaptive(counts, μ, hyperprior, start_κ; 
                                                 n_steps=n_steps, adapt_width=true)
        
        # Evaluate at endpoint
        log_p = log_density(κ_candidate)
        
        if log_p > best_log_p
            best_log_p = log_p
            best_κ = κ_candidate
        end
    end
    
    # Run longer chain from best point
    return update_kappa_slice_adaptive(counts, μ, hyperprior, best_κ; 
                                     n_steps=n_steps*2, adapt_width=true)
end

# Main hierarchical update function
function update_hierarchical!(chains::Vector{<:RJMCMCChain}, current_iteration::Int = 0; 
                            adaptive::Bool = true, diagnostic::Bool = false)
    # Check if all chains have hierarchical count priors
    hierarchical_priors = HierarchicalNegBinomialPrior[]
    
    for chain in chains
        if isa(chain.current_state.count_prior, HierarchicalNegBinomialPrior)
            push!(hierarchical_priors, chain.current_state.count_prior)
        else
            return  # No hierarchical updates needed
        end
    end
    
    if isempty(hierarchical_priors)
        return
    end
    
    # Use first prior as template
    template_prior = hierarchical_priors[1]
    current_μ = template_prior.μ
    current_κ = template_prior.κ
    current_τ² = template_prior.τ²
    
    # Collect all allocation counts
    all_counts = collect_emitter_counts(chains)
    
    if isempty(all_counts)
        return  # No data to update from
    end
    
    # Gibbs updates
    new_μ = update_mu_gibbs(all_counts, current_κ, template_prior.μ_hyperprior)
    
    # Use adaptive or multi-start for κ if requested
    if adaptive
        # Check if data looks Poisson-like (low overdispersion)
        emp_mean = mean(all_counts)
        emp_var = var(all_counts)
        variance_ratio = emp_var / emp_mean
        
        # Use multi-start if variance ratio suggests low overdispersion
        # or after some burn-in for general robustness
        if variance_ratio < 2.0 || current_iteration > 500
            if diagnostic
                println("  Using multi-start (var_ratio=$variance_ratio, iter=$current_iteration)")
            end
            new_κ = update_kappa_multistart(all_counts, new_μ, template_prior.κ_hyperprior, current_κ; 
                                          n_starts=5, n_steps=10, diagnostic=diagnostic)
        else
            if diagnostic
                println("  Using regular slice sampling")
            end
            new_κ = update_kappa_slice(all_counts, new_μ, template_prior.κ_hyperprior, current_κ)
        end
    else
        new_κ = update_kappa_slice(all_counts, new_μ, template_prior.κ_hyperprior, current_κ)
    end
    
    new_τ² = update_tau_squared_gibbs(chains, template_prior.τ²_hyperprior)
    
    # Print diagnostic info if requested
    if diagnostic
        m = mean(all_counts)
        v = var(all_counts)
        println("Hierarchical update at iteration $current_iteration:")
        println("  Data: mean=$(round(m, digits=2)), var=$(round(v, digits=2))")
        println("  Old: μ=$(round(current_μ, digits=2)), κ=$(round(current_κ, digits=2))")
        println("  New: μ=$(round(new_μ, digits=2)), κ=$(round(new_κ, digits=2))")
        println("  Theoretical var: $(round(new_μ + new_μ^2/new_κ, digits=2))")
    end
    
    # Create new hierarchical prior
    new_prior = HierarchicalNegBinomialPrior(
        new_μ, new_κ, new_τ²,
        template_prior.μ_hyperprior,
        template_prior.κ_hyperprior,
        template_prior.τ²_hyperprior
    )
    
    # Update all chains
    for chain in chains
        # Record update in history
        push!(chain.hierarchical_history, HierarchicalUpdate(
            current_iteration,
            new_μ,
            new_κ,
            new_τ²,
            mean(all_counts),
            length(all_counts)
        ))
        
        # Update chain state
        old_state = chain.current_state
        chain.current_state = BaGoLState(
            old_state.emitters,
            old_state.localizations,
            old_state.allocations,
            old_state.latent_positions,
            old_state.spatial_prior,
            new_prior,  # Updated count prior
            new_τ²,     # Updated τ²
            old_state.log_likelihood
        )
        
        # Also update the chain-level count prior for consistency
        chain.count_prior = new_prior
    end
end
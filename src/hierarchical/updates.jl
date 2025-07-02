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

# Collect all emitter counts across chains
function collect_emitter_counts(chains::Vector{<:RJMCMCChain})
    all_counts = Int[]
    for chain in chains
        for sample in chain.samples
            counts = count_allocations(sample)
            append!(all_counts, counts)
        end
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

Gibbs update for τ² (additional localization uncertainty) parameter.

# Mathematical Model
For model: loc ~ N(emitter_pos, diag(σx² + τ², σy² + τ²))
This is NOT conjugate with InverseGamma prior, so we use Metropolis-within-Gibbs.

# Prior Specification  
τ² ~ InverseGamma(a_τ, b_τ) where hyperprior = (a_τ, b_τ)

# Metropolis-within-Gibbs Update
Since the full conditional is not in closed form, we use M-H sampling
with a proposal distribution and accept/reject based on the posterior ratio.

# Arguments
- `chains`: Vector of RJMCMC chains containing current states
- `hyperprior`: Tuple (a_τ, b_τ) for InverseGamma prior parameters

# Returns
- New τ² sample from posterior distribution

# Physical Interpretation
τ² captures systematic errors like stage drift, PSF calibration errors,
and environmental effects not captured in per-localization uncertainties.
"""
function update_tau_squared_gibbs(chains::Vector{<:RJMCMCChain}, hyperprior::Tuple{Real,Real})
    a_τ, b_τ = hyperprior
    
    # Get current τ² from first chain (should be same across all chains)
    current_τ² = chains[1].current_state.τ²
    
    # Metropolis-within-Gibbs proposal
    # Use log-normal proposal to stay positive
    proposal_scale = 0.1  # Tune this for acceptance rate
    log_τ² = log(current_τ²)
    log_τ²_new = log_τ² + proposal_scale * randn()
    τ²_new = exp(log_τ²_new)
    
    # Calculate log posterior ratio
    log_prior_ratio = (a_τ + 1) * (log(current_τ²) - log(τ²_new)) + 
                     (τ²_new - current_τ²) / b_τ
    
    # Calculate likelihood ratio
    log_likelihood_ratio = 0.0
    for chain in chains
        state = chain.current_state
        for (i, loc) in enumerate(state.localizations)
            if 1 ≤ state.allocations[i] ≤ length(state.emitters)
                emitter = state.emitters[state.allocations[i]]
                
                # Current likelihood
                σx_total_old² = loc.σx^2 + current_τ²
                σy_total_old² = loc.σy^2 + current_τ²
                log_like_old = -0.5 * ((loc.x - emitter.x)^2 / σx_total_old² + 
                                      (loc.y - emitter.y)^2 / σy_total_old² + 
                                      log(σx_total_old²) + log(σy_total_old²))
                
                # Proposed likelihood  
                σx_total_new² = loc.σx^2 + τ²_new
                σy_total_new² = loc.σy^2 + τ²_new
                log_like_new = -0.5 * ((loc.x - emitter.x)^2 / σx_total_new² + 
                                      (loc.y - emitter.y)^2 / σy_total_new² + 
                                      log(σx_total_new²) + log(σy_total_new²))
                
                log_likelihood_ratio += log_like_new - log_like_old
            end
        end
    end
    
    # Jacobian for log transformation
    log_jacobian = log_τ²_new - log_τ²
    
    # Accept/reject
    log_acceptance_prob = log_prior_ratio + log_likelihood_ratio + log_jacobian
    
    if log(rand()) < log_acceptance_prob
        return τ²_new
    else
        return current_τ²
    end
end

# Slice sampling for κ (following math spec Section 8)
function update_kappa_slice(counts::Vector{Int}, μ::Real, hyperprior::Tuple{Real,Real}, 
                           current_κ::Real; n_steps::Int=10)
    c₀, d₀ = hyperprior
    n = length(counts)
    sum_counts = sum(counts)
    
    # Log density for κ (equation from Section 8)
    function log_density(κ)
        if κ ≤ 0
            return -Inf
        end
        
        log_p = (c₀ - 1) * log(κ) - d₀ * κ
        
        # Sum over emitters
        for n_j in counts
            log_p -= loggamma(κ)
            log_p += loggamma(n_j + κ)
        end
        
        log_p -= (sum_counts + n * κ) * log(κ + μ)
        
        return log_p
    end
    
    # Simple slice sampling
    κ = current_κ
    for _ in 1:n_steps
        # Sample height
        log_y = log_density(κ) + log(rand())
        
        # Find slice interval
        width = 2.0
        lower = max(0.1, κ - width * rand())
        upper = κ + width * rand()
        
        # Expand interval
        while lower > 0.1 && log_density(lower) > log_y
            lower = max(0.1, lower - width)
        end
        while log_density(upper) > log_y
            upper = upper + width
        end
        
        # Sample from slice
        while true
            κ_new = lower + (upper - lower) * rand()
            if log_density(κ_new) > log_y
                κ = κ_new
                break
            else
                if κ_new < κ
                    lower = κ_new
                else
                    upper = κ_new
                end
            end
        end
    end
    
    return κ
end

# Main hierarchical update function
function update_hierarchical!(chains::Vector{<:RJMCMCChain}, current_iteration::Int = 0)
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
    new_κ = update_kappa_slice(all_counts, new_μ, template_prior.κ_hyperprior, current_κ)
    new_τ² = update_tau_squared_gibbs(chains, template_prior.τ²_hyperprior)
    
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
            old_state.spatial_prior,
            new_prior,  # Updated count prior
            new_τ²,     # Updated τ²
            old_state.log_likelihood
        )
        
        # Also update the chain-level count prior for consistency
        chain.count_prior = new_prior
    end
end
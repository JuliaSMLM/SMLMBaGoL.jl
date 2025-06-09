# RJMCMC move emitter functionality - position updates
using Random
using SMLMData: Emitter2D

"""
Move an emitter to a new position using precision-weighted sampling.
"""
function move_emitter!(
    state::Vector{Emitter2D{T}},
    allocations::Vector{Int},
    observations::Vector{O},
    prior::HierarchicalPrior{T},
    log_prob_current::T,
    rng::AbstractRNG
) where {T<:AbstractFloat, O}
    
    n_emitters = length(state)
    n_emitters == 0 && return false, log_prob_current
    
    # Select emitter to move
    j = rand(rng, 1:n_emitters)
    
    # Get observations allocated to this emitter
    obs_indices = findall(==(j), allocations)
    
    # Store old position
    old_emitter = state[j]
    
    if isempty(obs_indices)
        # No allocated observations - sample from spatial prior
        # (mixture of Gaussians centered on all observations)
        i = rand(rng, 1:length(observations))
        obs = observations[i]
        σ_x = hasproperty(obs, :σ_x) ? obs.σ_x : T(0.1)
        σ_y = hasproperty(obs, :σ_y) ? obs.σ_y : T(0.1)
        x_new = obs.x + σ_x * randn(rng, T)
        y_new = obs.y + σ_y * randn(rng, T)
    else
        # Precision-weighted posterior sampling
        x_sum = zero(T)
        y_sum = zero(T)
        x_precision = zero(T)
        y_precision = zero(T)
        
        for i in obs_indices
            obs = observations[i]
            if hasproperty(obs, :σ_x)
                w_x = one(T) / (obs.σ_x^2)
                w_y = one(T) / (obs.σ_y^2)
            else
                w_x = w_y = T(100.0)  # High precision if no uncertainty
            end
            x_sum += w_x * obs.x
            y_sum += w_y * obs.y
            x_precision += w_x
            y_precision += w_y
        end
        
        # Posterior mean and variance
        x_mean = x_sum / x_precision
        y_mean = y_sum / y_precision
        x_var = one(T) / x_precision
        y_var = one(T) / y_precision
        
        # Sample from posterior
        x_new = x_mean + sqrt(x_var) * randn(rng, T)
        y_new = y_mean + sqrt(y_var) * randn(rng, T)
    end
    
    # Update state
    state[j] = Emitter2D(x_new, y_new, old_emitter.photons)
    
    # Compute new log probability
    log_prob_new = compute_log_posterior(state, allocations, observations, prior)
    
    # Metropolis-Hastings accept/reject
    log_ratio = log_prob_new - log_prob_current
    
    if log(rand(rng)) < log_ratio
        return true, log_prob_new
    else
        # Revert
        state[j] = old_emitter
        return false, log_prob_current
    end
end
# Likelihood configuration types

struct StandardLikelihood <: AbstractLikelihoodConfig end

struct ConsistencyLikelihood <: AbstractLikelihoodConfig
    α::Float64  # Penalty strength for variance mismatch
    
    ConsistencyLikelihood(α::Float64=1.0) = new(α)
end

struct AdaptiveConsistencyLikelihood <: AbstractLikelihoodConfig end

# Compute log likelihood based on configuration
function compute_log_likelihood(state::BaGoLState, config::StandardLikelihood)
    return log_likelihood(state)
end

function compute_log_likelihood(state::BaGoLState, config::ConsistencyLikelihood)
    return consistency_log_likelihood(state; α=config.α)
end

function compute_log_likelihood(state::BaGoLState, config::AdaptiveConsistencyLikelihood)
    return adaptive_consistency_likelihood(state)
end
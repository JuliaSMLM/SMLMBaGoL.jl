# Modified acceptance functions that use configured likelihood

function accept_probability_with_config(move_type::Type{<:AbstractRJMCMCMove}, current, proposed, likelihood_config::AbstractLikelihoodConfig)
    proposed === nothing && return 0.0
    log_ratio = log_acceptance_ratio_with_config(move_type, current, proposed, likelihood_config)
    return exp(min(0.0, log_ratio))
end

function log_acceptance_ratio_with_config(::Type{<:AbstractRJMCMCMove}, current, proposed, likelihood_config::AbstractLikelihoodConfig)
    log_posterior_with_config(proposed, likelihood_config) - log_posterior_with_config(current, likelihood_config)
end

function log_posterior_with_config(state::BaGoLState, likelihood_config::AbstractLikelihoodConfig)
    compute_log_likelihood(state, likelihood_config) + log_prior(state)
end
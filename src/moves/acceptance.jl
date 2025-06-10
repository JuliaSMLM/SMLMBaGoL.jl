function accept_probability(move_type::Type{<:AbstractRJMCMCMove}, current, proposed)
    proposed === nothing && return 0.0
    log_ratio = log_acceptance_ratio(move_type, current, proposed)
    return exp(min(0.0, log_ratio))
end

function log_acceptance_ratio(::Type{<:AbstractRJMCMCMove}, current, proposed)
    log_posterior(proposed) - log_posterior(current)
end

function log_acceptance_ratio(::Type{Birth}, current::BaGoLState, proposed::BaGoLState)
    log_acceptance_ratio_birth(current, proposed)
end

function log_acceptance_ratio(::Type{Death}, current::BaGoLState, proposed::BaGoLState)
    -log_acceptance_ratio_birth(proposed, current)  # Mathematical inverse!
end

function log_posterior(state::BaGoLState)
    log_likelihood(state) + log_prior(state)
end
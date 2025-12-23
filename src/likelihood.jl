# Likelihood calculations for BaGoL

"""
Compute log-likelihood for a single localization assigned to an emitter.
Uses Gaussian likelihood with effective variance = σ² + τ².
"""
function log_likelihood_single(
    loc::SMLMData.AbstractEmitter,
    emitter::Emitter,
    τ::Float64
)
    σ_x = loc.σ_x
    σ_y = loc.σ_y

    var_x = σ_x^2 + τ^2
    var_y = σ_y^2 + τ^2

    dx = loc.x - emitter.x
    dy = loc.y - emitter.y

    # 2D Gaussian log-likelihood
    return -0.5 * (dx^2 / var_x + dy^2 / var_y + log(2π * var_x) + log(2π * var_y))
end

"""
Compute total log-likelihood for all localizations given current emitter configuration.
"""
function compute_log_likelihood(
    locs::Vector{<:SMLMData.AbstractEmitter},
    state::BaGoLState,
    τ::Float64
)
    ll = 0.0
    for emitter in state.emitters
        for idx in emitter.allocated
            ll += log_likelihood_single(locs[idx], emitter, τ)
        end
    end
    return ll
end

"""
Compute log-likelihood contribution from localizations assigned to a specific emitter.
"""
function log_likelihood_emitter(
    locs::Vector{<:SMLMData.AbstractEmitter},
    emitter::Emitter,
    τ::Float64
)
    ll = 0.0
    for idx in emitter.allocated
        ll += log_likelihood_single(locs[idx], emitter, τ)
    end
    return ll
end

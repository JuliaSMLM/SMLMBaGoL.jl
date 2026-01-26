# Likelihood calculations for BaGoL

"""
Compute log-likelihood for a single localization assigned to an emitter.
Uses 2D Gaussian likelihood with full covariance matrix when available.

The covariance matrix is Σ = [σ_x² σ_xy; σ_xy σ_y²].
When σ_xy = 0 (uncorrelated or older SMLMData), this reduces to the diagonal case.
"""
function log_likelihood_single(
    loc::SMLMData.AbstractEmitter,
    emitter::Emitter
)
    var_x = loc.σ_x^2
    var_y = loc.σ_y^2
    σ_xy = get_cov_xy(loc)

    dx = loc.x - emitter.x
    dy = loc.y - emitter.y

    # Determinant of covariance matrix
    det_Σ = var_x * var_y - σ_xy^2

    # Guard against non-positive-definite (fall back to diagonal)
    if det_Σ <= 0
        return -0.5 * (dx^2 / var_x + dy^2 / var_y + log(2π * var_x) + log(2π * var_y))
    end

    # Quadratic form: d' * Σ⁻¹ * d
    inv_det = 1 / det_Σ
    quad = inv_det * (var_y * dx^2 - 2 * σ_xy * dx * dy + var_x * dy^2)

    # 2D Gaussian log-likelihood: -0.5 * (quad + log((2π)² * det(Σ)))
    return -0.5 * (quad + log(4π^2 * det_Σ))
end

"""
Compute total log-likelihood for all localizations given current emitter configuration.
"""
function compute_log_likelihood(
    locs::Vector{<:SMLMData.AbstractEmitter},
    state::BaGoLState
)
    ll = 0.0
    for emitter in state.emitters
        for idx in emitter.allocated
            ll += log_likelihood_single(locs[idx], emitter)
        end
    end
    return ll
end

"""
Compute log-likelihood contribution from localizations assigned to a specific emitter.
"""
function log_likelihood_emitter(
    locs::Vector{<:SMLMData.AbstractEmitter},
    emitter::Emitter
)
    ll = 0.0
    for idx in emitter.allocated
        ll += log_likelihood_single(locs[idx], emitter)
    end
    return ll
end

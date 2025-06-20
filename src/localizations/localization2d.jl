struct Localization2D{T<:Real} <: AbstractLocalization
    x::T
    y::T
    σx::T
    σy::T
    frame::Int
end

"""
Return log-likelihood of a single localisation at squared distance d2
from an emitter, given per-point σ (px or nm units).
Assumes a Rayleigh model of the radial spread.
"""
@inline function loglikelihood_rayleigh(d2::T, σ::T) where T<:Real
    r = sqrt(d2) / σ
    # Add small regularization to handle r≈0 case
    # Use 1e-6 instead of eps(T) to avoid extreme negative values
    r_reg = max(r, T(1e-6))
    return log(r_reg) - 0.5*r^2 - 2*log(σ)
end

# Drop-in replacement using Rayleigh likelihood
function log_likelihood(emitter::Emitter2D, loc::Localization2D)
    # Calculate squared distance
    d2 = (loc.x - emitter.x)^2 + (loc.y - emitter.y)^2
    
    # Use geometric mean of σx and σy for isotropic approximation
    σ = sqrt(loc.σx * loc.σy)
    
    # Rayleigh likelihood
    return loglikelihood_rayleigh(d2, σ) - log(2π)
end
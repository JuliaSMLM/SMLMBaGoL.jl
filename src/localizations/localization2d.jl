struct Localization2D{T<:Real} <: AbstractLocalization
    x::T
    y::T
    σx::T
    σy::T
    frame::Int
end

# Log-likelihood for Emitter2D-Localization2D pair with τ²
function log_likelihood(emitter::Emitter2D, loc::Localization2D, τ²::Real)
    # Total variance is observed variance plus additional systematic variance
    σx_total² = loc.σx^2 + τ²
    σy_total² = loc.σy^2 + τ²
    
    # Normalized squared distances
    dx² = (loc.x - emitter.x)^2 / σx_total²
    dy² = (loc.y - emitter.y)^2 / σy_total²
    
    # Log of normalizing constant
    log_norm = 0.5 * (log(2π) + log(σx_total²) + log(2π) + log(σy_total²))
    
    return -0.5 * (dx² + dy²) - log_norm
end
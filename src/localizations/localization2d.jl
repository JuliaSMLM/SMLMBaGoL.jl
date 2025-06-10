struct Localization2D{T<:Real} <: AbstractLocalization
    x::T
    y::T
    σx::T
    σy::T
    frame::Int
end

function log_likelihood(emitter::Emitter2D, loc::Localization2D)
    dx = (loc.x - emitter.x) / loc.σx
    dy = (loc.y - emitter.y) / loc.σy
    return -0.5 * (dx^2 + dy^2) - log(2π * loc.σx * loc.σy)
end
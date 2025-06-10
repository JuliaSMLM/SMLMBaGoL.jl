struct Emitter2D{T<:Real} <: AbstractEmitter
    x::T
    y::T
    σx::T
    σy::T
    id::Int
end
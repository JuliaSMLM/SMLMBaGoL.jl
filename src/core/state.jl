struct BaGoLState{E<:AbstractEmitter, L<:AbstractLocalization, T<:Real} <: AbstractChainState
    emitters::Vector{E}
    localizations::Vector{L}
    allocations::Vector{Int}  # maps localization i to emitter allocations[i]
    prior::AbstractPrior
    log_likelihood::T
end

mutable struct RJMCMCChain{T<:Real, E<:AbstractEmitter, L<:AbstractLocalization, P<:AbstractPrior}
    localizations::Vector{L}
    current_state::BaGoLState{E,L,T}
    prior::P
    move_weights::Dict{Type{<:AbstractRJMCMCMove}, Float64}
    samples::Vector{BaGoLState{E,L,T}}
    burn_in::Int
    thin::Int
    rng::AbstractRNG
end
import Base: length

mutable struct Params
    emitters::Vector{<:AbstractEmitter}
end
length(θ::Params) = length(θ.emitters)

struct RJMCMC_Chain
    states::Vector{Params}
end
length(chain::RJMCMC_Chain) = length(chain.states)

struct RJMCMC_ROI
    obs::SMLMBaGoL.Observations
    prior_y::Distributions.Distribution
    prior_k::Distributions.Distribution
    p_jump::Distributions.Categorical    
    emitter_type::Type
    prior_λ::Distributions.Distribution
end


mutable struct Emitter3D{T} <: AbstractEmitter
    μ_x::T
    μ_y::T
    μ_z::T
end

mutable struct Emitter2D_Drift{T} <: AbstractEmitter
    μ_x::T
    μ_y::T
    α_x::T
    α_y::T
end

mutable struct Emitter3D_Drift{T} <: AbstractEmitter
    μ_x::T
    μ_y::T
    μ_z::T
    α_x::T
    α_y::T
    α_z::T
end





struct Localization2D_Time{T} <: AbstractObservation
    x::T
    y::T
    σ_x::T
    σ_y::T
    framenum::Vector{Int}
end

struct Localization3D{T} <: AbstractObservation
    x::T
    y::T
    z::T
    σ_x::T
    σ_y::T
    σ_z::T
    framenum::Vector{Int}    
end


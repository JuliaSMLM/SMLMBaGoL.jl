
import Base: length

mutable struct Allocations 
    idx::Vector{Int}
end
length(z::Allocations) = length(z.idx)

abstract type AbstractEmitter end

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


mutable struct Params
    emitters::Vector{<:AbstractEmitter}
end
length(θ::Params) = length(θ.emitters)

abstract type AbstractObservation end 

mutable struct Observations
    ŷ::Vector{<:AbstractObservation}
end
length(obs::Observations) = length(obs.ŷ)


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

struct RJMCMC_Chain
    states::Vector{Params}
end
length(chain::RJMCMC_Chain) = length(chain.states)

struct RJMCMC_ROI
    obs::Observations
    prior_y::Distributions.Distribution
    prior_k::Distributions.Distribution
    p_jump::Distributions.Categorical    
    emitter_type::Type
end



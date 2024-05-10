import Base: length

abstract type AbstractEmitter end

struct Params{T<:AbstractEmitter}
    emitters::Vector{T}
end
function Params(emitters::Vector{T}) where T<:AbstractEmitter
    return Params{T}(copy(emitters))
end
length(θ::Params) = length(θ.emitters)

function Base.deepcopy(θ::Params{T}) where T<:AbstractEmitter
    return Params{T}(deepcopy(θ.emitters))
end


abstract type AbstractObservation end 

struct Observations{T<:AbstractObservation}
    ŷ::Vector{T}
end
length(obs::Observations) = length(obs.ŷ)

mutable struct Allocations 
    idx::Vector{Int}
end
length(z::Allocations) = length(z.idx)

abstract type Posterior end

mutable struct Posterior2D <: Posterior
    post_arr::Array{Float64, 2} 
    pixelsize::Float64
    x_start::Float64        
    y_start::Float64
    x_size::Int
    y_size::Int
end
function Posterior2D(smld::SMLMData.SMLD2D;
    pixel_size::Float64=1.0
    )
    x_start = 0.5
    y_start = 0.5
    x_size = Int(round((smld.datasize[2])/pixel_size))
    y_size = Int(round((smld.datasize[1])/pixel_size))
    post_arr = zeros(Float64, y_size, x_size)
    return Posterior2D(post_arr, pixel_size, x_start, y_start, x_size, y_size)
end

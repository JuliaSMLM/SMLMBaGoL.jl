import Base: length

abstract type AbstractObservation end 

mutable struct Observations
    ŷ::Vector{<:AbstractObservation}
end
length(obs::Observations) = length(obs.ŷ)

mutable struct Allocations 
    idx::Vector{Int}
end
length(z::Allocations) = length(z.idx)


abstract type AbstractObservation end 

mutable struct Observations
    ŷ::Vector{<:AbstractObservation}
end
length(obs::Observations) = length(obs.ŷ)

import Base: length
using Distributions
using SMLMData: AbstractEmitter, Emitter2D

# Using AbstractEmitter from SMLMData

struct Params{T<:AbstractEmitter}
    emitters::Vector{T}
end
function Params(θ::Params{T}) where T<:AbstractEmitter
    emitters = T[emit for emit in θ.emitters]
    return Params{T}(emitters)
end

# function Params(θ::Params{T}) where T<:AbstractEmitter
#     return Params{T}(deepcopy(θ.emitters))
# end
length(θ::Params) = length(θ.emitters)

function Base.deepcopy(θ::Params{T}) where T<:AbstractEmitter
    return Params{T}(deepcopy(θ.emitters))
end


abstract type AbstractObservation end 

struct Observations{T<:AbstractObservation}
    ŷ::Vector{T}
end
length(obs::Observations) = length(obs.ŷ)

mutable struct Allocations 
    idx::Vector{Int}
end
length(z::Allocations) = length(z.idx)

# Localization2D observation type from refactor-rjmcmc
struct Localization2D{T} <: AbstractObservation
    y::T
    x::T
    σ_y::T
    σ_x::T
end

# Constructor for Localization2D using type of Emitter2D
function Localization2D{T}(emitter::Emitter2D{T}, σ_y::T, σ_x::T) where T
    return Localization2D(emitter.y, emitter.x, σ_y, σ_x)
end

# Log likelihood function for Localization2D
function log_p_z_given_y(loc::Localization2D, emitter::Emitter2D)
    return logpdf(Normal(loc.x, loc.σ_x), emitter.x) +
           logpdf(Normal(loc.y, loc.σ_y), emitter.y)
end

# Hierarchical prior for gamma distribution on λ (localizations per emitter)
struct HierarchicalPrior{T<:AbstractFloat}
    α::T      # Shape parameter
    β::T      # Rate parameter
    a₀::T     # Hyperprior on α (shape)
    b₀::T     # Hyperprior on α (rate)
    c₀::T     # Hyperprior on β (shape)
    d₀::T     # Hyperprior on β (rate)
end

# Default uninformative prior
HierarchicalPrior{T}() where T = HierarchicalPrior{T}(
    one(T),    # α = 1
    one(T),    # β = 1
    one(T),    # a₀ = 1
    T(0.1),    # b₀ = 0.1
    one(T),    # c₀ = 1
    T(0.1)     # d₀ = 0.1
)

# Chain type that works with the new types
struct BaGoLChain{T<:AbstractFloat, O}
    states::Vector{Vector{Emitter2D{T}}}    # Internal states (position only)
    log_probs::Vector{T}                    # Log probabilities
    allocations::Vector{Vector{Int}}        # Allocation vectors
    observations::Vector{O}                 # Original observations (any emitter type)
end

# Main result type
struct BaGoLResult{T<:AbstractFloat, O}
    chains::Vector{BaGoLChain{T, O}}
    posterior::Matrix{T}
    mapn_emitters::Vector{Emitter2D{T}}
    log_evidence::T
    updated_prior::HierarchicalPrior{T}
end

# RJMCMC move types
@enum MoveType begin
    MOVE_EMITTER = 1
    ADD_EMITTER = 2
    REMOVE_EMITTER = 3
    SPLIT_EMITTER = 4
    MERGE_EMITTER = 5
    REALLOCATE = 6
end

# Move probabilities structure
struct MoveProbs{T<:AbstractFloat}
    move::T
    add::T
    remove::T
    split::T
    merge::T
    reallocate::T
    
    function MoveProbs{T}(move, add, remove, split, merge, reallocate) where T
        total = move + add + remove + split + merge + reallocate
        @assert abs(total - one(T)) < eps(T) "Move probabilities must sum to 1"
        new{T}(move, add, remove, split, merge, reallocate)
    end
end

# Default move probabilities
MoveProbs{T}() where T = MoveProbs{T}(
    T(0.3),   # move
    T(0.15),  # add
    T(0.15),  # remove
    T(0.1),   # split
    T(0.1),   # merge
    T(0.2)    # reallocate
)

abstract type Posterior end

mutable struct Posterior2D <: Posterior
    post_arr::Array{Float64, 2} 
    y_start::Float64
    x_start::Float64        
    y_size::Int
    x_size::Int
    pixelsize::Float64
end
function Posterior2D(smld::SMLMData.SMLD;
    pixelsize::Float64=1.0
    )
    y_start = 0.5
    x_start = 0.5
    # Get camera dimensions from pixel edges
    cam = smld.camera
    y_size = Int(round((cam.pixel_edges_y[end] - cam.pixel_edges_y[1])/pixelsize))
    x_size = Int(round((cam.pixel_edges_x[end] - cam.pixel_edges_x[1])/pixelsize))
    post_arr = zeros(Float64, y_size, x_size)
    return Posterior2D(post_arr, y_start, x_start, y_size, x_size, pixelsize)
end
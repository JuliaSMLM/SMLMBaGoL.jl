# Core type definitions for SMLMBaGoL
using SMLMData: AbstractEmitter, Emitter2D, Emitter2DFit

# Generic chain type that works with any observation emitter type
struct BaGoLChain{T<:AbstractFloat, O}
    states::Vector{Vector{Emitter2D{T}}}    # Internal states (position only)
    log_probs::Vector{T}                    # Log probabilities
    allocations::Vector{Vector{Int}}        # Allocation vectors
    observations::Vector{O}                 # Original observations (any emitter type)
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
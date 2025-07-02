# Abstract types for SMLMBaGoL package

abstract type AbstractLocalization end
# Use AbstractEmitter from SMLMData instead of defining our own
# abstract type AbstractEmitter end  
abstract type AbstractPrior end
abstract type AbstractRJMCMCMove end
abstract type AbstractChainState end

# Hierarchical update tracking structure
struct HierarchicalUpdate{T<:Real}
    iteration::Int
    μ::T                    # mean localizations per emitter
    κ::T                    # overdispersion parameter
    mean_count::T          # empirical mean from data
    n_emitters::Int        # number of emitters in update
end
using Base

# This file contains several functions/methods useful for manipulating chains
# and state structures.

"""
    addstate!(chain::SMLMBaGoL.BaGoLChain2D, 
              state::SMLMBaGoL.BaGoLState2D, 
              accepted::Bool)
        
Add `state` to the end of `chain`.

# Inputs
-`chain`: Chain of states.
-`state`: State to be added at the end of `chain`.
-`accepted`: Boolean indicating acceptance of the `state` being added to
             `chain`.
"""
function addstate!(chain::SMLMBaGoL.BaGoLChain2D, 
                   state::SMLMBaGoL.BaGoLState2D, 
                   accepted::Bool)
    # Update the chain to include `state`.
    push!(chain.states, state)
    push!(chain.accepted, accepted)
    chain.n += 1
end

"""
    removestate!(chain::SMLMBaGoL.BaGoLChain2D, remove)
        
Remove the state directed to by `remove` from the `chain`.

# Inputs
-`chain`: Chain of states.
-`remove`: State to be removed from `chain`, defined in any way allowed by the
           input `inds` in the Julia method deleteat!().
"""
function removestate!(chain::SMLMBaGoL.BaGoLChain2D, remove)
    # Update the chain remove the state `remove`.
    deleteat!(chain.states, remove)
    deleteat!(chain.accepted, remove)
    chain.n = Base.length(chain.states)
end

"""
    removeemitter!(state::SMLMBaGoL.BaGoLState2D, k::Int)
        
Remove the emitter indexed as `k` from the given `state`.

# Inputs
-`state`: State of a Markov chain.
-`k`: Emitter index of the emitter to be removed from `state`.
"""
function removeemitter!(state::SMLMBaGoL.BaGoLState2D, k::Int)
    # Remove the k-th emitter and ensure `z` consists of integers 1:k_emitters.
    keepind = setdiff(1:state.k, k)
    state.k -= 1
    state.μ = state.μ[keepind, :]
    state.a = state.a[keepind, :]
    state.z[state.z .> state.k] .-= 1
end

"""
    removeemitter!(state::SMLMBaGoL.BaGoLState2D, k::Vector{Int})
        
Remove the emitters indexed as `k` from the given `state`.  Note that no 
reallocation is performed, so if an emitter is removed which has localizations
allocated to it, `state.z` is set to -1 for those localizations.

# Inputs
-`state`: State of a Markov chain.
-`k`: Emitter indices of emitters to be removed from `state`.
"""
function removeemitter!(state::SMLMBaGoL.BaGoLState2D, k)
    # Remove the k-th emitter and ensure `z` consists of integers 1:k_emitters.
    keepind = setdiff(1:state.k, k)
    state.k = Base.length(keepind)
    state.μ = state.μ[keepind, :]
    state.a = state.a[keepind, :]
    for ii = 1:Base.length(state.z)
        if !in(state.z[ii], keepind)
            state.z[ii] = -1
        end
    end
end

"""
    chain = cat(chain1::SMLMBaGoL.BaGoLChain2D, chain2::SMLMBaGoL.BaGoLChain2D)

Concatenate `chain1` and `chain2`, with `chain2` added at the end of `chain1`.

# Inputs
-`chain1`: Initial chain.
-`chain2`: Chain to be concatenated at the end of `chain1`.

# Outputs
-`chain`: Concatenation of `chain1` and `chain2`.
"""
function cat(chain1::SMLMBaGoL.BaGoLChain2D, chain2::SMLMBaGoL.BaGoLChain2D)
    # Concatenate `chain2` at the end of `chain1`.
    chain = SMLMBaGoL.BaGoLChain2D(chain1.n + chain2.n)
    chain.states = [chain1.states; chain2.states]
    chain.accepted = [chain1.accepted; chain2.accepted]
    chain.n = chain1.n + chain2.n

    return chain
end

"""
    chain = cat(chain1::SMLMBaGoL.BaGoLChain2D, chain2::SMLMBaGoL.BaGoLChain2D)

Concatenate `chain1` and `chain2`, with `chain2` added at the end of `chain1`.

# Inputs
-`chain1`: Initial chain.
-`chain2`: Chain to be concatenated at the end of `chain1`.

# Outputs
-`chain`: Concatenation of `chain1` and `chain2`.
"""
function cat!(chain1::SMLMBaGoL.BaGoLChain2D, chain2::SMLMBaGoL.BaGoLChain2D)
    # Concatenate `chain2` at the end of `chain1`.
    chain1.states = [chain1.states; chain2.states]
    chain1.accepted = [chain1.accepted; chain2.accepted]
    chain1.n = chain1.n + chain2.n
end
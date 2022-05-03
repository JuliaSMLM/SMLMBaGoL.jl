using Base

# This file contains several functions/methods useful for manipulating chains
# and state structures.

"""
    addstate!(chain::SMLMBaGoL.BaGoLChain2D, 
              state::SMLMBaGoL.BaGoLState2D, 
              accepted::Bool)
        
Add `state` to the end of `chain`.

# Inputs
- `chain`: Chain of states.
- `state`: State to be added at the end of `chain`.
- `accepted`: Boolean indicating acceptance of the `state` being added to
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
- `chain`: Chain of states.
- `remove`: State to be removed from `chain`, defined in any way allowed by the
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
        
Remove the emitter indexed as `k` from the given `state`.  Note that no 
reallocation is performed, so if an emitter is removed which has localizations
allocated to it, `state.z` is set to -1 for those localizations.

# Inputs
- `state`: State of a Markov chain.
- `k`: Emitter index of emitter to be removed from `state`.
"""
function removeemitter!(state::SMLMBaGoL.BaGoLState2D, k::Int)
    # Remove the k-th emitter and ensure `z` consists of integers 1:k_emitters.
    keepind = setdiff(1:state.k, k)
    state.k = Base.length(keepind)
    state.μ = state.μ[keepind, :]
    state.a = state.a[keepind, :]
    state.z[state.z .== k] .= -1
end

"""
    removeemitter!(state::SMLMBaGoL.BaGoLState2D, k::Vector{Int})
        
Remove the emitters indexed as `k` from the given `state`.  Note that no 
reallocation is performed, so if an emitter is removed which has localizations
allocated to it, `state.z` is set to -1 for those localizations.

# Inputs
- `state`: State of a Markov chain.
- `k`: Emitter indices of emitters to be removed from `state`.
"""
function removeemitter!(state::SMLMBaGoL.BaGoLState2D, k::Vector{Int})
    # Remove the emitters and ensure `z` consists of integers 1:k_emitters.
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
- `chain1`: Initial chain.
- `chain2`: Chain to be concatenated at the end of `chain1`.

# Outputs
- `chain`: Concatenation of `chain1` and `chain2`.
"""
function Base.cat(chain1::SMLMBaGoL.BaGoLChain2D, chain2::SMLMBaGoL.BaGoLChain2D)
    # Concatenate `chain2` at the end of `chain1`.
    chain = SMLMBaGoL.BaGoLChain2D(chain1.n + chain2.n)
    chain.states = [chain1.states; chain2.states]
    chain.accepted = [chain1.accepted; chain2.accepted]
    chain.n = chain1.n + chain2.n

    return chain
end

"""
    chain = cat(chain1::Vector{SMLMBaGoL.BaGoLChain2D}, 
                chain2::Vector{SMLMBaGoL.BaGoLChain2D})

Concatenate `chain1` and `chain2`, with `chain2` added at the end of `chain1`.

# Inputs
- `chain1`: Vector of chains with indices matching those of `chain2` (e.g., 
            chain2[nn] will be concatenated with chain1[nn]).
- `chain2`: Chains to be concatenated at the end of `chain1`.

# Outputs
- `chain`: Concatenation of `chain1` and `chain2`.
"""
function Base.cat(chain1::Vector{SMLMBaGoL.BaGoLChain2D}, 
    chain2::Vector{SMLMBaGoL.BaGoLChain2D})

    # Concatenate `chain2` at the end of `chain1`.
    chain = Vector{SMLMBaGoL.BaGoLChain2D}(undef, length(chain1))
    for ii = 1:length(chain1)
        chain[ii] = SMLMBaGoL.BaGoLChain2D(chain1[ii].n + chain2[ii].n)
        chain[ii].states = [chain1[ii].states; chain2[ii].states]
        chain[ii].accepted = [chain1[ii].accepted; chain2[ii].accepted]
        chain[ii].n = chain1[ii].n + chain2[ii].n
    end

    return chain
end

"""
    k, z, μ, a = catfields(states::Vector{SMLMBaGoL.BaGoLState2D})

Extract fields from each state in the vector `states` and concatenate.

# Inputs
- `states`: Vector of states from which we'll extract the fields and 
            concatenate.

# Outputs
- `k`: `state.k` concatenated across all states in `states`.
- `z`: `state.z` concatenated across all states in `states`.
- `μ`: `state.μ` concatenated across all states in `states`.
- `a`: `state.a` concatenated across all states in `states`.
"""
function catfields(states::Vector{SMLMBaGoL.BaGoLState2D})
    k = Vector{Int}(undef, Base.length(states))
    z = Vector{Vector{Int}}(undef, Base.length(states))
    μ = Matrix{Float64}(undef, 0, 2)
    a = Matrix{Float64}(undef, 0, 2)
    for ii = 1:Base.length(states)
        k[ii] = states[ii].k
        z[ii] = states[ii].z
        μ = vcat(μ, states[ii].μ)
        a = vcat(a, states[ii].a)
    end

    return k, z, μ, a
end

"""
    k, z μ, a = catfields(chain::SMLMBaGoL.BaGoLChain2D)

Extract fields from each state of the `chain` and concatenate them into arrays
representing all states.

# Inputs
- `chain`: Chain from which we'll extract the fields of each state.

# Outputs
- `k`: `state.k` concatenated across all states in `states`.
- `z`: `state.z` concatenated across all states in `states`.
- `μ`: `state.μ` concatenated across all states in `states`.
- `a`: `state.a` concatenated across all states in `states`.
"""
function catfields(chain::SMLMBaGoL.BaGoLChain2D)
    return catfields(chain.states)
end

"""
    k, z, μ, a = catfields(chain::Vector{SMLMBaGoL.BaGoLChain2D})

Extract fields from each state of the `chain` and concatenate them into arrays
representing all states.

# Inputs
- `chain`: Vector of chains from which we'll extract the fields of each state.

# Outputs
- `k`: `state.k` concatenated across all states in `states`.
- `z`: `state.z` concatenated across all states in `states`.
- `μ`: `state.μ` concatenated across all states in `states`.
- `a`: `state.a` concatenated across all states in `states`.
"""
function catfields(chain::Vector{SMLMBaGoL.BaGoLChain2D})
    k = Vector{Int}(undef, 0)
    z = Vector{Vector{Int}}(undef, 0)
    μ = Matrix{Float64}(undef, 0, 2)
    a = Matrix{Float64}(undef, 0, 2)
    for ii = 1:Base.length(chain)
        knew, znew, μnew, anew = SMLMBaGoL.catfields(chain[ii])
        k = vcat(k, knew)
        z = vcat(z, znew)
        μ = vcat(μ, μnew)
        a = vcat(a, anew)
    end

    return k, z, μ, a
end

"""
    k, z, μ, a = catfields(chain::Matrix{Vector{SMLMBaGoL.BaGoLChain2D}})

Extract fields from each state of the `chain` and concatenate them into arrays
representing all states.

# Inputs
- `chain`: Matrix of vectorsof chains from which we'll extract the fields of 
           each state.

# Outputs
- `k`: `state.k` concatenated across all states in `states`.
- `z`: `state.z` concatenated across all states in `states`.
- `μ`: `state.μ` concatenated across all states in `states`.
- `a`: `state.a` concatenated across all states in `states`.
"""
function catfields(chain::Matrix{Vector{SMLMBaGoL.BaGoLChain2D}})
    k = Vector{Int}(undef, 0)
    z = Vector{Vector{Int}}(undef, 0)
    μ = Matrix{Float64}(undef, 0, 2)
    a = Matrix{Float64}(undef, 0, 2)
    for ii = 1:prod(size(chain))
        knew, znew, μnew, anew = SMLMBaGoL.catfields(chain[ii])
        k = vcat(k, knew)
        z = vcat(z, znew)
        μ = vcat(μ, μnew)
        a = vcat(a, anew)
    end

    return k, z, μ, a
end
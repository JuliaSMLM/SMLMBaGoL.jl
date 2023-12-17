

function test_ids(θ::Params, z::Allocations, message::String )
    
    # for i in 1:length(θ)
    #     if !has_allocation(z, i)
    #         @warn "Warning: emitter $i not in z.idx: $message"
    #         println("z.idx = $(z.idx)")
    #         println("maximum of z.idx = $(maximum(z.idx))")
    #         println("length of θ = $(length(θ)) ")
    #     end
    # end
    
    for i in unique(z.idx)
        if i > length(θ)
            @warn "Warning: z.idx contains $i but there are only $(length(θ)) emitters: $message"
            println("z.idx = $(z.idx)")
            println("maximum of z.idx = $(maximum(z.idx))")
            println("length of θ = $(length(θ)) ")
            return false
        end
    end
    return true
end


function clean_params!(θ::Params, z::Allocations)
    # Remove emitters that do not have any allocated observations and update z.idx
    for id in reverse(eachindex(θ.emitters))
        if id ∉ z.idx
            deleteat!(θ.emitters, id)
            z.idx[z.idx .> id] .-= 1
        end
    end
end


function add_emitter(θ::Params, new_emitter::AbstractEmitter)
    θ_new = deepcopy(θ)
    push!(θ_new.emitters, new_emitter)
    return θ_new
end

function remove_emitter(θ::Params, id::Int)
    θ_new = deepcopy(θ)
    deleteat!(θ_new.emitters, id)
    return θ_new
end

function propose_add_emitter(θ::Params, obs::Observations, prior_y::Distributions.Distribution)

    # Sample a new emitter position from the prior and create new emitter
    new_emitter = typeof(θ.emitters[1])(rand(prior_y))

    # Create a new parameter vector with the new emitter
    θ_test = add_emitter(θ, new_emitter)

    # Allocation
    z_test = allocate(obs, θ_test)

    test_ids(θ_test, z_test, "propose_add_emitter")
    return θ_test, z_test, length(θ_test)
end

function propose_remove_emitter(θ::Params, obs::Observations, prior_y::Distributions.Distribution)

    # Sample an emitter to remove
    id = rand(1:length(θ))

    # println("remove_emitter id = $id out of $(length(θ))")

    # Create a new parameter vector without the emitter
    θ_test = remove_emitter(θ, id)

    # Allocation
    z_test = allocate(obs, θ_test)

    θ_length = length(θ)
    θ_test_length = length(θ_test)
    # println("z_test.idx after remove_emitter to go from $θ_length to $θ_test_length  = $(z_test.idx)")

    # test_ids(θ_test, z_test, "propose_remove_emitter")
    return θ_test, z_test, id
end


function p_accept_common(θ::Params, θ_test::Params, obs::Observations, z::Allocations, z_test::Allocations, prior_k::Distributions.Distribution)

    # Prior on the number of emitters
    k = length(θ)
    prior_ratio = pdf(prior_k, k + 1) / pdf(prior_k, k)

    # Check that the allocation is correct
    a = test_ids(θ, z, "p_accept_add: original")
    b = test_ids(θ_test, z_test, "p_accept_add: test")

    if !a || !b
        println("z.idx in p_accept_add = $(z.idx)")
        println("z_test.idx in p_accept_add  = $(z_test.idx)")
    end

    # Compute the likelihood ratio
    # Note all allocations are different, so we need to compute the likelihood ratio for each observation
    log_likelihood_ratio = 1.0

    for i in 1:length(obs)
        # Compute the likelihood ratio for this observation
        id = z.idx[i]
        id_test = z_test.idx[i]
        log_likelihood_ratio += log_p_z_given_y(obs.ŷ[i], θ_test.emitters[id_test]) -
                                log_p_z_given_y(obs.ŷ[i], θ.emitters[id])
    end
  
    likelihood_ratio = exp(log_likelihood_ratio)
  
    # Compute the acceptance ratio
    return prior_ratio * likelihood_ratio
end

function p_accept_add(θ::Params, θ_test::Params, obs::Observations, z::Allocations, z_test::Allocations, prior_k::Distributions.Distribution, prior_y::Distributions.Distribution, area::Real)
    coord = [θ_test.emitters[end].y, θ_test.emitters[end].x]
    proposal_ratio = area * pdf(prior_y, coord)
    return p_accept_common(θ, θ_test, obs, z, z_test, prior_k) * proposal_ratio
end


function p_accept_remove(θ::Params, θ_test::Params, obs::Observations, z::Allocations, z_test::Allocations, prior_λ::Distributions.Distribution)
    
    a = test_ids(θ, z, "p_accept_remove: original")
    b = test_ids(θ_test, z_test, "p_accept_remove: test")

    if !a || !b
        println("z.idx in p_accept_add = $(z.idx)")
        println("z_test.idx in p_accept_add  = $(z_test.idx)")
    end
    # println("remove to go from $(length(θ)) to $(length(θ_test))")
    # Use the same function as for add with the arguments swapped and take the inverse
    return 1 / p_accept_common(θ_test, θ, obs, z_test, z, prior_λ) 
end


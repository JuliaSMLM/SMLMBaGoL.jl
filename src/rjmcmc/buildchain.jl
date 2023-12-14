

function take_jumps!(z::Allocations, θ::Params, roi::RJMCMC_Chain, n_jumps::Int)
    
    θ = copy(θ)
    for i in 1:n_jumps
        # Propose a new state
        jump_type = rand(roi.p_jump)
        if jump_type == 1 # move emitters
            move!(θ, roi.obs, z)
        elseif jump_type == 2 # allocate 
            allocate!(z, roi.obs, θ)
        elseif jump_type == 3 # add emitter
            θ_test, z_test = propose_add_emitter!(θ, roi.obs, z, roi.prior_y)
            p_accept = p_accept_add(θ, θ_test, roi.obs, z, z_test, roi.prior_k)
            if rand() < p_accept
                θ .= θ_test
                z .= z_test
            end
        elseif jump_type == 4 # remove emitter
            θ_test, z_test = propose_remove_emitter!(θ, roi.obs, z, roi.prior_y)
            p_accept = p_accept_remove(θ, θ_test, roi.obs, z, z_test, roi.prior_k)
            if rand() < p_accept
                θ .= θ_test
                z .= z_test
            end
        end
    end
    return θ
end


function buildchain(roi::RJMCMC_ROI, n_burnin::Int, n_jumps::Int; θ::Union{Nothing,Params} = nothing)

    # Initialize the chain
    if isnothing(θ)
        # draw k from prior
        k = rand(prior_k)
        # generate a Params object with k emitters of the correct type (based on chain)
        θ = Params([emitter_type(rand(prior_y)) for i in 1:k])
    end

    # Allocate
    z = allocate(obs, θ)
       
    # Run a burn-in period without saving the states
    θ = take_jumps!(z, θ, roi, n_burnin)
    
    # Initialize the chain
    chain.states[1] = θ

    for i in 2:n_jumps
        chain.states[i] = take_jumps!(z, θ, roi, 1)
    end

    return chain
end
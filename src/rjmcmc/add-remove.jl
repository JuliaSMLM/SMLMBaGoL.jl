
function clean_params!(θ::Params, z::Allocations)
    # Remove emitters that do not have any allocated observations and update z.idx
    for id in reverse(eachindex(θ.emitters))
        if id ∉ z.idx
            deleteat!(θ.emitters, id)
            z.idx[z.idx.>id] .-= 1
        end
    end
end

function add_emitter(θ::Params, new_emitter::AbstractEmitter)
    θ_new = Params(θ.emitters)
    push!(θ_new.emitters, new_emitter)
    return θ_new
end

function remove_emitter(θ::Params, id::Int)
    θ_new = Params(θ.emitters)
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

    return θ_test, z_test, length(θ_test)
end

function propose_remove_emitter(θ::Params, obs::Observations, prior_y::Distributions.Distribution)
    # Sample an emitter to remove
    id = rand(1:length(θ))

    # Create a new parameter vector without the emitter
    θ_test = remove_emitter(θ, id)

    # Allocation
    z_test = allocate(obs, θ_test)

    return θ_test, z_test, id
end

function loglikelihood_position(obs::Observations, θ::Params, z::Allocations)
    loglikelihood = 0.0
    for i in 1:length(obs)
        # Compute the likelihood for this observation
        id = z.idx[i]
        loglikelihood += log_p_z_given_y(obs.ŷ[i], θ.emitters[id])
    end
    return loglikelihood
end

function loglikelihood_number(θ::Params, z::Allocations, prior_λ::Distributions.Distribution)
    loglikelihood = 0.0
    for i in 1:length(θ)
        loglikelihood += logpdf(prior_λ, length(z.idx[z.idx.==i]))
    end
    return loglikelihood
end

function p_accept_add(θ::Params, θ_test::Params, obs::Observations, z::Allocations,
    z_test::Allocations, prior_k::Distributions.Distribution, prior_λ::Distributions.Distribution)

    n = length(θ)

    # These are left here to show mathematically what is happening
    # coord = [θ_test.emitters[end].y, θ_test.emitters[end].x]
    # proposal_ratio = (n+1)/(n*pdf(prior_y, coord)) 
    # prior_ratio_position = pdf(prior_y, coord) # note other terms cancel out

    prior_ratio_k = pdf(prior_k, n + 1) / pdf(prior_k, n)
    prior_proposal_ratio = prior_ratio_k * (n + 1) / n

    likelihood_ratio_position = exp(loglikelihood_position(obs, θ_test, z_test)
                                    -
                                    loglikelihood_position(obs, θ, z))

    # likelihood_ratio_number = exp(loglikelihood_number(θ_test, z_test, prior_λ)
    #     - loglikelihood_number(θ, z, prior_λ))

    likelihood_ratio_number = exp(log_dirichlet_multinomial_pmf(θ_test, z_test, prior_λ)
                                  -
                                  log_dirichlet_multinomial_pmf(θ, z, prior_λ))

    
    α = prior_proposal_ratio * likelihood_ratio_position * likelihood_ratio_number
    # println("likelihood_ratio_number = $likelihood_ratio_number")
    return α
end



function p_accept_remove(θ::Params, θ_test::Params, obs::Observations, z::Allocations,
    z_test::Allocations, prior_k::Distributions.Distribution, prior_λ::Distributions.Distribution)

    n = length(θ)

    # These are left here to show mathematically what is happening
    # coord = [θ_test.emitters[idx].y, θ_test.emitters[idx].x]
    # proposal_ratio = n/(n-1)*pdf(prior_y, coord) 
    # prior_ratio_position = 1/pdf(prior_y, coord) # note other terms cancel out

    prior_ratio_k = pdf(prior_k, n - 1) / pdf(prior_k, n)
    prior_proposal_ratio = prior_ratio_k * n / (n - 1)

    likelihood_ratio_position = exp(loglikelihood_position(obs, θ_test, z_test)
                                    -
                                    loglikelihood_position(obs, θ, z))

    # likelihood_ratio_number = exp(loglikelihood_number(θ_test, z_test, prior_λ)
    #     - loglikelihood_number(θ, z, prior_λ))

    likelihood_ratio_number = exp(log_dirichlet_multinomial_pmf(θ_test, z_test, prior_λ)
                                  -
                                  log_dirichlet_multinomial_pmf(θ, z, prior_λ))


    α = prior_proposal_ratio * likelihood_ratio_position * likelihood_ratio_number
    return α
end

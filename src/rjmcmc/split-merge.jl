function propose_split_emitter(θ::Params, obs::Observations, z::Allocations)
    # Sample an emitter to split
    id = rand(1:length(θ))

    # Create two new emitters with the same position as the original emitter
    emitter = θ.emitters[id]
    new_emitter1 = typeof(emitter)(emitter)
    new_emitter2 = typeof(emitter)(emitter)

    # Create a new parameter vector with the split emitters
    θ_test = remove_emitter(θ, id)
    θ_test = add_emitter(θ_test, new_emitter1)
    θ_test = add_emitter(θ_test, new_emitter2)

    # Allocation
    z_test = allocate(obs, θ_test)
    # show number of observations allocated to each emitter
    # for i in 1:length(θ_test)
    #     println("Emitter $i: ", sum(z_test.idx .== i))
    # end
    return θ_test, z_test, id
end

function propose_merge_emitters(θ::Params, obs::Observations, z::Allocations)
    # Sample an emitter to merge
    id1 = rand(1:length(θ))

    # Find the nearest emitter to merge with
    id2 = findmin([sqrt((θ.emitters[id1].x - θ.emitters[i].x)^2 + (θ.emitters[id1].y - θ.emitters[i].y)^2) for i in 1:length(θ) if i != id1])[2]
    if id2 >= id1
        id2 += 1
    end

    # Merge the emitters using the merge_emitters function
    emitter1 = θ.emitters[id1]
    emitter2 = θ.emitters[id2]
    new_emitter = merge_emitters(emitter1, emitter2)

    # Create a new parameter vector with the merged emitter
    if id1 > id2
        id1, id2 = id2, id1
    end
    θ_test = remove_emitter(θ, id2)
    θ_test = remove_emitter(θ_test, id1)
    θ_test = add_emitter(θ_test, new_emitter)

    # Allocation
    z_test = allocate(obs, θ_test)

    return θ_test, z_test, (id1, id2)
end

function p_accept_split(θ::Params, θ_test::Params, obs::Observations, z::Allocations,
    z_test::Allocations, prior_k::Distributions.Distribution, prior_λ::Distributions.Distribution)

    n = length(θ)

    prior_ratio_k = pdf(prior_k, n + 1) / pdf(prior_k, n)
    proposal_ratio = n / (n + 1)
    prior_proposal_ratio = prior_ratio_k * proposal_ratio

    likelihood_ratio_position = exp(loglikelihood_position(obs, θ_test, z_test)
                                    -
                                    loglikelihood_position(obs, θ, z))

    likelihood_ratio_number = exp(log_dirichlet_multinomial_pmf(θ_test, z_test, prior_λ)
                                  -
                                  log_dirichlet_multinomial_pmf(θ, z, prior_λ))

    α = prior_proposal_ratio * likelihood_ratio_position * likelihood_ratio_number
    # println("likelihood_ratio_number = $likelihood_ratio_number")
    # println("α = $α")
    return α
end

function p_accept_merge(θ::Params, θ_test::Params, obs::Observations, z::Allocations,
    z_test::Allocations, prior_k::Distributions.Distribution, prior_λ::Distributions.Distribution)

    n = length(θ)

    prior_ratio_k = pdf(prior_k, n - 1) / pdf(prior_k, n)
    proposal_ratio = (n - 1) / n
    prior_proposal_ratio = prior_ratio_k * proposal_ratio

    likelihood_ratio_position = exp(loglikelihood_position(obs, θ_test, z_test)
                                    -
                                    loglikelihood_position(obs, θ, z))

    likelihood_ratio_number = exp(log_dirichlet_multinomial_pmf(θ_test, z_test, prior_λ)
                                  -
                                  log_dirichlet_multinomial_pmf(θ, z, prior_λ))

    α = prior_proposal_ratio * likelihood_ratio_position * likelihood_ratio_number

    return α
end
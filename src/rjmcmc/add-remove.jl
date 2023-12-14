
function propose_add_emitter!(θ::Params, obs::Observations, z::Allocations, prior_y::Distributions.Distribution)

    # Sample a new emitter position from the prior and create new emitter
    new_emitter = typeof(θ.emitters[1])(rand(prior_y))
    
    # Create a new parameter vector with the new emitter
    θ_test = add_emitter(θ, new_emitter)

    # Allocation
    z_test = allocate(obs, θ_test)

    return θ_test, z_test
end

function propose_remove_emitter!(θ::Params, obs::Observations, z::Allocations, prior_y::Distributions.Distribution)

    # Sample an emitter to remove
    id = rand(1:length(θ.emitters))

    # Create a new parameter vector without the emitter
    θ_test = remove_emitter(θ, id)

    # Allocation
    z_test = allocate(obs, θ_test)

    return θ_test, z_test
end


function p_accept_add(θ::Params, θ_test::Params, obs::Observations, z::Allocations, z_test::Allocations, prior_k::Distributions.Distribution)
    
    # Prior on the number of emitters
    k = length(θ)
    prior_ratio = pdf(prior_k, k+1) / pdf(prior_k, k)

    # Compute the likelihood ratio
    # Note all allocations are different, so we need to compute the likelihood ratio for each observation
    log_likelihood_ratio = 1.0
    for i in 1:length(obs)
        # Compute the likelihood ratio for this observation
        id = z[i]
        id_test = z_test[i]
        log_likelihood_ratio += log(p_z_given_y(obs[i], θ_test[id_test])) - log(p_z_given_y(obs[i], θ[id]))
    end
    likelihood_ratio = exp(log_likelihood_ratio)

    # When drawing from the prior, the proposal ratio is 1 (for the position of the new emitter)
    proposal_ratio = 1.0

    # Compute the acceptance ratio
    return prior_ratio * likelihood_ratio * proposal_ratio
end

function p_accept_remove(θ::Params, θ_test::Params, obs::Observations, z::Allocations, z_test::Allocations, prior_λ::Distributions.Distribution)
   # Use the same function as for add with the arguments swapped and take the inverse
   return 1/p_accept_add(θ_test, θ, obs, z_test, z, prior_λ)
end


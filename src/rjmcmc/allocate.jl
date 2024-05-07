

# https://juliastats.org/Distributions.jl/stable/convolution/
function convolve_distribution(distribution::UnivariateDistribution, k::Int)
    convolved_distribution = distribution
    for _ in 2:k
        convolved_distribution = convolve(convolved_distribution, distribution)
    end
    return convolved_distribution
end

function build_prior_k(obs::SMLMBaGoL.Observations, prior_λ::Distributions.UnivariateDistribution)
    max_k = length(obs.ŷ)
    N = max_k
    probabilities = zeros(max_k + 1)
    k_vec = 0:max_k
    for k in 0:max_k
        convolved_distribution = convolve_distribution(prior_λ, k)
        probabilities[k+1] = pdf(convolved_distribution, N)
    end
    probabilities ./= sum(probabilities)  # Normalize to make it a valid probability distribution
    return DiscreteNonParametric(k_vec, probabilities)
end


function has_allocation(z::Allocations, id::Int)
    return id in z.idx
end


function allocate!(z::Allocations, obs::Observations, θ::Params)
    k = length(θ)
    log_p = zeros(k)

    for i in 1:length(obs)   
        for j in 1:k
            log_p[j] = log_p_z_given_y(obs.ŷ[i], θ.emitters[j])
        end
        log_p .-= maximum(log_p)
        log_p .= exp.(log_p) .+ eps()
        log_p ./= sum(log_p) 
        z.idx[i] = rand(Categorical(log_p))
    end
end

function allocate(obs::Observations, θ::Params)
    z = Allocations(zeros(Int, length(obs.ŷ)))
    allocate!(z, obs, θ)
    return z
end

function estimate_alpha(k, prior_λ)
    # Estimate the concentration parameter α based on the prior distribution and observed data
    # You can use methods like method of moments, maximum likelihood estimation, or Bayesian inference
    # For example, using the method of moments:
    E_k = mean(prior_λ)
    Var_k = var(prior_λ)
    α = (E_k * (E_k - 1)) / Var_k
    return α
end

function estimate_concentration_params(k_vec, prior_λ)
    N = length(k_vec)
    α_vec = zeros(N)
    for i in 1:N
        α_vec[i] = estimate_alpha(k_vec[i], prior_λ)
    end
    return α_vec
end

function log_dirichlet_multinomial_pmf(n_obs, k_vec, α_vec)
    N = length(k_vec)
    
    # Calculate the log-PMF of the Dirichlet-multinomial distribution
    log_pmf = logfactorial(n_obs) - sum(logfactorial.(k_vec))
    log_pmf += loggamma(sum(α_vec)) - loggamma(n_obs + sum(α_vec))
    for i in 1:N
        log_pmf += loggamma(k_vec[i] + α_vec[i]) - loggamma(α_vec[i])
    end
    
    return log_pmf
end

function log_dirichlet_multinomial_pmf(θ::Params, z::Allocations, prior_λ::Distributions.Distribution)
    n_obs = length(z.idx)
    unique_emitters = unique(z.idx)
    k_vec = [count(==(i), z.idx) for i in unique_emitters]
    α_vec = estimate_concentration_params(k_vec, prior_λ)
    return log_dirichlet_multinomial_pmf(n_obs, k_vec, α_vec)
end


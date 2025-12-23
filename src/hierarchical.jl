# Hierarchical Bayesian updates for BaGoL

"""
Update μ (mean localizations per emitter) via conjugate Gibbs sampling.

Model:
  λ_j ~ Gamma(α, α/μ)     # Per-emitter rate
  n_j ~ Poisson(λ_j)      # Count given rate
  μ ~ Gamma(a, b)         # Hyperprior on mean

With the NegBinomial marginal (integrating out λ_j), the conjugate update is:
  μ | {n_j}, α ~ Gamma(a + Σn_j, b + K*α)

This is equivalent to treating the sum of counts as Gamma-distributed data.
"""
function update_mu_gibbs!(chain::RJMCMCChain)
    config = chain.config
    state = chain.current_state

    # Hyperprior parameters
    a = config.μ_prior_a
    b = config.μ_prior_b
    α = config.α

    # Get counts for each emitter
    counts = [length(emitter.allocated) for emitter in state.emitters]
    k = length(counts)

    if k == 0
        # No emitters - sample from prior
        chain.μ = rand(Gamma(a, 1/b))
        return
    end

    sum_n = sum(counts)

    # Conjugate Gamma posterior
    posterior_shape = a + sum_n
    posterior_rate = b + k * α

    # Sample new μ (using scale parameterization: Gamma(shape, scale=1/rate))
    # Apply minimum to prevent collapse to single-localization emitters
    chain.μ = max(2.0, rand(Gamma(posterior_shape, 1 / posterior_rate)))
end

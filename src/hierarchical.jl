# Hierarchical Bayesian updates for BaGoL

"""
Update μ (mean localizations per emitter) via Gibbs sampling.

Model:
  n_j | μ ~ distribution with E[n_j] = μ
  μ ~ Gamma(a, b)         # Hyperprior on mean

Posterior update (moment-matching approach):
  μ | {n_j} ~ Gamma(a + Σn_j, b + K)

This gives posterior mean ≈ Σn_j / K = sample mean (for weak prior),
which correctly estimates the true mean localizations per emitter.
"""
function update_mu_gibbs!(chain::RJMCMCChain)
    config = chain.config
    state = chain.current_state

    # Hyperprior parameters
    a = config.μ_prior_a
    b = config.μ_prior_b

    # Get counts for each emitter
    counts = [length(emitter.allocated) for emitter in state.emitters]
    k = length(counts)

    if k == 0
        # No emitters - sample from prior
        chain.μ = rand(Gamma(a, 1/b))
        return
    end

    sum_n = sum(counts)

    # Gamma posterior: shape = a + Σn, rate = b + K
    # Posterior mean = (a + Σn) / (b + K) ≈ Σn/K for weak prior
    posterior_shape = a + sum_n
    posterior_rate = b + k

    # Sample new μ (using scale parameterization: Gamma(shape, scale=1/rate))
    # Apply minimum to prevent collapse to single-localization emitters
    chain.μ = max(2.0, rand(Gamma(posterior_shape, 1 / posterior_rate)))
end

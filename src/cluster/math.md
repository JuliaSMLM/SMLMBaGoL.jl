
Notation:
- Let $Y_{ij}$ denote the number of localizations for emitter $j$ in dataset $i$.
- Let $\alpha$ and $\beta$ be the shape and scale parameters of the gamma prior, respectively.
- Let $a_0$, $b_0$, $c_0$, and $d_0$ be the hyperparameters of the gamma hyperpriors for $\alpha$ and $\beta$.

Prior Distributions:
- Gamma prior for the number of localizations per emitter:
  $Y_{ij} \sim \text{Gamma}(\alpha, \beta)$
- Gamma hyperpriors for the shape and scale parameters:
  $\alpha \sim \text{Gamma}(a_0, b_0)$
  $\beta \sim \text{Gamma}(c_0, d_0)$

Likelihood:
- The likelihood of the observed data (number of localizations per emitter) given the gamma prior is:
  $\mathcal{L}(Y|\alpha, \beta) = \prod_{i=1}^{N} \prod_{j=1}^{M_i} \frac{\beta^{\alpha}}{\Gamma(\alpha)} Y_{ij}^{\alpha-1} e^{-\beta Y_{ij}}$
  where $N$ is the number of datasets and $M_i$ is the number of emitters in dataset $i$.

Posterior Distributions:
- The posterior distribution of $\alpha$ given the data and the hyperprior is:
  $p(\alpha|Y) \propto \mathcal{L}(Y|\alpha, \beta) \times p(\alpha)$
  $p(\alpha|Y) \propto \left(\prod_{i=1}^{N} \prod_{j=1}^{M_i} Y_{ij}^{\alpha-1}\right) \times \alpha^{a_0-1} e^{-b_0 \alpha}$
- The posterior distribution of $\beta$ given the data and the hyperprior is:
  $p(\beta|Y) \propto \mathcal{L}(Y|\alpha, \beta) \times p(\beta)$
  $p(\beta|Y) \propto \beta^{N \sum_{i=1}^{N} M_i \alpha - 1} e^{-\beta \left(\sum_{i=1}^{N} \sum_{j=1}^{M_i} Y_{ij} + d_0\right)}$

Updating the Gamma Prior:
1. After running the RJMCMC chains, collect the number of localizations per emitter $Y_{ij}$ from each chain.
2. Update the posterior distributions of $\alpha$ and $\beta$ using the collected data and the gamma hyperpriors.
3. Sample new values for $\alpha$ and $\beta$ from their respective posterior distributions using techniques such as Markov Chain Monte Carlo (MCMC) methods (e.g., Metropolis-Hastings algorithm or Gibbs sampling).
4. Update the gamma prior with the sampled values of $\alpha$ and $\beta$.
5. Use the updated gamma prior for subsequent RJMCMC runs.


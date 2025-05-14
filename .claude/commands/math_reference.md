# Mathematical Reference for SMLMBaGoL

This document provides a comprehensive reference for the mathematical and statistical foundations of the SMLMBaGoL package (Bayesian Grouping of Localizations for Single-Molecule Localization Microscopy).

## Table of Contents

1. [Introduction to SMLM and BaGoL](#introduction-to-smlm-and-bagol)
2. [Statistical Foundation](#statistical-foundation)
3. [Reversible Jump Markov Chain Monte Carlo](#reversible-jump-markov-chain-monte-carlo)
4. [Hierarchical Bayesian Model](#hierarchical-bayesian-model)
5. [Mathematical Algorithms](#mathematical-algorithms)
6. [Physical Model](#physical-model)
7. [References](#references)

## Introduction to SMLM and BaGoL

Single-Molecule Localization Microscopy (SMLM) is a super-resolution imaging technique that relies on the precise localization of individual fluorescent molecules. The precision of these localizations is limited by the number of photons collected from each molecule. By collecting multiple localizations from the same molecule (through blinking or binding events), the precision can be improved.

The Bayesian Grouping of Localizations (BaGoL) algorithm addresses this challenge by:

1. Identifying which localizations likely originate from the same physical emitter
2. Grouping these localizations to obtain a more precise estimate of the emitter position
3. Handling the uncertain number of underlying emitters

The algorithm is based on a Bayesian statistical framework that explicitly models the uncertainty in both the number of emitters and their positions. BaGoL can achieve sub-nanometer precision under dense labeling conditions and is compatible with both dSTORM and DNA-PAINT data.

## Statistical Foundation

### Model Variables

- $Y = \{y_1, y_2, \ldots, y_N\}$: Set of observed localizations
- $\Sigma = \{\sigma_1, \sigma_2, \ldots, \sigma_N\}$: Set of localization uncertainties
- $T = \{t_1, t_2, \ldots, t_N\}$: Set of time stamps
- $K$: Number of emitters
- $\theta_K$: Emitter positions and movements
- $Z$: Allocation of localizations to emitters (assignments)

Each localization $y_i$ is characterized by:
- Position coordinates $(x_i, y_i)$
- Uncertainty estimates $(\sigma_{x,i}, \sigma_{y,i})$
- Frame number $t_i$

### Bayesian Framework

The posterior probability distribution according to Fazel et al. is:

$$\pi(K, (\theta_K, Z)|Y, \Sigma, T) \propto P(Y, \Sigma, T|K, Z, \theta) \cdot P(K) \cdot P(Z|K) \cdot P(\theta_K|K, Z)$$

where:
- $P(Y, \Sigma, T|K, Z, \theta)$ is the likelihood of observing localizations $Y$ with uncertainties $\Sigma$ at times $T$
- $P(Z|K)$ is the prior on allocations
- $P(\theta_K|K, Z)$ is the prior on emitter positions
- $P(K)$ is the prior on the number of emitters

### Likelihood Function

The likelihood function models the probability of observing the localizations given the model parameters:

$$P(Y, \Sigma, T|K, Z, \theta) = \prod_{i=1}^{N} P(y_i|Z(i), \theta_K)$$

For a static emitter model, the likelihood of observing localization $y_i$ given it is associated with emitter $\theta_j$ is modeled as a Gaussian:

$$P(y_i|\theta_j, Z(i)=j) = \frac{1}{2\pi\sigma_{x,i}\sigma_{y,i}}\exp\left(-\frac{(x_i-x_j)^2}{2\sigma_{x,i}^2}-\frac{(y_i-y_j)^2}{2\sigma_{y,i}^2}\right)$$

For computational efficiency, this is often calculated in log space:

$$\log P(y_i|\theta_j, Z(i)=j) = \log\mathcal{N}(x_i | x_j, \sigma_{x,i}^2) + \log\mathcal{N}(y_i | y_j, \sigma_{y,i}^2)$$

### Prior Distributions

#### Prior on Number of Emitters $P(K)$

The prior on the number of emitters $K$ can be modeled using a Poisson or Gamma distribution:

$$P(K) \propto \frac{(\lambda\gamma)^K \exp(-\lambda\gamma)}{\Gamma(K+1)}$$

where:
- $\lambda$ is the mean number of localizations per emitter
- $\gamma$ is a scale parameter

#### Prior on Emitter Positions $P(\theta_K|K, Z)$

The prior on emitter positions is constructed as a sum of Gaussians centered on each localization, using their respective uncertainties:

$$P(\theta_j) \propto \sum_{i=1}^{N} \mathcal{N}(\theta_j | y_i, \Sigma_i)$$

where $\Sigma_i$ is the covariance matrix associated with localization $y_i$. This informative prior effectively creates a spatial probability distribution that follows the density of the observed localizations, weighted by their precision.

#### Prior on Allocations $P(Z|K)$

The prior on allocations is often modeled as a multinomial distribution with equal probabilities:

$$P(Z|K) \propto \left(\frac{1}{K}\right)^N$$

## Reversible Jump Markov Chain Monte Carlo

The RJMCMC algorithm, introduced by Green (1995), explores the posterior distribution over the variable-dimension parameter space. This is essential for BaGoL since the number of emitters is unknown and must be estimated along with their positions and localization assignments.

### Detailed Balance

The RJMCMC algorithm satisfies the detailed balance condition, which ensures that the Markov chain converges to the target posterior distribution. For any two states $(\theta, Z, K)$ and $(\theta', Z', K')$, detailed balance requires:

$$\pi(\theta, Z, K)P((\theta, Z, K) \to (\theta', Z', K')) = \pi(\theta', Z', K')P((\theta', Z', K') \to (\theta, Z, K))$$

where $\pi(\theta, Z, K)$ is the target distribution and $P((\theta, Z, K) \to (\theta', Z', K'))$ is the transition probability from state $(\theta, Z, K)$ to $(\theta', Z', K')$.

In RJMCMC, when the dimension of the parameter space changes (e.g., adding or removing an emitter), the detailed balance condition becomes more complex due to the dimension-matching requirement. This is handled by defining complementary pairs of moves (e.g., birth-death, split-merge) that can reverse each other's action.

The algorithm employs several types of moves (jumps) to sample from the posterior distribution:

### Move Types

1. **Move**: Updates emitter positions using Gibbs sampling based on their allocated localizations
2. **Allocate**: Reassigns localizations to emitters
3. **Birth/Add**: Adds a new emitter to the model
4. **Death/Remove**: Removes an existing emitter from the model

The original implementation by Fazel et al. uses these four fundamental moves, while SMLMBaGoL may include additional moves like split/merge for improved mixing.

### Implementation Details

Typical RJMCMC parameters include:
- 1000 burn-in jumps, followed by 2000 post-burn-in jumps
- Equal jump probabilities for move, allocate, birth, and death (e.g., 0.25 each)

### Birth-Death Process

The birth-death process modifies the dimensionality of the parameter space by adding or removing emitters.

#### Birth Jump
For a birth jump that adds a new emitter:
1. Sample a new emitter position from the prior
2. Update allocations to include the new emitter
3. Calculate the acceptance probability:

$$\alpha_{birth} = \min\left(1, \frac{P(Y|\theta', Z', K+1) \cdot P(Z'|K+1) \cdot P(\theta'|K+1, Z') \cdot P(K+1)}{P(Y|\theta, Z, K) \cdot P(Z|K) \cdot P(\theta|K, Z) \cdot P(K)} \cdot \frac{q(\theta, Z, K|\theta', Z', K+1)}{q(\theta', Z', K+1|\theta, Z, K)}\right)$$

where $q$ represents the proposal distribution.

#### Death Jump
For a death jump that removes an emitter:
1. Select an emitter to remove
2. Reallocate its localizations to other emitters
3. Calculate the acceptance probability (inverse of birth jump)

### Acceptance Criteria and Jump Pairs

The RJMCMC method maintains detailed balance through carefully designed jump pairs that can reverse each other's actions. Each pair of jumps (like birth-death or split-merge) shares mathematical structures that ensure reversibility.

The general form of the acceptance probability for a proposed move from state $(\theta, Z, K)$ to state $(\theta', Z', K')$ is:

$$\alpha = \min\left(1, \frac{\pi(\theta', Z', K')}{\pi(\theta, Z, K)} \cdot \frac{q(\theta, Z, K|\theta', Z', K')}{q(\theta', Z', K'|\theta, Z, K)} \cdot |J|\right)$$

where:
- $\pi(\cdot)$ is the posterior distribution
- $q(\cdot|\cdot)$ is the proposal distribution
- $|J|$ is the Jacobian determinant of the transformation

For dimension-changing moves like birth-death pairs, the same mathematical framework calculates both acceptance probabilities. For example, the death move acceptance probability is derived from the birth acceptance probability by swapping the states and inverting the ratio:

$$\alpha_{death} = \min\left(1, \frac{\pi(\theta, Z, K)}{\pi(\theta', Z', K+1)} \cdot \frac{q(\theta', Z', K+1|\theta, Z, K)}{q(\theta, Z, K|\theta', Z', K+1)} \cdot |J|^{-1}\right)$$

This symmetry in the mathematics ensures that detailed balance is maintained even as the dimension of the parameter space changes. The designed reversibility of jumps allows the Markov chain to freely explore models with different numbers of parameters while eventually converging to the correct posterior distribution.

## Hierarchical Bayesian Model

SMLMBaGoL implements a hierarchical Bayesian approach to handle uncertainty in the distribution of localizations per emitter. This approach is integrated with the cluster-based analysis through periodic updates to the hyperparameters.

### Notation

- $Y_{ij}$: Number of localizations for emitter $j$ in dataset $i$
- $\alpha, \beta$: Shape and scale parameters of the gamma prior
- $a_0, b_0, c_0, d_0$: Hyperparameters of the gamma hyperpriors

### Prior Distributions

- Gamma prior for the number of localizations per emitter:
  $$Y_{ij} \sim \text{Gamma}(\alpha, \beta)$$
  
- Gamma hyperpriors for the shape and scale parameters:
  $$\alpha \sim \text{Gamma}(a_0, b_0)$$
  $$\beta \sim \text{Gamma}(c_0, d_0)$$

### Likelihood

The likelihood of the observed data given the gamma prior is:

$$\mathcal{L}(Y|\alpha, \beta) = \prod_{i=1}^{N} \prod_{j=1}^{M_i} \frac{\beta^{\alpha}}{\Gamma(\alpha)} Y_{ij}^{\alpha-1} e^{-\beta Y_{ij}}$$

where $N$ is the number of datasets and $M_i$ is the number of emitters in dataset $i$.

### Posterior Distributions

- Posterior distribution of $\alpha$:
  $$p(\alpha|Y) \propto \left(\prod_{i=1}^{N} \prod_{j=1}^{M_i} Y_{ij}^{\alpha-1}\right) \times \alpha^{a_0-1} e^{-b_0 \alpha}$$

- Posterior distribution of $\beta$:
  $$p(\beta|Y) \propto \beta^{N \sum_{i=1}^{N} M_i \alpha - 1} e^{-\beta \left(\sum_{i=1}^{N} \sum_{j=1}^{M_i} Y_{ij} + d_0\right)}$$

### Integrated Analysis and Hierarchical Updates

SMLMBaGoL integrates the RJMCMC analysis of clusters with periodic hierarchical Bayesian updates in a coordinated workflow:

1. **Initial Setup**:
   - Data is divided into disconnected clusters using DBSCAN
   - Initial hyperparameters $\alpha$ and $\beta$ are set based on prior knowledge or default values
   - Each cluster is assigned an independent RJMCMC chain with the same prior distributions

2. **Interleaved RJMCMC and Hierarchical Updates**:
   - Run RJMCMC chains on all clusters for a fixed number of iterations (`nsamples`)
   - Collect the current states from all chains across all clusters
   - Extract the number of localizations per emitter from these states
   - Update the posterior distributions of $\alpha$ and $\beta$ using the collected data
   - Sample new values of $\alpha$ and $\beta$ using a Metropolis-Hastings step
   - Update the gamma prior $\Gamma(\alpha, \beta)$ in all chains with these new parameter values
   - Continue RJMCMC sampling on all clusters with the updated prior
   - Repeat this cycle for a specified number of hierarchical updates

3. **Computational Advantages**:
   - The RJMCMC chains for different clusters run in parallel between hierarchical updates
   - Hierarchical updates use information pooled from all clusters
   - This approach allows information about the blinking statistics to propagate across clusters
   - The parallel nature of the algorithm maintains computational efficiency

4. **Mathematical Formulation**:
   - At hierarchical update step $t$, collect allocation counts $Y^{(t)} = \{Y_{ij}^{(t)}\}$ from all clusters
   - Update the posterior distribution of hyperparameters:
     $$p(\alpha, \beta | Y^{(1:t)}) \propto p(Y^{(1:t)} | \alpha, \beta) \cdot p(\alpha) \cdot p(\beta)$$
   - Sample new hyperparameters $(\alpha^{(t+1)}, \beta^{(t+1)})$ from this posterior
   - Update the prior on localizations per emitter for the next RJMCMC iterations:
     $$p(Y | \alpha^{(t+1)}, \beta^{(t+1)}) = \Gamma(Y | \alpha^{(t+1)}, \beta^{(t+1)})$$

This interleaved approach allows the algorithm to adaptively learn the distribution of localizations per emitter from the data while maintaining the computational efficiency of parallel processing.

## Mathematical Algorithms

### Pre-clustering and Subregion Analysis

The BaGoL algorithm employs pre-clustering strategies to manage computational complexity and enable parallelization. The core RJMCMC algorithm scales as $O(N^2)$ with the number of localizations, making it computationally expensive for large datasets.

#### DBSCAN Clustering

SMLMBaGoL uses Density-Based Spatial Clustering of Applications with Noise (DBSCAN) to break the problem into disconnected clusters:

1. Localizations are treated as points in a spatial graph
2. Points are connected if they are within distance $\epsilon$ of each other, where $\epsilon$ is typically set to a multiple of the mean localization uncertainty (e.g., $\epsilon = 4\sigma$)
3. Clusters are formed as connected components in this graph
4. Each cluster is processed independently using the RJMCMC algorithm

The mathematical justification for this approach is that localizations separated by large distances (relative to their uncertainties) have negligible probability of originating from the same emitter.

The clustering approach transforms the computational complexity from $O(N^2)$ for the entire dataset to $O(\sum_{i=1}^k n_i^2)$, where $k$ is the number of clusters and $n_i$ is the number of localizations in cluster $i$. When clusters are roughly equal in size with $n_i \approx N/k$, this reduces to $O(N^2/k)$, providing a significant speedup.

#### Parallel Processing

After dividing the data into clusters using DBSCAN, SMLMBaGoL processes each cluster independently:

```julia
Threads.@threads for i in eachindex(subregions)
    subregion = subregions[i]
    chain, mapn_coords = rjmcmc(subregion.obs, prior_λ)
    subregion.chains[1] = chain
end
```

This parallel processing approach enables effective utilization of multiple CPU cores, significantly accelerating the analysis of large datasets.

#### Combining Results

To create the final posterior distribution, results from all clusters or subregions are merged by adding their contributions to a discretized posterior image:

$$P(\theta) = \sum_{i=1}^k P_i(\theta)$$

where $P_i(\theta)$ is the posterior contribution from cluster $i$, normalized appropriately.

### Allocation Algorithm

The allocation of localizations to emitters follows:

1. For each localization $y_i$:
   - Calculate the log-likelihood of assignment to each emitter $\theta_j$:
     $$\log P(Z(i) = j | y_i, \theta_j) = \log P(y_i | \theta_j, Z(i) = j) + \log P(Z(i) = j | \theta_j)$$
   - Normalize these probabilities: 
     $$P(Z(i) = j | y_i, \theta) = \frac{\exp(\log P(Z(i) = j | y_i, \theta_j))}{\sum_{k=1}^{K} \exp(\log P(Z(i) = k | y_i, \theta_k))}$$
   - Sample the allocation from this categorical distribution

### Emitter Position Update

When updating emitter positions:

1. For each emitter $\theta_j$:
   - Calculate the weighted mean of its allocated localizations:
     $$\hat{x}_j = \frac{\sum_{i: Z(i)=j} \frac{x_i}{\sigma_{x,i}^2}}{\sum_{i: Z(i)=j} \frac{1}{\sigma_{x,i}^2}}$$
     $$\hat{y}_j = \frac{\sum_{i: Z(i)=j} \frac{y_i}{\sigma_{y,i}^2}}{\sum_{i: Z(i)=j} \frac{1}{\sigma_{y,i}^2}}$$
   - Calculate the posterior variance:
     $$\sigma_{\hat{x}_j}^2 = \left(\sum_{i: Z(i)=j} \frac{1}{\sigma_{x,i}^2}\right)^{-1}$$
     $$\sigma_{\hat{y}_j}^2 = \left(\sum_{i: Z(i)=j} \frac{1}{\sigma_{y,i}^2}\right)^{-1}$$
   - Sample the new position from:
     $$x_j \sim \mathcal{N}(\hat{x}_j, \sigma_{\hat{x}_j}^2)$$
     $$y_j \sim \mathcal{N}(\hat{y}_j, \sigma_{\hat{y}_j}^2)$$

### Posterior Image Construction

The posterior distribution of emitter positions is constructed as:

1. Create a discretized grid with pixel size typically smaller than localization precision
2. For each state in the MCMC chain:
   - Add a count to each pixel where an emitter is located
3. Normalize the resulting image to sum to 1.0

### MAPN Generation: Addressing Particle Identity

A key challenge in RJMCMC analysis is that particle identity is not preserved across the chain. Since emitters can be added, removed, or reallocated between iterations, tracking individual emitters becomes problematic. SMLMBaGoL addresses this challenge through a sophisticated Maximum A Posteriori Number (MAPN) emitter estimation approach:

1. **Finding the MAPN**:
   - Determine the most frequently occurring number of emitters across the chain states
   - Extract a sub-chain containing only states with this number of emitters

2. **Reference State Selection**:
   - Create a spatial histogram from all emitter positions in the MAPN sub-chain
   - Find the state with emitter configurations that best match this spatial distribution
   - Use this state as an initial reference

3. **Emitter Reordering with Hungarian Algorithm**:
   - For each state in the MAPN sub-chain:
     - Construct a cost matrix containing pairwise distances between emitters in the reference state and the current state
     - Apply the Hungarian algorithm to find the optimal assignment that minimizes total distance
     - Permute the emitters in the current state to match the reference ordering

4. **Iterative Refinement**:
   - Calculate mean positions for each emitter across all reordered states
   - Use these mean positions as a new reference state
   - Repeat the Hungarian assignment step for further refinement (typically 2 iterations)

5. **Statistical Estimation**:
   - For each emitter in the final ordered chain:
     - Calculate the mean position across all states
     - Compute the standard deviation as a measure of positional uncertainty

This approach effectively addresses the particle identity issue in RJMCMC by leveraging optimal assignment algorithms and statistical averaging. The mathematical basis lies in the bipartite matching problem, which the Hungarian algorithm solves in $O(n^3)$ time, where $n$ is the number of emitters.

The algorithm can be summarized by the following pseudo-equation for the position of emitter $j$:

$$\hat{\theta}_j = \frac{1}{|C_{MAPN}|} \sum_{i \in C_{MAPN}} \theta_{\sigma_i(j)}^{(i)}$$

where:
- $\hat{\theta}_j$ is the estimated position of emitter $j$
- $C_{MAPN}$ is the set of chain states with the MAPN number of emitters
- $\theta_{\sigma_i(j)}^{(i)}$ is the position of the emitter in state $i$ assigned to index $j$ by permutation $\sigma_i$
- $\sigma_i$ is the optimal assignment found by the Hungarian algorithm

## Physical Model

### Localization Precision

The localization precision in SMLM is related to the number of photons $N$ detected from the emitter:

$$\sigma_x \approx \frac{\sigma_{PSF}}{\sqrt{N}}$$

where $\sigma_{PSF}$ is the standard deviation of the point spread function.

### Precision Improvement

The expected precision improvement when combining $n$ localizations is:

$$\sigma_{combined} \approx \frac{\sigma_{individual}}{\sqrt{n}}$$

This relationship forms the theoretical basis for the precision improvement achieved by BaGoL.

### Blinking Process

The blinking behavior of fluorophores is modeled stochastically:
- The number of blinks per emitter follows a distribution (often geometric or gamma)
- The temporal distribution of blinks depends on the photophysics of the fluorophore
- For DNA-PAINT, the binding kinetics determine the temporal distribution of localizations

## References

1. Fazel M, Wester MJ, Rieger B, Jungmann R, Lidke KA. High-precision estimation of emitter positions using Bayesian grouping of localizations. Nature Communications. 2022;13(1):7152. [https://doi.org/10.1038/s41467-022-34894-2](https://doi.org/10.1038/s41467-022-34894-2)

2. Green PJ. Reversible jump Markov chain Monte Carlo computation and Bayesian model determination. Biometrika. 1995;82(4):711-732.

3. Richardson S, Green PJ. On Bayesian analysis of mixtures with an unknown number of components. Journal of the Royal Statistical Society, Series B. 1997;59(4):731-792.
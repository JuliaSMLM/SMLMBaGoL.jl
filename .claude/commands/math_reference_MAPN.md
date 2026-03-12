# MAP-N Estimation: Problem Analysis and Alternatives

---

## 1. Problem Statement

After running the collapsed Gibbs sampler on a partition, we must extract a point
estimate: how many emitters $K$, and where are they? The current approach — **global
MAP-N** — has a systematic bias that worsens with partition size.

### 1.1 Current Approach (Global MAP-N)

1. Run collapsed Gibbs for $T$ post-burn-in iterations, recording $K^{(t)}$ at each step
2. Build histogram of $K$ values; find the mode: $\hat{K} = \arg\max_k \sum_t \mathbf{1}[K^{(t)} = k]$
3. Filter to samples with $K^{(t)} = \hat{K}$
4. Iterative Hungarian matching across filtered samples to solve label switching
5. Median positions per matched emitter (robust to outliers)

### 1.2 The Bias Problem

Consider a partition with $K_\text{true}$ well-separated emitters. Each emitter $j$
has some small probability $p_j$ of being transiently split into two clusters on any
given chain iteration (e.g., a localization wanders to form its own singleton cluster
during a Gibbs sweep).

For well-separated emitters, $p_j$ is small but nonzero. At each iteration, the
observed $K^{(t)}$ is:

$$K^{(t)} = K_\text{true} + \sum_{j=1}^{K_\text{true}} S_j^{(t)}$$

where $S_j^{(t)} \in \{0, 1, \ldots\}$ counts extra clusters from emitter $j$ (usually
0, occasionally 1). Taking expectations:

$$\mathbb{E}[K^{(t)}] = K_\text{true} + \sum_j p_j \approx K_\text{true} + K_\text{true} \bar{p}$$

where $\bar{p}$ is the mean per-emitter split probability. The K histogram mode is
pulled above $K_\text{true}$ by approximately $K_\text{true} \bar{p}$.

**Key insight:** The bias scales linearly with partition size. A partition with 10
emitters and $\bar{p} = 0.05$ has bias $\approx 0.5$ (negligible). A partition with 100
emitters has bias $\approx 5$ (significant).

### 1.3 Cascading Damage from Wrong K

The damage from overestimating $K$ extends far beyond "a few extra emitters":

1. **Hungarian matching degrades.** With $\hat{K} > K_\text{true}$, the matching must
   place phantom emitters across samples. These phantoms have random, inconsistent
   positions across iterations, polluting the assignment problem.

2. **Label switching worsens.** The iterative Hungarian matching uses a reference
   sample and matches subsequent samples to it. With phantom emitters in the reference,
   real emitters can be mismatched, corrupting median positions even for correctly
   identified emitters.

3. **Median positions are biased.** When a real emitter is sometimes matched to one
   phantom and sometimes to another across iterations, its median position is computed
   from a mixture of correct and incorrect matches.

4. **Posterior covariances are inflated.** The spread from mismatching inflates the
   apparent posterior uncertainty, degrading the Rao-Blackwellized posterior image.

The net effect: overestimating $K$ by $\delta$ degrades the quality of *all*
$K_\text{true}$ emitters, not just the $\delta$ extra ones.

### 1.4 The Fundamental Issue

Global MAP-N treats $K$ as a single partition-level quantity. But in a partition with
many well-separated emitters, the $K$ posterior is broad — its standard deviation
grows as $\sqrt{K_\text{true}}$ or faster. For large partitions, the K histogram
becomes nearly flat near the mode, making MAP-N unreliable even in principle.

The problem is that a *global* question ("how many emitters total?") is being used to
answer what are really *local* questions ("do these specific localizations belong to
the same emitter?"). One ambiguous emitter pair — where the chain fluctuates between
1 and 2 clusters — shifts the entire K histogram, affecting the extraction of all
other emitters in the partition.

### 1.5 Desired Properties of a Replacement

A better extraction method should:

- **Be local:** Each emitter's membership is determined from its own localizations'
  chain behavior, independent of ambiguity elsewhere in the partition
- **Not require a global K estimate:** The number of emitters should emerge from local
  decisions, not be imposed top-down
- **Preserve posterior covariances:** The ClusterStats analytical posteriors are
  well-calibrated (scale factor $c \approx 1.08$, validated via Mahalanobis $\chi^2(2)$
  test); the replacement should use them
- **Handle the easy case trivially:** For well-separated emitters, localizations that
  are always co-assigned should be grouped with certainty
- **Degrade gracefully for ambiguous cases:** Where the chain genuinely fluctuates
  between 1 and 2 emitters, report the uncertainty rather than forcing a hard decision

---

## 2. Posterior Similarity Matrix (PSM)

The co-occupancy matrix we proposed is known in Bayesian nonparametrics as the
**Posterior Similarity Matrix** (PSM). It is the standard tool for summarizing MCMC
partition output, bypassing label switching entirely.

$$C_{ij} = P(z_i = z_j \mid \text{data}) \approx \frac{1}{T} \sum_{t=1}^{T} \mathbf{1}[z_i^{(t)} = z_j^{(t)}]$$

### 2.1 Properties

- $C_{ii} = 1$ always
- $C_{ij} \approx 1$ if locs $i, j$ are from the same well-identified emitter
- $C_{ij} \approx 0$ if locs $i, j$ are from different, well-separated emitters
- $C_{ij} \in (0.3, 0.7)$ indicates genuine posterior ambiguity
- $C$ is symmetric and positive semidefinite
- $C$ is invariant to cluster label permutations — no label switching problem

### 2.2 Accumulation

Accumulate online during the chain. Each iteration, for each cluster with members
$S_k$, increment counts for all pairs in $S_k$:

```
for each cluster k with members S_k:
    for (i, j) in combinations(S_k, 2):
        counts[i,j] += 1
```

Cost per iteration: $O(\sum_k n_k^2)$ where $n_k$ are cluster sizes. For our typical
partitions (N=50–500 locs, K=5–50 emitters), this is much less than $N^2$.

**Storage:** Dense $N \times N$ is fine for our scale. N=500 → 2 MB (Float64) or
0.5 MB (UInt32 counts, upper triangle). No need to optimize.

---

## 3. Loss Functions for Partition Estimation

Extracting a point estimate from MCMC partition samples is a **decision theory**
problem: choose a loss function $L(\hat{c}, c)$, then find the partition $\hat{c}$
minimizing the posterior expected loss.

### 3.1 Binder's Loss (Binder 1978)

Counts pairwise co-clustering disagreements:

$$L_B(\hat{c}, c) = \sum_{i < j} \left[ a \cdot \mathbf{1}[\hat{c}_i = \hat{c}_j] \cdot \mathbf{1}[c_i \neq c_j] + (2-a) \cdot \mathbf{1}[\hat{c}_i \neq \hat{c}_j] \cdot \mathbf{1}[c_i = c_j] \right]$$

where $a \in [0, 2]$ controls the asymmetry:
- $a = 1$: symmetric (equal cost for false splits and false merges)
- $a > 1$: penalizes false splits more → produces fewer, larger clusters
- $a < 1$: penalizes false merges more → produces more, smaller clusters

**Posterior expected Binder loss** decomposes through the PSM:

$$\hat{c}^* = \arg\min_{\hat{c}} \sum_{i<j} \left( \mathbf{1}[\hat{c}_i = \hat{c}_j] - C_{ij} \right)^2$$

This is the squared Frobenius distance between $\hat{c}$'s association matrix and
the PSM. The co-clustering threshold for pair $(i,j)$ is $C_{ij} > (2-a)/(2)$,
which equals 0.5 for symmetric loss.

**Critical flaw for SMLM:** Binder's loss dramatically overestimates $K$ as $N$ grows.
Wade & Ghahramani (2018) showed that for 4 true clusters, Binder estimates $K = 9$
at $N = 200$ and $K = 41$ at $N = 1600$. The mechanism: uncertain boundary points
(intermediate $C_{ij}$) get assigned to their own small clusters rather than merged
into a main cluster. This is our MAP-N problem in a different guise.

### 3.2 Variation of Information (Meila 2007)

An information-theoretic metric on partitions:

$$\text{VI}(c, \hat{c}) = H(c) + H(\hat{c}) - 2\,I(c, \hat{c})$$

where $H$ is entropy and $I$ is mutual information, computed from the contingency
table $n_{gh} = |C_g \cap \hat{C}_h|$:

$$H(c) = -\sum_g \frac{n_{g+}}{N} \log \frac{n_{g+}}{N}, \qquad H(c, \hat{c}) = -\sum_{g,h} \frac{n_{gh}}{N} \log \frac{n_{gh}}{N}$$

$$\text{VI}(c, \hat{c}) = 2\,H(c, \hat{c}) - H(c) - H(\hat{c})$$

Equivalently: $\text{VI} = H(c \mid \hat{c}) + H(\hat{c} \mid c)$ — the information
lost plus the information gained when moving between partitions.

**Why VI is better than Binder for our problem:**
- VI is a proper metric on the partition lattice
- VI is symmetric around $K = \sqrt{N}$ clusters; Binder is biased toward finer
  partitions at all $K$ (Property 3.6 of Wade & Ghahramani)
- VI does not overestimate $K$ as $N$ grows: in simulations with 4 true clusters,
  VI correctly finds $K = 4$ for all sample sizes $N = 200$ to $1600$
- VI-optimal $\hat{K}$ is not necessarily the posterior mode of $K$ — it minimizes
  the expected loss over the full posterior, which is more principled

**Jensen lower bound (practical computation):** The exact posterior expected VI
requires $O(M \cdot N^2)$ per candidate. A Jensen inequality lower bound reduces
this to $O(N^2)$ per candidate using only the PSM:

$$\hat{c}^* \approx \arg\min_{\hat{c}} \left\{ \sum_n \log\!\left(\sum_{n'} \mathbf{1}[\hat{c}_{n'} = \hat{c}_n]\right) - 2\sum_n \log\!\left(\sum_{n'} C_{nn'} \cdot \mathbf{1}[\hat{c}_{n'} = \hat{c}_n]\right) \right\}$$

### 3.3 PEAR: Posterior Expected Adjusted Rand (Fritsch & Ickstadt 2009)

Maximizes the posterior expected adjusted Rand index. The chance correction in ARI
penalizes partitions with many small clusters, providing **shrinkage toward fewer
clusters** compared to Binder. Requires either full MCMC samples (exact) or a
first-order approximation via the PSM.

### 3.4 Comparison Summary

| Loss function | Tends to... | K estimation | PSM-only? |
|---|---|---|---|
| Binder (a=1) | Over-split | Overestimates, worsens with N | Yes |
| Binder (a>1) | Fewer clusters | Tunable via a | Yes |
| PEAR | Shrink toward fewer | Better than Binder | Approx. yes |
| **VI** | **Correct K** | **Best across all N** | **Lower bound yes** |

**Recommendation for SMLMBaGoL: Use VI loss.** It directly addresses our problem
(K overestimation in large partitions) and has the strongest theoretical and
empirical support.

---

## 4. Search Algorithms

The space of all partitions has Bell number cardinality — finding the optimal
partition is NP-hard in general. Three practical approaches exist.

### 4.1 Dahl's Method (Dahl 2006)

Select the MCMC sample whose association matrix is closest to the PSM:

$$\hat{c}_\text{Dahl} = \arg\min_{c^{(t)}} \sum_{i,j} \left( \mathbf{1}[c_i^{(t)} = c_j^{(t)}] - C_{ij} \right)^2$$

**Pros:** Trivial to implement, guarantees a partition the chain actually visited.
**Cons:** Restricted to visited partitions. For large $N$, the optimal partition may
never appear in the chain. Equivalent to Binder loss (a=1) restricted to MCMC draws,
so inherits Binder's over-splitting tendency.

**Complexity:** $O(T \cdot N^2)$ — evaluate each of $T$ samples against the PSM.

### 4.2 Hierarchical Clustering on 1 - C

Use $1 - C$ as a distance matrix with average or complete linkage. Cut the dendrogram
where linkage distance crosses 0.5. Can find partitions not visited by the chain.

**Pros:** Simple, fast, explores beyond MCMC draws.
**Cons:** Ad hoc — not decision-theoretic. Fritsch & Ickstadt note it "performed
quite poorly" in some settings (Medvedovic et al. 2004). Single linkage especially
prone to chaining artifacts.

### 4.3 Greedy Search (Rastelli & Friel 2018; Wade & Ghahramani 2018)

Start from a partition (e.g., an MCMC sample), then iteratively reassign each
observation to the cluster (or a new singleton) that most reduces the expected loss.

**Algorithm (Rastelli & Friel 2018):**
```
1. Initialize partition a (e.g., from an MCMC sample with K near K_up)
2. Repeat until no change in a full sweep:
   a. For each loc i (random order):
      - For each possible cluster s = 1, ..., K_up:
        Compute Δψ = change in expected loss from moving i to cluster s
      - Move i to argmin_s Δψ
3. Return a
```

**Key efficiency trick:** Moving observation $i$ from cluster $r$ to cluster $s$
changes only two rows/columns of the contingency table. The per-sample change
$\Delta L^{(t)}$ is computable in $O(1)$, so evaluating $\Delta\psi$ for one
$(i, s)$ pair costs $O(T)$.

**Complexity per sweep:** $O(T \cdot N \cdot K_\text{up})$ — note this is
$O(N \cdot K)$, **not** $O(N^2)$, which is important for large partitions.

**Multiple restarts** with different initializations are needed to escape local
optima (success rates 65–100% in Rastelli & Friel's experiments).

**Practical timing (from Rastelli & Friel):** For $N = 435$, greedy takes ~70 sec
vs. 30 hours for MCMC. The optimization is always negligible vs. sampling time.

### 4.4 Recommended Approach for SMLMBaGoL

**Phase 1 (simple):** Dahl's method. Accumulate PSM online, then scan stored
assignment samples for the closest match. Trivial to implement as a new accumulator.

**Phase 2 (if needed):** Greedy VI search. Start from Dahl's partition, then
run greedy reassignment sweeps under VI loss. More complex but avoids Binder's
over-splitting.

---

## 5. Posterior Positions and Covariances

### 5.1 Conditional ClusterStats (Recommended)

Given the final hard partition $\hat{c}$:

1. For each cluster $k$ in $\hat{c}$, instantiate a fresh `ClusterStats`
2. `add_loc` all localizations assigned to cluster $k$
3. Call `posterior_mean(cs)` and `posterior_cov(cs)`

This gives exact analytical posteriors **conditional on the assignment**. Since the
ClusterStats posteriors are well-calibrated (scale factor $c \approx 1.08$, validated
via Mahalanobis $\chi^2(2)$ test), this preserves calibration.

### 5.2 Why Not Incorporate Assignment Uncertainty?

The law of total variance gives the unconditional covariance:

$$\Sigma_\text{total} = \mathbb{E}[\Sigma_j^{(t)}] + \text{Var}(\mu_j^{(t)})$$

But this requires stable cluster correspondence across iterations (matching by
localization overlap, not Hungarian position matching). For ambiguous clusters
($C_{ij} \approx 0.5$), the $\text{Var}(\mu_j)$ term dominates and produces
unnaturally blurred estimates.

**Recommendation:** Use conditional ClusterStats for the hard partition. Report
ambiguous regions (low stability scores) separately rather than inflating covariances.

### 5.3 Stability Scores

For each emitter (cluster) in the final partition, compute a stability score:

$$s_k = \frac{1}{|S_k|^2} \sum_{i,j \in S_k} C_{ij}$$

where $S_k$ is the set of localizations in cluster $k$. This is the mean pairwise
co-occupancy within the cluster:
- $s_k \approx 1$: cluster is stable across the chain (high confidence)
- $s_k < 0.8$: cluster contains localizations that sometimes separate (flag as ambiguous)

---

## 6. Literature

### 6.1 Core References

**Binder, D. A. (1978).** "Bayesian cluster analysis." *Biometrika* 65, 31–38.
Introduced the pairwise loss function for partition estimation.

**Dahl, D. B. (2006).** "Model-Based Clustering for Expression Data via a Dirichlet
Process Mixture Model." In *Bayesian Inference for Gene Expression and Proteomics*,
Cambridge University Press, pp. 201–218. DOI: 10.1017/CBO9780511584589.011.
Introduced the least-squares partition estimate (select MCMC sample closest to PSM).
Equivalent to Binder loss restricted to visited partitions.

**Fritsch, A. and Ickstadt, K. (2009).** "Improved criteria for clustering based on
the posterior similarity matrix." *Bayesian Analysis* 4(2), 367–391.
DOI: 10.1214/09-BA414.
Introduced PEAR (posterior expected adjusted Rand), showed Binder over-splits,
proposed hierarchical clustering on 1-PSM and asymmetric Binder loss.

**Wade, S. and Ghahramani, Z. (2018).** "Bayesian Cluster Analysis: Point Estimation
and Credible Balls." *Bayesian Analysis* 13(2), 559–626. DOI: 10.1214/17-BA1073.
The definitive paper. Proved VI loss gives correct $K$ while Binder overestimates.
Introduced credible balls for partition uncertainty. Showed MAP-K and MAP partition
are both poor estimators. Proposed greedy lattice search.

**Rastelli, R. and Friel, N. (2018).** "Optimal Bayesian estimators for latent variable
cluster models." *Statistics and Computing* 28, 1169–1186.
DOI: 10.1007/s11222-017-9786-y.
Efficient greedy algorithm for any loss function. $O(T \cdot N \cdot K)$ per sweep.
Confirmed VI gives best $K$ estimation; Binder overestimates dramatically.

### 6.2 Additional References

**Dahl, D. B., Johnson, D. J., and Mueller, P. (2022).** "Search Algorithms and Loss
Functions for Bayesian Clustering." *JCGS* 31(4). DOI: 10.1080/10618600.2022.2069779.
SALSO algorithm: embarrassingly parallel greedy search supporting generalized VI and
asymmetric Binder. R package `salso`.

**Meila, M. (2007).** "Comparing Clusterings — an Information Based Distance."
*Journal of Multivariate Analysis* 98(5), 873–895.
Introduced Variation of Information as a metric on partitions.

**Miller, J. W. and Harrison, M. T. (2014).** "Inconsistency of Pitman-Yor and
Normalized Inverse-Gaussian Process Mixture Models for the Number of Components."
*JMLR* 15, 3333–3370.
Showed that DP/PY posterior on $K$ is inconsistent (overestimates). Motivates
partition-level estimation over marginal $K$ estimation.

### 6.3 Software

- **R:** `mcclust` (Fritsch), `mcclust.ext` (Wade), `salso` (Dahl et al.), `GreedyEPL` (Rastelli)
- **Julia:** No existing implementation — to be developed in SMLMBaGoL

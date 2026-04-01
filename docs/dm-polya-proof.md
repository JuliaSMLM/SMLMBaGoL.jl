# DM/Polya Partition Prior: Derivation and Physical Implications

**Why the Dirichlet-Multinomial partition prior arises from independent NegBin emitter counts, and why it may be the wrong model for BaGoL.**

For the general derivation starting from an arbitrary iid emitter-count pmf
`q(n)`, see [count-family-partition-prior.md](count-family-partition-prior.md).

---

## 1. Physical Model

Each of $K$ emitters independently generates localizations:

$$n_j \sim \text{NegBin}(\alpha, p), \quad j = 1, \ldots, K$$

with $p = \alpha/(\alpha + \mu)$, giving $E[n_j] = \mu$ and $\text{Var}[n_j] = \mu(1 + \mu/\alpha)$.

The total count $N = \sum_j n_j$ is observed. The assignment vector $z$ records which emitter generated each localization: $z_i \in \{1, \ldots, K\}$.

---

## 2. Derivation: NegBin Counts Imply DM Partition Prior

### Step 1: Joint count distribution (iid NegBin)

$$P(n_1, \ldots, n_K) = \prod_j \frac{\Gamma(n_j + \alpha)}{n_j!\, \Gamma(\alpha)}\, p^\alpha\, (1-p)^{n_j}$$

### Step 2: Marginal of $N$ (NegBin additivity)

$$P(N \mid K) = \frac{\Gamma(N + K\alpha)}{N!\, \Gamma(K\alpha)}\, p^{K\alpha}\, (1-p)^N$$

### Step 3: Conditional count vector given $N$

Dividing Step 1 by Step 2, the $p^{K\alpha}$ and $(1-p)^N$ terms cancel:

$$P(n_1, \ldots, n_K \mid N, K) = \frac{N!}{\prod_j n_j!} \cdot \frac{\Gamma(K\alpha)}{\Gamma(N + K\alpha)} \cdot \prod_j \frac{\Gamma(n_j + \alpha)}{\Gamma(\alpha)}$$

This is the **Dirichlet-Multinomial (DM)** distribution with concentration $\alpha$ per component.

### Step 4: From count vectors to assignment vectors

Given the count vector $(n_1, \ldots, n_K)$, there are $N!/\prod_j n_j!$ distinct assignment vectors consistent with those counts. By exchangeability of the localization labels (before observing positions), each is equally likely:

$$P(z \mid n_1, \ldots, n_K) = \frac{\prod_j n_j!}{N!}$$

### Step 5: Prior on the assignment vector

$$P(z \mid N, K) = P(z \mid \mathbf{n}) \cdot P(\mathbf{n} \mid N, K) = \frac{\prod n_j!}{N!} \cdot \text{DM}(\mathbf{n}; \alpha)$$

The multinomial coefficients cancel exactly:

$$\boxed{P(z \mid N, K) = \frac{\Gamma(K\alpha)}{\Gamma(N + K\alpha)} \cdot \prod_j \frac{\Gamma(n_j + \alpha)}{\Gamma(\alpha)} = \text{Polya}(z; \alpha)}$$

where $n_j = |\{i : z_i = j\}|$ are the cluster sizes induced by $z$.

### Verification

The identity $\text{DM}(\mathbf{n}) \times P(N \mid K) = \text{Polya}(z) \times \binom{N}{\mathbf{n}} \times P(N \mid K) = \prod_j \text{NegBin}(n_j; \alpha, p)$ holds exactly. The per-emitter NegBin likelihood decomposes into the Polya prior on $z$, the multinomial multiplicity of assignment vectors, and the marginal count likelihood.

### What this does and does not assume

It is important to be precise about the hidden assumption here.

The derivation does **not** assume a distribution over $\alpha$ and then
integrate $\alpha$ out. Here, $\alpha$ is fixed.

What is assumed is the factorization

$$P(n_1, \ldots, n_K \mid K, \alpha, p) = \prod_j \text{NegBin}(n_j; \alpha, p).$$

That factorized finite-$\alpha$ NegBin model is enough by itself to imply the
DM count prior and the Polya prior on $z$ after conditioning on $N$.

An equivalent latent-variable interpretation is:

$$\lambda_j \stackrel{\text{iid}}{\sim} \text{Gamma}(\alpha, \alpha/\mu), \qquad
n_j \mid \lambda_j \sim \text{Poisson}(\lambda_j).$$

Marginally, each emitter count is still Negative Binomial. But after
conditioning on total count $N$, the normalized shares

$$\pi_j = \lambda_j / \sum_\ell \lambda_\ell$$

follow a Dirichlet distribution,

$$\pi \sim \text{Dir}(\alpha, \ldots, \alpha),$$

and

$$\mathbf{n} \mid N, \pi \sim \text{Multinomial}(N, \pi).$$

Integrating out $\pi$ gives the DM distribution on count vectors and the Polya
prior on assignment vectors.

So the physical content of `DM/Polya` is not "random $\alpha$." It is
"emitter-specific random count shares, integrated out."

---

## 3. The Gibbs Full Conditional

For fixed $K$, the conditional assignment probability is:

$$P(z_i = k \mid z_{-i}, N, K) = \frac{n_{-i,k} + \alpha}{(N - 1) + K\alpha}$$

This is the **Polya urn predictive**. The $(n_{-i,k} + \alpha)$ weighting — the "rich get richer" dynamic — is a mathematical consequence of the DM prior, not a separate modeling choice.

---

## 4. On the Source of the Reinforcement

An important subtlety: the rich-get-richer behavior is **not** solely an artifact of conditioning on $N$.

In the equivalent Dirichlet-mixture representation:

$$\pi \sim \text{Dir}(\alpha, \ldots, \alpha), \quad z_i \mid \pi \sim \text{Cat}(\pi)$$

conditioned on $\pi$, the assignments $z_i$ are iid. But after integrating out $\pi$, the assignments are **exchangeable but not independent** — they exhibit Polya-urn reinforcement regardless of whether $N$ is fixed.

Conditioning on $N$ makes the dependence among counts explicit (it forces the count vector onto the simplex $\sum n_j = N$), but the reinforcement is already present in the collapsed Dirichlet-mixture representation before $N$ is observed.

---

## 5. Practical Consequences for BaGoL

The Polya prior on $z$ with $\gamma = \alpha$ creates **systematic downward pressure on $K$** through every move type in the sampler:

### 5.1 Gibbs allocation sweep

The allocation weight $(n_k + \gamma) \times \text{predictive}$ implements the Polya conditional exactly. When neighboring emitters have similar spatial predictives (intermediate separations, $d/\sigma \approx 2$), the size-based $(n_k + \gamma)$ term dominates. This creates a rich-get-richer dynamic that destabilizes balanced partitions.

**Empirical evidence:** At $K = 8$ with $\text{NN} = 1.9\sigma$ and $\gamma = 5$, the Gibbs sweep drives balanced sizes $[5,5,5,5,5,5,5,5]$ to unbalanced equilibrium $[9,7,7,6,4,3,2,2]$ with 80--95% of localizations misassigned.

### 5.2 Polya density favors unbalanced sizes

Since $\log\Gamma$ is convex, $\sum_k \log\Gamma(n_k + \gamma)$ is **minimized** at balanced sizes. Unbalanced partitions have higher Polya density. This means the "correct" DM prior actively favors unbalanced cluster sizes.

### 5.3 DM penalty in MH moves

The $\Delta_\text{partition} = \log P_\text{DM}(z' \mid K') - \log P_\text{DM}(z \mid K)$ term appears in split/merge and birth/death acceptance ratios. For $N = 40$, $\gamma = 5$:

| Move | $\Delta_\text{DM}$ | Effect |
|------|--------------------:|--------|
| Merge $5+5 \to 10$ ($K{=}8 \to 7$) | $+6.43$ | **Favors** merge |
| Split $10 \to 5+5$ ($K{=}7 \to 8$) | $-6.43$ | **Opposes** split |
| Birth (singleton from balanced $K{=}8$) | $-3.94$ | **Opposes** birth |
| Death (absorb singleton) | $+3.94$ | **Favors** death |

Every K-increasing move is penalized. Every K-decreasing move is rewarded. The count model ($\Delta_\text{count}$) and spatial term ($\Delta_\text{spatial}$) must overcome this at **every step** of the $K{=}1 \to K{=}8$ climb.

### 5.4 Compounding across K transitions

For dimers ($K{=}2$), one barrier of ${\sim}{-}6$ nats is surmountable. For octamers ($K{=}8$), seven barriers compound. Combined with Gibbs destabilization of balanced partitions and near-100% death acceptance for singletons, the sampler systematically under-splits.

**Empirical evidence:** Fixed $N = 40$, exactly 5 locs/emitter, $\text{NN} = 1.9\sigma$, nohier:
- Q-PAINT: $K = 8$ recovery = **100%**
- BaGoL from oracle $K{=}8$: drops to $\bar{K} \approx 5.8$, $K{=}8$ recovery = **0%**
- Converged $P(K)$: $K{=}5$ (32%), $K{=}6$ (40%), $K{=}7$ (19%), $K{=}8$ (4%)

---

## 6. The Core Problem

The derivation in Section 2 is mathematically correct. The DM/Polya IS the right prior on $z$ given the NegBin count model conditioned on $N$.

However, the resulting target distribution produces a posterior that **systematically under-estimates $K$** at intermediate emitter separations, performing worse than count-only estimation (Q-PAINT). This occurs because:

1. The Polya prior favors unbalanced cluster sizes (Section 5.2)
2. The Gibbs sweep implements this preference as rich-get-richer dynamics (Section 5.1)
3. The MH penalties compound across K-transitions (Section 5.3--5.4)
4. The sampler correctly samples from a target that doesn't match the physical reality

The question is whether the NegBin count model with its implied DM partition prior is the right model for the physics. If the true blink-count heterogeneity is weaker than $\text{NegBin}(\alpha{=}5)$ suggests — or if the conditioning on $N$ introduces correlations that don't exist in the measurement process — then the model should be revised.

### Possible directions

- **Poisson limit** ($\alpha \to \infty$): DM concentration $\to \infty$, allocation weights become nearly uniform, rich-get-richer vanishes. Appropriate if blink counts have low overdispersion.
- **Decoupled target**: Use uniform $P(z \mid K) = 1/K^N$ with per-emitter NegBin in MH acceptance only. Gibbs becomes predictive-only. Changes the target — requires re-validation.
- **MH-corrected Gibbs**: Keep DM target but propose allocations proportional to predictive only, accept/reject with DM ratio. Changes dynamics without changing target.

---

## References

- Fazel et al., "High-Precision Estimation of Emitter Positions using Bayesian Grouping of Localizations," *Nature Communications* 13, 7152 (2022).
- Jain & Neal, "A Split-Merge Markov Chain Monte Carlo Procedure for the Dirichlet Process Mixture Model," *JCGS* 13(1), 158--182 (2004).

# Count Family to Partition Prior: General Derivation

**How an iid per-emitter count model `q(n)` induces a prior on the allocation vector `z`, and why the key issue is the induced allocation ratio, not the family label by itself.**

---

## 1. Setup

Suppose there are `K` emitters and emitter `k` produces `n_k` localizations.
Assume only that the emitter counts are iid draws from some pmf `q(n)`:

$$n_1, \ldots, n_K \stackrel{\text{iid}}{\sim} q(n), \quad n \in \{0,1,2,\ldots\}$$

Let

$$N = \sum_{k=1}^K n_k$$

be the observed total number of localizations, and let the assignment vector

$$z = (z_1, \ldots, z_N), \quad z_i \in \{1, \ldots, K\}$$

record which emitter generated each localization.

If `n_k = |\{i : z_i = k\}|`, then the count vector induced by `z` is

$$\mathbf{n}(z) = (n_1, \ldots, n_K).$$

The goal is to derive the prior on `z` implied by the iid count family `q(n)`.

---

## 2. Count Vectors and Assignment Vectors

### 2.1 Joint prior on the count vector

By independence,

$$P(n_1, \ldots, n_K \mid K) = \prod_{k=1}^K q(n_k).$$

### 2.2 Conditional count vector given total `N`

Conditioning on the observed total `N`,

$$P(n_1, \ldots, n_K \mid N, K)
=
\frac{\prod_{k=1}^K q(n_k)}{P(N \mid K)}
\;\mathbf{1}\!\left[\sum_k n_k = N\right],$$

where

$$P(N \mid K) = \sum_{\substack{n_1,\ldots,n_K \ge 0 \\ \sum_k n_k = N}} \prod_{k=1}^K q(n_k).$$

### 2.3 Assignment vectors given the count vector

Given the count vector `(n_1, ..., n_K)`, there are

$$\frac{N!}{\prod_k n_k!}$$

assignment vectors consistent with those counts. By exchangeability of the
localization labels before looking at positions,

$$P(z \mid n_1, \ldots, n_K) = \frac{\prod_k n_k!}{N!}.$$

---

## 3. General Prior on the Assignment Vector

Combining the previous pieces,

$$P(z \mid N, K) = P(z \mid \mathbf{n}) P(\mathbf{n} \mid N, K).$$

Substituting the formulas above gives

$$P(z \mid N, K)
=
\frac{\prod_k n_k!}{N!}
\cdot
\frac{\prod_k q(n_k)}{P(N \mid K)}.$$

For fixed `N` and `K`, the factors `N!` and `P(N | K)` are constants, so

$$\boxed{
P(z \mid N, K) \propto \prod_{k=1}^K \Big[n_k! \, q(n_k)\Big]
}$$

with `n_k = |\{i : z_i = k\}|`.

This is the general answer.

The induced one-step predictive for reassigning one localization is

$$\boxed{
P(z_i = k \mid z_{-i}, N, K)
\propto
(n_{-i,k} + 1)\,
\frac{q(n_{-i,k} + 1)}{q(n_{-i,k})}
}$$

because only cluster `k` changes when localization `i` is assigned there.

So the full conditional is determined entirely by the ratio

$$\frac{q(n+1)}{q(n)}.$$

Equivalently, if we define

$$a_n := n!\,q(n),$$

then

$$P(z \mid N, K) \propto \prod_k a_{n_k}$$

and the predictive is driven by

$$\frac{a_{n+1}}{a_n}.$$

This is the key quantity. The important question is not the family label
"Poisson" or "NegBin" by itself, but whether `a_{n+1}/a_n` is constant or
depends on `n`.

---

## 4. Special Cases

### 4.1 Poisson counts

If

$$q(n) = \text{Poisson}(n; \mu) = e^{-\mu} \frac{\mu^n}{n!},$$

then

$$n! \, q(n) = e^{-\mu} \mu^n.$$

Therefore

$$P(z \mid N, K) \propto \prod_k e^{-\mu} \mu^{n_k}
= e^{-K\mu} \mu^N,$$

which is constant over all assignment vectors with the same `N` and `K`.

So

$$\boxed{P(z \mid N, K) \propto 1}$$

and the one-step predictive is uniform over emitters:

$$P(z_i = k \mid z_{-i}, N, K) \propto \mu.$$

After normalization, each emitter has probability `1/K`.

**Interpretation:** iid Poisson emitter counts imply no rich-get-richer effect
after conditioning on total `N` because `a_{n+1}/a_n` is constant.

### 4.2 Negative Binomial counts

If

$$q(n) = \text{NegBin}(n; \alpha, p)
=
\frac{\Gamma(n+\alpha)}{n!\,\Gamma(\alpha)} p^\alpha (1-p)^n,$$

then

$$n! \, q(n) \propto \Gamma(n+\alpha).$$

Therefore

$$\boxed{
P(z \mid N, K) \propto \prod_{k=1}^K \Gamma(n_k + \alpha)
}$$

which is exactly the Polya / Dirichlet-Multinomial form.

The predictive becomes

$$P(z_i = k \mid z_{-i}, N, K) \propto n_{-i,k} + \alpha.$$

This is the usual rich-get-richer factor because `a_{n+1}/a_n` grows with `n`.

### 4.2.1 What is being marginalized in the NegBin case?

This point is easy to misread, so it is worth stating explicitly.

The `DM/Polya` prior does **not** come from putting a distribution on `alpha`
and then integrating `alpha` out. In this derivation, `alpha` is a fixed
hyperparameter that controls the shape of the emitter-level count law.

What is being assumed is the factorized count model

$$P(n_1, \ldots, n_K \mid K, \alpha, p) = \prod_{k=1}^K \text{NegBin}(n_k; \alpha, p).$$

That assumption alone is enough to force the `DM/Polya` prior after
conditioning on `N`.

There is also an equivalent latent-variable representation that makes the
source of the reinforcement easier to see:

$$\lambda_k \stackrel{\text{iid}}{\sim} \text{Gamma}(\alpha, \alpha/\mu),$$

$$n_k \mid \lambda_k \sim \text{Poisson}(\lambda_k).$$

Marginalizing the emitter-specific intensities `lambda_k` gives

$$n_k \sim \text{NegBin}(\alpha, p), \qquad p = \frac{\alpha}{\alpha+\mu}.$$

Now condition on the total count `N = \sum_k n_k`. Define the normalized
shares

$$\pi_k = \frac{\lambda_k}{\sum_j \lambda_j}.$$

Then

$$\pi = (\pi_1, \ldots, \pi_K) \sim \text{Dir}(\alpha, \ldots, \alpha),$$

and

$$ (n_1, \ldots, n_K) \mid N, \pi \sim \text{Multinomial}(N; \pi_1, \ldots, \pi_K). $$

Integrating out `pi` gives the Dirichlet-Multinomial count prior, and then
integrating out the multinomial count-vector multiplicity gives the Polya
prior on `z`.

So the hidden structural assumption behind `DM/Polya` is:

- each emitter has its own latent count share / latent intensity;
- these emitter-specific shares are random;
- after integrating them out, assignments inherit reinforcement.

This is an *equivalent representation* of iid finite-`alpha` NegBin counts.
It is not an extra assumption on top of them. But it is often the clearest way
to interpret what the model is saying physically.

### 4.3 Geometric counts

The geometric distribution is the special case `alpha = 1` of the
Negative Binomial:

$$q(n) = p(1-p)^n.$$

Then

$$n! \, q(n) \propto n!$$

and

$$P(z \mid N, K) \propto \prod_k n_k!,$$

with predictive

$$P(z_i = k \mid z_{-i}, N, K) \propto n_{-i,k} + 1.$$

This is the strongest finite-`alpha` Polya reinforcement in the NegBin family.

### 4.4 Poisson limit of the Negative Binomial

For `alpha -> infinity` with mean `mu` fixed, the Negative Binomial converges
to Poisson. Correspondingly,

$$\Gamma(n+\alpha) / \Gamma(\alpha) \approx \alpha^n,$$

so after conditioning on total `N`, the Polya factor becomes effectively
constant across assignments. The rich-get-richer effect vanishes in the
Poisson limit.

### 4.5 Deterministic counts

If every emitter produces exactly `m` localizations,

$$q(n) = \mathbf{1}[n = m],$$

then only count vectors with `n_k = m` for all `k` have support. Thus

$$N = Km$$

must hold exactly, and `P(z | N, K)` is uniform over the assignments
consistent with those fixed counts.

---

## 5. What "Identical Emitters" Does and Does Not Imply

This derivation uses two assumptions:

1. all emitters share the same count law `q(n)`;
2. the emitter counts are independent draws from that law.

The first assumption means the emitters are *identical in distribution*.
The second is stronger: it specifies the full joint distribution of the count
vector as a product.

That distinction matters:

- Identical + independent counts with constant `a_{n+1}/a_n` give a uniform
  `P(z | N, K)`.
- Identical + independent counts with `a_{n+1}/a_n` increasing in `n` give a
  rich-get-richer prior on `z`.
- Finite-`alpha` NegBin is one important example of the second case.
- Poisson, and the `alpha -> infinity` limit of NegBin, are examples of the
  first case.

So "the emitters are identical" does **not** by itself imply `DM/Polya`.
What implies `DM/Polya` is the specific factorization

$$P(n_1, \ldots, n_K \mid K) = \prod_k \text{NegBin}(n_k; \alpha, p).$$

with finite `alpha`, which makes `a_{n+1}/a_n` depend on `n`.

Another equivalent way to say the same thing is:

- `DM/Polya` assumes emitter-specific latent count shares survive after
  conditioning on `N`;
- it does **not** assume a distribution over `alpha`;
- `alpha` is fixed and controls how variable those latent shares are.

Large `alpha` means the latent shares are nearly equal and the induced
allocation prior is close to uniform. Small finite `alpha` means the latent
shares are more uneven and the induced allocation prior shows stronger
reinforcement.

---

## 6. Where the Modeling Question Actually Lives

If the physical process really supports iid finite-`alpha` Negative Binomial
counts per emitter, then the `DM/Polya` prior on `z` is mathematically forced.

If that rich-get-richer prior feels unphysical, then the assumption to revisit
is not the algebra above. It is the emitter-level count model itself, or the
independence assumption behind it.

In particular:

- If the correct emitter-level count model gives nearly constant
  `a_{n+1}/a_n`, then `P(z | N, K)` should be close to uniform.
- If the pooled empirical count histogram looks Negative Binomial, that alone
  does **not** prove the count vector factorizes as iid Negative Binomial
  within one cluster.
- Any alternative physical model that reproduces the same marginal count
  histogram but changes the joint count vector can induce a different partition
  prior on `z`.

For example, a model with a *shared* latent acquisition factor can make each
emitter's marginal count distribution overdispersed while still canceling out
when conditioning on `N`. In that case the emitters remain identical, but the
allocation prior need not be `DM/Polya`.

That is the key modeling fork for BaGoL.

---

## 7. Relation to the Existing DM/Polya Note

The file [dm-polya-proof.md](dm-polya-proof.md) is the `q(n) = NegBin`
specialization of this general derivation.

Its conclusion remains correct:

$$\text{iid NegBin emitter counts} \quad \Longrightarrow \quad \text{DM/Polya prior on } z.$$

What this broader note makes explicit is that this conclusion depends on the
effective allocation ratio `a_{n+1}/a_n`, and not on "identical emitters"
alone or on the distribution family name in isolation.

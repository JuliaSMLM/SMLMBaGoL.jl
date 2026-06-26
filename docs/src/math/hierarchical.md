```@meta
CurrentModule = SMLMBaGoL
```

# Hierarchical Learning

The count-distribution hyperparameters are not fixed — they are learned from the data every
`sync_interval` iterations (default 100), gated by `learn_distribution` (`hierarchical.jl`).

## ``\mu`` and shape ``\alpha`` — Metropolis–Hastings

Both are updated by a **log-normal random-walk Metropolis–Hastings** step against the
per-cluster negative-binomial likelihood and a Gamma prior. For ``\mu`` (shape ``\alpha`` is
analogous, with both NegBin parameters depending on it):

```math
\mu' = \mu\, e^{s\xi},\ \xi \sim \mathcal{N}(0,1), \qquad
\log\alpha_{\text{acc}} = \big[\log\mathcal{L}(\mu') - \log\mathcal{L}(\mu)\big]
   + \big[\log p(\mu') - \log p(\mu)\big] + \underbrace{(\log\mu' - \log\mu)}_{\text{Jacobian}},
```

where ``\mathcal{L}(\mu) = \prod_{k}\mathrm{NegBin}(n_k;\ \alpha,\ \alpha/(\alpha+\mu))`` over
active clusters, and the multiplicative walk contributes the log-Jacobian
``\log\mu' - \log\mu``. The priors are ``\mu \sim \mathrm{Gamma}(2, 5)`` (mean 10) and
``\alpha \sim \mathrm{Gamma}(2, 1)`` (mean 2); proposals are bounded to ``[1, 500]`` and
``[0.5, 50]`` respectively.

Two regimes exist:

- **Partitioned [`run_bagol`](@ref) path** — an **``N``-step (50) adaptive** kernel that pools
  counts across all partitions and tunes the step size ``s`` by a Robbins–Monro rule toward
  ≈ 0.30 acceptance **during burn-in**; the step size is then frozen, preserving ergodicity.
- **Standalone [`run_collapsed_chain`](@ref)** — a single-step, fixed-scale (``s = 0.3``)
  version.

!!! info "Figure · pipeline · `assets/hier_convergence.png`"
    The ``K`` / ``\mu`` / shape traces over iterations with the burn-in line, from
    [`plot_report`](@ref)'s `convergence` panel. The hyperparameters settle as the chain
    learns the count distribution — the standard burn-in diagnostic.

## ``\rho`` — conjugate Gibbs (flat model only)

The Poisson-``K``-prior density ``\rho`` (used only under `spatial_model = :flat`) has a
conjugate Gamma posterior and is drawn exactly — no accept/reject — from

```math
\rho \mid K, A \;\sim\; \mathrm{Gamma}\big(\text{shape } a + K,\ \text{rate } b + A\big),
```

with ``\rho \sim \mathrm{Gamma}(2, 1)`` as prior. Under the default `:locmix` model there is
no Poisson ``K`` prior, so this step **never fires** — ``\rho`` stays at its prior mean.

## How learning closes the loop

Learned ``\mu`` and ``\alpha`` feed straight back into the [count model](priors.md#Count:-the-negative-binomial-blink-model)
that scores every ``K``-changing [move](moves.md): a larger learned ``\mu`` (more
localizations per emitter) lowers the count-model penalty for *fewer*, larger clusters, and
vice versa. The convergence trace above is the direct readout of this feedback.

!!! info "Figure · pipeline · `assets/hier_countdist.png`"
    The learned count distribution overlaid on the true one (and the empirical histogram),
    from [`plot_report`](@ref)'s `count_distribution`. Confirms the hyperparameter MH is
    recovering the right NegBin.

# Split/Merge Move Reference

*Updated: K!-correction round (2026-07-02). Added the K! occupied-label multiplicity
term ($\Delta_{K!}$) and the Fix A merge seed-compatibility guard.
Update this file at end of each sampler research round.*

**Code:** `propose_split_merge!` in `src/collapsed_moves.jl`.

---

## Overview

RJMCMC split/merge with $|\Delta K| = 1$ random proposals, Jain-Neal restricted Gibbs scans, and full MH acceptance. Constitutes 25% of move selection.

---

## Phase 1: Direction Selection

$$b_K = P(\text{split at } K), \quad d_K = 1 - b_K$$

- $K = 1$: $b_K = 1$ (always split)
- $K \geq N$: $d_K = 1$ (always merge)
- Otherwise: $b_K = d_K = 0.5$

---

## Phase 2: Execute Split or Merge

### Split ($K \to K+1$)

1. **Select parent** cluster uniformly: $1/K$
2. **Random seed selection:** pick two members (ordered pair, probability $1/(m(m-1))$).
   Seed 1 $\to$ sub-A, Seed 2 $\to$ sub-B.
3. **Canonical ordering** of remaining members (sorted by loc index)
4. **Sequential predictive allocation (launch):** for each remaining member $j$:

$$\log w_A = \log(n_A + \gamma) + \log p_{\text{pred}}(d_j \mid \text{sub-A})$$
$$\log w_B = \log(n_B + \gamma) + \log p_{\text{pred}}(d_j \mid \text{sub-B})$$

Sample: $j \to B$ with probability $p_B = w_B / (w_A + w_B)$.

5. **Restricted Gibbs scans (Jain-Neal):** `n_restricted_scans - 1` intermediate sweeps (no density tracking) + 1 final sweep (density tracked $\to q_{\text{alloc}}$). Only the final sweep enters the MH ratio. Default `n_restricted_scans = 5`.

6. Forward structural density: $q_{\text{fwd}} = (1/K) \times q_{\text{alloc}}$
7. Reverse structural density: $q_{\text{rev}} = 1/\binom{K+1}{2}$

### Merge ($K \to K-1$)

1. **Select pair** uniformly: $1/\binom{K}{2}$
2. **Random seed selection** matching the split bijection. Seed density cancels.
3. **Canonical ordering,** `is_in_b` relative to sub-cluster labels
4. **Reverse allocation density** via Jain-Neal:
   a. Sample launch via sequential allocation from merged members
   b. Run `n_restricted_scans - 1` intermediate sweeps on launch
   c. Compute transition density: intermediate $\to$ current allocation using hybrid state
   d. **Fix A (seed-compatibility guard):** if `is_in_b[1]` — both drawn seeds fell in the
      SAME pre-merge cluster — the reverse split cannot recreate this seed pair (a split
      pins seed1$\to$A, seed2$\to$B), so $q_{\text{rev}} = 0$ ($\log q_{\text{alloc\_rev}} = -\infty$).
      Without the guard the Jain-Neal scan (which never re-scans the seeds) fabricates a
      *positive* reverse density → DB violation → over-merge → $K$ biased **down**. Do NOT
      "fix" this by resampling seeds until compatible: that changes the seed law
      $1/(m(m-1)) \to 1/(2 n_A n_B)$ and silently re-breaks DB.
5. Forward: $q_{\text{fwd}} = 1/\binom{K}{2}$
6. Reverse: $q_{\text{rev}} = (1/(K-1)) \times q_{\text{alloc\_rev}}$
7. Execute merge: move all locs from slot B into slot A, deactivate B.

---

## Phase 3: MH Acceptance

$$\log \alpha = \Delta_{\text{spatial}} + \Delta_{\text{partition}} + \Delta_{\text{proposal}} + \Delta_{\text{count}} + \Delta_{\text{move\_type}} + \Delta_{K!}$$

| Term | Definition |
|------|-----------|
| $\Delta_{\text{spatial}}$ | $\sum_k \log\text{ML}_k^{\text{new}} - \sum_k \log\text{ML}_k^{\text{old}}$ |
| $\Delta_{\text{partition}}$ | $\log P_{\text{DM}}(z' \mid K') - \log P_{\text{DM}}(z \mid K)$ |
| $\Delta_{\text{proposal}}$ | $\log q_{\text{rev}} - \log q_{\text{fwd}}$ |
| $\Delta_{\text{count}}$ | $\log P(N \mid K') - \log P(N \mid K)$ (uses **fixed** $\mu_0$) |
| $\Delta_{\text{move\_type}}$ | $\log(d_{K'}/b_K)$ for splits, $\log(b_{K'}/d_K)$ for merges |
| $\Delta_{K!}$ | $\log K'! - \log K!$ ($=+\log(K{+}1)$ split, $-\log K$ merge). **Only under a Poisson $K$ prior** (`_uses_poisson_k_prior`); cancels the prior's $-\log K!$ so the kernel targets the coherent $K$-posterior $T_{\text{fac}} = K!\,T_1$ instead of the canonical $T_1$. Exact at fixed hyperparameters ($e^{\lambda m}$ carry-through only matters when learning $\mu/\text{shape}/\rho$). No-op for `:locmix`. |

On rejection: full rollback from saved state.

---

## Key Invariants

- **Fixed $\mu$ in count ratio:** Uses initial $\mu$ (prior mean), not adaptive, to prevent $\mu$-$K$ feedback.
- **Seed density cancellation:** $1/(m(m-1))$ appears in both split and merge via RJMCMC bijection and cancels. Must NOT be included explicitly (see KB #12).
- **Restricted Gibbs:** Intermediate sweeps are "free" (Jain-Neal 2004). Only final sweep density enters MH.
- **Hybrid state for merge reverse:** Members already processed use TARGET assignment; later members use INTERMEDIATE state.

---

## Performance (Round 9 brute-force, N=6)

| Test | Split acc. | Merge acc. |
|------|-----------|------------|
| Well-separated (d/σ=10) | ~7% | ~7% |
| Close dimer (d/σ=3) | ~5% | ~10% |
| Single emitter | ~3% | ~15% |
| SM-only well-sep | ~7% | ~7% |

SM-only brute-force FAILS at 1.76x max ratio (well-separated). BD compensates.

---

## Constants

| Value | What | Justification |
|-------|------|---------------|
| 50/50 | Split/merge coin flip | Equal $K\pm 1$; boundary-aware |
| $\gamma = \alpha$ | DM concentration | From NegBin count model |
| 5 | `n_restricted_scans` | 4 intermediate + 1 final Jain-Neal |

---

## Known Limitations

The per-allocation DM penalty ($\approx -2.5$ to $-5$ per split) creates an energy barrier that compounds at large $N$. Even though the marginal $P(K)$ favors high $K$, individual split proposals face this barrier. See `docs/math_reference.md` Section 9.1.

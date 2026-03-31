# Birth/Death Move Reference

*Updated: Round 11 (2026-03-30). No code changes in Round 11.
Update this file at end of each sampler research round.*

**Code:** `propose_birth_death!` in `src/collapsed_moves.jl`.

---

## Overview

Incremental $K \pm 1$ transitions with lower energy barriers than split/merge. The DM penalty per birth is $\approx -1.2$ (vs $-2.5$ to $-5$ per split). Constitutes 25% of move selection, with `n_bd_substeps` (default 5) sequential attempts per selection (**BD burst**).

---

## Direction Selection

| Condition | $p_{\text{birth}}$ | $p_{\text{death}}$ |
|-----------|--------------------|--------------------|
| Both feasible | 0.5 | 0.5 |
| Only birth ($n_{\text{singletons}} = 0$ or $K \leq 1$) | 1.0 | 0.0 |
| Only death ($N_{\text{eligible}} = 0$ or $K \geq N$) | 0.0 | 1.0 |

---

## Birth ($K \to K+1$)

1. Count non-sole-occupant locs: $N_{\text{eligible}} = N - n_{\text{singletons}}$
2. Pick random non-sole-occupant loc: $1/N_{\text{eligible}}$
3. Remove from parent cluster, create singleton cluster
4. Subsequent Gibbs sweeps grow the singleton by attracting nearby locs

---

## Death ($K \to K-1$)

1. Pick random singleton cluster: $1/n_{\text{singletons}}$
2. Compute DM-weighted predictive for each destination:

$$w(k) = (n_k + \gamma) \times p_{\text{pred}}(d_i \mid \text{cluster}_k)$$

3. Sample destination proportional to $w(k)$
4. Absorb singleton loc into destination

---

## Proposal Densities

**Birth forward:**
$$q_{\text{birth}}(x \to x') = p_{\text{birth}}(x) \times \frac{1}{N_{\text{eligible}}(x)}$$

**Death reverse (from $x'$):**
$$q_{\text{death\_rev}}(x' \to x) = p_{\text{death}}(x') \times \frac{1}{n_{\text{singletons}}(x')} \times \frac{w(\text{dest})}{\sum_k w(k)}$$

Death forward and birth reverse are analogous with roles swapped.

---

## MH Acceptance

$$\log \alpha = \Delta_{\text{spatial}} + \Delta_{\text{partition}} + \Delta_{\text{proposal}} + \Delta_{\text{count}}$$

No $\Delta_{\text{move\_type}}$ --- the $p_{\text{birth}}/p_{\text{death}}$ boundary handling is already in $q_{\text{fwd}}/q_{\text{rev}}$.

| Term | Definition |
|------|-----------|
| $\Delta_{\text{spatial}}$ | $\sum_k \log\text{ML}_k^{\text{new}} - \sum_k \log\text{ML}_k^{\text{old}}$ |
| $\Delta_{\text{partition}}$ | $\log P_{\text{DM}}(z' \mid K') - \log P_{\text{DM}}(z \mid K)$ |
| $\Delta_{\text{proposal}}$ | $\log q_{\text{rev}} - \log q_{\text{fwd}}$ |
| $\Delta_{\text{count}}$ | $\log P(N \mid K') - \log P(N \mid K)$ (uses **fixed** $\mu_0$) |

---

## BD Burst

When birth/death is selected (25% of iterations), `n_bd_substeps` (default 5) sequential BD attempts are made. Each is independent MH with proper acceptance/rejection. The target distribution is unchanged.

**Why burst helps:** For co-located emitters, no proposal weighting can improve birth acceptance (locs are exchangeable). The only lever is more attempts. With 5 substeps, K-transition rate increases $\sim 5\times$.

---

## Performance (Round 9 brute-force, N=6)

| Test | Birth acc. | Death acc. |
|------|-----------|------------|
| Well-separated (d/σ=10) | ~15% | 100% |
| Close dimer (d/σ=3) | ~10% | 100% |
| Single emitter | ~5% | 100% |

BD burst (n=5) achieved 4/4 brute-force PASS (n=3 gave 3/4).

---

## Constants

| Value | What | Justification |
|-------|------|---------------|
| 50/50 | Birth/death coin flip | Equal $K\pm 1$; boundary-aware |
| 5 | `n_bd_substeps` | 4/4 brute-force PASS at $N=6$; 3 was 3/4 |

---

## Known Limitations

- **Death acceptance 100%:** Singletons are always absorbed because the DM + spatial terms favor merging. This creates an asymmetry: births are rare (~5-15%) but deaths are certain.
- **Gibbs destabilizes births:** After a successful birth, the next Gibbs sweep may create another singleton (rich-get-richer from DM), which is immediately killed by the next death.
- **Large-N scaling:** Birth acceptance doesn't scale with $N$ because each birth faces the per-allocation DM penalty. At $N=40$, births occur but the chain cannot sustain high $K$ (see `docs/math_reference.md` Section 9.1).

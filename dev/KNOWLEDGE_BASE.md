# Sampler Knowledge Base

Structured catalog of approaches tried, dead ends hit, and techniques that work. Read before proposing a new sampler design — if your idea resembles a documented dead end, explain why it's different.

---

## Dead Ends

### 1. Full Spatial ML + Restricted Gibbs Scan

**What was tried:** Use the full spatial marginal likelihood (including per-cluster -log(A) term from uniform prior) in the split/merge acceptance ratio. After proposing K_new, run a restricted Gibbs scan to reallocate.

**Why it failed:** Runaway splitting. The uniform prior's -log(A) per cluster is small relative to the likelihood gain from splitting a cluster into two well-positioned subclusters. The sampler systematically over-splits, especially at high density.

**What was learned:** The spatial ML under uniform prior has an inherent bias toward larger K. Any approach using uniform spatial prior needs an explicit penalty to compensate.

**Branch/commit:** Early loc-mixture-prior development.

---

### 2. Full Spatial ML + Deterministic Nearest-Anchor

**What was tried:** After proposing K_new from the count model, use deterministic nearest-anchor assignment (each loc goes to closest cluster center) instead of Gibbs.

**Why it failed:** mu-K feedback loop. When K increases, the nearest-anchor assignment creates small clusters. Small clusters → higher inferred μ → count model favors even higher K → positive feedback. The chain runs away to K_max.

**What was learned:** Deterministic allocation + adaptive μ creates unstable feedback. Either the allocation must be stochastic (breaking the deterministic link) or μ must be fixed during K proposals.

**Branch/commit:** Early loc-mixture-prior development.

---

### 3. Full Spatial ML + Nearest-Anchor + Fixed μ₀

**What was tried:** Same as #2 but with μ fixed at prior mean μ₀ = μ_prior_shape × μ_prior_scale during K proposals. This breaks the μ-K feedback loop.

**Why it failed:** Occam barrier blocks co-located splits. When emitters are co-located (d ≈ 0), the spatial ML improvement from splitting is near zero, but the -log(A) penalty from the uniform prior makes the MH ratio always reject. The count model correctly proposes K_new > K_old, but the spatial term vetoes it.

**What was learned:** The uniform spatial prior creates an inherent Occam penalty per cluster. For co-located emitters (the hardest case), this penalty dominates the acceptance ratio. Led directly to the locmix prior idea.

**Branch/commit:** loc-mixture-prior, before locmix was introduced.

---

### 4. Count-Only K Proposal + Jain-Neal Pair Selection

**What was tried:** Propose K from count model posterior (no spatial info). For split/merge, use Jain-Neal (2004) restricted Gibbs: pick a random pair of locs, propose split/merge of their clusters.

**Why it failed:** Mixing problems at high K. Jain-Neal pair selection is efficient for K=1↔2 but scales poorly — when K is large, the probability of selecting a pair that straddles the right cluster boundary is O(1/K²). The chain gets stuck at whatever K it reaches first.

**What was learned:** Pair-based split/merge selection doesn't combine well with direct K sampling. The K proposal jumps to the right K, but the spatial rearrangement can't keep up.

**Branch/commit:** Early locmix-grid development.

---

### 5. Count-Only K Proposal + Direct K Sampling (No Spatial Info)

**What was tried:** Propose K from count model posterior. Heuristic split/merge to adjust allocation. No spatial MH correction — accept all K changes that the count model proposes.

**Why it succeeded (partially):** Matches Q-PAINT exactly for co-located emitters. The count model alone gives the information-theoretic optimal K estimate when there's no spatial separation.

**Why it's insufficient:** No spatial information used. For well-separated emitters (d/σ > 3), the spatial evidence strongly favors the correct K, but this approach ignores it. BaGoL should beat Q-PAINT when spatial information is available.

**What was learned:** The count-model K proposal is the right foundation. The question is how to layer spatial information on top without introducing bias.

**Branch/commit:** Intermediate locmix-grid state.

---

### 6. Direct K Sampling + Spatial MH (Current Approach)

**What was tried:** Propose K from count model. Heuristic split (random 50/50 of largest cluster) or merge (smallest into nearest). 5 Gibbs relaxation sweeps. Spatial MH acceptance using locmix prior (Δ_fit only, no area term).

**What works:** Beats Q-PAINT for separated emitters. The spatial MH correction filters out bad K proposals. Locmix prior eliminates the Occam barrier.

**What's broken:** Brute-force enumeration detects bias. The heuristic split/merge is not reversible — forward (split) and reverse (merge) have different proposal structures. The MH correction accounts for the spatial fit difference but not the proposal asymmetry.

**Current status:** Superseded by approach #11 (proper RJMCMC). The heuristic split/merge has intractable proposal density due to Gibbs relaxation.

**Branch/commit:** locmix-grid, merged to main. Replaced in Round 2.

---

### 7. SMC-Based Split Moves (smc-split branch)

**What was tried:** MFM (Mixture of Finite Mixtures) model with SMC-based split moves. Sequential Monte Carlo to propose the allocation after a split, giving a proper proposal density.

**Outcome:** Explored on `smc-split` branch (2026-03-27). Not merged — parallel investigation.

**Status:** Unknown (not evaluated against brute-force). Worth revisiting.

---

### 8. Round 1 Root Cause Analysis (2026-03-29) — SUPERSEDED by Round 2

**What was found (Round 1):** Over-splitting attributed to missing Δ_proposal in MH.

**What was actually wrong (Round 2):** The root cause analysis was INCOMPLETE. The dominant issue was a **target distribution mismatch**: the brute-force included a Dirichlet-Multinomial partition prior P(z|K) but the sampler didn't have one. Without the DM prior, the implicit partition prior was uniform-per-label, which strongly favors higher K due to combinatorial explosion (S(6,3)=90 partitions vs S(6,1)=1).

**Three interacting issues identified in Round 2:**
1. Missing DM partition prior in Gibbs sweep → wrong conditional distribution at each K
2. Missing DM partition prior in MH → wrong acceptance ratio for K changes
3. Intractable proposal density (Gibbs relaxation) → can't compute Δ_proposal

Adding DM to Gibbs alone (without MH) → no change (MH dynamics dominate)
Adding DM to both Gibbs + MH (without Δ_proposal) → severe under-splitting (DM penalty + no Gibbs relaxation to compensate)
Proper RJMCMC (DM + computable Δ_proposal, no Gibbs relaxation) → 7-10x improvement

**Key lesson:** When brute-force says "sampler is biased," first verify that the brute-force target MATCHES the sampler's target. In this case, they were different distributions entirely.

---

### 9. DM-Weighted Gibbs Without MH Correction (Round 2 intermediate)

**What was tried:** Add (n_k+γ) weighting to Gibbs allocation sweep but keep the split/merge MH using only Δ_spatial.

**Why it failed:** No change from baseline. The DM weighting changes the allocation at fixed K but doesn't affect K transitions. The split/merge dynamics (which control P(K)) dominate and are unchanged by the Gibbs weighting alone.

**What was learned:** The partition prior must enter BOTH the Gibbs sweep AND the MH acceptance to have any effect on the K distribution.

---

### 10. DM in Both Gibbs + MH, Without Δ_proposal (Round 2 intermediate)

**What was tried:** Add (n_k+γ) to Gibbs sweep AND add Δ_partition to split/merge MH. No Δ_proposal (still using heuristic split + Gibbs relaxation).

**Why it failed:** Severe under-splitting. Split acceptance dropped to 0.5-2.3% (from 37-42%). The DM partition prior penalizes splits (Γ is superadditive), and without Δ_proposal to compensate, virtually all splits are rejected. The chain gets stuck at K_true.

**What was learned:** Δ_partition and Δ_proposal partially cancel. Including one without the other creates worse bias than including neither. The Gibbs relaxation doesn't just improve split quality — it implicitly provides a Δ_proposal correction through the quality of the proposed allocation.

---

### 11. Proper RJMCMC with Sequential Allocation (Round 2)

**What was tried:** Replace heuristic split (random 50/50 + 5 Gibbs relaxation) with sequential predictive allocation. Each loc assigned proportional to (n_sub + γ) × spatial predictive. Proposal density is a product of conditional probabilities — fully tractable. Full MH: Δ_spatial + Δ_partition + Δ_proposal.

**What works:** Marginal P(K) KL divergence improved 7-10x. Well-separated dimer nearly passes brute-force test. Bias direction flipped from over-splitting to mild under-splitting (better failure mode for MAP-N).

**What's still broken:** Fixed seed ordering (member[1]→A, member[2]→B) makes some partitions 35x under-visited in the split-merge-only diagnostic. Fixed in Round 3 by random seeds.

**Branch/commit:** main, Round 2 changes.

---

### 12. Random Seed Selection with Canceling Density (Round 3)

**What was tried:** Replace fixed seeds (member[1]→A, member[2]→B) with random seed selection from all cluster members. Key insight: in the RJMCMC bijection framework, the seed selection density 1/(m(m-1)) appears as an auxiliary variable in BOTH split and merge proposals and cancels in the MH ratio. The merge also randomly selects seeds (matching the bijection), with is_in_b recomputed relative to sub-cluster labels.

**What works:** SM-only per-partition KL improved 14x (0.29 → 0.02). The 35x under-visited partitions are now at 1.4x. Full MCMC improved on 3/4 tests, close dimer KL halved. K=2 test accuracy 53.5% → 80%.

**What's still broken:** Residual under-splitting bias (2-3x on K>K_true). ~50% of merge proposals pick "bad" seeds (both from same original cluster), giving near-zero q_alloc_rev — these proposals are wasted.

**Key mathematical detail:** FIRST implementation attempt INCORRECTLY included -log(m(m-1)) in both log_q_fwd (split) and log_q_rev (merge). This created a massive asymmetry (m(m-1) ≈ 90 for m=10), causing all splits to be accepted and 0% K accuracy. The correct approach is that the seed density CANCELS and should NOT appear in the MH ratio.

**Branch/commit:** main, Round 3 changes.

---

### 13. Jain-Neal Restricted Gibbs Scans (Round 4 — CURRENT)

**What was tried:** After the initial sequential allocation (launch state), run `n_restricted_scans - 1` intermediate Gibbs sweeps to improve the allocation, then one final sweep produces the proposal with computable density. Only the final sweep's density enters the MH ratio; intermediate sweeps are "free" (Jain-Neal 2004).

For splits: launch → intermediate sweeps → final sweep (sample + density = log_q_fwd).
For merge reverse density: launch from merged cluster → intermediate sweeps → transition density from intermediate state to current allocation (log_q_rev).

The transition density for the merge reverse uses a "hybrid state" at each step: members already processed use the TARGET assignment, members not yet processed use the INTERMEDIATE state assignment. This correctly models the Gibbs sweep transition kernel.

**What works:** First brute-force PASS on well-separated dimer (KL 0.014, max ratio 1.48x). 3/4 full MCMC tests improved. K=1 test accuracy converged from 92% to 80% (matching 79.5% theory — R3 was biased toward K=1). K=4 improved from 76% to 84%.

**What doesn't change much:** Acceptance rates are nearly identical to Round 3 (split: 2-12%). The restricted Gibbs improves proposal QUALITY, not acceptance rates. Better proposals mean the MH correction is smaller, but the proposal density also increases, so these effects partially cancel.

**Limitation:** For small clusters (m=3, typical for N=6 K=2), there's only 1 non-seed member — restricted Gibbs can't improve the allocation beyond sequential. Benefit is stronger for larger clusters.

**Default:** n_restricted_scans=5 (4 intermediate + 1 final). Parameter available as kwarg in run_bagol, run_collapsed_chain, propose_split_merge!.

**Branch/commit:** main, Round 4 changes.

---

### 14. Informed Seed Selection for Merges (Round 5 — DEAD END)

**What was tried:** Replace uniform random seed selection in merge proposals with "informed" seeds — one seed from each sub-cluster (A and B). This eliminates ~50% of wasted proposals where both seeds come from the same original cluster, giving near-zero q_alloc_rev. The seed density changes from 1/(m(m-1)) (uniform) to 1/(n_a × n_b) (one-from-each).

**Why it failed:** The asymmetric seed density ratio m(m-1)/(n_a × n_b) enters the MH ratio as a 3-6× factor favoring splits. For typical cluster sizes (m=6, K=2 → n_a=n_b=3), m(m-1)/(n_a×n_b) = 30/9 ≈ 3.3. This overwhelms the spatial and partition terms in the MH ratio, causing catastrophic over-splitting. K-accuracy dropped from 80% to 16-32% on the K=2 test.

**Deeper issue:** In the RJMCMC bijection framework, the split and merge must use MATCHING seed selection mechanisms. Uniform-uniform cancels. Informed-informed would also cancel, but informed seed selection for SPLITS (picking seeds that are far apart) concentrates proposal density on good splits, which increases log_q_fwd without a matching increase in log_q_rev. Any asymmetry between split and merge seed mechanisms creates a multiplicative bias that scales with cluster size.

**What was learned:** Seed selection mechanisms must be symmetric between split and merge (same density, same mechanism). Asymmetric approaches that eliminate "bad" proposals on one side but not the other create density ratio factors that destroy K estimation. The ~50% wasted merge proposals are the price of correct detailed balance.

**Branch/commit:** Implemented and reverted within Round 5 (not committed).

---

### 15. Weakening DM Prior (γ < shape) to Fix Under-Splitting (Round 6 — DEAD END)

**What was tried:** Multi-prior brute-force enumeration comparing DM γ=2.0 (default), γ=1.0, γ=0.5, and uniform 1/S(N,K) partition priors. The hypothesis was that γ=shape=2 is "too strong" and a weaker γ would give better K recovery at d/σ=3.

**Why it's wrong:** The exact posterior under DM γ=2 gives the HIGHEST P(K_true) for the close dimer (d/σ=3). Lower γ shifts mass TOWARD K=1, not K=2. This is because:
1. DM γ=2 favors balanced partitions (n_k ≈ N/K) which have BETTER spatial fit (members near their emitter)
2. DM γ=0.5 favors unbalanced partitions (one large, one small cluster) which have WORSE spatial fit (the large cluster's mean is between emitters)
3. The DM concentration and spatial evidence are cooperative, not antagonistic

**Uniform partition (1/S(N,K)) catastrophically over-splits:** MAP-K=4 for both d/σ=3 and d/σ=10 dimers. Removing the DM penalty entirely reintroduces the combinatorial over-splitting from KB #1.

**What was learned:** The DM prior with γ=shape is the right choice. The under-splitting is a mixing problem (sampler can't cross the DM energy barrier), not a model problem (DM posterior itself gives correct answers). Adjusting γ is not a principled fix — it changes the correct target distribution for the worse.

**Key data (d/σ=3, N=6):** P_exact(K=2): DM γ=2→0.603, γ=1→0.599, γ=0.5→0.579, Uniform→0.249

**Branch/commit:** Analysis script `dev/prior_sensitivity.jl` added in Round 6. No sampler changes.

---

### 16. Hierarchical μ/shape Feedback Under Under-Splitting (Round 6 — OBSERVATION)

**What was observed:** In smlmsim_highdensity (true μ≈8.7, shape≈1.5), the hierarchical learner converged to μ=17.44, shape=21.92. Because the sampler under-splits (K too low), each inferred cluster has more locs → the learner infers higher μ and shape. This creates a secondary feedback loop: higher μ shifts the count model toward lower K, reinforcing under-splitting.

**Not a dead end per se** — fixing K-mixing would fix this. But it's a mechanism to be aware of: the hierarchical learner can mask or amplify mixing problems.

---

### 17. |ΔK|=1 Random Split/Merge (Round 7 — CURRENT)

**What was tried:** Replace count-model K sampling (which could propose |ΔK|>1) with random single-step proposals: 50/50 split/merge coin flip with boundary handling (K=1 always split, K≥N always merge). Since K is no longer proposed from π_count, the count-model ratio Δ_count = log P(N|K') - log P(N|K) enters the MH acceptance explicitly, along with a birth/death rate correction Δ_move_type for boundary cases.

**What works:**
1. Split acceptance doubled (2.7% → 6.7% on well-separated SM-only)
2. Close dimer max ratio improved (2.18x → 2.01x) — Δ_count helps drive K=1→2 splits
3. ESS improved ~12% (SM-only: 33171 → 37088)
4. Eliminated multi-step waste (~25% of proposals in R6 were |ΔK|>1 and almost never accepted)
5. Code simpler — no chain-of-splits/merges loops

**What doesn't change:** The fundamental DM energy barrier per split (-2.5 to -5) persists. Single-step split acceptance is still only 5-7%. Practical smlmsim recall unchanged (~59%).

**Trade-off:** The old count-model proposal was an independence sampler (global K jumps); the new one is a local random walk (K±1). This trades global K mobility for elimination of dead-on-arrival proposals. For the under-splitting problem (where K is stuck near K_mode), the local walk with Δ_count is a net win because the wasted multi-step proposals weren't helping anyway.

**Key data (SM-only well-separated):**
- R6: split 2.7%, merge 9.2%, ESS 33171, KL 0.020
- R7: split 6.7%, merge 6.7%, ESS 37088, KL 0.019

**Branch/commit:** main, Round 7 changes.

---

## Working Techniques

### A. Localization Mixture Prior

**What:** Replace P(θ) = 1/A with P(θ) = (1/N) Σⱼ N(θ; dⱼ, Σⱼ). Each localization contributes a Gaussian component.

**Why it works:** Eliminates the -log(A) Occam penalty. At d=0, the prior density at the cluster posterior mean is high (data supports itself), so splitting is not penalized. At d>0, the prior still peaks near data, so spatial information flows through.

**Performance:** Grid-based O(1) via LocmixGrid. Zero allocation, type-stable.

**Caveat:** Saddle-point approximation — evaluates prior at posterior mean, not integrated over posterior. May be inaccurate for very wide posteriors (n=1 cluster).

### B. NegBin Count Model (No Separate K Prior)

**What:** P(K|N) ∝ NegBin(N; K*shape, p). No Poisson prior on K.

**Why it works:** The NegBin likelihood alone regularizes K. Adding a Poisson prior on K over-penalizes large K.

**Key property:** Closed under summation. If n_j ~ NegBin(shape, p), then N = Σn_j ~ NegBin(K*shape, p).

### C. Gibbs Allocation Sweep (DM-weighted)

**What:** At fixed K, randomly reassign each loc to clusters proportional to `(n_{-i,k} + γ) × predictive`.

**Why it works:** Exact Gibbs targeting the DM-weighted posterior. The `(n_k + γ)` factor implements the Dirichlet-Multinomial partition prior, compensating for the combinatorial explosion of partitions at higher K. Without it, the sampler has an implicit uniform-per-label prior that strongly favors higher K (S(N,3) >> S(N,1)).

**Key detail:** Sole-occupant locs are skipped to maintain K. K changes only through split/merge.

### E. Sequential Predictive Allocation with Random Seeds + Restricted Gibbs (for RJMCMC)

**What:** For splits, randomly select two seeds from cluster members, then: (1) sequential allocation → launch state, (2) `n-1` intermediate restricted Gibbs sweeps, (3) one final restricted Gibbs sweep produces proposal + computable density. Seed selection density cancels via RJMCMC bijection.

For merge reverse density: same launch mechanism, then intermediate sweeps, then transition density from intermediate state to current allocation via hybrid-state Gibbs sweep evaluation.

**Why it works:** Sequential allocation provides a reasonable launch. Intermediate Gibbs sweeps improve it by allowing each member to see ALL other members (not just those processed earlier). The final sweep's density is the product of conditional probabilities — fully tractable. Intermediate sweeps don't enter the density (Jain-Neal 2004).

**Critical detail:** The seed density 1/(m(m-1)) must NOT be included in the MH ratio — it cancels between the split and merge auxiliary variables. Including it creates an m(m-1) ≈ 90 factor asymmetry that destroys K estimation.

**Performance note:** Acceptance rates don't change much because better proposals also have higher proposal density, which partially cancels in the MH ratio. The benefit is in proposal quality — accepted moves are better.

### F. |ΔK|=1 Random Split/Merge with Count-Model Ratio (Round 7)

**What:** Replace count-model K proposal (which sampled K_new from π_count and could propose |ΔK|>1) with random ±1 proposals. 50/50 split/merge, boundary-aware. MH includes Δ_count = log P(N|K') - log P(N|K) and Δ_move_type for boundary corrections.

**Why it works:** Eliminates multi-step proposals that compound the DM penalty and are almost never accepted. The count model enters the acceptance ratio directly, helping drive splits when the data supports higher K. Split acceptance doubled from ~3% to ~7%.

**Trade-off:** Local random walk instead of global independence sampler. Loses ability to jump multiple K steps at once, but those jumps were almost never accepted anyway.

### G. Birth/Death Moves (Round 8 — CURRENT)

**What:** Birth detaches a random non-sole-occupant loc as a singleton (K+1). Death absorbs a random singleton into the best-fit cluster via DM-weighted predictive (K-1). Standard MH with fully tractable proposal densities:
- q_birth = p_birth × 1/N_eligible
- q_death = p_death × 1/n_singletons × w(dest)/Σw where w(k) = (n_k+γ) × pred(i|k)

**Why it works:** The DM partition penalty per birth is ~-1.2 (vs -2.5 to -5 for split), halving the energy barrier. After a birth, Gibbs sweeps attract nearby locs to the new singleton, growing it into a proper cluster. This decomposes the monolithic split (propose full allocation in one shot) into incremental steps (birth creates seed → Gibbs grows it).

**Results:** Brute-force improved from 1/4 PASS to 3/4 PASS. Close dimer KL improved 2.8× (0.036→0.013), large dimer KL improved 15× (0.038→0.003). Birth acceptance 5-20% (varies by test), death acceptance 100%.

**Move mix:** 50% Gibbs / 25% split-merge / 25% birth-death. Split/merge still needed for balanced splits; birth/death supplements with cheap K-mobility.

**Limitation (resolved by Round 9 BD burst):** Single emitter test failed with n_bd_substeps=1 (KL=0.024, max ratio 1.97×). BD burst (n=5) fixes this by providing more K-transition attempts per iteration.

### H. BD Burst (Round 9 — CURRENT)

**What:** When the birth/death move is selected (25% of iterations), run `n_bd_substeps` (default 5) sequential BD attempts instead of 1. Each substep is independent MH with proper acceptance/rejection.

**Why it works:** For co-located emitters, no proposal weighting can improve birth acceptance — all births have identical acceptance probability (the locs are exchangeable). The only lever is more attempts. With 5 substeps, the effective K-transition rate increases 5×, the ESS for K increases proportionally, and the max ratio deviation decreases by ~√5.

**Key insight:** Codex review identified that targeted birth (inverse-predictive weighting) doesn't help co-located cases because (1) all locs have similar predictive, and (2) MH correction exactly compensates any proposal bias. The mathematically correct approach is simply more attempts.

**Results:** 4/4 brute-force PASS (was 3/4). Single emitter: 1.97×→1.40×. ESS improved 16-179%. Practical recall unchanged (~59%) because large partitions are bottlenecked by split/merge, not BD.

**Calibration:** n=3 gives 3/4 PASS (single emitter 1.76×, just above 1.65× threshold). n=5 gives comfortable 4/4 PASS (single emitter 1.40×). Cost: ~2× per outer iteration (BD is cheap relative to Gibbs/SM).

### D. Hierarchical μ/shape Learning

**What:** Global MH updates for count distribution parameters, pooling cluster sizes across all partitions.

**Why it works:** Adapts to the actual blink distribution in the data. Important when the user's prior on μ is wrong.

**Interaction:** μ and shape affect K proposals through the count model. Learning is synchronized across partitions at sync_interval.

---

### 18. DM/Polya Partition Prior: Correct Prior, Mixing Problem at Large N (Rounds 9-11)

**What was found:** The DM partition prior with γ=shape is provably the correct prior on assignment vectors z given the NegBin count model conditioned on N (see `docs/dm-polya-proof.md`). γ=shape is NOT a tuning knob — it is a mathematical consequence of the count model and must not be changed.

**Critical conceptual point (Codex, Round 10):** P(N|K) and the DM P(z|K,N) are NOT two independent forces fighting over K. They are a factorization of the same NegBin count model. After summing P_DM(z|K,N) over all allocations z at fixed K, the DM collapses back into P(N|K).

**Round 11 resolution:** The target IS correct. The marginal P(K|data) at d=0 was computed exactly by composition enumeration (K=1..6) and monotonically increases through K=6, consistent with Q-PAINT MAP K=8. The sampler at K≈3 is far below the target — a severe MIXING failure.

**Why specific-allocation scoring was misleading (Round 11 lesson):** K=4 gibbs-opt scores +26 above K=8 oracle for a SPECIFIC allocation (the MAP at each K). But the MARGINAL sums over ALL allocations weighted by DM. The entropy at higher K (many more allocations) compensates for the per-allocation DM penalty. Comparing MAP allocations across K does NOT give P(K|data).

**Partition-dependent Occam residual (Codex correction):** The partition-dependent part of the spatial ML at d=0 (equal σ) is `-Σ_k log(n_k)`, NOT `-K log(2π) - Σ log(n_k)`. The `log(2π)` terms cancel because `Σn_k = N` is fixed. The `-Σlog(n_k)` creates a per-allocation energy barrier for K-increasing moves, but this is compensated in the MARGINAL by the DM's allocation entropy at higher K.

**Mixing failure mechanism:** Each K-increasing move (birth, split) faces the per-allocation DM penalty (≈-6 to -30 depending on cluster sizes). Even though the marginal favors high K, the chain's transition moves see the per-allocation landscape, not the marginal. The chain oscillates locally (K≈3-6) but cannot make sustained progress toward the target peak (K≈7-8). This failure scales with N — at N=6 (brute-force), the barriers are small enough for BD burst to overcome; at N=40, they are insurmountable with current moves.

**What was learned:**
1. Do NOT conclude "target is wrong" from specific-allocation scoring. Always compute the marginal.
2. Convergence from both K=1 and K=8 to the same K does NOT prove the target peaks there — it can indicate a metastable mixing trap accessible from both directions.
3. The brute-force at N=6 correctly showed the target is OK. Trust it.
4. Saddle-point locmix is fine (<0.02/cluster). The approximation is not the issue.

---

## Theoretical Limits

### Co-Located MAP-N Accuracy

For co-located emitters with count data alone:
- MAP-K = argmax_K P(N|K) where N ~ NegBin(K*shape, p)
- P(correct) ≈ μ/(2σ_N) where σ_N = √(K*μ*(μ+shape)/shape)

Example (shape=5, μ=20): K=1→84%, K=2→53%, K=4→39%, K=8→28%.

This is the information-theoretic ceiling. No algorithm can exceed it for co-located emitters with count data alone. Use as pass/fail threshold: sampler should achieve ≥80% of theoretical maximum.

### MAP Estimator Bias

MAP (mode) is biased low relative to posterior mean for discrete distributions with K-dependent variance. This is inherent to argmax, not a sampler defect.

---

## Key Design Principles

1. **BaGoL ≥ Q-PAINT always.** At d=0, reduce to count-only. At d>0, spatial helps.
2. **Area invariance.** No -log(A) terms anywhere. Locmix prior ensures this.
3. **Decouple K mixing from spatial mixing.** Count model handles K; spatial MH only corrects fit quality.
4. **Validate numerically before recording claims.** Run brute_force_enumeration and detailed_balance_check.
5. **Let diagnostics lead.** The bias was found by brute-force, not theory. Trust the numbers.

# Diagnostics Module — Goals, Math, and Procedures

This module provides tools to answer four questions about any MCMC sampler
operating on the partition space:

1. **Is detailed balance provably correct?**
2. **Does the kernel preserve the target distribution?**
3. **Is the chain irreducible?**
4. **How efficiently does it mix?**

Plus supporting tools for count-model baselines and partition comparison.

### State Space Convention

The sampler operates on **labeled allocations**: vectors z in {1,...,M}^N
where z_i is the slot index of localization i. The target density pi(z) is
**label-invariant**: pi(z) = pi(sigma(z)) for any label permutation sigma.

For diagnostics, we work with **canonical partitions**: relabelings where
labels appear in order of first occurrence (e.g., [2,1,2,1] -> [1,2,1,2]).
Each canonical partition with K clusters corresponds to exactly K! distinct
labeled allocations, all with identical target density and (for a
label-symmetric kernel) identical transition behavior.

This means:
- `log_target(td, z, ...)` evaluates pi for a single labeled allocation
- When comparing to canonicalized MCMC samples, the exact posterior must
  weight each canonical partition by K!: P_canonical(z) ~ K! * pi(z)
- The K! correction assumes (a) label-invariant target, (b) label-symmetric
  kernel, (c) the slot-reuse mechanism doesn't break this symmetry. If any
  of these fail, the K! factor is wrong and cross-K comparisons are invalid.

### Shared oracle risk

Every test in sections 2-6 compares the sampler against `log_target` or
an exact posterior derived from it. If `log_target` has a bug, all of
these tests validate the sampler against the wrong distribution and can
PASS while the sampler is wrong. This is the single most dangerous
failure mode of the diagnostic suite. It is not detectable from within
the suite itself. Mitigate by independently unit-testing `log_target`:
normalization on tiny cases, known analytic limits, comparison between
target variants (DecoupledTarget vs MFMTarget should agree on limiting
cases), and label-symmetry spot checks.

---

## 1. Target Density Abstraction (`target.jl`)

### Goal
Decouple diagnostic tools from specific posterior formulations.

### Targets

**DecoupledTarget** (current `locmix-grid` branch):

    log pi(z) = log P_count(N | K, mu, alpha) + sum_k log ML_locmix(D_k)

- P_count(N | K) = NegBin(N; K*alpha, alpha/(alpha+mu))
- ML_locmix(D_k) = (1/N) sum_j ML_flat(D_k union {virtual_j})
- Flat allocation prior: all labeled Z with K groups equally likely
- Support restriction: N >= K (every cluster has >= 1 localization)

**MFMTarget** (Mixture of Finite Mixtures):

    log pi(z) = log P_count(N | K) + log P_partition(z | K) + sum_k log ML_locmix(D_k)

P_partition is the Dirichlet-Multinomial for labeled allocations:

    P(z | K, gamma) = Gamma(K*gamma) / [Gamma(gamma)^K * Gamma(N+K*gamma)] * prod_k Gamma(n_k + gamma)

with gamma = alpha (NegBin shape). This is for one labeled allocation, not counts.

**UniformPriorTarget**:

    log pi(z) = log P_count(N | K) + sum_k log ML_uniform(D_k)

ML_uniform includes -log|R| per cluster.

---

## 2. Detailed Balance Verification (`detailed_balance.jl`)

### Goal
For samplers with tractable proposal densities, **prove** DB holds.

### Math

    Delta_DB = [log pi(z') + log q(z'->z)] - [log pi(z) + log q(z->z')]

If |Delta_DB - Delta_code| < tol for the sampler's computed acceptance ratio,
DB holds for this transition.

### Procedure
1. `generate_split_transitions(N, K_max)` -- enumerate all K->K+1 canonical splits
2. For each (z_from, z_to): compute log pi, optionally check against code's ratio
3. Default mode (log_q = 0): reports target density ratio only, **no verdict**
4. Full mode (log_q + log_ratio_code): checks |Delta_DB - Delta_code| < tol, PASS/FAIL

### Interpreting results

**Default mode output is NOT a DB check.** It only shows that `log_target`
gives self-consistent density ratios across partition pairs. This tells you
nothing about whether the sampler's acceptance ratio is correct.

**Full mode PASS on auto-generated splits is weak evidence.** The generator
only covers canonical K->K+1 splits. It misses merge transitions, same-K
Gibbs reallocations, multi-step jumps, relabel-sensitive code paths, and
data-dependent edge cases. A PASS on auto-generated cases is not evidence
for the full sampler unless those are the only move types and every code
path has been exercised. For a sampler with multiple move types, you must
supply transitions covering every move type and every proposal branch.

**FAIL means:** The sampler's acceptance ratio does not match the theoretical
DB ratio for this transition. Either the sampler has a bug, or `log_target`
has a bug. Both must be investigated. There is no benign interpretation.

---

## 3. Kernel Invariance Test (`enumeration.jl`)

### Goal
Empirical test: does the MCMC kernel preserve the target distribution?

### Math

    P[i,j] = P(z_{t+1} = j | z_t = i)

estimated from n_steps_per_state one-step chains from each state.

    ||pi^T P - pi^T||_TV = (1/2) sum_j |sum_i pi_i P_{ij} - pi_j|

### Interpreting results

**TV < 0.05:** The estimated transition matrix approximately preserves
the target. This is evidence, not proof. The transition matrix is estimated
from finite samples; with n_steps_per_state = 200, per-row noise is of order
1/sqrt(200) ~ 0.07. A kernel with small systematic bias (a few percent per
transition) can pass. PASS also cannot detect a shared bug between the
sampler and `log_target` -- if both are wrong in the same way, the test
agrees with itself.

**TV > 0.05:** The tested sampler/oracle combination failed the invariance
check. Treat the sampler as unfit until you explain the discrepancy. The
failure could be: broken kernel, broken `log_target`, incorrect K!
correction, truncation from overflow, or (less likely) insufficient
n_steps_per_state for the estimation noise to average out. Do not dismiss
this as noise without ruling out the other causes.

**Irreducibility (support graph connected):** This means the sampled
one-step transition graph is connected. It does NOT mean every state was
reached from every other state. It means that in the sampled edges, there
exists a path. Rare transitions (probability << 1/n_steps_per_state)
will be missing from the sampled graph. A disconnected support graph is
strong evidence of irreducibility failure. A connected support graph is
weak evidence of irreducibility.

**Any nonzero overflow compromises the test.** The transition matrix rows
are renormalized after dropping K > K_max transitions. This conditions on
a truncated kernel. If the excluded transitions are where the kernel
misbehaves, the test misses them. The 5% threshold is a pragmatic stop
sign, not a safety boundary. Even 1% overflow concentrated in high-pi
states can materially bias the invariance check.

**This test only covers the z-marginal kernel** with fixed (mu, shape). It
says nothing about hierarchical updates. A bug in the mu/shape MH kernel
will not be detected.

---

## 4. Brute-Force Enumeration Test (`enumeration.jl`)

### Goal
For small systems (N <= 8): does the chain's long-run distribution match
the exact posterior?

### Math

    P_canonical(z) ~ K! * pi(z)

Compare via KL(exact || empirical), TV distance, K-marginals.

### Interpreting results

**PASS (KL_marginal < 0.05, TV < 0.10):** The chain's empirical frequencies
are consistent with the exact posterior for this specific test configuration.
This does not mean the sampler is correct. These thresholds are loose
(TV < 0.10 is ~7x the iid noise floor for 100K samples over ~100 states).
A 5% bias in P(K) spread across many partitions can pass. PASS also cannot
detect shared oracle bugs (see "Shared oracle risk" above).

**PASS on N=4 does NOT mean the sampler works on N=20.** Small-N tests
exercise a tiny fraction of the move space. Bugs triggered by large
clusters, high K, specific spatial configurations, or rare proposal
branches will not be caught.

**FAIL:** The tested run does not match the exact posterior. Either: the
sampler is wrong, the exact posterior is wrong (bug in `log_target` or K!
correction), or the run was too short relative to the chain's mixing time.
Do not assume "run longer" is the fix. First check the exact posterior
independently, then check overflow, then check ESS. If ESS is reasonable
and the exact posterior is correct, the sampler is broken.

**KL_marginal failing while per-partition TV is acceptable:** The sampler
gets within-K relative frequencies right but cross-K balance wrong. This
points to a bug in the K-changing move (split/merge), not in the Gibbs
sweep.

---

## 5. Reachability Test (`enumeration.jl`)

### Goal
Does the chain visit all canonical partitions with nontrivial probability?

### Interpreting results

**PASS (all reachable states visited):** The chain reached every partition
with P >= threshold from this initialization in this many iterations. This
is weak evidence of irreducibility. A chain that reaches state A from
initialization but cannot reach A from state B is not irreducible.

**FAIL (unvisited states):** The chain did not visit a state with nontrivial
exact probability. Posterior estimates from this run are not trustworthy for
any quantity that depends on the unvisited partition mass. Possible causes:
- The chain is not irreducible (structural failure)
- The chain is irreducible but the state's mixing time exceeds the run length
- The initialization is far from the missing state

Do not assume "run longer" is sufficient. First check the kernel invariance
test's support graph. If the support graph is disconnected, the chain is
structurally unable to reach those states and no amount of runtime will fix it.

---

## 6. Combined Stationarity Test (`enumeration.jl`)

### Interpreting results

**Overall PASS:** For this one small-N configuration, the chain appears to
sample from the right distribution, visit expected states, and approximately
preserve pi under one-step transitions. This does not validate the sampler.
Specifically:
- It only checks the z-marginal with fixed (mu, shape). Hierarchical
  parameter updates are not tested.
- It can pass against a wrong target if `log_target` has the same bug as
  the sampler.
- It covers one (N, mu, shape, spatial layout) configuration. A sampler
  that passes N=3 and fails N=6 has a bug; you just didn't see it.

You need PASS across multiple configurations before concluding the
z-marginal kernel is likely correct. Even then, the hierarchical kernel
and numerical behavior at scale are untested.

---

## 7. Chain Mixing Diagnostics (`chain_diagnostics.jl`)

### Goal
Assess mixing efficiency. These tools say nothing about correctness.
Good mixing of a wrong distribution is useless.

### Diagnostics

**ESS (batch means):**

    ESS = n * var_sample / (batch_size * var_batch_means)

**ESS (initial positive sequence, Geyer 1992):**

    ESS = n / tau,  tau = 1 + 2 sum_m [rho(2m+1) + rho(2m+2)]

**ACF (biased, PSD-guaranteed):**

    rho(lag) = [sum_{t=1}^{n-lag} (x_t - mu)(x_{t+lag} - mu)] / (n * sigma^2)

**R-hat:** Classic + split (2m half-chains).

**Indicator ESS:** ESS on 1[K=k] for MAP-K +/- 2 neighbors.

**Run-length test:** Heuristic.

### Interpreting results

**ESS is on the K trace only.** High ESS for K does not mean the partition
structure is well-explored. The chain can mix well in K while being trapped
among very different spatial allocations with the same K value. This
within-K metastability is common in partition samplers and will not be
detected by any K-summary diagnostic. Indicator ESS helps (catches cases
where P(K=k) estimates are noisy) but does not detect within-K trapping.

**R-hat < 1.1 is a coarse screening threshold, not a convergence standard.**
On a discrete, low-dimensional summary like K, R-hat can look excellent
while the underlying partition structure is badly mixed. If all chains are
stuck in the same wrong mode, R-hat will be near 1. Current best practice
(Vehtari et al. 2021) uses 1.01 as a stricter threshold. The 1.1 threshold
used here will miss moderate convergence failures.

**Split R-hat >> classic R-hat:** The chains are non-stationary (trending).
This usually means burn-in was too short. Do not interpret any downstream
results from this run.

**Run-length test verdict is a heuristic.** The longest chain is the
reference, not ground truth. If the longest chain is biased, every
comparison inherits that bias. `:mixing` means KL to the reference decreases
with length. `:systematic` means it doesn't. Neither verdict is reliable
without independent verification of the reference. Use `run_enumeration_test`
for correctness; use this only as a first-pass sanity check.

---

## 8. Count Model Analysis (`count_model.jl`)

### Goal
Q-PAINT baseline: best K estimation using count information alone.

### Math

**Posterior:** P(K | N) ~ NegBin(N; K*alpha, alpha/(alpha+mu)), uniform prior on K.

**Recovery rate:**

    P(correct) = sum_{N=1}^{N_max} P(N|K_true) * 1[MAP_K(N) = K_true] / (1 - P(N=0))

**Confusion matrix:** C[i,j] = P(MAP_K = j | K_true = i).

**Gaussian approximation** (heuristic):

    P(correct) ~ erf(mu / (2*sigma_N*sqrt(2))),  sigma_N = sqrt(K*mu*(1 + mu/alpha))

### Interpreting results

**These are theoretical limits, not performance measures.** The Q-PAINT
recovery rate is the best K-estimation possible without spatial information.
If BaGoL's K-accuracy is below the Q-PAINT rate for co-located emitters,
the spatial model is actively making things worse.

**The Gaussian approximation underestimates.** At K=1, alpha=1 it's 35% low.
Use the exact recovery rate, not the approximation.

**Confusion matrix off-diagonals show systematic bias direction.** If
C[2,1] >> C[2,3], the count model underestimates K=2 as K=1. These errors
persist even with a perfect sampler.

---

## 9. Partition Comparison Metrics (`partition_metrics.jl`)

### Math

**VI (Meila 2007):**

    VI(z1, z2) = H(z1|z2) + H(z2|z1)

- overseg = H(z1|z2): estimate has finer structure than oracle
  (splits what oracle keeps together)
- underseg = H(z2|z1): estimate has coarser structure than oracle
  (merges what oracle keeps separate)
- Normalized: vi_total / log(N) in [0, 1]

**EPL:** EPL(a) = (1/T) sum_t L(a, Z^(t)) with L = VI total.

### Interpreting results

**VI = 0 means perfect agreement.** Use normalized VI to compare across N.

**overseg > underseg:** The estimator splits too aggressively (too many
small clusters). This happens when mu is too low, the spatial prior is
too tight, or the count model overestimates K.

**underseg > overseg:** The estimator merges too aggressively (too few
large clusters). This happens when mu is too high or the spatial prior
is too loose.

**EPL(Dahl) > EPL(Oracle):** The Dahl consensus is a worse posterior
summary than the ground truth partition. This can mean: the sampler is
not exploring the posterior well (missed modes), the posterior genuinely
disagrees with the oracle (model misspecification), or there is a labeling
mismatch between oracle and posterior. Do not automatically conclude "sampler
bug" -- check whether the model and oracle are compatible first.

**Large regret (EPL(Dahl) - EPL(best)):** Dahl is a poor summary of the
posterior samples. The posterior may be multimodal or diffuse. Consider PSM
or VI-greedy estimators instead.

---

## Test Hierarchy (for evaluating a new sampler)

**Phase 0: Validate the oracle independently**
- Unit-test `log_target`: check normalization sums to 1 on tiny N,
  verify known analytic limits, confirm label symmetry (pi(z) = pi(sigma(z))
  for random permutations sigma)
- Verify K! correction: for N=3 K_max=2, confirm K! * pi(z) yields the
  same distribution as direct enumeration over labeled allocations
- Check canonicalization: verify that `canonicalize` is a proper equivalence
  relation (idempotent, groups all relabelings)
- Compare target variants on shared limiting cases (DecoupledTarget and
  MFMTarget should agree when gamma -> infinity)

Without Phase 0, every subsequent phase can pass while validating against
the wrong distribution.

**Phase 1: Analytic DB verification** (cheap, local; skip if proposals intractable)
- `check_detailed_balance` with actual proposal densities and the sampler's
  acceptance ratio, covering every move type and proposal branch
- This is only possible for samplers with fully computable Q(z->z')
- Auto-generated split transitions alone are not sufficient

**Phase 2: Finite-state empirical validation** (expensive, global)
- `run_stationarity_test` across multiple configurations:
  - Vary N (3, 4, 5, 6), mu (5, 10, 20), shape (1, 2, 5)
  - Vary spatial layout (co-located, well-separated, overlapping)
  - A sampler that passes some configurations and fails others has a bug
  - A sampler that passes all tested configurations has survived these tests;
    it is not proven correct

**Phase 3: Label symmetry and canonicalization**
- Verify the kernel is label-symmetric: from the same canonical partition,
  run many one-step transitions, canonicalize results, check that the
  distribution is the same regardless of which labeled allocation was used
  as the starting point
- Verify slot-reuse does not break symmetry: from z=[1,1,3,3] (gap at slot 2)
  vs z=[1,1,2,2], check identical canonical transition distributions

**Phase 4: Hierarchical kernel validation**
- The z-marginal tests in Phase 2 hold (mu, shape) fixed
- If the sampler updates mu and shape, those MH kernels need separate
  validation: fixed-z chains that update only mu/shape, checking that
  the marginal mu/shape posterior matches the analytic conditional

**Phase 5: Efficiency assessment** (any N)
- `run_mixing_test` -- ESS, R-hat, indicator ESS
- These do not tell you if the sampler is correct
- Poor mixing is a problem; good mixing is not a guarantee

**Phase 6: Scale and stress testing**
- Run on N=50, 100, 500 with known ground truth
- Test extreme parameters: shape=0.5 (overdispersed), shape=50 (near-Poisson),
  mu=1 (few locs per emitter), mu=100 (many locs)
- Test pathological layouts: all locs at same position, two very close
  clusters, one distant outlier
- Check for numerical issues: NaN/Inf in acceptance ratios, negative
  log-likelihoods, overflow in logsumexp

**Phase 7: Baselines and quality**
- Count model baselines -- theoretical floor BaGoL must match
- `partition_diagnostics` -- VI, EPL, regret vs ground truth

### What this hierarchy cannot do

No finite set of diagnostic tests can prove an MCMC sampler is correct for
all inputs. The tests validate on small state spaces and specific
configurations. Bugs that manifest only at scale, in rare code paths, or
for untested parameter combinations will not be caught.

The strongest achievable guarantee: Phase 0 (oracle correct) + Phase 1
(DB proven for all transitions in all move types) + Phase 2 (empirical
invariance confirmed across multiple configurations) + Phase 3 (label
symmetry verified) + Phase 4 (hierarchical kernel validated). This still
does not cover numerical instability at scale or code paths not exercised
by the test configurations.

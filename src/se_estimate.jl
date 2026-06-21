# ============================================================================
# estimate_se_adjust — data-driven per-blink systematic-error τ finder
# ============================================================================
#
# The "find" companion to `apply_se_adjust` / `run_bagol(; se_adjust)` (the
# "apply" side). Estimates the per-blink systematic localization error τ — the
# value you pass as `se_adjust` — directly FROM DATA: no ground truth, no
# simulation, blink-count-free, identity-free.
#
# PRINCIPLE (distribution comparison, NOT MLE):
#   At the true τ and the correct grouping, the within-BaGoL-group scaled
#   neighbor distances
#       z = d / sqrt(σ_a² + σ_b² + 2τ²)        (d = |y_a − y_b|)
#   are Rayleigh(1)  (CDF 1 − exp(−z²/2)).  Pick the τ whose z best matches
#   Rayleigh(1) by the Kolmogorov–Smirnov statistic.
#
# ADAPTIVE OVER-MERGE DESCENT (Keith + Codex, 2026-06-20):
#   E-step  group the localizations with BaGoL at the current τ (expensive).
#   M-step  on that FIXED grouping, the τ best matching Rayleigh(1) — the
#           unscaled distances d are fixed, so this is a cheap 1-D KS search
#           (no BaGoL), right-biased (see `_mstep_rb`).
#   Start deliberately OVER-merged (`g_start`, a large se_adjust → the full
#   same-molecule spread, no truncation, only catchable contamination), then
#   JUMP to the M-step estimate each step (g ← m). Approaching the truth FROM
#   ABOVE means no under-group truncation, so it converges without the ascending
#   creep — in ~2–5 BaGoL runs. Stop when the grouping is self-consistent
#   (|m − g| ≤ tol). KS(τ) is not provably unimodal (discrete grouping + BaGoL
#   stochasticity), so a bracketing search (golden-section) is unsafe;
#   descend-from-above + the right-biased tie rule is the safe family.
#
# Method developed and validated in SMLMClustering dev scratch (nanoruler
# τ = 0/4/8 nm → recovers 0.3/3.8/8.5 nm; circularity refuted via re-grouped /
# frozen-group / ground-truth-label KS curves all minimizing at the true τ;
# Codex + agents reviewed). Ported here as the in-package finder.
#
# UNITS: SMLMBaGoL works in μm, but the finder's validated numerics live in nm
# (the grids/tolerances are nm-natural). So the internals compute in nm and the
# PUBLIC boundary is μm with `_um`-suffixed names — `tau_hat_um` etc. feed
# straight into `run_bagol(; se_adjust = tau_hat_um)`.
#
# GROUPING backends (`grouping` kwarg):
#   :mapn_proxy  KDTree of the MAP-N emitters + nearest-emitter assignment of
#                each localization. External proxy (how it ran outside BaGoL);
#                can truncate a molecule's largest within-group displacements
#                and steal locs across emitters in crowded regions. Stage-1
#                default; marked experimental.
#   :dahl        BaGoL's actual Dahl-consensus per-loc assignment (more
#                faithful; stage 2 — see plumbing in rjmcmc.jl). Not yet wired.
# ============================================================================

# Pack emitter xy into a 2×N matrix (μm) for KDTree / distance work.
function _emitter_xy(ems)
    M = Matrix{Float64}(undef, 2, length(ems))
    @inbounds for (j, e) in enumerate(ems)
        M[1, j] = e.x
        M[2, j] = e.y
    end
    M
end

# Group the input localizations `em` given the BaGoL MAP-N emitters `EM`.
# Returns a Vector of localization-index groups.
function _assign_groups(em, EM, grouping::Symbol)
    if grouping === :mapn_proxy
        etree = KDTree(_emitter_xy(EM))
        idx, _ = nn(etree, _emitter_xy(em))      # nearest MAP-N emitter per loc
        grp = Dict{Int, Vector{Int}}()
        for (i, a) in enumerate(idx)
            push!(get!(grp, a, Int[]), i)
        end
        return collect(values(grp))
    elseif grouping === :dahl
        throw(ArgumentError("grouping=:dahl (BaGoL Dahl-consensus assignment) is stage 2 — " *
                            "the per-loc assignment plumbing in run_bagol is not yet wired. Use :mapn_proxy."))
    else
        throw(ArgumentError("grouping must be :mapn_proxy or :dahl (got :$grouping)"))
    end
end

# E-step: group at τ (nm) with BaGoL, return within-group UNSCALED distances
# d (nm), pair variance s2 = σ_a² + σ_b² (nm²), and pair-midpoint positions (μm)
# for the spatial-block bootstrap. `se_adjust` is force-applied so a candidate τ
# is never skipped by the already-σ-corrected guard; posterior image is disabled
# (these runs are throwaway).
function _group_distances(smld, tau_nm; n_iterations, burn_in, grouping, bagol_kwargs)
    t = tau_nm / 1000                            # nm → μm
    res, _ = run_bagol(smld; se_adjust = (t, t), force_se_adjust = true,
                       n_iterations = n_iterations, burn_in = burn_in,
                       posterior_pixel_size = 0.0, verbose = false, bagol_kwargs...)
    EM = res.emitters
    isempty(EM) && return (Float64[], Float64[], NTuple{2, Float64}[], 0)
    em = smld.emitters
    groups = _assign_groups(em, EM, grouping)
    d = Float64[]; s2 = Float64[]; pos = NTuple{2, Float64}[]
    for ks in groups
        length(ks) < 2 && continue
        for a in 1:length(ks) - 1, b in a + 1:length(ks)
            ea = em[ks[a]]; eb = em[ks[b]]
            push!(d, hypot(ea.x - eb.x, ea.y - eb.y) * 1000)     # μm → nm
            push!(s2, (ea.σ_x^2 + eb.σ_x^2) * 1e6)               # μm² → nm²
            push!(pos, ((ea.x + eb.x) / 2, (ea.y + eb.y) / 2))   # μm
        end
    end
    d, s2, pos, length(EM)
end

# KS of the scaled-distance distribution against Rayleigh(1), at assumed τ (nm),
# over an index set (cheap — the M-step kernel).
function _ks_rayleigh1(d, s2, tau_nm, idxs)
    isempty(idxs) && return NaN
    t2 = tau_nm^2
    z = sort!([d[k] / sqrt(s2[k] + 2t2) for k in idxs])
    n = length(z)
    maximum(abs(i / n - (1 - exp(-z[i]^2 / 2))) for i in 1:n)
end

# M-step: τ minimizing the KS over a fine grid (nm), no BaGoL. The basic kernel.
_mstep(d, s2, grid_nm, idxs) = grid_nm[argmin([_ks_rayleigh1(d, s2, τ, idxs) for τ in grid_nm])]

# RIGHT-BIASED M-step (Codex's tie rule): among τ within `ks_noise` of the min
# KS, take the LARGEST — preserves the same-molecule spread and avoids the
# under-group truncation that causes the ascending creep. This is the M-step the
# descent and the bootstrap use.
function _mstep_rb(d, s2, grid_nm, idxs; ks_noise = 0.01)
    ks = [_ks_rayleigh1(d, s2, τ, idxs) for τ in grid_nm]
    kmin = minimum(ks)
    maximum(grid_nm[i] for i in eachindex(grid_nm) if ks[i] ≤ kmin + ks_noise)
end

# UPPER-TAIL KS of z vs Rayleigh(1), over the [qlo,qhi] quantile band ONLY. The
# FULL KS is faked low by tight over-split groups (they look ~Rayleigh at small τ —
# that IS the collapse); the tail can't be faked: over-split TRUNCATES it (light),
# over-merge CONTAMINATES it (heavy), truth matches. So band-tail KS is U-shaped
# along the descent with its MIN at truth. (Keith + smlmclustering, 2026-06-20.)
function _ks_rayleigh1_tail(d, s2, tau_nm, idxs; qlo = 0.85, qhi = 0.99)
    isempty(idxs) && return NaN
    t2 = tau_nm^2
    z = sort!([d[k] / sqrt(s2[k] + 2t2) for k in idxs])
    n = length(z)
    lo = max(1, ceil(Int, qlo * n)); hi = min(n, floor(Int, qhi * n))
    lo ≥ hi && return NaN
    maximum(abs(i / n - (1 - exp(-z[i]^2 / 2))) for i in lo:hi)
end

# SIGNED upper-tail ratio: mean empirical z over [qlo,qhi] vs the Rayleigh(1)
# quantile there. r_tail ≈ 1 at truth, < 1 OVER-SPLIT (tail truncated/light),
# > 1 OVER-MERGE (tail heavy). Gives the crossing DIRECTION — stop at r_tail=1,
# robust to the band-KS min's noise.
function _r_tail(d, s2, tau_nm, idxs; qlo = 0.97, qhi = 0.99)
    isempty(idxs) && return NaN
    t2 = tau_nm^2
    z = sort!([d[k] / sqrt(s2[k] + 2t2) for k in idxs])
    n = length(z)
    lo = max(1, ceil(Int, qlo * n)); hi = min(n, floor(Int, qhi * n))
    lo > hi && return NaN
    emp = sum(@view z[lo:hi]) / (hi - lo + 1)
    p = (qlo + qhi) / 2
    emp / sqrt(-2 * log(1 - p))
end

# Spatial-block bootstrap: resample `block_um` tiles of pair-midpoints (the
# independent unit — pairs within a group are NOT independent) → a resampled
# index list for the M-step.
function _block_indices(pos, block_um, rng)
    isempty(pos) && return Int[]
    x0 = minimum(p[1] for p in pos); y0 = minimum(p[2] for p in pos)
    tile(p) = (floor(Int, (p[1] - x0) / block_um), floor(Int, (p[2] - y0) / block_um))
    tiles = Dict{Tuple{Int, Int}, Vector{Int}}()
    for (i, p) in enumerate(pos)
        push!(get!(tiles, tile(p), Int[]), i)
    end
    kv = collect(keys(tiles)); out = Int[]
    for _ in 1:length(kv)
        append!(out, tiles[kv[rand(rng, 1:length(kv))]])
    end
    out
end

"""
    estimate_se_adjust(smld::SMLMData.SMLD; kwargs...) -> NamedTuple

Estimate the per-blink systematic localization error τ (the `se_adjust` value)
directly from data, by distribution comparison to Rayleigh(1) via an adaptive
over-merge descent. The "find" companion to [`apply_se_adjust`](@ref) /
`run_bagol(; se_adjust)`.

`smld` must hold **raw-σ** localizations (e.g. frame-connected blinks). τ is the
correction to *find*, so an already-σ-corrected input
(`metadata["sigma_corrected"]==true`) is rejected.

# Returns
A `NamedTuple` with τ values in **μm**:
- `tau_hat_um` — point estimate (feeds `run_bagol(; se_adjust = tau_hat_um)`)
- `ci_lo_um`, `ci_hi_um` — spatial-block-bootstrap 95% CI
- `ks_at_hat` — KS-to-Rayleigh(1) at the estimate (lower = better fit)
- `n_bagol` — number of BaGoL E-steps run (the descent length)
- `path_um` — the descent trajectory as `(g, m)` pairs (grouping τ, M-step τ)
- `grouping` — the grouping backend used
- `diagnostics` — `nothing`, or (with `return_diagnostics=true`) the arrays the
  `BaGoLMakieExt` plots consume, so the extension never re-runs BaGoL.

# Keyword arguments
- `g_start_um = 0.012` — over-merged starting τ for the descent (μm)
- `stop_tol_um = 5e-4` — self-consistency tolerance `|m−g|` to stop (μm)
- `max_steps = 6` — descent step cap
- `ks_noise = 0.01` — KS tie band for the right-biased M-step
- `grid_um = 0.0:0.0001:0.014` — fine M-step τ grid (μm)
- `n_iterations = 2000`, `burn_in = 500` — per-E-step BaGoL chain length
- `n_boot = 200` — bootstrap resamples for the CI
- `block_um = 1.0` — spatial-block tile size (μm)
- `seed = 1` — bootstrap RNG seed (a local RNG; BaGoL's own E-step stochasticity
  is only controlled as far as the sampler permits)
- `grouping = :mapn_proxy` — `:mapn_proxy` (stage 1) or `:dahl` (stage 2)
- `return_diagnostics = false` — attach the diagnostic arrays
- `bagol_kwargs...` — forwarded to `run_bagol` (e.g. `partition_sigma`, `shape`,
  `allocation_model`). **Reserved** (managed by the finder, do not pass):
  `se_adjust`, `force_se_adjust`, `posterior_pixel_size`, `verbose`.

!!! warning "Experimental grouping"
    The default `:mapn_proxy` backend is an external nearest-emitter proxy that
    can bias τ̂ downward in crowded regions. The faithful `:dahl` backend (stage
    2) uses BaGoL's actual assignment posterior.
"""
function estimate_se_adjust(smld::SMLMData.SMLD;
        g_start_um = 0.012,
        stop_tol_um::Float64 = 5e-4,
        max_steps::Int = 6,
        ks_noise::Float64 = 0.01,
        grid_um = 0.0:0.0001:0.014,
        n_iterations::Int = 2000,
        burn_in::Int = 500,
        n_boot::Int = 200,
        block_um::Float64 = 1.0,
        seed::Int = 1,
        grouping::Symbol = :mapn_proxy,
        return_diagnostics::Bool = false,
        bagol_kwargs...)

    # τ is the σ-correction to FIND; finding it on already-corrected σ is ill-posed.
    md = hasproperty(smld, :metadata) ? smld.metadata : Dict{String, Any}()
    if get(md, "sigma_corrected", false) == true
        throw(ArgumentError("estimate_se_adjust expects raw-σ localizations, but the input SMLD is " *
                            "already σ-corrected (metadata sigma_corrected=true). τ is the correction " *
                            "to find; estimating it on corrected σ is ill-posed."))
    end

    # Fail fast on grouping (before any BaGoL run); :dahl is stage 2 (see _assign_groups).
    if grouping === :dahl
        throw(ArgumentError("grouping=:dahl (BaGoL Dahl-consensus assignment) is stage 2 — " *
                            "the per-loc assignment plumbing in run_bagol is not yet wired. Use :mapn_proxy."))
    elseif grouping !== :mapn_proxy
        throw(ArgumentError("grouping must be :mapn_proxy or :dahl (got :$grouping)"))
    end

    # Work in nm internally (preserves the validated numerics).
    grid     = 1000 .* collect(Float64, grid_um)
    g        = 1000 * g_start_um                 # nm — start over-merged
    stop_tol = 1000 * stop_tol_um

    # ADAPTIVE OVER-MERGE DESCENT: jump to the right-biased M-step each step
    # (g ← m); descent from above ⇒ no under-group truncation ⇒ no creep.
    n_bagol = 0; τ = g
    d = Float64[]; s2 = Float64[]; pos = NTuple{2, Float64}[]
    path = Tuple{Float64, Float64}[]
    # Per-descent-g instrumentation. No sim reproduces the real-data collapse, so
    # the REAL ruler is the only gate: log every candidate stop-signal at τ=g and
    # pick the winner empirically — ks_full (faked low by over-split), ks_tail +
    # r_tail (over-split-proof: tail truncates), n_emit (over-split count/elbow).
    instr = NamedTuple[]
    for _ in 1:max_steps
        d, s2, pos, n_emit = _group_distances(smld, g; n_iterations, burn_in, grouping, bagol_kwargs)
        n_bagol += 1
        isempty(d) && break
        m = _mstep_rb(d, s2, grid, eachindex(d); ks_noise)
        ix = eachindex(d)
        push!(instr, (; g_nm = g, m_nm = m, n_pairs = length(d), n_emit,
                      ks_full = _ks_rayleigh1(d, s2, g, ix),
                      ks_tail = _ks_rayleigh1_tail(d, s2, g, ix),
                      r_tail  = _r_tail(d, s2, g, ix)))
        push!(path, (g, m)); τ = m
        abs(m - g) ≤ stop_tol && break           # grouping self-consistent
        g = m                                    # jump (descend from above)
    end

    # CI: reuse the final grouping; bootstrap the right-biased M-step over spatial
    # blocks (the independent unit).
    rng = MersenneTwister(seed); boots = Float64[]
    for _ in 1:n_boot
        bi = _block_indices(pos, block_um, rng)
        isempty(bi) || push!(boots, _mstep_rb(d, s2, grid, bi; ks_noise))
    end
    lo, hi = isempty(boots) ? (NaN, NaN) : quantile(boots, [0.025, 0.975])
    ks_at = isempty(d) ? NaN : _ks_rayleigh1(d, s2, τ, eachindex(d))

    diagnostics = nothing
    if return_diagnostics
        # Compact arrays for BaGoLMakieExt — it PLOTS these, never re-runs BaGoL.
        diagnostics = (; d_nm = d, s2_nm2 = s2, pos_um = pos, tau_hat_nm = τ,
                       grid_nm = grid,
                       ks_path = [_ks_rayleigh1(d, s2, τ′, eachindex(d)) for τ′ in grid])
    end

    (; tau_hat_um = τ / 1000, ci_lo_um = lo / 1000, ci_hi_um = hi / 1000,
       ks_at_hat = ks_at, n_bagol,
       path_um = [(gᵢ / 1000, mᵢ / 1000) for (gᵢ, mᵢ) in path],
       instr, grouping, diagnostics)
end

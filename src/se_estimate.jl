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
# ADAPTIVE OVER-MERGE ROOT SEARCH (Keith + Codex, 2026-06-20; root search 0.4.6):
#   E-step  group the localizations with BaGoL at the current τ (expensive).
#   M-step  on that FIXED grouping, the τ best matching Rayleigh(1) — the
#           unscaled distances d are fixed, so this is a cheap 1-D KS search
#           (no BaGoL), right-biased (see `_mstep_rb`) with the threshold crossing
#           interpolated between grid points to remove grid quantization. Extend
#           the initial grid when its top qualifies, up to the numerical cap
#           `tau_max_um` (default 50 nm); this evaluates the same KS rule. m(g)
#           remains only approximately continuous (grouping changes, E-step noise).
#   The estimate is the self-consistent τ, m(τ) = τ: the root of r(g) = m(g) − g.
#   Start deliberately OVER-merged (`g_start`, a large se_adjust → the full
#   same-molecule spread, no truncation, only catchable contamination). Starting
#   below the root can instead find a spurious, nearly self-consistent under-merged
#   grouping. Report whether the first M-step has m(g_start) < g_start and warn
#   otherwise. Safeguarded
#   secant / regula falsi steps stop at an exact r = 0 or a measured sign-change
#   bracket of width ≤ 2·stop_tol; τ̂ interpolates inside that bracket. A still-
#   qualifying M-step top at the cap is censored: τ̂ is a lower bound, not a root,
#   and the search stops without convergence. For a root
#   inside the bracket, |τ̂ − g*| ≤ its width ≤ 2·stop_tol. The bracket guarantees
#   a sign change of measured r, not continuity or uniqueness. Agreement across
#   starts is measured behavior on genmab fields and sr-signaling, not a guarantee
#   for starts below the root (planted 16 nm data have spurious roots near 8–11 nm). KS(τ) is
#   not provably unimodal (discrete grouping + BaGoL stochasticity), so a
#   golden-section search is unsafe.
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
    else
        # :dahl does not use _assign_groups — it groups via _dahl_label_groups on the
        # _dahl_out labels (see _group_distances). Any other value is invalid.
        throw(ArgumentError("_assign_groups supports :mapn_proxy only (got :$grouping)"))
    end
end

# Group ORIGINAL loc indices by their global Dahl-consensus label (from _dahl_out).
# Label 0 = overlap-dup loc (skipped by the plumbing); excluded from grouping.
function _dahl_label_groups(labels)
    grp = Dict{Int, Vector{Int}}()
    for (i, l) in enumerate(labels)
        l == 0 && continue
        push!(get!(grp, l, Int[]), i)
    end
    collect(values(grp))
end

# Total data log-likelihood at the grouping under the model at se_adjust=g
# (σ' = √(σ²+g²), precision-weighted cluster mean). The distribution-shape signals
# are faked because they IGNORE complexity; the Bayesian-Occam / BIC score
# −2·loglik + n_emit·log(N) does NOT: over-split tightens the fit (loglik ↑) but
# pays the n_emit penalty, so the score's MINIMUM is the Occam balance ≈ the true
# count — where every shape signal collapsed. (Keith + smlmclustering, 2026-06-20.)
function _data_loglik(em, groups, g_nm)
    g2 = (g_nm / 1000)^2                          # μm²
    ll = 0.0
    for ks in groups
        isempty(ks) && continue
        wx = 0.0; wy = 0.0; mx = 0.0; my = 0.0
        for i in ks
            e = em[i]; px = 1 / (e.σ_x^2 + g2); py = 1 / (e.σ_y^2 + g2)
            wx += px; wy += py; mx += px * e.x; my += py * e.y
        end
        mx /= wx; my /= wy
        for i in ks
            e = em[i]; s2x = e.σ_x^2 + g2; s2y = e.σ_y^2 + g2
            ll += -0.5 * ((e.x - mx)^2 / s2x + log(2π * s2x)) -
                   0.5 * ((e.y - my)^2 / s2y + log(2π * s2y))
        end
    end
    ll
end

# E-step: group at τ (nm) with BaGoL, return within-group UNSCALED distances
# d (nm), pair variance s2 = σ_a² + σ_b² (nm²), and pair-midpoint positions (μm)
# for the spatial-block bootstrap. `se_adjust` is force-applied so a candidate τ
# is never skipped by the already-σ-corrected guard; posterior image is disabled
# (these runs are throwaway).
function _group_distances(smld, tau_nm; n_iterations, burn_in, grouping, bagol_kwargs)
    t = tau_nm / 1000                            # nm → μm
    em = smld.emitters
    if grouping === :dahl
        # Use BaGoL's OWN per-loc Dahl-consensus assignment (faithful), captured via
        # the _dahl_out plumbing in _run_bagol_collapsed — NOT the nearest-emitter proxy.
        dref = Base.RefValue(Int[])
        res, _ = _run_bagol_collapsed(smld; se_adjust = (t, t), force_se_adjust = true,
                           n_iterations = n_iterations, burn_in = burn_in,
                           posterior_pixel_size = 0.0, verbose = false,
                           _dahl_out = dref, bagol_kwargs...)
        n_emit = length(res.emitters)
        n_emit == 0 && return (Float64[], Float64[], NTuple{2, Float64}[], 0, NaN)
        groups = _dahl_label_groups(dref[])
    else
        res, _ = run_bagol(smld; se_adjust = (t, t), force_se_adjust = true,
                           n_iterations = n_iterations, burn_in = burn_in,
                           posterior_pixel_size = 0.0, verbose = false, bagol_kwargs...)
        n_emit = length(res.emitters)
        n_emit == 0 && return (Float64[], Float64[], NTuple{2, Float64}[], 0, NaN)
        groups = _assign_groups(em, res.emitters, grouping)
    end
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
    ll = _data_loglik(em, groups, tau_nm)
    d, s2, pos, n_emit, ll
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
# descent and the bootstrap use. The rule is a statement about continuous τ; the
# grid only discretizes it, and a grid-quantized m has plateaus where several g are
# exact fixed points of the finder's map. `interpolate=true` locates the threshold
# crossing by linear interpolation of the KS curve between the last qualifying grid
# point and the next one (none when the last qualifying point is the grid end).
function _mstep_rb(d, s2, grid_nm, idxs; ks_noise = 0.01, interpolate = false)
    ks = [_ks_rayleigh1(d, s2, τ, idxs) for τ in grid_nm]
    _mstep_rb_pick(ks, grid_nm; ks_noise, interpolate)
end

# Apply the same right-biased rule to precomputed KS values (also used after extension).
# Nonfinite points carry no information; interpolate between neighboring finite points.
function _mstep_rb_pick(ks, grid_nm; ks_noise = 0.01, interpolate = false)
    finite = findall(isfinite, ks)
    isempty(finite) && return NaN
    kmin = minimum(ks[i] for i in finite)
    thr = kmin + ks_noise
    p = argmax([ks[i] ≤ thr ? grid_nm[i] : -Inf for i in finite])
    j = finite[p]
    (!interpolate || p == lastindex(finite)) && return grid_nm[j]
    k = finite[p + 1]
    !(ks[k] > thr) && return grid_nm[j]
    grid_nm[j] + (thr - ks[j]) / (ks[k] - ks[j]) * (grid_nm[k] - grid_nm[j])
end

# A finite top also guarantees a nonempty set for the finite-only minimum.
_ks_top_qualifies(ks, ks_noise) =
    isfinite(ks[end]) && ks[end] ≤ minimum(k for k in ks if isfinite(k)) + ks_noise

# Extend only when the top qualifies, reusing all previously computed KS values.
# Own the grid so independent bootstrap threads never mutate their shared input.
function _mstep_rb_ext(d, s2, grid_nm, idxs; ks_noise = 0.01, cap_nm)
    grid = collect(Float64, grid_nm)
    length(grid) ≥ 2 || throw(ArgumentError("the M-step grid must have at least two points"))
    cap_nm ≥ maximum(grid) || throw(ArgumentError("cap_nm must be at least the M-step grid top"))
    ks = [_ks_rayleigh1(d, s2, τ, idxs) for τ in grid]
    while _ks_top_qualifies(ks, ks_noise) && grid[end] < cap_nm
        top = grid[end]
        step = top - grid[end - 1]
        step > 0 || throw(ArgumentError("the M-step grid must have a positive last step"))
        target = min(cap_nm, max(2top, top + step))
        new_points = collect((top + step):step:target)
        if target == cap_nm
            # Snap roundoff at the cap; include the cap even when it is off-grid.
            if !isempty(new_points) && isapprox(new_points[end], cap_nm; rtol = 8eps(Float64))
                new_points[end] = cap_nm
            else
                push!(new_points, Float64(cap_nm))
            end
        end
        append!(grid, new_points)
        append!(ks, [_ks_rayleigh1(d, s2, τ, idxs) for τ in new_points])
    end
    m = _mstep_rb_pick(ks, grid; ks_noise, interpolate = true)
    censored = grid[end] == cap_nm && _ks_top_qualifies(ks, ks_noise)
    (; m, grid, censored)
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

# Build a precision-DBSCAN neighbor graph for reuse within its τ ceiling.
# Positions are fixed across the search — τ only changes the σ_eff neighbor
# threshold — so cache the candidate-pair geometry
# at the COARSEST σ_eff (τ = cache ceiling), a provable superset up to that τ.
# Rebuild if a later E-step exceeds the ceiling; otherwise re-threshold exactly.
# Returns nothing (→ per-call KDTree build, unchanged) for an empty SMLD or partition_sigma=Inf.
function _build_finder_neighbor_graph(smld, tau_ceiling_um::Float64, partition_sigma::Float64)
    ems = smld.emitters
    (isempty(ems) || !isfinite(partition_sigma)) && return nothing
    md = hasproperty(smld, :metadata) ? smld.metadata : Dict{String, Any}()
    # σ_eff at the ceiling τ, via the SAME per-axis inflation the E-steps apply.
    infl, _, _ = _maybe_apply_se_adjust(ems, md, (tau_ceiling_um, tau_ceiling_um), true)
    σ_eff_ceiling = maximum(mean_sigma(e) for e in infl)
    max_radius = partition_sigma * 2 * σ_eff_ceiling
    coords = Float64.(coords_matrix(ems))
    return build_precision_neighbor_graph(coords, max_radius)
end

# Root search for the finder's self-consistent τ (nm): the root g* of r(g) = m(g) − g.
# `evalm(g)` runs the E-step at g and the M-step on its grouping and returns m(g) in
# nm, or `nothing` when the E-step is empty or the M-step is censored at its cap
# (the search ends, `converged=false`). It is the only place the search touches
# BaGoL, so tests drive it with synthetic maps.
# The search is bounded by `range_nm`, independently of the initial M-step grid.
# A censored M-step must not supply a zero residual at the upper bound;
# grouping changes and E-step noise can still make r discontinuous. Convergence
# requires an exact r = 0 or a measured sign-change bracket [a, b] of width ≤ 2·tol.
# τ̂ linearly interpolates inside the final bracket, giving |τ̂ − g*| ≤ b − a for a
# root inside it. Steps: a separated jump, then secant on the last two points while
# the slope of r is resolved and negative, regula falsi (Illinois) inside the bracket,
# bisection when a proposal is unusable. Bracketed steps stay at least tol from
# either end (Dekker's minimum step). Returns `(; tau_nm, converged, path,
# method_path, n_evals, bracket_nm)`; `bracket_nm` is (NaN, NaN) without a sign-change
# bracket (including an exact root found before bracketing). `path` holds the (g, m)
# pairs and `method_path` how each g was chosen (:start, :jump, :secant, :illinois, :bisect).
function _tau_root_search(evalm, g0_nm; grid_nm, stop_tol_nm, max_steps,
                          range_nm = extrema(grid_nm), probe_nm = 0.5, resolve_nm = 0.05)
    lo, hi = range_nm
    resolve = min(resolve_nm, stop_tol_nm / 2)     # resolve steps at the stopping scale
    g = clamp(Float64(g0_nm), lo, hi); prop = g; method = :start
    path = Tuple{Float64, Float64}[]; method_path = Symbol[]
    gs = Float64[]; rs = Float64[]
    ga = NaN; ra = NaN; fa = NaN                 # bracket end with r > 0 (g smaller)
    gb = NaN; rb = NaN; fb = NaN                 # bracket end with r < 0; f = Illinois-weighted r
    side = 0                                     # end replaced last: +1 a, −1 b
    exact = NaN; converged = false
    jump(gk, rk) = clamp(gk + (rk == 0 ? -1 : sign(rk)) * max(abs(rk), probe_nm), lo, hi)
    near(x) = any(h -> abs(h - x) < resolve, gs)
    for _ in 1:max_steps
        m = evalm(g)
        (m === nothing || !isfinite(m)) && break
        r = m - g
        push!(path, (g, m)); push!(method_path, method); push!(gs, g); push!(rs, r)
        if r == 0
            exact = g; converged = true
            break
        elseif r > 0 && (isnan(gb) || g < gb) && (isnan(ga) || g > ga)
            ga = g; ra = r; fa = r
            side == 1 && (fb /= 2)               # Illinois: b kept twice
            side = 1
        elseif r < 0 && (isnan(ga) || g > ga) && (isnan(gb) || g < gb)
            gb = g; rb = r; fb = r
            side == -1 && (fa /= 2)
            side = -1
        end
        bracket = !isnan(ga) && !isnan(gb)
        if bracket
            if gb - ga ≤ 2stop_tol_nm
                converged = true
                break
            end
            x = ga - fa * (gb - ga) / (fb - fa); method = :illinois
            # Near an end, aim tol past the interpolated root toward the other end.
            if x - ga ≤ stop_tol_nm
                x += stop_tol_nm
            elseif gb - x ≤ stop_tol_nm
                x -= stop_tol_nm
            end
            # Dekker: shrink the bracket from either side by at least stop_tol.
            x = clamp(x, ga + stop_tol_nm, gb - stop_tol_nm)
            if !(ga < x < gb) || near(x)
                x = (ga + gb) / 2; method = :bisect
            end
        else
            x = jump(g, r); method = :jump
            if length(gs) ≥ 2
                # Keep the second point separated. Later closing probes must be
                # resolved and reach at least m(g), or flat residuals make them crawl.
                if abs(r) < probe_nm
                    step = max(2stop_tol_nm, 2resolve)
                    xp = clamp(g + sign(r) * step, lo, hi)
                    abs(r) ≤ step && !near(xp) && (x = xp)
                end
                Δg = gs[end] - gs[end - 1]; Δr = rs[end] - rs[end - 1]
                if abs(Δg) ≥ resolve && Δr / Δg < 0
                    # Aim past the secant root to avoid approaching from one side.
                    xs = clamp(g - r * Δg / Δr + sign(r) * stop_tol_nm, lo, hi)
                    near(xs) || ((x, method) = (xs, :secant))
                end
            end
        end
        prop = x
        # A clamped search-range end can be closer than resolve and still be an
        # unevaluated exact root. Only stall if that end has already been tried.
        new_edge = (x == lo || x == hi) && !(x in gs)
        if near(x) && !new_edge
            prop = g
            break
        end
        g = x
    end
    tau = !isnan(exact) ? exact :
          !isnan(ga) && !isnan(gb) ? ga + ra / (ra - rb) * (gb - ga) :
          prop
    bracket_nm = !isnan(ga) && !isnan(gb) ? (ga, gb) : (NaN, NaN)
    (; tau_nm = clamp(tau, lo, hi), converged, path, method_path,
       n_evals = length(path), bracket_nm)
end

# τ̂ lies inside the sign-change bracket, so its nearest evaluated g is a bracket
# end; without a bracket, τ̂ is the last proposal. Only the largest positive-r g
# below the smallest negative-r g, that negative-r end, and the latest E-step can
# supply the CI grouping. Drop superseded arrays immediately (at most three kept).
function _keep_finder_eval!(evals, e)
    push!(evals, e)
    b = minimum((p.g for p in evals if p.m < p.g); init = Inf)
    a = maximum((p.g for p in evals if p.m > p.g && p.g < b); init = -Inf)
    filter!(p -> p === e || p.g == a || p.g == b, evals)
    evals
end

# A censored τ̂ is the cap, not a root: use the E-step that triggered censoring,
# rather than selecting an older grouping nearer that cap after pruning.
function _select_finder_eval(evals, tau_nm, censored)
    isempty(evals) && return nothing
    censored ? last(evals) : evals[argmin([abs(e.g - tau_nm) for e in evals])]
end

"""
    estimate_se_adjust(smld::SMLMData.SMLD; kwargs...) -> NamedTuple

Estimate the per-blink systematic localization error τ (the `se_adjust` value)
directly from data, by distribution comparison to Rayleigh(1): the self-consistent τ
(the grouping at τ gives an M-step estimate equal to τ) is found by root search from an
over-merged start. The "find" companion to [`apply_se_adjust`](@ref) /
`run_bagol(; se_adjust)`.

Interpolating the KS threshold crossing removes grid quantization; `m(g)` remains
only approximately continuous because groupings change and E-steps are noisy. The
bracket guarantees a sign change of the measured `r(g) = m(g) − g`, not continuity
or uniqueness of its root. Agreement across starts is measured behavior on genmab
fields and sr-signaling, not a guarantee for starts below the root: under-merged
groupings can be self-consistent at a too-small τ. Start above the expected τ.

The initial M-step grid extends when its top qualifies, up to `tau_max_um`.
Extension changes no statistic: it evaluates the same continuous right-biased KS
rule over the range the data need. The cap is numerical, not a prior. If the
M-step still prefers τ above the cap, the result is censored and `tau_hat_um` is
the cap, a lower bound rather than a self-consistent root.

`smld` must hold **raw-σ** localizations (e.g. frame-connected blinks). τ is the
correction to *find*, so an already-σ-corrected input
(`metadata["sigma_corrected"]==true`) is rejected.

# Returns
A `NamedTuple` with τ values in **μm**:
- `tau_hat_um` — point estimate (feeds `run_bagol(; se_adjust = tau_hat_um)`);
  a censored result is a lower bound, not a root
- `ci_lo_um`, `ci_hi_um` — spatial-block-bootstrap 95% CI, conditional on the grouping of
  the E-step nearest `tau_hat_um`, or the censoring E-step when `censored=true`
  (it omits E-step variability; nonfinite bootstrap estimates are excluded)
- `ks_at_hat` — KS-to-Rayleigh(1) at the estimate (lower = better fit)
- `n_bagol` — number of BaGoL E-steps run (the search length)
- `path_um` — the search trajectory as `(g, m)` pairs (grouping τ, M-step τ)
- `converged` — `true` at an exact `r = 0` or when the measured sign-change bracket
  has width ≤ `2·stop_tol_um`; otherwise `false` (cap, empty/nonfinite E-step, or stalled
  search), and `tau_hat_um` is then the best estimate so far
- `censored` — `true` if an E-step's M-step still qualifies at `tau_max_um`;
  implies `converged = false`
- `start_overmerged` — `true` iff the first E-step has `m(g_start) < g_start`;
  `false` when there is no evaluation. A start at or below the root emits a warning
- `grid_top_um` — final M-step grid top (μm)
- `tau_max_um` — numerical upper bound of the search and grid extension (μm)
- `n_boot_censored` — number of bootstrap replicates censored at the same cap
- `method_path` — how each step's g was chosen (`:start`, `:jump`, `:secant`, `:illinois`,
  `:bisect`)
- `grouping` — the grouping backend used
- `bagol_kwargs` — the BaGoL kwargs the E-steps ran with (a `NamedTuple`; excludes the
  internal neighbor graph)
- `diagnostics` — `nothing`, or (with `return_diagnostics=true`) the arrays the
  `BaGoLMakieExt` plots consume, so the extension never re-runs BaGoL.
- `bracket_um::Tuple{Float64,Float64}` — final measured sign-change bracket;
  `(NaN, NaN)` without one, including an exact root found before bracketing. For a
  root `g*` inside `[a, b]`, `|tau_hat_um − g*| ≤ b − a ≤ 2·stop_tol_um` on a width stop.

# Keyword arguments
- `g_start_um = 0.008` — starting τ (μm), which must be above the root (the method's
  premise), clamped into `(minimum(grid_um), tau_max_um)`. The 8 nm default suits
  data whose τ is below 8 nm (dSTORM 4.5–5.7 nm; genmab fields 5.1–7.7 nm).
  For data whose τ may be larger (e.g. SeqSR, about 16 nm), pass a larger start such as `0.03`
- `stop_tol_um = 1e-4` — the search stops when the root is bracketed within
  2·stop_tol_um; τ̂ interpolates inside that bracket (an exact `r = 0` also stops)
- `max_steps = 12` — E-step cap; if it is reached without the stop rule, `converged = false`
- `ks_noise = 0.01` — KS tie band for the right-biased M-step
- `grid_um = 0.0:0.0001:0.014` — initial M-step τ grid (μm), at least two points;
  extended at its last step when the top qualifies, up to `tau_max_um`
- `tau_max_um = 0.05` — numerical upper bound (50 nm), at least `maximum(grid_um)`
- `n_iterations = 4000`, `burn_in = 2000` — per-E-step BaGoL chain length
- `n_boot = 200` — bootstrap resamples for the CI
- `block_um = 1.0` — spatial-block tile size (μm)
- `seed = 1` — bootstrap RNG seed (a local RNG; BaGoL's own E-step stochasticity
  is only controlled as far as the sampler permits)
- `grouping = :dahl` — `:dahl` (BaGoL consensus assignment, default) or `:mapn_proxy` (legacy nearest-emitter proxy)
- `return_diagnostics = false` — attach the diagnostic arrays
- `bagol_kwargs...` — forwarded to `run_bagol` (e.g. `partition_sigma`, `shape`,
  `allocation_model`). **Reserved** (managed by the finder, do not pass):
  `se_adjust`, `force_se_adjust`, `posterior_pixel_size`, `verbose`.

!!! note "Grouping backend"
    The default `:dahl` backend groups by BaGoL's own per-loc consensus assignment
    (faithful). The legacy `:mapn_proxy` (external nearest-emitter) biases τ̂
    downward and collapses in crowded regions — kept only for comparison.
"""
function estimate_se_adjust(smld::SMLMData.SMLD;
        g_start_um = 0.008,
        stop_tol_um::Float64 = 1e-4,
        max_steps::Int = 12,
        ks_noise::Float64 = 0.01,
        grid_um = 0.0:0.0001:0.014,
        tau_max_um::Float64 = 0.05,
        n_iterations::Int = 4000,
        burn_in::Int = 2000,
        n_boot::Int = 200,
        block_um::Float64 = 1.0,
        seed::Int = 1,
        grouping::Symbol = :dahl,
        return_diagnostics::Bool = false,
        bagol_kwargs...)

    # τ is the σ-correction to FIND; finding it on already-corrected σ is ill-posed.
    md = hasproperty(smld, :metadata) ? smld.metadata : Dict{String, Any}()
    if get(md, "sigma_corrected", false) == true
        throw(ArgumentError("estimate_se_adjust expects raw-σ localizations, but the input SMLD is " *
                            "already σ-corrected (metadata sigma_corrected=true). τ is the correction " *
                            "to find; estimating it on corrected σ is ill-posed."))
    end

    # Fail fast on an invalid grouping (before any BaGoL run).
    if grouping !== :mapn_proxy && grouping !== :dahl
        throw(ArgumentError("grouping must be :mapn_proxy or :dahl (got :$grouping)"))
    end

    # Work in nm internally (preserves the validated numerics).
    initial_grid_um = collect(Float64, grid_um)
    length(initial_grid_um) ≥ 2 || throw(ArgumentError("grid_um must have at least two points"))
    tau_max_um ≥ maximum(initial_grid_um) || throw(ArgumentError(
        "tau_max_um must be at least maximum(grid_um)"))
    grid     = 1000 .* initial_grid_um
    cap_nm   = 1000 * tau_max_um
    range_nm = (minimum(grid), cap_nm)
    g        = clamp(1000 * g_start_um, range_nm...) # nm — start over-merged
    stop_tol = 1000 * stop_tol_um

    # ROOT SEARCH on r(g) = m(g) − g from an over-merged start (see `_tau_root_search`);
    # keep only bracket ends and the latest E-step for the CI grouping nearest τ̂.
    n_bagol = Ref{Int}(0)
    censored = false
    start_overmerged = false
    evals = NamedTuple[]
    # Per-descent-g instrumentation. No sim reproduces the real-data collapse, so
    # the REAL ruler is the only gate: log every candidate stop-signal at τ=g and
    # pick the winner empirically — ks_full (faked low by over-split), ks_tail +
    # r_tail (over-split-proof: tail truncates), n_emit (over-split count/elbow).
    instr = NamedTuple[]
    # Reuse the graph within its ceiling, doubling it when the search needs more.
    _kw_used = (; bagol_kwargs...)                     # settings the E-steps run with (reported below)
    graph_ceiling_nm = max(maximum(grid), g)
    partition_sigma = Float64(get(bagol_kwargs, :partition_sigma, 3.0))
    _ng = _build_finder_neighbor_graph(smld, graph_ceiling_nm / 1000, partition_sigma)
    if _ng !== nothing
        bagol_kwargs = (; bagol_kwargs..., neighbor_graph = _ng)
    end
    # E-step + M-step at g. Interpolating the KS threshold crossing removes grid
    # quantization; grouping changes and E-step noise still affect continuity of m(g).
    function evalm(g)
        local d, s2, pos, n_emit, loglik
        if _ng !== nothing && g > graph_ceiling_nm
            graph_ceiling_nm = min(cap_nm, max(2graph_ceiling_nm, g))
            _ng = _build_finder_neighbor_graph(smld, graph_ceiling_nm / 1000, partition_sigma)
            bagol_kwargs = (; bagol_kwargs..., neighbor_graph = _ng)
        end
        d, s2, pos, n_emit, loglik = _group_distances(smld, g; n_iterations, burn_in, grouping, bagol_kwargs)
        n_bagol[] += 1
        isempty(d) && return nothing
        step = _mstep_rb_ext(d, s2, grid, eachindex(d); ks_noise, cap_nm)
        grid = step.grid
        m = step.m
        if n_bagol[] == 1
            start_overmerged = m < g
            if m ≥ g
                @warn "The finder start is not over-merged (below or at the self-consistent tau): g_start=$(g) nm, m(g_start)=$(m) nm. The search then climbs through under-merged groupings, which can be self-consistent at a too-small tau. Pass a g_start_um above the expected tau."
            end
        end
        _keep_finder_eval!(evals, (; g, m, d, s2, pos))
        ix = eachindex(d)
        # BIC/Occam evidence score = −2·loglik + k·log(N), k = 2·n_emit (x,y per
        # emitter): MIN ≈ true count — over-split-proof (it pays the complexity the
        # shape signals ignore). loglik + n_emit are logged separately so any penalty
        # is recomputable; the penalty constant shifts the peak, so confirm with the
        # proper ClusterStats marginal. (smlmclustering 2026-06-20.)
        bic = -2 * loglik + 2 * n_emit * log(length(smld.emitters))
        push!(instr, (; g_nm = g, m_nm = m, n_pairs = length(d), n_emit, loglik, bic,
                      ks_full = _ks_rayleigh1(d, s2, g, ix),
                      ks_tail = _ks_rayleigh1_tail(d, s2, g, ix),
                      r_tail  = _r_tail(d, s2, g, ix)))
        if step.censored
            censored = true
            @warn "No self-consistent τ at or below tau_max_um at this grouping: the M-step prefers τ above the cap (over-merging or a model mismatch). τ̂ is a censored lower bound, not a root." tau_max_um
            return nothing
        end
        m
    end
    found = _tau_root_search(evalm, g; grid_nm = grid, range_nm, stop_tol_nm = stop_tol, max_steps)
    τ = censored ? cap_nm : found.tau_nm; path = found.path

    # CI, KS, bootstrap censor counts and diagnostics share the selected grouping.
    selected = _select_finder_eval(evals, τ, censored)
    if selected === nothing
        d = Float64[]; s2 = Float64[]; pos = NTuple{2, Float64}[]
    else
        (; d, s2, pos) = selected
    end

    # CI: reuse that grouping (so it is conditional on it: E-step variability is not in
    # it); bootstrap the right-biased M-step over spatial blocks (the independent unit).
    # The n_boot bootstrap resamples are independent and `_mstep_rb_ext` is pure (reads
    # d/s2/grid, allocates its own scratch), so thread them. Draw the block-index sets
    # SERIALLY first — this keeps the exact MersenneTwister(seed) stream, so the CI is
    # bit-identical to the serial loop — then parallelize only the expensive M-steps.
    # This was the dominant single-threaded tail of the finder (200 sequential grid×KS
    # searches pinned one core for minutes on large cells).
    rng = MersenneTwister(seed)
    bis = [_block_indices(pos, block_um, rng) for _ in 1:n_boot]
    boots_raw = Vector{Float64}(undef, n_boot)
    ok = fill(false, n_boot)
    boot_censored = fill(false, n_boot)            # byte arrays: independent thread writes
    Threads.@threads for b in 1:n_boot
        bi = bis[b]
        if !isempty(bi)
            step = _mstep_rb_ext(d, s2, grid, bi; ks_noise, cap_nm)
            boots_raw[b] = step.m
            boot_censored[b] = step.censored
            ok[b] = isfinite(step.m)
        end
    end
    boots = boots_raw[ok]
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
       ks_at_hat = ks_at, n_bagol = n_bagol[],
       path_um = [(gᵢ / 1000, mᵢ / 1000) for (gᵢ, mᵢ) in path],
       converged = found.converged, method_path = found.method_path,
       censored, start_overmerged, grid_top_um = maximum(grid) / 1000, tau_max_um,
       n_boot_censored = count(boot_censored),
       instr, grouping, bagol_kwargs = _kw_used, diagnostics,
       bracket_um = (found.bracket_nm[1] / 1000, found.bracket_nm[2] / 1000))
end

# ============================================================================
# PER-AXIS finder — the anisotropy fix (estimate_se_adjust_peraxis)
# ============================================================================
# The isotropic descent collapses on anisotropic-residual data (a Y-excess + an
# honest X cannot fit ONE Rayleigh → no fixed point → runaway). The per-axis
# finder groups with an ANISOTROPIC se_adjust=(g_x,g_y) and scores EACH axis
# against Half-Normal(1): |d_a|/√(σ_a²+2g_a²) ~ HalfNormal(1) at the true g_a;
# it descends g_x,g_y separately. Validated in sim (smlmclustering per_axis_test.jl):
# converges (g_x→0 honest X, g_y→Y-excess) where the isotropic finder collapsed.
# CANDIDATE — the real-data per-axis run gates the ship (does over-split fake
# Half-Normal PER-AXIS on real, or converge?). (Keith + smlmclustering, 2026-06-21.)

# Half-Normal(1) CDF = P(|Z|≤z) for Z~N(0,1) = 2·Φ(z)−1 (exact, via Distributions
# which the module already imports).
_halfnormal_cdf(z) = 2 * cdf(Normal(), z) - 1

# KS of the per-axis scaled |d| against Half-Normal(1), at assumed g (nm).
function _ks_halfnormal(d, s2, g_nm, idxs)
    isempty(idxs) && return NaN
    g2 = g_nm^2
    z = sort!([abs(d[k]) / sqrt(s2[k] + 2g2) for k in idxs])
    n = length(z)
    maximum(abs(i / n - _halfnormal_cdf(z[i])) for i in 1:n)
end

# Per-axis M-step: g minimizing the Half-Normal KS over the grid. `ks_noise>0`
# applies the right-biased tie rule (largest g within ks_noise of the min) — the
# per-axis analog of the isotropic `_mstep_rb`, to counter within-group truncation
# under-bias (smlmclustering flagged g_y biased low). Default 0 ≈ argmin (faithful
# to the validated sim).
function _mstep_ax(d, s2, grid_nm, idxs; ks_noise = 0.0)
    ks = [_ks_halfnormal(d, s2, g, idxs) for g in grid_nm]
    kmin = minimum(ks)
    maximum(grid_nm[i] for i in eachindex(grid_nm) if ks[i] ≤ kmin + ks_noise)
end

# Per-axis E-step: group with ANISOTROPIC se_adjust=(gx,gy); return signed per-axis
# differences dx/dy (nm), pair variances s2x/s2y (nm²), and the MAP-N emitter count.
function _group_peraxis(smld, gx_nm, gy_nm; n_iterations, burn_in, bagol_kwargs)
    res, _ = run_bagol(smld; se_adjust = (gx_nm / 1000, gy_nm / 1000), force_se_adjust = true,
                       n_iterations = n_iterations, burn_in = burn_in,
                       posterior_pixel_size = 0.0, verbose = false, bagol_kwargs...)
    EM = res.emitters
    isempty(EM) && return (Float64[], Float64[], Float64[], Float64[], 0)
    em = smld.emitters
    etree = KDTree(_emitter_xy(EM))
    idx, _ = nn(etree, _emitter_xy(em))
    grp = Dict{Int, Vector{Int}}()
    for (i, a) in enumerate(idx); push!(get!(grp, a, Int[]), i); end
    dx = Float64[]; dy = Float64[]; s2x = Float64[]; s2y = Float64[]
    for ks in values(grp)
        length(ks) < 2 && continue
        for a in 1:length(ks) - 1, b in a + 1:length(ks)
            ea = em[ks[a]]; eb = em[ks[b]]
            push!(dx, (ea.x - eb.x) * 1000); push!(dy, (ea.y - eb.y) * 1000)
            push!(s2x, (ea.σ_x^2 + eb.σ_x^2) * 1e6); push!(s2y, (ea.σ_y^2 + eb.σ_y^2) * 1e6)
        end
    end
    dx, dy, s2x, s2y, length(EM)
end

"""
    estimate_se_adjust_peraxis(smld::SMLMData.SMLD; kwargs...) -> NamedTuple

Per-axis `se_adjust` finder (the anisotropy fix). Groups with an anisotropic
`se_adjust=(g_x,g_y)` and descends g_x,g_y separately, scoring each axis against
Half-Normal(1). For anisotropic residual (e.g. a Y-axis excess over an honest X)
where the isotropic [`estimate_se_adjust`](@ref) collapses. **Candidate** — gate
the result on a real-data run + a magnitude-calibration sweep before shipping.

Returns `(; se_x_um, se_y_um, path_um, n_bagol, instr)` (τ in μm). `instr` logs
per step `(gx_nm, gy_nm, mx_nm, my_nm, n_pairs, n_emit, ks_x, ks_y)`.

`ks_noise>0` enables the right-biased per-axis tie rule (de-bias). `bagol_kwargs`
forwarded to `run_bagol`; reserved: `se_adjust`/`force_se_adjust`/`posterior_pixel_size`/`verbose`.
"""
function estimate_se_adjust_peraxis(smld::SMLMData.SMLD;
        g_start_um = 0.008, stop_tol_um::Float64 = 5e-4, max_steps::Int = 8,
        ks_noise::Float64 = 0.0, grid_um = 0.0:0.0001:0.014,
        n_iterations::Int = 4000, burn_in::Int = 2000, bagol_kwargs...)
    md = hasproperty(smld, :metadata) ? smld.metadata : Dict{String, Any}()
    if get(md, "sigma_corrected", false) == true
        throw(ArgumentError("estimate_se_adjust_peraxis expects raw-σ localizations, but the input SMLD " *
                            "is already σ-corrected (metadata sigma_corrected=true)."))
    end
    grid = 1000 .* collect(Float64, grid_um)
    gx = 1000 * g_start_um; gy = 1000 * g_start_um
    stop_tol = 1000 * stop_tol_um
    n_bagol = 0; instr = NamedTuple[]
    for _ in 1:max_steps
        dx, dy, s2x, s2y, n_emit = _group_peraxis(smld, gx, gy; n_iterations, burn_in, bagol_kwargs)
        n_bagol += 1
        isempty(dx) && break
        mx = _mstep_ax(dx, s2x, grid, eachindex(dx); ks_noise)
        my = _mstep_ax(dy, s2y, grid, eachindex(dy); ks_noise)
        push!(instr, (; gx_nm = gx, gy_nm = gy, mx_nm = mx, my_nm = my, n_pairs = length(dx), n_emit,
                      ks_x = _ks_halfnormal(dx, s2x, gx, eachindex(dx)),
                      ks_y = _ks_halfnormal(dy, s2y, gy, eachindex(dy))))
        conv = abs(mx - gx) ≤ stop_tol && abs(my - gy) ≤ stop_tol
        gx = mx; gy = my
        conv && break
    end
    (; se_x_um = gx / 1000, se_y_um = gy / 1000,
       path_um = [(s.gx_nm / 1000, s.gy_nm / 1000) for s in instr],
       n_bagol, instr)
end

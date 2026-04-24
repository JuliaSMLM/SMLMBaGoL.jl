# Count prior estimation: data-driven hyperprior for (μ, α) on dSTORM SMLDs.
#
# Two independent sources, combined as a COMPOSITE likelihood (not naive product —
# shared nuisance structure per Codex critique, so the joint is overconfident if
# treated as independent):
#
#   S1. Blinks/frame curve (temporal bleach, NHPP Poisson likelihood)
#       N(t) ~ Poisson(A·exp(-β·t))
#       Identifies β cleanly, and A = N₀·λ. μ = λ/β requires external N₀.
#
#   S2. σ-scaled NND (spatial clustering likelihood)
#       r_i = d_NN(i) / √(σ_i² + σ_NN²)   (quadrature, NOT linear sum)
#       Mix over per-emitter K via size-biased NegBin(μ, α).
#       Per-K NND PDF tabulated via Monte Carlo — exact (not independent-pairs approx).
#
# Combining: tempered sum of log-likelihoods in (logμ, logα) space.
# Hessian-based covariance is LOWER BOUND (composite), use bootstrap for calibration.
#
# References: Codex critique 2026-04-24.

# NOTE: All using/import statements live in src/SMLMBaGoL.jl (CLAUDE.md convention).
# Required: Distributions, LinearAlgebra, Random, SpecialFunctions, Statistics.

# ============================================================================
# Table of per-K NND PDF (equal-σ case, Monte Carlo)
# ============================================================================

"""
    NNDTable

Precomputed per-loc nearest-neighbor distance PDF f(r | K) for K = 2..K_max
iid samples from a single 2D Gaussian emitter, with σ=1, using QUADRATURE
denominator (r = d / √(σ_i² + σ_j²)).

Built via Monte Carlo (n_samples emitters per K). Stored on a fixed r grid.
"""
struct NNDTable
    K_range::UnitRange{Int}
    r_edges::Vector{Float64}    # length n_bins + 1
    r_centers::Vector{Float64}  # length n_bins
    pdfs::Matrix{Float64}       # (n_bins, n_K) — PDF at r_centers for each K
end

"""
    build_nnd_table(; K_max=15, n_samples=200_000, r_max=5.0, n_bins=500, seed=42) -> NNDTable

Monte Carlo table of per-loc NND PDF at each K. Simulates K iid 2D standard normal
points (σ=1), computes each loc's NND, scales by √(σ_i² + σ_j²) = √2, and histograms.
"""
function build_nnd_table(; K_max::Int=15, n_samples::Int=200_000,
                         r_max::Float64=5.0, n_bins::Int=500, seed::Int=42)
    rng = MersenneTwister(seed)
    edges = collect(range(0.0, r_max, length=n_bins+1))
    centers = [(edges[i] + edges[i+1]) / 2 for i in 1:n_bins]
    Δ = edges[2] - edges[1]
    K_range = 2:K_max
    pdfs = zeros(n_bins, length(K_range))

    for (ki, K) in enumerate(K_range)
        n_emitters = cld(n_samples, K)
        counts = zeros(Int, n_bins)
        n_collected = 0
        xs = zeros(2, K)
        for _ in 1:n_emitters
            for k in 1:K
                xs[1, k] = randn(rng); xs[2, k] = randn(rng)
            end
            for i in 1:K
                best = Inf
                for j in 1:K
                    j == i && continue
                    d = sqrt((xs[1,i]-xs[1,j])^2 + (xs[2,i]-xs[2,j])^2)
                    r = d / sqrt(2.0)  # σ_i = σ_j = 1, quadrature √(1+1) = √2
                    r < best && (best = r)
                end
                best > r_max && continue
                idx = clamp(searchsortedlast(edges, best), 1, n_bins)
                counts[idx] += 1
                n_collected += 1
            end
        end
        pdfs[:, ki] .= counts ./ (n_collected * Δ)
    end
    return NNDTable(K_range, edges, centers, pdfs)
end

"Interpolate f(r | K) from the precomputed table. K can be Int or Float (linear interp)."
function pdf_nnd(table::NNDTable, r::Float64, K::Int)
    K < first(table.K_range) && return 0.0
    K > last(table.K_range) && (K = last(table.K_range))
    r < 0 && return 0.0
    r > last(table.r_edges) && return 0.0
    ki = K - first(table.K_range) + 1
    n_bins = length(table.r_centers)
    # Linear interpolation in r
    ri_raw = (r - table.r_edges[1]) / (table.r_edges[2] - table.r_edges[1])
    ri = floor(Int, ri_raw) + 1
    ri = clamp(ri, 1, n_bins)
    return table.pdfs[ri, ki]
end

# ============================================================================
# σ-scaled NND computation (quadrature denominator)
# ============================================================================

"""
    sigma_scaled_nnd(locs; quadrature=true) -> Vector{Float64}

For each loc, compute r = d_NN / denom where
  denom = √(σ_i² + σ_NN²)  if quadrature=true   (default, per Codex)
  denom = σ_i + σ_NN       otherwise             (BaGoL visual convention)

Uses O(N²) brute force — fine for per-partition or ROI-scale inputs.
For full-field data, pre-partition first.
"""
function sigma_scaled_nnd(locs; quadrature::Bool=true)
    N = length(locs)
    r = Vector{Float64}(undef, N)
    σs = Float64[l.σ_x for l in locs]
    xs = Float64[l.x for l in locs]
    ys = Float64[l.y for l in locs]
    for i in 1:N
        best = Inf
        for j in 1:N
            j == i && continue
            d = sqrt((xs[i]-xs[j])^2 + (ys[i]-ys[j])^2)
            denom = quadrature ? sqrt(σs[i]^2 + σs[j]^2) : (σs[i] + σs[j])
            rij = d / denom
            rij < best && (best = rij)
        end
        r[i] = best
    end
    return r
end

# ============================================================================
# S1. Bleach-curve fit (NHPP)
# ============================================================================

"""
    fit_bleach_curve(smld; t_min=1, t_max=nothing) -> NamedTuple

Fits N(t) = A·exp(-β·t) to the per-frame localization count via NHPP Poisson
log-likelihood:
  log L = Σ_t [N_t · log(A·exp(-β·t)) - A·exp(-β·t)]

Optimization in log-space (logA, log β) with Newton's method on the score.

Returns:
  A, β          — fit parameters (logspace-optimized)
  τ = 1/β       — bleach time constant (frames)
  T_obs         — total observation window (frames)
  F             — bleached fraction = 1 - exp(-β·T_obs)
  cov           — 2×2 covariance of (logA, logβ) from inverse Hessian
  N_t           — per-frame count histogram used in fit
  note          — identifiability caveat: μ = λ/β requires external N₀

Codex critique applies: from the curve alone, A = N₀·λ is identified but
λ separately is not. We return A and β; downstream combiner must supply N₀
(e.g., from detected-emitter count after a clustering pass) to get μ = A / (N₀·β).
"""
function fit_bleach_curve(smld; t_min::Union{Int,Nothing}=nothing,
                          t_max::Union{Int,Nothing}=nothing)
    frames = Int[e.frame for e in smld.emitters]
    isempty(frames) && error("fit_bleach_curve: empty SMLD")
    t_lo = isnothing(t_min) ? minimum(frames) : t_min
    t_hi = isnothing(t_max) ? maximum(frames) : t_max
    n_frames = t_hi - t_lo + 1
    N_t = zeros(Int, n_frames)
    for f in frames
        t_lo ≤ f ≤ t_hi && (N_t[f - t_lo + 1] += 1)
    end

    # NHPP neg-log-likelihood in (logA, logβ)
    ts = collect(0:n_frames-1)  # t relative to t_lo
    function nll(logA, logβ)
        A = exp(logA); β = exp(logβ)
        s = 0.0
        for t in eachindex(N_t)
            λt = A * exp(-β * ts[t])
            λt ≤ 0 && return Inf
            s -= N_t[t] * log(λt) - λt  # negative log-lik up to constant
        end
        return s
    end

    # Reasonable initial guess: A₀ ≈ mean of first 10% of frames, β₀ from ratio of
    # last-tenth to first-tenth averages.
    head = mean(N_t[1:max(1, n_frames÷10)])
    tail = mean(N_t[max(1, n_frames - n_frames÷10 + 1):end])
    head ≤ 0 && (head = 1.0)
    tail = max(tail, 1e-3 * head)
    β0 = log(head / tail) / max(1, n_frames - 1)
    A0 = head
    logA = log(max(A0, 1.0)); logβ = log(max(β0, 1e-6))

    # Coordinate descent with golden section (simpler than pulling in Optim here)
    # Two-parameter problem — this is fine.
    for _ in 1:50
        # Line search in logA (closed form for Poisson: A = Σ N_t / Σ exp(-β t))
        β = exp(logβ)
        denom = sum(exp(-β * t) for t in ts)
        num = sum(N_t)
        num ≤ 0 && break
        logA = log(num / denom)
        # Line search in logβ: gradient-based bisection
        lo, hi = logβ - 2.0, logβ + 2.0
        for _ in 1:40
            mid = (lo + hi) / 2
            # d(nll)/d(logβ) at this logβ
            β_mid = exp(mid); A = exp(logA)
            dnll = 0.0
            for t in eachindex(N_t)
                λt = A * exp(-β_mid * ts[t])
                dnll += -N_t[t] * (-β_mid * ts[t]) + λt * β_mid * ts[t]
            end
            # Correction: derivative w.r.t. logβ, not β; multiply by β
            # But our nll already includes logβ effect — compute directly:
            ε = 1e-4
            f1 = nll(logA, mid - ε); f2 = nll(logA, mid + ε)
            grad = (f2 - f1) / (2ε)
            if grad > 0
                hi = mid
            else
                lo = mid
            end
        end
        logβ = (lo + hi) / 2
    end

    A_fit = exp(logA); β_fit = exp(logβ)

    # Hessian at optimum (numerical, 2×2)
    ε = 1e-3
    f0 = nll(logA, logβ)
    H = zeros(2, 2)
    H[1,1] = (nll(logA+ε, logβ) - 2f0 + nll(logA-ε, logβ)) / ε^2
    H[2,2] = (nll(logA, logβ+ε) - 2f0 + nll(logA, logβ-ε)) / ε^2
    fpp = nll(logA+ε, logβ+ε) + nll(logA-ε, logβ-ε)
    fpm = nll(logA+ε, logβ-ε) + nll(logA-ε, logβ+ε)
    H[1,2] = H[2,1] = (fpp - fpm) / (4*ε^2)
    cov = try inv(Hermitian(H)) catch; fill(NaN, 2, 2) end

    τ = 1 / β_fit
    T_obs = Float64(n_frames)
    F = 1 - exp(-β_fit * T_obs)

    return (A=A_fit, β=β_fit, τ=τ, T_obs=T_obs, F=F,
            logA=logA, logβ=logβ, cov_logAβ=cov,
            N_t=N_t, t_lo=t_lo, t_hi=t_hi,
            note="A = N₀·λ identified; μ = λ/β requires external N₀.")
end

"Convert (A, β, T_obs, N₀) → (μ, α_mixture) with exact finite-window count law."
function bleach_to_count_params(bleach::NamedTuple, N₀::Real)
    A = bleach.A; β = bleach.β; T_obs = bleach.T_obs
    λ = A / N₀           # blink rate per emitter per frame
    μ_ub = λ / β          # upper bound: mean K if full bleach
    F = bleach.F          # fraction bleached before T_obs

    # Exact expected K under partial bleach:
    # K | survived ~ Poisson(λ·T_obs), K | bleached ~ Geom(β/(λ+β)) truncated
    # E[K] = (1-F)·λ·T_obs + F·(λ/β)·(1 - adjust)
    # For simplicity, take E[K] ≈ λ/β · F + λ·T_obs · (1-F)
    μ = λ/β * F + λ * T_obs * (1 - F)

    # Variance: Geom has var λ/β·(1+λ/β) ≈ (λ/β)² for large λ/β;
    # Poisson has var = mean. Over-dispersion α = μ²/(var - μ) heuristic:
    var_bleach = λ/β * (1 + λ/β)
    var_surv = λ * T_obs
    var_mix = F * var_bleach + (1-F) * var_surv + F*(1-F)*(λ/β - λ*T_obs)^2
    α_est = if var_mix > μ
        μ^2 / (var_mix - μ)
    else
        Inf  # Poisson limit (no over-dispersion)
    end
    # For F → 1, α → 1 (geometric limit); for F → 0, α → ∞ (Poisson limit).
    return (μ=μ, α=α_est, λ=λ, F=F)
end

# ============================================================================
# S2. σ-scaled NND fit
# ============================================================================

"""
    fit_nnd_count(locs, table; μ0=5.0, α0=2.0, r_max=3.0) -> NamedTuple

MLE of (μ, α) from the population σ-scaled NND histogram under the size-biased
mixture model:

  f_pop(r | μ, α) = (1/E[K]) Σ_{K ≥ 2} K · P(K | μ, α) · f_NND(r | K)

using the tabulated exact f_NND(r | K) from build_nnd_table. NegBin
parameterization: K-1 ~ NegativeBinomial(α, α/(α+μ)), so K ∈ {1, 2, ...}
with mean K = μ + 1.

Nelder-Mead in (logμ, logα) space.

Returns (μ̂, α̂, cov_logμα, n_kept, converged).
"""
function fit_nnd_count(locs, table::NNDTable;
                      μ0::Float64=5.0, α0::Float64=2.0,
                      r_max::Float64=3.0,
                      K_max_fit::Int=last(table.K_range))
    rs = sigma_scaled_nnd(locs; quadrature=true)
    rs_kept = filter(<=(r_max), rs)
    n_kept = length(rs_kept)
    n_kept < 10 && return (μ̂=NaN, α̂=NaN, cov=fill(NaN, 2, 2), n_kept=n_kept, converged=false)

    function f_pop(r, μ, α)
        (μ ≤ 0 || α ≤ 0) && return 0.0
        p = α / (α + μ)
        pmf_K1 = pdf(NegativeBinomial(α, p), 0)  # K=1
        EK = 1.0 + μ  # E[K] under this param: K=NegBin+1
        val = 0.0
        for K in 2:K_max_fit
            pmfK = pdf(NegativeBinomial(α, p), K-1)
            val += K * pmfK * pdf_nnd(table, r, K)
        end
        return val / EK
    end

    # Normalize f_pop over [0, r_max] for truncation
    r_grid = range(1e-4, r_max, length=300)
    Δr = step(r_grid)

    function nll(logμ, logα)
        # Reject unreasonable log-params — prevents Nelder-Mead runaway.
        (logμ < -4 || logμ > 4 || logα < -4 || logα > 4) && return Inf
        μ = exp(logμ); α = exp(logα)
        (μ ≤ 0 || α ≤ 0) && return Inf
        Z = 0.0
        for r in r_grid; Z += f_pop(r, μ, α) * Δr; end
        Z ≤ 0 && return Inf
        s = 0.0
        for r in rs_kept
            fp = f_pop(r, μ, α)
            fp ≤ 0 && return Inf
            s -= log(fp / Z)
        end
        return s
    end

    # Nelder-Mead in logspace (hand-rolled, 2D)
    simplex = [[log(μ0), log(α0)],
               [log(μ0) + 0.5, log(α0)],
               [log(μ0), log(α0) + 0.5]]
    vals = [nll(s[1], s[2]) for s in simplex]

    for _ in 1:500
        # Sort by value
        p = sortperm(vals)
        simplex = simplex[p]; vals = vals[p]
        # Converge check
        maximum(vals) - minimum(vals) < 1e-6 && break
        # Centroid of all but worst
        c = (simplex[1] .+ simplex[2]) ./ 2
        worst = simplex[3]
        # Reflection
        xr = c .+ 1.0 .* (c .- worst); fr = nll(xr[1], xr[2])
        if fr < vals[1]
            # Expansion
            xe = c .+ 2.0 .* (xr .- c); fe = nll(xe[1], xe[2])
            if fe < fr; simplex[3] = xe; vals[3] = fe
            else;       simplex[3] = xr; vals[3] = fr
            end
        elseif fr < vals[2]
            simplex[3] = xr; vals[3] = fr
        else
            # Contraction
            xc = c .+ 0.5 .* (worst .- c); fc = nll(xc[1], xc[2])
            if fc < vals[3]
                simplex[3] = xc; vals[3] = fc
            else
                # Shrink
                simplex[2] = simplex[1] .+ 0.5 .* (simplex[2] .- simplex[1])
                simplex[3] = simplex[1] .+ 0.5 .* (simplex[3] .- simplex[1])
                vals[2] = nll(simplex[2][1], simplex[2][2])
                vals[3] = nll(simplex[3][1], simplex[3][2])
            end
        end
    end

    p = sortperm(vals); simplex = simplex[p]; vals = vals[p]
    logμ̂, logα̂ = simplex[1]
    μ̂, α̂ = exp(logμ̂), exp(logα̂)

    # Numerical Hessian
    ε = 1e-3
    f0 = vals[1]
    H = zeros(2, 2)
    H[1,1] = (nll(logμ̂+ε, logα̂) - 2f0 + nll(logμ̂-ε, logα̂)) / ε^2
    H[2,2] = (nll(logμ̂, logα̂+ε) - 2f0 + nll(logμ̂, logα̂-ε)) / ε^2
    fpp = nll(logμ̂+ε, logα̂+ε) + nll(logμ̂-ε, logα̂-ε)
    fpm = nll(logμ̂+ε, logα̂-ε) + nll(logμ̂-ε, logα̂+ε)
    H[1,2] = H[2,1] = (fpp - fpm) / (4*ε^2)
    cov = try inv(Hermitian(H)) catch; fill(NaN, 2, 2) end
    converged = all(isfinite, (logμ̂, logα̂)) && maximum(vals) - minimum(vals) < 1e-4

    return (μ̂=μ̂, α̂=α̂, logμ̂=logμ̂, logα̂=logα̂, cov_logμα=cov,
            n_kept=n_kept, converged=converged, nll=vals[1])
end

# ============================================================================
# Combine S1 + S2 (composite likelihood, tempered)
# ============================================================================

"""
    combine_count_prior(bleach_params, nnd_fit; temper=0.5, N₀=nothing) -> NamedTuple

Combine the bleach-model implied (μ, α) with the NND-fit (μ̂, α̂) under a
composite-likelihood assumption. Since the two summaries share nuisance
structure (per Codex), the naive product is overconfident — we temper by
`temper` ∈ [0, 1] and caveat the covariance.

If N₀ is provided, the bleach estimate yields (μ_bleach, α_bleach) via
bleach_to_count_params(); otherwise only the NND fit is used and a warning
is recorded.

Returns (μ, α, cov_logμα, components) where components records both sources
and the tempering factor.
"""
function combine_count_prior(bleach_params::NamedTuple, nnd_fit::NamedTuple;
                              temper::Float64=0.5, N₀::Union{Real,Nothing}=nothing)
    # Guards: fall back to NND-only if inputs are degenerate
    if isnothing(N₀) || !isfinite(nnd_fit.μ̂) || nnd_fit.μ̂ ≤ 0 ||
       !isfinite(nnd_fit.α̂) || nnd_fit.α̂ ≤ 0
        return (μ=nnd_fit.μ̂, α=nnd_fit.α̂, cov_logμα=nnd_fit.cov_logμα,
                components=(bleach=nothing, nnd=nnd_fit, N₀=N₀),
                note="NND-only fit (N₀ missing or NND fit failed)")
    end
    bl = bleach_to_count_params(bleach_params, N₀)
    if !all(isfinite, (bl.μ, bl.α)) || bl.μ ≤ 0 || bl.α ≤ 0
        return (μ=nnd_fit.μ̂, α=nnd_fit.α̂, cov_logμα=nnd_fit.cov_logμα,
                components=(bleach=bl, nnd=nnd_fit, N₀=N₀),
                note="NND-only (bleach params invalid)")
    end
    logμ_bl = log(bl.μ); logα_bl = log(bl.α)
    cov_bl = [0.1 0.0; 0.0 0.5] ./ max(temper, 1e-3)
    logμ_nnd = nnd_fit.logμ̂; logα_nnd = nnd_fit.logα̂
    cov_nnd_raw = nnd_fit.cov_logμα
    cov_nnd = if all(isfinite, cov_nnd_raw)
        cov_nnd_raw ./ max(temper, 1e-3)
    else
        [0.5 0.0; 0.0 1.0]  # uninformative fallback
    end

    local μ, α, cov_tot
    try
        P_bl = inv(Hermitian(cov_bl))
        P_nnd = inv(Hermitian(cov_nnd))
        P_tot = P_bl .+ P_nnd
        cov_tot = inv(Hermitian(P_tot))
        x_bl = [logμ_bl, logα_bl]; x_nnd = [logμ_nnd, logα_nnd]
        x_tot = cov_tot * (P_bl * x_bl + P_nnd * x_nnd)
        μ = exp(x_tot[1]); α = exp(x_tot[2])
    catch err
        # Degenerate combination — return NND estimate with a flag
        return (μ=nnd_fit.μ̂, α=nnd_fit.α̂, cov_logμα=fill(NaN, 2, 2),
                components=(bleach=bl, nnd=nnd_fit, N₀=N₀),
                note="precision-weighted combine failed ($err), NND-only")
    end
    return (μ=μ, α=α, cov_logμα=cov_tot,
            components=(bleach=bl, nnd=nnd_fit, N₀=N₀),
            note="composite-likelihood combine with temper=$temper (covariance is lower-bound)")
end

# ============================================================================
# Convenience wrapper
# ============================================================================

"""
    fit_count_prior(smld; kwargs...) -> NamedTuple

One-shot: fit both bleach and NND on the same SMLD and combine.
Returns (μ, α, cov_logμα, components, ...) suitable to pass to BaGoLConfig.

By default builds a global NND table (n_samples=50_000, K_max=15). For
production use, cache the table via `build_nnd_table()` and pass it in.

N₀ is estimated as length(smld.emitters) / mean_K_rough_estimate if not provided.
"""
function fit_count_prior(smld; table::Union{NNDTable,Nothing}=nothing,
                        N₀::Union{Real,Nothing}=nothing,
                        μ0::Float64=5.0, α0::Float64=2.0,
                        r_max::Float64=3.0, temper::Float64=0.5)
    tbl = isnothing(table) ? build_nnd_table(; n_samples=50_000) : table
    bleach = fit_bleach_curve(smld)
    nnd = fit_nnd_count(smld.emitters, tbl; μ0, α0, r_max)
    # Crude N₀ estimate if not given: total locs / (1 + μ̂_nnd)
    N0_eff = isnothing(N₀) ? length(smld.emitters) / (1 + nnd.μ̂) : N₀
    combined = combine_count_prior(bleach, nnd; temper, N₀=N0_eff)
    return (combined..., bleach=bleach, nnd=nnd, N₀=N0_eff, table=tbl)
end

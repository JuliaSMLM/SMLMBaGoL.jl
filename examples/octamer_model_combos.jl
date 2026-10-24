# Octamer — model-combination comparison
# =======================================
# Simulates ONE 8-emitter ring (an "octamer") and runs BaGoL under eight valid
# combinations of {spatial prior} × {allocation model} × {K prior}, then compares
# the recovered emitter count and localization accuracy. This shows how the
# modeling choices interact on identical data.
#
# Data are simulated with a Poisson(10) blink model. All BaGoL arms use the fixed
# correct count prior (learn_distribution=false, μ=10, shape=1000), with no tau
# finder and no motion. The flat / ipp + Poisson-K arms with "fixed ρ" additionally fix
# the emitter density ρ over the unpadded field bounding box (learn_rho=false, ρ = N/A_field) — the correct
# fixed-ρ setup after the 1/A area cancellation; the plain ipp arm keeps ρ learning.
#
# Takeaway (one seeded run per arm; seed 42, ρ fixed at N/A_field so the Poisson(ρA) prior has mean 8):
# locmix recovers N=8 (count-driven). With the Poisson-K prior at the correct fixed ρ, the flat and
# ipp arms land at N=9-10 (flat + DM, ipp + DM: 9; flat + categorical: 10), and with ρ learned the
# ipp arm sits low (N=5). Without a K prior, flat + categorical sits low (N=6) and flat + decoupled
# (no K penalty) sits high (N=11). So the flat-family result is set by the K prior and ρ, not by
# locmix's count-driven target. (An earlier version of this script fixed ρ from the padded
# partition area, which made the intended Poisson mean about 0.4 instead of 8 and made every
# fixed-ρ arm look low; that conclusion was an artifact.) The flat marginal still carries an Occam
# volume factor (−½·log det Λ) that penalizes splitting close/co-located emitters, which the
# localization-mixture prior of locmix cancels (≡ flat with ρ = 1/(2πσ²), one emitter per
# resolution cell). At g ≈ 1 over a 40 nm ring (h = 0.2 µm) the ipp arms equal the flat arms.
# Arm order matters a little: this script reseeds per arm, but a later arm can land on a different N.
#
# The comparison grid (all entries pass run_bagol validation):
#   spatial_model    ∈ {:ipp, :locmix, :flat}   (:locmix is the run_bagol default)
#   allocation_model ∈ {:dm, :decoupled, :categorical}
#   k_prior          ∈ {:auto, :poisson, :none}   (:poisson valid with :flat and :ipp;
#                                                   :locmix carries no K prior)
#
# Run with: julia --threads=auto --project=examples examples/octamer_model_combos.jl

using Pkg; Pkg.activate(@__DIR__)
using SMLMBaGoL
using SMLMData
using CairoMakie      # activates BaGoLMakieExt
using Random, Statistics, Printf

# =============================================================================
# Parameters
# =============================================================================

const SEED        = 42
const N_EMITTERS  = 8
const DIAMETER    = 0.040     # μm ring diameter (40 nm; ~15 nm adjacent spacing)
const PSF_SIGMA   = 0.130     # μm (130 nm)
const PHOTON_MEAN = 500.0
const PHOTON_MIN  = 100.0
const BLINK_MEAN  = 10.0      # localizations per emitter
const N_ITER      = 10000
const BURN_IN     = 2000
const MATCH_THRESH = 0.030    # μm, Hungarian match radius to truth

# =============================================================================
# Simulate one octamer
# =============================================================================

Random.seed!(SEED)
sim = simulate_nmer(;
    n=N_EMITTERS, diameter=DIAMETER,
    mean_count=BLINK_MEAN, count_model=:poisson, psf_sigma=PSF_SIGMA,
    mean_photons=PHOTON_MEAN, min_photons=PHOTON_MIN,
    pixel_size=0.100, field_size=25.6,
)
print_simulation_summary(sim)
fov = compute_fov(sim.smld)

# Fixed emitter density over the analyzed field: ρ = N/A_field so the Poisson(ρ·A) K prior
# peaks at the true count. A_field is the unpadded bounding box of all localizations, the exposure
# run_bagol uses for the flat / ipp Poisson-K prior (a single partition's share is the whole field).
# (Flat / ipp + Poisson-K arms only; locmix has no ρ.)
_xs = extrema(e.x for e in sim.smld.emitters); _ys = extrema(e.y for e in sim.smld.emitters)
const A_FIELD   = (_xs[2] - _xs[1]) * (_ys[2] - _ys[1])
const RHO_FIXED = N_EMITTERS / A_FIELD

# =============================================================================
# The model combinations
# =============================================================================

combos = [
    (label="ipp + DM (learned ρ)",            spatial=:ipp,    alloc=:dm,          kp=:auto,    learn_rho=true,  rho=nothing),
    (label="ipp + DM + Poisson-K (fixed ρ)",           spatial=:ipp,    alloc=:dm,          kp=:poisson, learn_rho=false, rho=RHO_FIXED),
    (label="locmix + DM (count-driven)",               spatial=:locmix, alloc=:dm,          kp=:auto,    learn_rho=true,  rho=nothing),
    (label="locmix + categorical",                     spatial=:locmix, alloc=:categorical, kp=:auto,    learn_rho=true,  rho=nothing),
    (label="flat + DM + Poisson-K (fixed ρ)",          spatial=:flat,   alloc=:dm,          kp=:poisson, learn_rho=false, rho=RHO_FIXED),
    (label="flat + categorical + Poisson-K (fixed ρ)", spatial=:flat,   alloc=:categorical, kp=:poisson, learn_rho=false, rho=RHO_FIXED),
    (label="flat + categorical, no K prior",           spatial=:flat,   alloc=:categorical, kp=:none,    learn_rho=false, rho=nothing),
    (label="flat + decoupled (no K penalty)",          spatial=:flat,   alloc=:decoupled,   kp=:none,    learn_rho=false, rho=nothing),
]

# =============================================================================
# Run BaGoL under each combination on the SAME data
# =============================================================================

results = NamedTuple[]
for c in combos
    Random.seed!(SEED)   # identical RNG start per arm → fair comparison
    smld, _ = run_bagol(sim.smld;
        n_iterations=N_ITER, burn_in=BURN_IN,
        partition_sigma=Inf,               # single octamer, no partitioning
        # fixed correct count prior matched to the (Poisson) generative model; no
        # hierarchical learning, no tau finder, no motion. The sim redraws dim
        # localizations so the observed per-emitter mean equals mean_count.
        μ=sim.count_params.μ, shape=sim.count_params.shape, learn_distribution=false,
        learn_rho=c.learn_rho, rho=c.rho,
        se_adjust=0.0, motion=:none,
        spatial_model=c.spatial, allocation_model=c.alloc, k_prior=c.kp,
        posterior_pixel_size=0.001,
        posterior_xlim=(fov[1], fov[2]), posterior_ylim=(fov[3], fov[4]),
    )
    m = match_positions(smld.emitters, sim.true_positions; threshold=MATCH_THRESH)
    nmatched = count(>(0), m.assignments)
    rmse_nm  = isempty(m.matched_distances) ? NaN :
               1000 * sqrt(mean(m.matched_distances .^ 2))
    push!(results, (combo=c, smld=smld,
                    n=length(smld.emitters), nmatched=nmatched, rmse=rmse_nm))
    @printf("  %-42s  N=%2d (true %d)   matched=%d/%d   RMSE=%.1f nm\n",
            c.label, length(smld.emitters), N_EMITTERS, nmatched, N_EMITTERS, rmse_nm)
end

# =============================================================================
# Summary table + comparison figure
# =============================================================================

output_dir = joinpath(@__DIR__, "output", "octamer_combos")
rm(output_dir; force=true, recursive=true); mkpath(output_dir)

open(joinpath(output_dir, "summary.txt"), "w") do io
    @printf(io, "Octamer model-combination comparison (true N = %d)\n\n", N_EMITTERS)
    @printf(io, "%-42s %8s %10s %10s\n", "combination", "recov.N", "matched", "RMSE(nm)")
    for r in results
        @printf(io, "%-42s %8d %10s %10.1f\n",
                r.combo.label, r.n, "$(r.nmatched)/$(N_EMITTERS)", r.rmse)
    end
end

lx = 1000 .* [e.x for e in sim.smld.emitters]   # raw locs (nm), for context
ly = 1000 .* [e.y for e in sim.smld.emitters]
sig_nm = [1000*sqrt(e.σ_x*e.σ_y) for e in sim.smld.emitters]   # 1σ (nm); circles drawn at radius=1σ (diameter 2σ)
tx = 1000 .* [p[1] for p in sim.true_positions]
ty = 1000 .* [p[2] for p in sim.true_positions]

nrows = cld(length(results), 3)
fig = Figure(size=(1290, 410 * nrows + 40))
Label(fig[0, 1:3], "Octamer (true N = $N_EMITTERS): BaGoL MAP-N under each model combination";
      fontsize=19, font=:bold)
for (i, r) in enumerate(results)
    row, col = fldmod1(i, 3)
    ax = Axis(fig[row, col]; aspect=DataAspect(),
              title=@sprintf("%s\nrecovered N = %d   (matched %d/%d, RMSE %.1f nm)",
                             r.combo.label, r.n, r.nmatched, N_EMITTERS, r.rmse),
              titlesize=12, xlabel="x (nm)", ylabel="y (nm)")
    scatter!(ax, lx, ly; marker=:circle, markersize=2 .* sig_nm, markerspace=:data,  # radius=1σ
             color=(:steelblue,0.10), strokecolor=(:steelblue,0.45), strokewidth=0.5)
    scatter!(ax, tx, ty; color=:black, marker=:circle, markersize=15,        # truth
             strokecolor=:black, strokewidth=1.4, glowcolor=:white)
    ex = 1000 .* [e.x for e in r.smld.emitters]
    ey = 1000 .* [e.y for e in r.smld.emitters]
    scatter!(ax, ex, ey; color=:crimson, marker=:xcross, markersize=13)      # MAP-N
end
Legend(fig[nrows + 1, 1:3],
    [MarkerElement(color=(:steelblue,0.5), marker=:circle, markersize=8),
     MarkerElement(color=:black, marker=:circle, markersize=12),
     MarkerElement(color=:crimson, marker=:xcross, markersize=12)],
    ["localizations (1σ circles)", "true emitters", "BaGoL MAP-N emitters"];
    orientation=:horizontal, framevisible=false)

figpath = joinpath(output_dir, "octamer_combos.png")
save(figpath, fig; px_per_unit=2)
try; run(`show-result $figpath`); catch; end

println("\nResults in $output_dir")

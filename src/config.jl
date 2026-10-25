# BaGoLConfig — analysis settings forwarded to run_bagol
#
# Ecosystem convention: config struct fields match analysis kwargs
# with sensible defaults. Serializable to TOML for reproducibility.
# Runtime-computed values (posterior_xlim, posterior_ylim) are excluded.

"""
    BaGoLConfig <: SMLMData.AbstractSMLMConfig

Configuration for BaGoL analysis. Fields correspond to `run_bagol` keywords.
Hyperpriors (`μ_prior_shape`, `μ_prior_scale`, `shape_prior_shape`, `shape_prior_scale`,
`ρ_prior_shape`, `ρ_prior_rate`) are keywords of the bare `run_bagol` entry point,
not config fields.

# Count Model
- `μ=10.0`: Mean localizations per emitter. The keyword entry point's `μ=nothing`
  uses `μ_prior_shape * μ_prior_scale`, also 10.0 with the default hyperpriors.
- `shape=2.0`: Gamma shape (1=exponential/dSTORM, >1=peaked/DNA-PAINT)
- `learn_distribution=true`: Learn count params. `true`=both, `false`=fix both,
  `:mu`=learn μ only, `:shape`=learn shape only
- `gamma=nothing`: DM concentration. `nothing`=use shape (default), `Float64`=fixed
- `learn_rho=true`: Learn Poisson-K emitter density ρ independently of μ/shape.
  Partitioned `run_bagol` with the `:ipp` or `:flat` spatial model and Poisson K prior:
  partition i gets the exposure A_i = A_field·n_i/N_all (A_field = bounding box of all
  localizations, n_i = its localizations including overlap copies, N_all = all input
  localizations) in both its spatial area and its Poisson K prior, so they cancel at
  fixed ρ, and ρ ~ Gamma(a + ΣK_i, 1/(b + ΣA_i)) over all active clusters: the emitter
  density.
  The default prior Gamma(2, 1) acts as 1 μm² of pseudo-area: negligible on whole cells,
  material on fields below a few μm² (use the bare `run_bagol` keyword
  `ρ_prior_rate`; it is not a config field). The single-chain
  `run_collapsed_chain` update is unchanged.
  For fully fixed hyperparameters, set both `learn_distribution=false` and
  `learn_rho=false`.
- `rho=nothing`: Initial ρ in emitters per unit area. If provided with
  `learn_rho=false`, hold this fixed for Fazel-style fixed-ρ-over-bounding-box
  flat Poisson-K runs. When `nothing` (`:ipp`, `:flat`), ρ starts at `N_all/(μ₀·A_field)`.

# MCMC
- `n_iterations=4000`: Total MCMC iterations
- `burn_in=2000`: Burn-in before accumulating
- `sync_interval=100`: Iterations between global hierarchical updates
- `allocation_model=:dm`: `:dm` (Dirichlet-Multinomial), `:decoupled`
  (no partition prior), or `:categorical` (labeled K^(-N) — Fazel-equivalent
  when paired with `spatial_model=:flat` + `k_prior=:none`)
- `spatial_model=:locmix`: `:locmix` (default; localization mixture, normalized per
  partition, so its output depends on the partitioning: dense fields are under-counted),
  `:ipp`
  (inhomogeneous Poisson process with a fixed pilot intensity g and global density ρ;
  partition-invariant, calibrated for dense fields, but under-splits isolated structures
  such as rulers and origami), or `:flat` (uniform)
- `intensity_bandwidth=0.2`: Kernel bandwidth h (μm) of the `:ipp` pilot intensity g
- `k_prior=:auto`: K-prior gating. `:auto` uses spatial-model default
  (Poisson(ρA) under `:ipp` and `:flat`, none under `:locmix`); `:poisson` always
  include (valid with `:ipp` and `:flat`); `:none` always disable
- `n_restricted_scans=5`: Jain-Neal restricted Gibbs scans for split/merge
- `n_bd_substeps=5`: Birth/death substeps per BD selection

# Partitioning
- `partition_sigma=3.0`: DBSCAN threshold in sigma units (Inf=no partitioning)
- `min_partition_size=0`: Minimum locs per partition
- `max_partition_size=1000`: METIS-split partitions larger than this
- `skip_partition_size=typemax(Int)`: Skip partitions larger than this
- `overlap=:auto`: Overlap width for bisected partitions (`:auto` or Float64 in μm)
- `bridge_ratio=0.0`: Optional bridge refinement strength for DBSCAN clusters
- `min_split_size=3`: Minimum core component size for bridge refinement

# Uncertainty Correction (standalone; 0 in the integrated pipeline)
- `se_adjust=:auto`: Deliberately differs from the bare keyword default `0.0`, so
  bare and test calls do not run the finder. `:auto` runs the finder
  (`estimate_se_adjust`) and uses τ̂ (skipped if the SMLD is already σ-corrected).
  The finder's internal BaGoL E-steps use
  this config's partitioning and model settings; `n_iterations`/`burn_in` stay the
  finder's own. `:auto` is not supported with `motion=:linear` (throws; give a numeric
  τ). `0.0` = no correction; a numeric value = manual τ (μm) added in quadrature to
  per-loc σ (σ²+τ²) — scalar,
  `(τx,τy)` tuple, length-N vector, or `(τx_vec,τy_vec)`.
- `force_se_adjust=false`: Apply even if the SMLD is already σ-corrected.

# Output
- `posterior_pixel_size=0.001`: Rao-Blackwellized posterior image pixel size in μm (1 nm; 0.0=disable)
- `archive_path=nothing`: Mmap chain archive path (nothing=disable)
- `progress_file=nothing`: Write progress to file (nothing=disable)
- `verbose=true`: Print progress to stdout
"""
Base.@kwdef struct BaGoLConfig <: SMLMData.AbstractSMLMConfig
    # Count model
    μ::Float64 = 10.0
    shape::Float64 = 2.0
    learn_distribution::Union{Bool, Symbol} = true
    gamma::Union{Nothing, Float64} = nothing
    learn_rho::Bool = true
    rho::Union{Nothing, Float64} = nothing

    # MCMC
    n_iterations::Int = 4000
    burn_in::Int = 2000
    sync_interval::Int = 100
    allocation_model::Symbol = :dm
    spatial_model::Symbol = :locmix
    # Pilot-intensity kernel bandwidth (μm) for spatial_model=:ipp.
    intensity_bandwidth::Float64 = DEFAULT_INTENSITY_BANDWIDTH
    # K prior gating (:auto = use spatial-model default; :poisson = always include
    # Poisson(ρA); :none = never). ρ updates are controlled by learn_rho.
    # :auto preserves existing behavior.
    k_prior::Symbol = :auto
    n_restricted_scans::Int = 5
    n_bd_substeps::Int = 5

    # Per-emitter linear motion (opt-in; default off). `:linear` lets each emitter
    # drift linearly in time (θ(t)=μ+v·δ, v integrated out), recovering emitters whose
    # blinks span a moving track instead of over-splitting them; `motion_sigma` is the
    # per-axis end-to-end drift prior SD in μm. Needs per-loc `frame` spanning the
    # acquisition. Reported velocities + mean±std land in the diagnostics Info.
    motion::Symbol = :none
    motion_sigma::Float64 = 0.002

    # Partitioning
    partition_sigma::Float64 = 3.0
    min_partition_size::Int = 0
    max_partition_size::Int = 1000
    skip_partition_size::Int = typemax(Int)
    overlap::Union{Float64, Symbol} = :auto
    bridge_ratio::Float64 = 0.0
    min_split_size::Int = 3

    # Uncertainty correction (standalone; leave 0 in the integrated pipeline,
    # where σ is corrected upstream and stamped metadata["sigma_corrected"]=true)
    se_adjust::Union{Float64, Tuple{Float64,Float64}, AbstractVector{Float64}, Symbol} = :auto
    force_se_adjust::Bool = false

    # Output
    posterior_pixel_size::Float64 = 0.001
    archive_path::Union{Nothing, String} = nothing
    progress_file::Union{Nothing, String} = nothing
    verbose::Bool = true
end

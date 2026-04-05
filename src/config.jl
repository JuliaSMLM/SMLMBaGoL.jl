# BaGoLConfig — 1:1 mapping with run_bagol kwargs
#
# Ecosystem convention: config struct fields match all interface kwargs
# with sensible defaults. Serializable to TOML for reproducibility.
# Runtime-computed values (posterior_xlim, posterior_ylim) are excluded.

"""
    BaGoLConfig <: SMLMData.AbstractSMLMConfig

Configuration for BaGoL analysis. All fields correspond 1:1 to `run_bagol` kwargs.

# Count Model
- `μ=10.0`: Mean localizations per emitter
- `shape=2.0`: Gamma shape (1=exponential/dSTORM, >1=peaked/DNA-PAINT)
- `learn_distribution=true`: Learn count params. `true`=both, `false`=fix both,
  `:mu`=learn μ only, `:shape`=learn shape only
- `gamma=nothing`: DM concentration. `nothing`=use shape (default), `Float64`=fixed

# MCMC
- `n_iterations=10000`: Total MCMC iterations
- `burn_in=2000`: Burn-in before accumulating
- `sync_interval=500`: Iterations between global hierarchical updates
- `allocation_model=:dm`: `:dm` (Dirichlet-Multinomial) or `:decoupled`
- `spatial_model=:locmix`: `:locmix` (localization mixture) or `:flat`
- `n_restricted_scans=5`: Jain-Neal restricted Gibbs scans for split/merge
- `n_bd_substeps=5`: Birth/death substeps per BD selection

# Partitioning
- `partition_sigma=3.0`: DBSCAN threshold in sigma units (Inf=no partitioning)
- `min_partition_size=0`: Minimum locs per partition
- `max_partition_size=1000`: METIS-split partitions larger than this
- `skip_partition_size=typemax(Int)`: Skip partitions larger than this
- `overlap=:auto`: Overlap width for bisected partitions (`:auto` or Float64 in μm)

# Output
- `posterior_pixel_size=0.002`: Rao-Blackwellized posterior image pixel size (0.0=disable)
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

    # MCMC
    n_iterations::Int = 10000
    burn_in::Int = 2000
    sync_interval::Int = 500
    allocation_model::Symbol = :dm
    spatial_model::Symbol = :locmix
    n_restricted_scans::Int = 5
    n_bd_substeps::Int = 5

    # Partitioning
    partition_sigma::Float64 = 3.0
    min_partition_size::Int = 0
    max_partition_size::Int = 1000
    skip_partition_size::Int = typemax(Int)
    overlap::Union{Float64, Symbol} = :auto

    # Output
    posterior_pixel_size::Float64 = 0.002
    archive_path::Union{Nothing, String} = nothing
    progress_file::Union{Nothing, String} = nothing
    verbose::Bool = true
end

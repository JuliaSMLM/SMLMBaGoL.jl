# Main BaGoL API

# ============================================================================
# se_adjust — independent (residual) localization-error correction
# σ²_eff = σ²_CRLB + τ²  (quadrature variance add, per axis; τ in μm). Standalone
# path; in the integrated pipeline σ is corrected upstream (SMLMClustering) and
# se_adjust stays 0, self-guarded against an already-σ-corrected SMLD.
# ============================================================================

"""
    _resolve_tau(se_adjust, n) -> (τx, τy)

Normalize an `se_adjust` spec into per-localization τ vectors (length `n`, μm):
`Real` → both axes/all locs; `(τx,τy)` tuple → per-axis; length-`n` vector →
per-loc (both axes); `(τx_vec, τy_vec)` → per-loc per-axis. All τ must be ≥ 0.
"""
function _resolve_tau(se, n::Int)
    τx, τy = if se isa Real
        v = Float64(se); (fill(v, n), fill(v, n))
    elseif se isa Tuple{<:Real, <:Real}
        (fill(Float64(se[1]), n), fill(Float64(se[2]), n))
    elseif se isa Tuple{<:AbstractVector, <:AbstractVector}
        (length(se[1]) == n && length(se[2]) == n) ||
            throw(ArgumentError("se_adjust per-axis vectors must each have length n_locs=$n"))
        (Float64.(se[1]), Float64.(se[2]))
    elseif se isa AbstractVector
        length(se) == n ||
            throw(ArgumentError("se_adjust per-loc vector length $(length(se)) != n_locs $n"))
        v = Float64.(se); (v, copy(v))
    else
        throw(ArgumentError("se_adjust must be a scalar, (τx,τy) tuple, length-n vector, or (τx_vec,τy_vec); got $(typeof(se))"))
    end
    (any(<(0.0), τx) || any(<(0.0), τy)) &&
        throw(ArgumentError("se_adjust τ values must be ≥ 0 (μm)"))
    return τx, τy
end

"""
    _inflate_sigma(e, τx, τy) -> Emitter2DFit{Float64}

Copy of `e` with σ inflated in quadrature per axis: σ_x'=√(σ_x²+τx²),
σ_y'=√(σ_y²+τy²). σ_xy and all non-σ fields unchanged.
"""
function _inflate_sigma(e::SMLMData.Emitter2DFit, τx::Real, τy::Real)
    return SMLMData.Emitter2DFit(
        Float64(e.x), Float64(e.y), Float64(e.photons), Float64(e.bg),
        sqrt(Float64(e.σ_x)^2 + τx^2), sqrt(Float64(e.σ_y)^2 + τy^2), Float64(e.σ_xy),
        Float64(e.σ_photons), Float64(e.σ_bg), e.frame, e.dataset, e.track_id, e.id)
end

"""
    _maybe_apply_se_adjust(locs, metadata, se_adjust, force_se_adjust)
        -> (locs, tau_applied, msg::String)

Apply se_adjust σ-inflation unless τ is all-zero, or the SMLD is already
σ-corrected (`metadata["sigma_corrected"]==true`) and `force_se_adjust` is false
(apply-once guard). `tau_applied` is `(τx, τy)` (μm; representative max for per-loc
τ) when applied, else `nothing`. Returns possibly-new locs, `tau_applied`, log line.
"""
function _maybe_apply_se_adjust(locs, metadata, se_adjust, force_se_adjust::Bool)
    n = length(locs)
    n == 0 && return locs, nothing, ""
    τx, τy = _resolve_tau(se_adjust, n)
    (any(>(0.0), τx) || any(>(0.0), τy)) || return locs, nothing, ""   # all-zero → no-op
    already = get(metadata, "sigma_corrected", false) === true
    if already && !force_se_adjust
        @warn "run_bagol: input SMLD is already σ-corrected (metadata sigma_corrected=true); skipping se_adjust to avoid double-counting τ. Pass force_se_adjust=true to override."
        return locs, nothing, ""
    end
    already && @warn "run_bagol: applying se_adjust on top of an already σ-corrected SMLD (force_se_adjust=true) — τ is intentionally double-counted."
    inflated = [_inflate_sigma(locs[i], τx[i], τy[i]) for i in 1:n]
    tau_applied = (maximum(τx), maximum(τy))
    msg = "  se_adjust applied: σ inflated in quadrature (max τx=$(round(1000*tau_applied[1], digits=1)) nm, max τy=$(round(1000*tau_applied[2], digits=1)) nm)"
    return inflated, tau_applied, msg
end

"""
    apply_se_adjust(smld, se_adjust; force_se_adjust=false) -> BasicSMLD

Return a copy of `smld` with per-localization σ inflated in quadrature by
`se_adjust` (σ² → σ² + τ², per axis; see `run_bagol`'s `se_adjust`). No-op
(returns `smld` unchanged) if `se_adjust` is 0 or the SMLD is already σ-corrected
(unless `force_se_adjust`). Used by `run_bagol` and by renders to display the
localization uncertainty BaGoL actually used.
"""
function apply_se_adjust(smld::SMLMData.SMLD, se_adjust; force_se_adjust::Bool=false)
    md = hasproperty(smld, :metadata) ? smld.metadata : Dict{String,Any}()
    locs, tau, _ = _maybe_apply_se_adjust(smld.emitters, md, se_adjust, force_se_adjust)
    tau === nothing && return smld
    return SMLMData.BasicSMLD(locs, smld.camera, smld.n_frames, smld.n_datasets, md)
end

# ============================================================================
# Main API: run_bagol returns (BasicSMLD, BaGoLDiagnostics)
# ============================================================================

"""
    run_bagol(smld::SMLD; kwargs...) -> (BasicSMLD, BaGoLDiagnostics)

Run BaGoL analysis on an SMLD. Returns grouped emitters as BasicSMLD and diagnostics.

Uses precision-weighted DBSCAN to partition localizations, then runs parallel
collapsed Gibbs MCMC on each partition with global hierarchical updates.

# Count Model
n_j ~ Gamma(shape, μ/shape) where:
- μ = mean locs per emitter
- shape = 1: exponential (dSTORM), shape > 1: peaked (DNA-PAINT)

# Partitioning Arguments
- `partition_sigma=3.0`: DBSCAN threshold in sigma units (Inf = no partitioning)
- `min_partition_size=0`: Minimum locs per partition (smaller clusters dropped as noise)
- `max_partition_size=1000`: Split partitions larger than this
- `skip_partition_size=typemax(Int)`: Skip partitions larger than this

# Uncertainty Correction (standalone — leave 0 in the integrated pipeline)
- `se_adjust=0.0`: Independent extra position error τ (μm) added in quadrature to
  the per-loc CRLB σ (σ²_eff = σ² + τ²). Accepts a scalar (both axes), a `(τx, τy)`
  tuple (per-axis), a length-N vector (per-loc), or `(τx_vec, τy_vec)`. Corrects
  BaGoL under-grouping when CRLB underestimates the true localization error. If the
  input SMLD is already σ-corrected (`metadata["sigma_corrected"]==true`), se_adjust
  is skipped with a warning unless `force_se_adjust=true`.
- `force_se_adjust=false`: Override the already-corrected guard (intentional double-count).

# MCMC Arguments
- `sync_interval=500`: Iterations between global μ/shape updates
- `n_iterations=10000`: Total MCMC iterations
- `burn_in=2000`: Burn-in iterations before recording
- `shape=2.0`: Initial Gamma shape (1=exponential, higher=more peaked)
- `learn_distribution=true`: Control count distribution learning.
  `true`=learn both μ and shape, `false`=fix both,
  `:mu`=learn μ only (fix shape), `:shape`=learn shape only (fix μ)
- `allocation_model=:dm`: `:dm` (Dirichlet-Multinomial), `:decoupled`
  (no partition prior), or `:categorical` (labeled K^(-N) — Fazel-equivalent
  when paired with `spatial_model=:flat` + `k_prior=:none`)
- `spatial_model=:locmix`: `:locmix` (localization mixture) or `:flat`
- `k_prior=:auto`: K-prior gating. `:auto` uses spatial-model default
  (Poisson(ρA) under `:flat`, none under `:locmix`); `:poisson` always
  include (only valid with `:flat`); `:none` always disable
- `verbose=true`: Print progress

# Posterior Image
- `posterior_pixel_size=0.002`: Rao-Blackwellized posterior image pixel size in μm (0.0 to disable)
- `posterior_xlim=nothing`: Override x bounds for posterior image
- `posterior_ylim=nothing`: Override y bounds for posterior image

# Archive
- `archive_path=nothing`: Enable mmap chain archive for post-hoc analysis

# Returns
- `BasicSMLD`: Grouped emitter positions with uncertainties
- `BaGoLDiagnostics`: n_emitters, posterior_k, acceptance_rates, final parameters
"""
function run_bagol(
    smld::SMLMData.SMLD;
    # Count model
    μ::Union{Nothing, Float64} = nothing,
    shape::Float64 = 2.0,
    learn_distribution::Union{Bool, Symbol} = true,
    gamma::Union{Nothing, Float64} = nothing,
    # MCMC
    n_iterations::Int = 10000,
    burn_in::Int = 2000,
    sync_interval::Int = 500,
    allocation_model::Symbol = :dm,
    spatial_model::Symbol = :locmix,
    k_prior::Symbol = :auto,
    n_restricted_scans::Int = 5,
    n_bd_substeps::Int = 5,
    # Partitioning
    partition_sigma::Float64 = 3.0,
    min_partition_size::Int = 0,
    max_partition_size::Int = 1000,
    skip_partition_size::Int = typemax(Int),
    overlap::Union{Float64, Symbol} = :auto,
    # Uncertainty correction (standalone; leave 0 in the integrated pipeline)
    se_adjust::Union{Real, Tuple, AbstractVector} = 0.0,
    force_se_adjust::Bool = false,
    # Output
    posterior_pixel_size::Float64 = 0.002,
    posterior_xlim::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    posterior_ylim::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    archive_path::Union{Nothing, String} = nothing,
    progress_file::Union{Nothing, String} = nothing,
    verbose::Bool = true,
    # Hyperpriors (rarely changed)
    μ_prior_shape::Float64 = 2.0,
    μ_prior_scale::Float64 = 5.0,
    shape_prior_shape::Float64 = 2.0,
    shape_prior_scale::Float64 = 1.0,
    ρ_prior_shape::Float64 = 2.0,
    ρ_prior_rate::Float64 = 1.0,
)
    return _run_bagol_collapsed(smld;
        partition_sigma, min_partition_size, max_partition_size, skip_partition_size,
        overlap, se_adjust, force_se_adjust, sync_interval, n_iterations, burn_in, shape, learn_distribution,
        posterior_pixel_size, posterior_xlim, posterior_ylim,
        archive_path, progress_file, verbose,
        μ=μ, gamma=gamma, allocation_model=allocation_model,
        spatial_model=spatial_model, k_prior=k_prior,
        n_restricted_scans=n_restricted_scans,
        n_bd_substeps=n_bd_substeps,
        μ_prior_shape=μ_prior_shape, μ_prior_scale=μ_prior_scale,
        shape_prior_shape=shape_prior_shape, shape_prior_scale=shape_prior_scale,
        ρ_prior_shape=ρ_prior_shape, ρ_prior_rate=ρ_prior_rate)
end

"""
    run_bagol(smld, cfg::BaGoLConfig; posterior_xlim=nothing, posterior_ylim=nothing)

Run BaGoL from a config struct. Runtime-only kwargs (posterior bounds) are separate.
"""
function run_bagol(
    smld::SMLMData.SMLD,
    cfg::BaGoLConfig;
    posterior_xlim::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    posterior_ylim::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
)
    return run_bagol(smld;
        μ=cfg.μ, shape=cfg.shape, learn_distribution=cfg.learn_distribution,
        gamma=cfg.gamma,
        n_iterations=cfg.n_iterations, burn_in=cfg.burn_in,
        sync_interval=cfg.sync_interval,
        allocation_model=cfg.allocation_model, spatial_model=cfg.spatial_model,
        k_prior=cfg.k_prior,
        n_restricted_scans=cfg.n_restricted_scans, n_bd_substeps=cfg.n_bd_substeps,
        partition_sigma=cfg.partition_sigma,
        min_partition_size=cfg.min_partition_size,
        max_partition_size=cfg.max_partition_size,
        skip_partition_size=cfg.skip_partition_size,
        overlap=cfg.overlap,
        se_adjust=cfg.se_adjust, force_se_adjust=cfg.force_se_adjust,
        posterior_pixel_size=cfg.posterior_pixel_size,
        posterior_xlim=posterior_xlim, posterior_ylim=posterior_ylim,
        archive_path=cfg.archive_path, progress_file=cfg.progress_file,
        verbose=cfg.verbose)
end

# ============================================================================
# Per-cluster overlap masks for hierarchical learning
# ============================================================================

"""
    _compute_cluster_overlap_masks(states, partitions) -> Vector{BitVector}

Compute per-cluster overlap masks for all partitions. Returns masks where
`masks[i][j] = true` iff cluster j in partition i contains NO overlap locs.

Used to exclude overlap-contaminated clusters from hierarchical μ/shape/ρ
learning and from the reported cluster size histogram.
"""
function _compute_cluster_overlap_masks(states, partitions)
    masks = Vector{BitVector}(undef, length(states))
    for i in eachindex(states)
        n_slots = length(states[i].clusters)
        m = trues(n_slots)
        is_ov = partitions[i].is_overlap
        if any(is_ov)
            @inbounds for loc_idx in eachindex(states[i].assignments)
                if is_ov[loc_idx]
                    cj = states[i].assignments[loc_idx]
                    cj > 0 && (m[cj] = false)
                end
            end
        end
        masks[i] = m
    end
    return masks
end

# ============================================================================
# Collapsed Gibbs sampler dispatch
# ============================================================================

function _run_bagol_collapsed(
    smld::SMLMData.SMLD;
    partition_sigma::Float64 = 3.0,
    min_partition_size::Int = 0,
    max_partition_size::Int = 1000,
    skip_partition_size::Int = typemax(Int),
    overlap::Union{Float64, Symbol} = :auto,
    se_adjust::Union{Real, Tuple, AbstractVector} = 0.0,
    force_se_adjust::Bool = false,
    sync_interval::Int = 500,
    n_iterations::Int = 10000,
    burn_in::Int = 2000,
    shape::Float64 = 2.0,
    learn_distribution::Union{Bool, Symbol} = true,
    posterior_pixel_size::Float64 = 0.002,
    posterior_xlim::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    posterior_ylim::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    archive_path::Union{Nothing, String} = nothing,
    progress_file::Union{Nothing, String} = nothing,
    verbose::Bool = true,
    μ::Union{Nothing, Float64} = nothing,
    gamma::Union{Nothing, Float64} = nothing,
    allocation_model::Symbol = :dm,
    spatial_model::Symbol = :locmix,
    k_prior::Symbol = :auto,
    n_restricted_scans::Int = 5,
    n_bd_substeps::Int = 3,
    μ_prior_shape::Float64 = 2.0,
    μ_prior_scale::Float64 = 5.0,
    shape_prior_shape::Float64 = 2.0,
    shape_prior_scale::Float64 = 1.0,
    ρ_prior_shape::Float64 = 2.0,
    ρ_prior_rate::Float64 = 1.0,
    # Internal (τ-finder use): if a Ref is passed, the global per-loc Dahl-consensus
    # assignment is written to it (0 = unassigned / overlap-dup / skipped). Not public API.
    _dahl_out::Union{Nothing, Base.RefValue{Vector{Int}}} = nothing,
)
    # Convert bounds to Float64 (GPU fitters produce Float32 coordinates)
    posterior_xlim = posterior_xlim === nothing ? nothing : (Float64(posterior_xlim[1]), Float64(posterior_xlim[2]))
    posterior_ylim = posterior_ylim === nothing ? nothing : (Float64(posterior_ylim[1]), Float64(posterior_ylim[2]))

    # Validate learn_distribution
    if learn_distribution isa Symbol && learn_distribution ∉ (:mu, :shape)
        throw(ArgumentError("learn_distribution must be true, false, :mu, or :shape (got :$learn_distribution)"))
    end
    _learn_mu = learn_distribution === true || learn_distribution === :mu
    _learn_shape = learn_distribution === true || learn_distribution === :shape

    locs = smld.emitters
    camera = smld.camera

    # Progress logging: write directly to file (bypasses stdout buffering)
    _t_start = time()
    function _log_progress(msg::String)
        if progress_file !== nothing
            open(progress_file, "a") do io
                elapsed = round(time() - _t_start, digits=1)
                println(io, "[$(elapsed)s] $msg")
            end
        end
        verbose && println(msg)
    end

    # Independent-error correction (standalone path). No-op when se_adjust=0 or
    # the SMLD is already σ-corrected upstream (apply-once guard).
    _se_md = hasproperty(smld, :metadata) ? smld.metadata : Dict{String,Any}()
    locs, _se_tau, _se_msg = _maybe_apply_se_adjust(locs, _se_md, se_adjust, force_se_adjust)
    _se_tau !== nothing && _log_progress(_se_msg)

    _log_progress("Partitioning $(length(locs)) localizations (partition_sigma=$partition_sigma)...")

    partitions, skipped = partition_locs(locs; partition_sigma, min_size=min_partition_size,
                                          max_size=max_partition_size,
                                          skip_size=skip_partition_size,
                                          overlap=overlap)

    _log_progress("  Created $(length(partitions)) partitions")
    if !isempty(skipped)
        n_skipped = sum(length(p.locs) for p in skipped)
        _log_progress("  Skipped $(length(skipped)) oversized clusters ($n_skipped locs)")
    end

    if isempty(partitions)
        @warn "No valid partitions"
        empty_smld = SMLMData.BasicSMLD(SMLMData.Emitter2DFit{Float64}[], camera, 1, 1)
        empty_diag = BaGoLDiagnostics(0, Int[], Dict{Symbol,Float64}(), 0.0, shape, 0.0, 0, Int[], Int[], Int[], nothing, _se_tau,
            (; iters=Int[], K=Int[], mu=Float64[], shape=Float64[], rho=Float64[]))
        return empty_smld, empty_diag
    end

    n_overlap_locs = sum(count(p.is_overlap) for p in partitions)

    n_partitions = length(partitions)

    # Allocation model
    allocation_model in (:dm, :decoupled, :categorical) ||
        throw(ArgumentError("allocation_model must be :dm, :decoupled, or :categorical"))
    am = if allocation_model === :dm
        DMAllocation(gamma)
    elseif allocation_model === :decoupled
        DecoupledAllocation()
    else
        CategoricalAllocation()
    end

    spatial_model_sym = spatial_model
    spatial_model_sym in (:locmix, :flat) ||
        throw(ArgumentError("spatial_model must be :locmix or :flat"))

    k_prior in (:auto, :poisson, :none) ||
        throw(ArgumentError("k_prior must be :auto, :poisson, or :none (got :$k_prior)"))

    # Hyperprior config
    config_nt = (μ_prior_shape=μ_prior_shape, μ_prior_scale=μ_prior_scale,
                 shape_prior_shape=shape_prior_shape, shape_prior_scale=shape_prior_scale,
                 ρ_prior_shape=ρ_prior_shape, ρ_prior_rate=ρ_prior_rate)

    # Initialize collapsed states and accumulators per partition
    # Use concrete parametric type for the state vector
    _sp_type = spatial_model_sym === :locmix ? LocmixSpatial : FlatSpatial
    _am_type = if allocation_model === :dm
        DMAllocation
    elseif allocation_model === :decoupled
        DecoupledAllocation
    else
        CategoricalAllocation
    end
    states = Vector{CollapsedState{_sp_type, _am_type}}(undef, n_partitions)
    partition_accumulators = Vector{Vector{AbstractAccumulator}}(undef, n_partitions)
    count_hists = Vector{EmitterCountHist}(undef, n_partitions)
    partition_samples = Vector{PartitionSamples}(undef, n_partitions)
    partition_psms = Vector{PSMAccumulator}(undef, n_partitions)

    # Validate k_prior compatibility with spatial model BEFORE the @spawn
    # loop — otherwise the ArgumentError gets wrapped in CompositeException
    # and is hard to surface cleanly to callers.
    if k_prior === :poisson && spatial_model_sym !== :flat
        throw(ArgumentError("k_prior=:poisson is only valid with spatial_model=:flat (the prior is the flat-area-cancelled form). Got spatial_model=:$spatial_model_sym"))
    end

    @sync for i in 1:n_partitions
        Threads.@spawn begin
            p_locs = partitions[i].locs
            sp = if spatial_model_sym === :locmix
                LocmixSpatial(p_locs)
            else
                FlatSpatial(log(area(UniformSpatialPrior(p_locs))))
            end
            # Resolve k_prior into a concrete bool for this partition.
            # (Compatibility check already done outside the @spawn block.)
            use_kprior_i = if k_prior === :poisson
                true
            elseif k_prior === :none
                false
            else
                _uses_poisson_k_prior(sp)
            end
            states[i] = initialize_collapsed_state(p_locs, sp, am;
                use_poisson_k_prior=use_kprior_i)

            # Per-partition accumulators
            accs = AbstractAccumulator[]
            count_hist = EmitterCountHist()
            push!(accs, count_hist)
            count_hists[i] = count_hist

            ps_acc = PartitionSamples(thin=5)
            push!(accs, ps_acc)
            partition_samples[i] = ps_acc

            psm_acc = PSMAccumulator()
            push!(accs, psm_acc)
            partition_psms[i] = psm_acc

            if posterior_pixel_size > 0.0
                # Each partition gets a local posterior image (auto-sized from its locs).
                # These are merged into a full-field image after sampling.
                push!(accs, PosteriorImage(pixel_size=posterior_pixel_size))
            end

            partition_accumulators[i] = accs
        end
    end

    # Global μ, shape, ρ (shared across partitions)
    # Direct μ kwarg takes precedence over hyperprior product
    μ = μ !== nothing ? Float64(μ) : μ_prior_shape * μ_prior_scale
    current_shape = shape
    ρ = ρ_prior_shape / ρ_prior_rate  # Initial ρ from prior mean

    # Per-partition areas (for conjugate ρ update)
    partition_areas = [spatial_area(states[i].spatial) for i in 1:n_partitions]

    # Log cluster-level overlap stats (now that states exist)
    if n_overlap_locs > 0
        init_masks = _compute_cluster_overlap_masks(states, partitions)
        n_total_clusters = sum(s.n_active for s in states)
        n_excluded = sum(count(j -> states[i].active[j] && !init_masks[i][j],
                               eachindex(states[i].active))
                         for (i, _) in enumerate(states))
        _log_progress("  Overlap filtering: $n_excluded/$n_total_clusters clusters excluded ($n_overlap_locs overlap locs)")
    end

    # Initialize archive if requested
    archive = nothing
    if archive_path !== nothing
        archive = BaGoLArchive(archive_path, n_partitions,
                               [length(p.locs) for p in partitions])
    end

    # Synchronized outer loop
    n_outer = div(n_iterations, sync_interval)
    iter_counters = zeros(Int, n_partitions)

    # Per-partition timing (cumulative nanoseconds per partition)
    partition_times_ns = zeros(UInt64, n_partitions)

    # Per-partition acceptance tracking
    _zero() = (0, 0)
    partition_acceptance = [Dict{Symbol, Tuple{Int, Int}}(
        :gibbs_sweep => _zero(), :split => _zero(), :merge => _zero(),
        :birth => _zero(), :death => _zero()
    ) for _ in 1:n_partitions]

    # Convergence trace — global K + learned hyperparams recorded at each sync (resolution = sync_interval)
    trace_iters = Int[]; trace_K = Int[]; trace_mu = Float64[]; trace_shape = Float64[]; trace_rho = Float64[]

    for outer in 1:n_outer
        @sync for i in 1:n_partitions
            Threads.@spawn begin
                t0 = time_ns()
                iter_counters[i] = run_collapsed_iterations!(
                    states[i], partitions[i].locs, sync_interval,
                    μ, current_shape, ρ,
                    partition_accumulators[i], burn_in, iter_counters[i];
                    acceptance=partition_acceptance[i],
                    n_restricted_scans=n_restricted_scans,
                    n_bd_substeps=n_bd_substeps
                )
                partition_times_ns[i] += time_ns() - t0
            end
        end

        # Write archive samples if past burn-in
        if archive !== nothing && (outer * sync_interval) > burn_in
            for i in 1:n_partitions
                write_sample!(archive, i, states[i], μ, current_shape)
            end
        end

        # Per-cluster overlap masks (recomputed each sync — assignments change during MCMC)
        cluster_masks = _compute_cluster_overlap_masks(states, partitions)

        # Global hierarchical updates — exclude clusters containing overlap locs
        # to prevent overlap inflation from biasing learned μ/shape/ρ
        if _learn_mu
            μ = _update_mu_collapsed_global!(states, μ, current_shape, config_nt;
                                              cluster_masks=cluster_masks)
        end
        if _learn_shape
            current_shape = _update_shape_collapsed_global!(states, μ, current_shape, config_nt;
                                                             cluster_masks=cluster_masks)
        end
        # Conjugate ρ update — gated on spatial model. Only the FlatSpatial
        # legacy path uses a Poisson(ρA) K prior; LocmixSpatial target has no
        # ρ per docs/math_reference.md. Partitions share spatial-model type
        # within a run, so it's enough to check the first state.
        ρ = if !isempty(states) && _uses_poisson_k_prior(states[1])
            _update_rho_collapsed_global!(states, partition_areas, config_nt;
                                          cluster_masks=cluster_masks)
        else
            ρ
        end

        total_K = sum(s.n_active for s in states)
        shape_str = _learn_shape ? ", shape=$(round(current_shape, digits=2))" : ""
        iter_done = outer * sync_interval
        _log_progress("Sync $outer/$n_outer (iter $iter_done): K=$total_K, μ=$(round(μ, digits=2))$shape_str, ρ=$(round(ρ, digits=4))")
        push!(trace_iters, iter_done); push!(trace_K, total_K); push!(trace_mu, μ)
        push!(trace_shape, current_shape); push!(trace_rho, ρ)
    end

    # Run remaining iterations
    remaining = n_iterations - n_outer * sync_interval
    if remaining > 0
        @sync for i in 1:n_partitions
            Threads.@spawn begin
                iter_counters[i] = run_collapsed_iterations!(
                    states[i], partitions[i].locs, remaining,
                    μ, current_shape, ρ,
                    partition_accumulators[i], burn_in, iter_counters[i];
                    acceptance=partition_acceptance[i],
                    n_restricted_scans=n_restricted_scans,
                    n_bd_substeps=n_bd_substeps
                )
            end
        end
    end

    # Final trace point at n_iterations (after the remaining partial-sync iterations)
    if remaining > 0
        push!(trace_iters, n_iterations); push!(trace_K, sum(s.n_active for s in states))
        push!(trace_mu, μ); push!(trace_shape, current_shape); push!(trace_rho, ρ)
    end

    _log_progress("MCMC complete. Extracting MAP-N emitters ($n_partitions partitions, threaded)...")

    # Extract emitters via MAP-N from stored assignment samples (threaded)
    sigmas = [mean_sigma(loc) for loc in locs]
    # Scale margin with partition_sigma: partition gap ≈ partition_sigma*(σ_i+σ_j), so emitters
    # can only be duplicates if within ~partition_sigma*σ of boundary
    boundary_margin = partition_sigma * median(sigmas)

    # Per-partition results (filled in parallel, largest-first for load balance)
    partition_emitters = Vector{Vector{SMLMData.Emitter2DFit{Float64}}}(undef, n_partitions)
    partition_boundary = Vector{Vector{Bool}}(undef, n_partitions)
    partition_dahl = Vector{Vector{Int}}(undef, n_partitions)  # per-loc Dahl labels (τ-finder)

    # Sort by partition size descending so large (expensive) partitions start first.
    # Use @spawn (dynamic scheduling) for better load balance across heavy-tailed sizes.
    pid_order = sortperm([length(p.locs) for p in partitions], rev=true)

    mapn_done = Threads.Atomic{Int}(0)
    mapn_log_interval = max(1, n_partitions ÷ 20)  # ~20 log lines total
    mapn_t0 = time()

    @sync for pid in pid_order
        Threads.@spawn begin
            partition = partitions[pid]
            ps_acc = partition_samples[pid]
            psm_acc = partition_psms[pid]
            samples = accumulator_result(ps_acc)
            if isempty(samples)
                error("Partition $pid ($(length(partition.locs)) locs): no assignment samples collected. " *
                      "Check burn_in ($burn_in) < n_iterations ($n_iterations).")
            end
            # Dahl+overlap MAP-N: Dahl consensus partition + overlap Hungarian matching
            psm_result = accumulator_result(psm_acc)
            psm = psm_result.psm
            _, _, _, dahl_assignments = estimate_dahl(samples, partition.locs, psm)
            partition_dahl[pid] = dahl_assignments    # capture before samples are cleared (τ-finder)
            emitters_i, _ = estimate_mapn_overlap(samples, partition.locs, dahl_assignments)
            if isempty(emitters_i)
                k_dahl = length(unique(dahl_assignments))
                error("Partition $pid ($(length(partition.locs)) locs): estimate_mapn_overlap returned " *
                      "0 emitters from $(length(samples)) samples (k_dahl=$k_dahl).")
            end
            bflags = [emitter_near_boundary(e, partition, boundary_margin) for e in emitters_i]
            partition_emitters[pid] = emitters_i
            partition_boundary[pid] = bflags
            # Free sample/PSM memory after extraction (large for big partitions)
            partition_samples[pid] = PartitionSamples(thin=typemax(Int))
            partition_psms[pid] = PSMAccumulator()

            # Progress logging for MAP-N extraction
            n_done = Threads.atomic_add!(mapn_done, 1) + 1
            if n_done % mapn_log_interval == 0 || n_done == n_partitions
                pct = round(100 * n_done / n_partitions, digits=0)
                dt = round(time() - mapn_t0, digits=1)
                _log_progress("  MAP-N: $n_done/$n_partitions ($(Int(pct))%) in $(dt)s [pid=$pid, N=$(length(partition.locs))]")
            end
        end
    end

    # Internal: expose the global per-loc Dahl-consensus assignment for the τ finder
    # (estimate_se_adjust grouping=:dahl). Each loc is labeled once from its CORE
    # partition (overlap-dup rows skipped); labels are offset per partition for global
    # uniqueness. A molecule split across partitions becomes two sub-groups (loses
    # cross-boundary pairs, never contaminates) — the accepted boundary nuance.
    if _dahl_out !== nothing
        global_labels = zeros(Int, length(locs))
        offset = 0
        for pid in 1:n_partitions
            da = partition_dahl[pid]
            oi = partitions[pid].original_indices
            ov = partitions[pid].is_overlap
            for j in eachindex(da)
                ov[j] && continue
                global_labels[oi[j]] = offset + da[j]
            end
            offset += isempty(da) ? 0 : maximum(da)
        end
        _dahl_out[] = global_labels
    end

    # Filter overlap emitters via Dahl membership: discard emitters whose
    # member locs are majority-overlap (they belong to a sibling partition).
    # Uses the Dahl consensus assignment to determine which locs compose each emitter.
    n_overlap_discarded = 0
    for pid in 1:n_partitions
        any(partitions[pid].is_overlap) || continue  # no overlap locs in this partition
        partition = partitions[pid]
        ps_acc = partition_samples[pid]
        psm_acc = partition_psms[pid]
        # Dahl assignments already computed above — recompute from stored result
        # (samples were cleared, but we stored the emitters; use emitter positions
        # to determine core/overlap membership)
        is_ov = partition.is_overlap
        keep = Bool[]
        for e in partition_emitters[pid]
            # Find which locs are closest to this emitter (within its cluster)
            # Simple heuristic: emitter position near overlap region → discard
            # Better: check if emitter's position is closer to core locs than overlap locs
            # Best: use Dahl assignment to count core vs overlap members
            # For now: use position-based check — emitter is "core" if its nearest
            # core loc is closer than its nearest overlap loc
            min_d_core = Inf
            min_d_overlap = Inf
            for (li, loc) in enumerate(partition.locs)
                d = (e.x - loc.x)^2 + (e.y - loc.y)^2
                if is_ov[li]
                    d < min_d_overlap && (min_d_overlap = d)
                else
                    d < min_d_core && (min_d_core = d)
                end
            end
            push!(keep, min_d_core <= min_d_overlap)
        end
        n_discard = count(.!keep)
        if n_discard > 0
            partition_emitters[pid] = partition_emitters[pid][keep]
            partition_boundary[pid] = partition_boundary[pid][keep]
            n_overlap_discarded += n_discard
        end
    end
    if n_overlap_discarded > 0
        _log_progress("  Overlap discard: $n_overlap_discarded emitters removed (majority-overlap membership)")
    end

    # Flatten results (preserving partition order)
    all_emitters = SMLMData.Emitter2DFit{Float64}[]
    partition_ids = Int[]
    is_near_boundary = Bool[]
    for pid in 1:n_partitions
        for (j, emitter) in enumerate(partition_emitters[pid])
            push!(all_emitters, emitter)
            push!(partition_ids, pid)
            push!(is_near_boundary, partition_boundary[pid][j])
        end
    end
    # Deduplicate boundary emitters
    n_pre_dedup = length(all_emitters)
    n_boundary = count(is_near_boundary)
    if !isempty(all_emitters)
        merged_emitters = deduplicate_boundary_emitters(
            all_emitters, partition_ids, is_near_boundary, boundary_margin
        )
    else
        merged_emitters = SMLMData.Emitter2DFit{Float64}[]
    end
    n_deduped = n_pre_dedup - length(merged_emitters)
    _log_progress("  Dedup: $n_pre_dedup → $(length(merged_emitters)) emitters ($n_deduped removed, $n_boundary near boundary)")

    # Merge count histograms for posterior_k
    merged_count_hist = EmitterCountHist()
    for ch in count_hists
        accumulator_merge!(merged_count_hist, ch)
    end
    posterior_k = accumulator_result(merged_count_hist)

    # Merge posterior images if requested
    post_img = nothing
    if posterior_pixel_size > 0.0
        # Create full-field target image from user-specified or data-derived bounds
        xlim_final = posterior_xlim
        ylim_final = posterior_ylim
        if xlim_final === nothing || ylim_final === nothing
            all_x = [loc.x for loc in locs]
            all_y = [loc.y for loc in locs]
            pad = 3 * median([mean_sigma(loc) for loc in locs])
            xlim_final = (Float64(minimum(all_x) - pad), Float64(maximum(all_x) + pad))
            ylim_final = (Float64(minimum(all_y) - pad), Float64(maximum(all_y) + pad))
        end
        merged_post = PosteriorImage(pixel_size=posterior_pixel_size,
                                      xlim=xlim_final, ylim=ylim_final)
        for accs in partition_accumulators
            for acc in accs
                if acc isa PosteriorImage
                    accumulator_merge!(merged_post, acc)
                end
            end
        end
        pi_result = accumulator_result(merged_post)
        # Convert to integer counts for backward compat
        int_image = round.(Int, pi_result.image)
        post_img = (image=int_image, edges_x=pi_result.edges_x,
                   edges_y=pi_result.edges_y, pixel_size=pi_result.pixel_size)
        _log_progress("  Posterior image merged ($(size(int_image,1))×$(size(int_image,2)) px)")
    end

    # Aggregate acceptance rates across all partitions
    acceptance_totals = Dict{Symbol, Tuple{Int, Int}}()
    for pa in partition_acceptance
        for (move, (acc, tot)) in pa
            prev = get(acceptance_totals, move, (0, 0))
            acceptance_totals[move] = (prev[1] + acc, prev[2] + tot)
        end
    end
    acceptance_rates = Dict{Symbol, Float64}()
    for (move, (acc, tot)) in acceptance_totals
        acceptance_rates[move] = tot > 0 ? acc / tot : 0.0
    end

    # Pool cluster sizes from clusters without overlap locs (consistent with hierarchical learner)
    final_cluster_masks = _compute_cluster_overlap_masks(states, partitions)
    cluster_sizes = Int[]
    for (i, s) in enumerate(states)
        for (j, cs) in enumerate(s.clusters)
            s.active[j] || continue
            final_cluster_masks[i][j] || continue  # skip clusters with overlap locs
            push!(cluster_sizes, Int(cs.n))
        end
    end

    # Per-partition emitter counts (from MAP-N extraction, before deduplication)
    partition_k = [length(partition_emitters[pid]) for pid in 1:n_partitions]

    # Per-localization partition IDs (for partition-colored renders)
    loc_partition_ids = zeros(Int, length(locs))
    for pid in 1:n_partitions
        for idx in partitions[pid].original_indices
            loc_partition_ids[idx] = pid
        end
    end

    # Build diagnostics
    diagnostics = BaGoLDiagnostics(
        length(merged_emitters), posterior_k, acceptance_rates,
        μ, current_shape, ρ, n_partitions, cluster_sizes, partition_k,
        loc_partition_ids, post_img, _se_tau,
        (; iters=trace_iters, K=trace_K, mu=trace_mu, shape=trace_shape, rho=trace_rho)
    )
    result_smld = SMLMData.BasicSMLD(merged_emitters, camera, 1, 1)

    # Per-partition timing summary
    partition_sizes = [length(p.locs) for p in partitions]
    partition_secs = partition_times_ns ./ 1e9
    sort_idx = sortperm(partition_secs; rev=true)
    total_cpu_s = sum(partition_secs)
    _log_progress("  Partition timing: $(round(total_cpu_s, digits=1))s total CPU, " *
                  "$(round(maximum(partition_secs), digits=1))s slowest, " *
                  "$(round(median(partition_secs), digits=3))s median")
    # Top 10 slowest partitions
    n_show = min(10, n_partitions)
    for rank in 1:n_show
        i = sort_idx[rank]
        _log_progress("    #$rank: pid=$i, N=$(partition_sizes[i]), " *
                      "$(round(partition_secs[i], digits=2))s " *
                      "($(round(partition_secs[i]/total_cpu_s*100, digits=1))% of CPU)")
    end

    _log_progress("Done: $(length(merged_emitters)) emitters from $(length(locs)) localizations")

    # Close archive
    if archive !== nothing
        close(archive)
    end

    return result_smld, diagnostics
end

"""
    run_bagol(locs::Vector{<:AbstractEmitter}; camera, kwargs...) -> (BasicSMLD, BaGoLDiagnostics)

Run BaGoL on a vector of localizations. Requires camera argument.

See `run_bagol(smld::SMLD; ...)` for full documentation.
"""
function run_bagol(
    locs::Vector{<:SMLMData.AbstractEmitter};
    camera::SMLMData.AbstractCamera,
    kwargs...
)
    # Wrap in SMLD and dispatch
    smld = SMLMData.BasicSMLD(locs, camera, 1, 1)
    return run_bagol(smld; kwargs...)
end

# Keep partition dispatch for internal use
run_bagol(p::Partition; camera::SMLMData.AbstractCamera, kwargs...) =
    run_bagol(p.locs; camera=camera, kwargs...)

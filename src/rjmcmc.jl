# Main BaGoL API

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
- `nsigma=3.0`: DBSCAN threshold in sigma units (Inf = no partitioning)
- `min_partition_size=0`: Minimum locs per partition (smaller clusters dropped as noise)
- `max_partition_size=1000`: Split partitions larger than this
- `skip_partition_size=typemax(Int)`: Skip partitions larger than this

# MCMC Arguments
- `sync_interval=500`: Iterations between global μ/shape updates
- `n_iterations=10000`: Total MCMC iterations
- `burn_in=2000`: Burn-in iterations before recording
- `shape=2.0`: Initial Gamma shape (1=exponential, higher=more peaked)
- `learn_shape=true`: Whether to update shape during MCMC
- `verbose=true`: Print progress

# Posterior Image
- `posterior_pixel_size=0.0`: Enable Rao-Blackwellized posterior image (pixel size in μm)
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
    nsigma::Float64 = 3.0,
    min_partition_size::Int = 0,
    max_partition_size::Int = 1000,
    skip_partition_size::Int = typemax(Int),
    sync_interval::Int = 500,
    n_iterations::Int = 10000,
    burn_in::Int = 2000,
    shape::Float64 = 2.0,
    learn_shape::Bool = true,
    posterior_pixel_size::Float64 = 0.0,
    posterior_xlim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
    posterior_ylim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
    archive_path::Union{Nothing, String} = nothing,
    progress_file::Union{Nothing, String} = nothing,
    verbose::Bool = true,
    kwargs...
)
    return _run_bagol_collapsed(smld;
        nsigma, min_partition_size, max_partition_size, skip_partition_size,
        sync_interval, n_iterations, burn_in, shape, learn_shape,
        posterior_pixel_size, posterior_xlim, posterior_ylim,
        archive_path, progress_file, verbose, kwargs...)
end

# ============================================================================
# Collapsed Gibbs sampler dispatch
# ============================================================================

function _run_bagol_collapsed(
    smld::SMLMData.SMLD;
    nsigma::Float64 = 3.0,
    min_partition_size::Int = 0,
    max_partition_size::Int = 1000,
    skip_partition_size::Int = typemax(Int),
    sync_interval::Int = 500,
    n_iterations::Int = 10000,
    burn_in::Int = 2000,
    shape::Float64 = 2.0,
    learn_shape::Bool = true,
    posterior_pixel_size::Float64 = 0.0,
    posterior_xlim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
    posterior_ylim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
    archive_path::Union{Nothing, String} = nothing,
    progress_file::Union{Nothing, String} = nothing,
    verbose::Bool = true,
    kwargs...
)
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

    _log_progress("Partitioning $(length(locs)) localizations (nsigma=$nsigma)...")

    partitions, skipped = partition_locs(locs; nsigma, min_size=min_partition_size,
                                          max_size=max_partition_size,
                                          skip_size=skip_partition_size)

    _log_progress("  Created $(length(partitions)) partitions")
    if !isempty(skipped)
        n_skipped = sum(length(p.locs) for p in skipped)
        _log_progress("  Skipped $(length(skipped)) oversized clusters ($n_skipped locs)")
    end

    if isempty(partitions)
        @warn "No valid partitions"
        empty_smld = SMLMData.BasicSMLD(SMLMData.Emitter2DFit[], camera, 1, 1)
        empty_diag = BaGoLDiagnostics(0, Int[], Dict{Symbol,Float64}(), 0.0, shape, 0, Int[], nothing)
        return empty_smld, empty_diag
    end

    n_partitions = length(partitions)

    # Hyperprior config
    μ_prior_shape = get(kwargs, :μ_prior_shape, 2.0)
    μ_prior_scale = get(kwargs, :μ_prior_scale, 5.0)
    shape_prior_shape = get(kwargs, :shape_prior_shape, 2.0)
    shape_prior_scale = get(kwargs, :shape_prior_scale, 1.0)
    config_nt = (μ_prior_shape=μ_prior_shape, μ_prior_scale=μ_prior_scale,
                 shape_prior_shape=shape_prior_shape, shape_prior_scale=shape_prior_scale)

    # Initialize collapsed states and accumulators per partition
    states = Vector{CollapsedState}(undef, n_partitions)
    partition_accumulators = Vector{Vector{AbstractAccumulator}}(undef, n_partitions)
    count_hists = Vector{EmitterCountHist}(undef, n_partitions)
    partition_samples = Vector{PartitionSamples}(undef, n_partitions)
    partition_psms = Vector{PSMAccumulator}(undef, n_partitions)

    @sync for i in 1:n_partitions
        Threads.@spawn begin
            p_locs = partitions[i].locs
            spatial_prior = UniformSpatialPrior(p_locs)
            states[i] = initialize_collapsed_state(p_locs, spatial_prior)

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

    # Global μ, shape (shared across partitions)
    μ = μ_prior_shape * μ_prior_scale
    μ₀ = μ  # Fixed μ for split-merge acceptance (decouples K from μ adaptation)
    current_shape = shape

    # Initialize archive if requested
    archive = nothing
    if archive_path !== nothing
        archive = BaGoLArchive(archive_path, n_partitions,
                               [length(p.locs) for p in partitions])
    end

    # Synchronized outer loop
    n_outer = div(n_iterations, sync_interval)
    iter_counters = zeros(Int, n_partitions)

    # Per-partition acceptance tracking
    _zero() = (0, 0)
    partition_acceptance = [Dict{Symbol, Tuple{Int, Int}}(
        :gibbs_sweep => _zero(), :split => _zero(), :merge => _zero()
    ) for _ in 1:n_partitions]

    for outer in 1:n_outer
        @sync for i in 1:n_partitions
            Threads.@spawn begin
                λ_K_i = get(kwargs, :λ_K, Float64(length(partitions[i].locs)) / μ)
                iter_counters[i] = run_collapsed_iterations!(
                    states[i], partitions[i].locs, sync_interval,
                    μ, current_shape, λ_K_i,
                    partition_accumulators[i], burn_in, iter_counters[i];
                    acceptance=partition_acceptance[i], μ₀=μ₀
                )
            end
        end

        # Write archive samples if past burn-in
        if archive !== nothing && (outer * sync_interval) > burn_in
            for i in 1:n_partitions
                write_sample!(archive, i, states[i], μ, current_shape)
            end
        end

        # Global hierarchical updates
        μ = _update_mu_collapsed_global!(states, μ, current_shape, config_nt)
        if learn_shape
            current_shape = _update_shape_collapsed_global!(states, μ, current_shape, config_nt)
        end

        total_K = sum(s.n_active for s in states)
        shape_str = learn_shape ? ", shape=$(round(current_shape, digits=2))" : ""
        iter_done = outer * sync_interval
        _log_progress("Sync $outer/$n_outer (iter $iter_done): K=$total_K, μ=$(round(μ, digits=2))$shape_str")
    end

    # Run remaining iterations
    remaining = n_iterations - n_outer * sync_interval
    if remaining > 0
        @sync for i in 1:n_partitions
            Threads.@spawn begin
                λ_K_i = get(kwargs, :λ_K, Float64(length(partitions[i].locs)) / μ)
                iter_counters[i] = run_collapsed_iterations!(
                    states[i], partitions[i].locs, remaining,
                    μ, current_shape, λ_K_i,
                    partition_accumulators[i], burn_in, iter_counters[i];
                    acceptance=partition_acceptance[i], μ₀=μ₀
                )
            end
        end
    end

    _log_progress("MCMC complete. Extracting MAP-N emitters ($n_partitions partitions, threaded)...")

    # Extract emitters via MAP-N from stored assignment samples (threaded)
    sigmas = [mean_sigma(loc) for loc in locs]
    # Scale margin with nsigma: partition gap ≈ nsigma*(σ_i+σ_j), so emitters
    # can only be duplicates if within ~nsigma*σ of boundary
    boundary_margin = nsigma * median(sigmas)

    # Per-partition results (filled in parallel, largest-first for load balance)
    partition_emitters = Vector{Vector{SMLMData.Emitter2DFit}}(undef, n_partitions)
    partition_boundary = Vector{Vector{Bool}}(undef, n_partitions)

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

    # Flatten results (preserving partition order)
    all_emitters = SMLMData.Emitter2DFit[]
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
        merged_emitters = SMLMData.Emitter2DFit[]
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
            xlim_final = (minimum(all_x) - pad, maximum(all_x) + pad)
            ylim_final = (minimum(all_y) - pad, maximum(all_y) + pad)
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

    # Pool cluster sizes from all partition final states (what the Gamma was fit to)
    cluster_sizes = Int[]
    for s in states
        for (j, cs) in enumerate(s.clusters)
            s.active[j] || continue
            push!(cluster_sizes, Int(cs.n))
        end
    end

    # Build diagnostics
    diagnostics = BaGoLDiagnostics(
        length(merged_emitters), posterior_k, acceptance_rates,
        μ, current_shape, n_partitions, cluster_sizes, post_img
    )
    result_smld = SMLMData.BasicSMLD(merged_emitters, camera, 1, 1)

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

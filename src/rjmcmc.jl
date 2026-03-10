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
- `dm_concentration=1.0`: Dirichlet-Multinomial concentration parameter
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
    verbose::Bool = true,
    kwargs...
)
    dm_concentration = get(kwargs, :dm_concentration, 1.0)
    return _run_bagol_collapsed(smld;
        nsigma, min_partition_size, max_partition_size, skip_partition_size,
        sync_interval, n_iterations, burn_in, shape, learn_shape,
        posterior_pixel_size, posterior_xlim, posterior_ylim,
        archive_path, dm_concentration, verbose, kwargs...)
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
    dm_concentration::Float64 = 1.0,
    verbose::Bool = true,
    kwargs...
)
    locs = smld.emitters
    camera = smld.camera

    if verbose
        println("Partitioning $(length(locs)) localizations (nsigma=$nsigma)...")
    end

    partitions, skipped = partition_locs(locs; nsigma, min_size=min_partition_size,
                                          max_size=max_partition_size,
                                          skip_size=skip_partition_size)

    if verbose
        println("  Created $(length(partitions)) partitions")
        if !isempty(skipped)
            n_skipped = sum(length(p.locs) for p in skipped)
            println("  Skipped $(length(skipped)) oversized clusters ($n_skipped locs)")
        end
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

    Threads.@threads for i in 1:n_partitions
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

        if posterior_pixel_size > 0.0
            push!(accs, PosteriorImage(pixel_size=posterior_pixel_size,
                                        xlim=posterior_xlim, ylim=posterior_ylim))
        end

        partition_accumulators[i] = accs
    end

    # Global μ, shape (shared across partitions)
    μ = μ_prior_shape * μ_prior_scale
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
        :gibbs_sweep => _zero(), :block_birth => _zero(),
        :block_death => _zero(), :split => _zero(), :merge => _zero()
    ) for _ in 1:n_partitions]

    β = dm_concentration
    for outer in 1:n_outer
        Threads.@threads for i in 1:n_partitions
            λ_K_i = get(kwargs, :λ_K, Float64(length(partitions[i].locs)) / μ)
            iter_counters[i] = run_collapsed_iterations!(
                states[i], partitions[i].locs, sync_interval,
                μ, current_shape, λ_K_i, β,
                partition_accumulators[i], burn_in, iter_counters[i];
                acceptance=partition_acceptance[i]
            )
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

        if verbose && outer % max(1, n_outer ÷ 5) == 0
            total_K = sum(s.n_active for s in states)
            shape_str = learn_shape ? ", shape=$(round(current_shape, digits=2))" : ""
            println("Sync $outer/$n_outer: total K=$total_K, μ=$(round(μ, digits=2))$shape_str")
        end
    end

    # Run remaining iterations
    remaining = n_iterations - n_outer * sync_interval
    if remaining > 0
        Threads.@threads for i in 1:n_partitions
            λ_K_i = get(kwargs, :λ_K, Float64(length(partitions[i].locs)) / μ)
            iter_counters[i] = run_collapsed_iterations!(
                states[i], partitions[i].locs, remaining,
                μ, current_shape, λ_K_i, β,
                partition_accumulators[i], burn_in, iter_counters[i];
                acceptance=partition_acceptance[i]
            )
        end
    end

    if verbose
        println("\nMerging partition results...")
    end

    # Extract emitters via MAP-N from stored assignment samples
    all_emitters = SMLMData.Emitter2DFit[]
    partition_ids = Int[]
    is_near_boundary = Bool[]

    sigmas = [mean_sigma(loc) for loc in locs]
    # Scale margin with nsigma: partition gap ≈ nsigma*(σ_i+σ_j), so emitters
    # can only be duplicates if within ~nsigma*σ of boundary
    boundary_margin = nsigma * median(sigmas)

    for (pid, (partition, ps_acc)) in enumerate(zip(partitions, partition_samples))
        samples = accumulator_result(ps_acc)
        if isempty(samples)
            error("Partition $pid ($(length(partition.locs)) locs): no assignment samples collected. " *
                  "Check burn_in ($burn_in) < n_iterations ($n_iterations).")
        end
        emitters, _ = estimate_mapn_collapsed(samples, partition.locs)
        if isempty(emitters)
            error("Partition $pid ($(length(partition.locs)) locs): estimate_mapn_collapsed returned " *
                  "0 emitters from $(length(samples)) samples.")
        end
        for emitter in emitters
            push!(all_emitters, emitter)
            push!(partition_ids, pid)
            near_boundary = emitter_near_boundary(emitter, partition, boundary_margin)
            push!(is_near_boundary, near_boundary)
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
    if verbose
        n_deduped = n_pre_dedup - length(merged_emitters)
        println("  Pre-dedup: $n_pre_dedup emitters ($n_boundary near boundary), " *
                "removed $n_deduped duplicates")
    end

    # Merge count histograms for posterior_k
    merged_count_hist = EmitterCountHist()
    for ch in count_hists
        accumulator_merge!(merged_count_hist, ch)
    end
    posterior_k = accumulator_result(merged_count_hist)

    # Merge posterior images if requested
    post_img = nothing
    if posterior_pixel_size > 0.0
        merged_post = nothing
        for accs in partition_accumulators
            for acc in accs
                if acc isa PosteriorImage
                    if merged_post === nothing
                        merged_post = acc
                    else
                        accumulator_merge!(merged_post, acc)
                    end
                end
            end
        end
        if merged_post !== nothing
            pi_result = accumulator_result(merged_post)
            # Convert to integer counts for backward compat
            int_image = round.(Int, pi_result.image)
            post_img = (image=int_image, edges_x=pi_result.edges_x,
                       edges_y=pi_result.edges_y, pixel_size=pi_result.pixel_size)
        end
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

    if verbose
        println("Result: $(length(merged_emitters)) emitters")
    end

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

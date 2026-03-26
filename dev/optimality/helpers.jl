# Simulation, count model, and trial runner for optimality workflow.
# All functions depend on constants from config.jl.

"""Generate dimer positions on circle of diameter d, centered at FOV center."""
function dimer_positions(d::Float64)
    if d == 0.0
        return [FOV_CENTER, FOV_CENTER]
    end
    r = d / 2
    cx, cy = FOV_CENTER
    return [(cx - r, cy), (cx + r, cy)]
end

"""Generate N-mer positions on circle of diameter d."""
function nmer_positions(n::Int, d::Float64)
    if d == 0.0
        return fill(FOV_CENTER, n)
    end
    r = d / 2
    cx, cy = FOV_CENTER
    return [(cx + r * cos(2π * (i-1) / n), cy + r * sin(2π * (i-1) / n)) for i in 1:n]
end

"""Simulate localizations from true positions. Returns (locs, N_pre_filter)."""
function simulate_locs(positions::Vector{Tuple{Float64,Float64}};
                       μ::Float64=BLINK_MEAN, α::Float64=TRUE_ALPHA, seed=nothing)
    seed !== nothing && Random.seed!(seed)

    p = α / (α + μ)
    blink_dist = NegativeBinomial(α, p)
    photon_dist = Exponential(PHOTON_MEAN)

    locs = SMLMData.Emitter2DFit[]
    loc_id = 1
    for (ei, (ex, ey)) in enumerate(positions)
        n_blinks = max(1, rand(blink_dist))
        for _ in 1:n_blinks
            N_phot = rand(photon_dist)
            N_phot < PHOTON_MIN && continue
            σ = PSF_SIGMA / sqrt(N_phot)
            x = ex + σ * randn()
            y = ey + σ * randn()
            push!(locs, SMLMData.Emitter2DFit(
                x, y, N_phot, 10.0, σ, σ, 0.0, sqrt(N_phot), 1.0,
                loc_id, 1, ei, loc_id))
            loc_id += 1
        end
    end
    N_pre = length(locs)
    filtered = filter(l -> max(l.σ_x, l.σ_y) <= PRECISION_MAX, locs)
    return filtered, N_pre
end

"""
Count-model MAP K and full posterior: argmax_K P(N|K,μ,shape).
Pure NegBin likelihood — no prior on K (removed with locmix branch).
Returns (map_k, posterior_vector) where posterior_vector[k] = P(K=k|N).
"""
function count_model_posterior(N::Int, μ::Float64, shape::Float64; K_max::Int=20)
    log_probs = Float64[]
    for K in 1:K_max
        lp = log_prior_total_count(N, K, μ, shape)
        push!(log_probs, lp)
    end
    # Normalize
    max_lp = maximum(log_probs)
    probs = exp.(log_probs .- max_lp)
    probs ./= sum(probs)
    map_k = argmax(probs)
    return map_k, probs
end

"""Q-PAINT recovery rate via Monte Carlo."""
function qpaint_recovery_rate(K_true::Int, μ::Float64, shape::Float64; n_mc::Int=10_000)
    p = shape / (shape + μ)
    dist = NegativeBinomial(shape, p)
    n_correct = 0
    for _ in 1:n_mc
        N = sum(max(1, rand(dist)) for _ in 1:K_true)
        map_k, _ = count_model_posterior(N, μ, shape)
        map_k == K_true && (n_correct += 1)
    end
    return n_correct / n_mc
end

"""Oracle RMSE: weighted mean of locs assigned by true track_id vs true positions."""
function compute_oracle_rmse(locs, true_positions)
    errors = Float64[]
    for (ei, (tx, ty)) in enumerate(true_positions)
        assigned = filter(l -> l.track_id == ei, locs)
        isempty(assigned) && continue
        sum_w = 0.0; sum_wx = 0.0; sum_wy = 0.0
        for loc in assigned
            w = 1.0 / ((loc.σ_x + loc.σ_y) / 2)^2
            sum_w += w; sum_wx += w * loc.x; sum_wy += w * loc.y
        end
        ox, oy = sum_wx / sum_w, sum_wy / sum_w
        push!(errors, sqrt((ox - tx)^2 + (oy - ty)^2))
    end
    isempty(errors) && return NaN
    return sqrt(mean(errors .^ 2))
end

"""Compute Mahalanobis d² for each matched emitter pair."""
function mahalanobis_d2(emitters, true_positions; threshold=MATCH_THRESHOLD)
    assignments, _, _ = match_positions(emitters, true_positions, threshold)
    d2s = Float64[]
    for (i, j) in enumerate(assignments)
        j == 0 && continue
        tx, ty = true_positions[j]
        e = emitters[i]
        dx = e.x - tx
        dy = e.y - ty
        σx2 = e.σ_x^2
        σy2 = e.σ_y^2
        σxy = e.σ_xy
        det = σx2 * σy2 - σxy^2
        det <= 0 && continue
        d2 = (σy2 * dx^2 - 2 * σxy * dx * dy + σx2 * dy^2) / det
        push!(d2s, d2)
    end
    return d2s
end

"""Run sampler on single cluster with given parameters. Returns NamedTuple or nothing."""
function run_sampler_trial(locs, true_positions;
                           μ_fix::Float64, shape_fix::Float64,
                           hierarchical::Bool, K_true::Int)
    N = length(locs)
    N < 2 && return nothing

    ps_acc = PartitionSamples(thin=5)
    psm_acc = PSMAccumulator()
    count_hist = EmitterCountHist()

    if hierarchical
        result = run_collapsed_chain(locs;
            n_iterations=N_ITERATIONS, burn_in=BURN_IN,
            shape=2.0, learn_shape=true,
            μ_prior_shape=2.0, μ_prior_scale=5.0,
            hierarchical_interval=100,
            accumulators=AbstractAccumulator[count_hist, ps_acc, psm_acc],
            verbose=false)
    else
        result = run_collapsed_chain(locs;
            n_iterations=N_ITERATIONS, burn_in=BURN_IN,
            shape=shape_fix, learn_shape=false,
            μ_prior_shape=μ_fix, μ_prior_scale=1.0,
            hierarchical_interval=N_ITERATIONS + 1,
            accumulators=AbstractAccumulator[count_hist, ps_acc, psm_acc],
            verbose=false)
    end

    samples = accumulator_result(ps_acc)
    psm = accumulator_result(psm_acc).psm
    dahl_emitters, posterior_k, _, dahl_z = estimate_dahl(samples, locs, psm)
    K_dahl = length(unique(dahl_z))
    K_mode = length(posterior_k) > 0 ? argmax(result.accumulators[1]) - 1 : 0

    oracle_z = Int16[loc.track_id for loc in locs]
    m = compute_all_metrics(dahl_emitters, true_positions; threshold=MATCH_THRESHOLD)
    ormse = compute_oracle_rmse(locs, true_positions)
    d2 = mahalanobis_d2(dahl_emitters, true_positions)
    pd = partition_diagnostics(dahl_z, oracle_z, samples)

    acc_split = result.acceptance[:split][2] > 0 ?
        result.acceptance[:split][1] / result.acceptance[:split][2] : 0.0
    acc_merge = result.acceptance[:merge][2] > 0 ?
        result.acceptance[:merge][1] / result.acceptance[:merge][2] : 0.0

    return (
        K_dahl = K_dahl, K_mode = K_mode,
        posterior_k = result.accumulators[1],
        emitters = dahl_emitters, dahl_z = dahl_z,
        jaccard = m.jaccard, precision = m.precision,
        recall = m.recall, f1 = m.f1,
        rmse = m.rmse, n_matched = m.n_matched,
        oracle_rmse = ormse, d2 = d2,
        vi_total = pd.vi_total, overseg = pd.overseg, underseg = pd.underseg,
        epl_dahl = pd.epl_dahl, epl_oracle = pd.epl_oracle,
        epl_best = pd.epl_best, regret = pd.regret_dahl,
        final_mu = result.μ, final_shape = result.shape,
        acc_split = acc_split, acc_merge = acc_merge,
    )
end

# --- Metric extraction helpers ---

"""Extract per-trial K values from a results vector."""
get_K_vals(results_vec) = [r === nothing ? NaN : Float64(r.K_dahl) for r in results_vec]

"""Extract scalar metric from results, optionally filtering to correct-K trials."""
function extract_metric(results_vec, field::Symbol; filter_correct_K::Bool=false)
    vals = Float64[]
    for r in results_vec
        r === nothing && continue
        filter_correct_K && r.K_dahl != K_TRUE && continue
        v = getfield(r, field)
        v isa Number && !isnan(v) && push!(vals, Float64(v))
    end
    return vals
end

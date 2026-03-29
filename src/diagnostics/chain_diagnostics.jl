# Chain mixing diagnostics
#
# ESS, autocorrelation, Gelman-Rubin R-hat, and higher-level
# mixing and run-length scaling tests.

# ============================================================================
# Chain diagnostic accumulator
# ============================================================================

"""
    ChainDiagnosticAccumulator <: AbstractAccumulator

Records K (emitter count) trace during the MCMC chain for post-hoc
mixing diagnostics. Lightweight — stores one Int per thinned iteration.

Use with `run_collapsed_chain` to capture chain dynamics:
```julia
diag = ChainDiagnosticAccumulator(; thin=1)
result = run_collapsed_chain(locs; accumulators=[diag], ...)
trace = accumulator_result(diag)
ess = effective_sample_size(trace.k_trace)
```
"""
mutable struct ChainDiagnosticAccumulator <: AbstractAccumulator
    k_trace::Vector{Int}
    thin::Int
    _counter::Int
end

ChainDiagnosticAccumulator(; thin::Int=1) =
    ChainDiagnosticAccumulator(Int[], thin, 0)

function accumulator_update!(acc::ChainDiagnosticAccumulator,
                              state::CollapsedState,
                              locs::Vector{<:SMLMData.AbstractEmitter},
                              μ::Float64, shape::Float64, iter::Int)
    acc._counter += 1
    if acc._counter % acc.thin == 0
        push!(acc.k_trace, state.n_active)
    end
end

function accumulator_result(acc::ChainDiagnosticAccumulator)
    return (k_trace=acc.k_trace,)
end

function accumulator_merge!(target::ChainDiagnosticAccumulator,
                             source::ChainDiagnosticAccumulator)
    # Chain traces from different partitions are independent — no merge
end

# ============================================================================
# Effective sample size
# ============================================================================

"""
    effective_sample_size(trace; method=:batch_means) -> Float64

Effective sample size of a scalar trace.

Methods:
- `:batch_means` (default): Divide trace into √n batches, compare batch
  variance to sample variance. ESS = n × (σ²_sample / σ²_batch_means).
- `:initial_sequence`: Geyer (1992) initial positive sequence estimator.
  Sum autocorrelations in pairs until a pair is negative.

ESS ≈ n for iid samples. ESS << n indicates slow mixing.
"""
function effective_sample_size(trace::AbstractVector{<:Real};
                                method::Symbol=:batch_means)
    n = length(trace)
    n < 4 && return Float64(n)

    if method == :batch_means
        return _ess_batch_means(trace)
    elseif method == :initial_sequence
        return _ess_initial_sequence(trace)
    else
        throw(ArgumentError("Unknown ESS method: $method. Use :batch_means or :initial_sequence"))
    end
end

function _ess_batch_means(trace::AbstractVector{<:Real})
    n = length(trace)
    n_batches = max(floor(Int, sqrt(n)), 2)
    batch_size = n ÷ n_batches

    # Overall mean
    μ = mean(trace)

    # Batch means
    batch_means = Vector{Float64}(undef, n_batches)
    for b in 1:n_batches
        i_start = (b - 1) * batch_size + 1
        i_end = b * batch_size
        batch_means[b] = mean(@view trace[i_start:i_end])
    end

    # Variance of batch means (estimates σ²/ESS_per_batch)
    σ²_batch = var(batch_means)
    σ²_sample = var(trace)

    if σ²_batch <= 0 || σ²_sample <= 0
        return Float64(n)
    end

    # ESS = n × σ²_sample / (batch_size × σ²_batch)
    ess = n * σ²_sample / (batch_size * σ²_batch)
    return clamp(ess, 1.0, Float64(n))
end

function _ess_initial_sequence(trace::AbstractVector{<:Real})
    n = length(trace)
    acf = autocorrelation(trace; max_lag=n ÷ 2)

    # Sum autocorrelations in consecutive pairs until sum is negative (Geyer 1992)
    tau = 1.0  # Start with lag-0 contribution
    max_lag = length(acf) - 1
    lag = 1
    while lag < max_lag
        pair_sum = acf[lag + 1]  # +1 for 1-indexing (lag 0 = index 1)
        if lag + 1 <= max_lag
            pair_sum += acf[lag + 2]
        end
        if pair_sum < 0
            break
        end
        tau += 2.0 * pair_sum
        lag += 2
    end

    ess = n / tau
    return clamp(ess, 1.0, Float64(n))
end

# ============================================================================
# Autocorrelation
# ============================================================================

"""
    autocorrelation(trace; max_lag=nothing) -> Vector{Float64}

Normalized autocorrelation function of a scalar time series.

Returns ACF[lag+1] for lag = 0, 1, ..., max_lag. ACF[1] = 1.0 by definition.
Default max_lag = min(n÷4, 500).
"""
function autocorrelation(trace::AbstractVector{<:Real};
                          max_lag::Union{Int, Nothing}=nothing)
    n = length(trace)
    if max_lag === nothing
        max_lag = min(n ÷ 4, 500)
    end
    max_lag = min(max_lag, n - 1)

    μ = mean(trace)
    σ² = var(trace; corrected=false)
    σ² == 0 && return ones(max_lag + 1)

    acf = Vector{Float64}(undef, max_lag + 1)
    acf[1] = 1.0  # lag 0

    for lag in 1:max_lag
        c = 0.0
        @inbounds for t in 1:(n - lag)
            c += (trace[t] - μ) * (trace[t + lag] - μ)
        end
        acf[lag + 1] = c / ((n - lag) * σ²)
    end

    return acf
end

# ============================================================================
# Gelman-Rubin R-hat
# ============================================================================

"""
    gelman_rubin(chains) -> Float64

Gelman-Rubin R-hat diagnostic for multi-chain convergence.

    R̂ = √((n-1)/n + B/(n×W))

where:
- B = between-chain variance of chain means
- W = mean within-chain variance
- n = chain length (shortest chain used)

R̂ near 1.0 indicates convergence. R̂ > 1.1 suggests insufficient convergence.
Requires at least 2 chains.
"""
function gelman_rubin(chains::AbstractVector{<:AbstractVector{<:Real}})
    m = length(chains)
    m < 2 && throw(ArgumentError("R-hat requires at least 2 chains"))

    n = minimum(length(c) for c in chains)
    n < 2 && return NaN

    # Chain means and within-chain variances
    chain_means = [mean(c[1:n]) for c in chains]
    chain_vars = [var(c[1:n]) for c in chains]

    # Between-chain variance
    grand_mean = mean(chain_means)
    B = n * var(chain_means)

    # Within-chain variance
    W = mean(chain_vars)

    if W <= 0
        return W == 0 && B == 0 ? 1.0 : Inf
    end

    # Pooled variance estimate
    V_hat = ((n - 1) / n) * W + (1 / n) * B

    return sqrt(V_hat / W)
end

# ============================================================================
# Higher-level mixing tests
# ============================================================================

"""
    run_mixing_test(locs; n_chains=4, n_iterations=10000, burn_in=2000,
                    μ=10.0, shape=2.0, seed=nothing, kwargs...) -> NamedTuple

Run multiple independent chains and assess mixing quality.

Returns:
- `ess`: Effective sample size (from longest chain)
- `rhat`: Gelman-Rubin R-hat across chains
- `k_traces`: Per-chain K time series (post-burn-in)
- `acceptance_rates`: Per-chain acceptance rate dictionaries
- `chain_means`: Mean K per chain
- `chain_vars`: Variance of K per chain
"""
function run_mixing_test(locs::Vector{<:SMLMData.AbstractEmitter};
                          n_chains::Int=4,
                          n_iterations::Int=10000,
                          burn_in::Int=2000,
                          μ_prior_shape::Float64=2.0,
                          μ_prior_scale::Float64=5.0,
                          shape::Float64=2.0,
                          learn_distribution::Union{Bool, Symbol}=false,
                          seed::Union{Int, Nothing}=nothing,
                          kwargs...)
    k_traces = Vector{Vector{Int}}(undef, n_chains)
    acceptance_rates = Vector{Dict{Symbol, Float64}}(undef, n_chains)

    for c in 1:n_chains
        if seed !== nothing
            Random.seed!(seed + c - 1)
        end

        diag_acc = ChainDiagnosticAccumulator(; thin=1)
        result = run_collapsed_chain(locs;
            n_iterations=n_iterations,
            burn_in=burn_in,
            shape=shape,
            learn_distribution=learn_distribution,
            μ_prior_shape=μ_prior_shape,
            μ_prior_scale=μ_prior_scale,
            accumulators=AbstractAccumulator[diag_acc],
            verbose=false,
            kwargs...)

        k_traces[c] = accumulator_result(diag_acc).k_trace

        # Extract acceptance rates
        rates = Dict{Symbol, Float64}()
        for (move, (acc, tot)) in result.acceptance
            rates[move] = tot > 0 ? acc / tot : 0.0
        end
        acceptance_rates[c] = rates
    end

    # Compute diagnostics
    ess = effective_sample_size(k_traces[1])
    rhat = gelman_rubin(k_traces)
    chain_means = [mean(Float64.(t)) for t in k_traces]
    chain_vars = [var(Float64.(t)) for t in k_traces]

    return (
        ess=ess,
        rhat=rhat,
        k_traces=k_traces,
        acceptance_rates=acceptance_rates,
        chain_means=chain_means,
        chain_vars=chain_vars,
    )
end

"""
    run_length_test(locs, td; lengths=[1000, 5000, 20000, 100000],
                    n_chains=2, μ=10.0, shape=2.0, burn_fraction=0.2,
                    K_max=nothing, seed=nothing) -> NamedTuple

Test whether bias decreases with chain length (mixing vs systematic bias).

For each chain length, runs `n_chains` chains and compares the empirical
K distribution to a reference (longest chain). If KL divergence decreases
with length, the bias is a mixing issue. If it plateaus, there's a
systematic bias (e.g., detailed balance violation).

Returns:
- `lengths`: Chain lengths tested
- `kl_divs`: KL divergence at each length (relative to longest)
- `tv_dists`: Total variation distance at each length
- `ess_values`: ESS at each length
- `rhat_values`: R-hat at each length
- `verdict`: :mixing (bias decreases) or :systematic (bias plateaus)
"""
function run_length_test(locs::Vector{<:SMLMData.AbstractEmitter},
                          td::AbstractTargetDensity;
                          lengths::Vector{Int}=[1000, 5000, 20000, 100000],
                          n_chains::Int=2,
                          μ::Float64=10.0,
                          shape::Float64=2.0,
                          burn_fraction::Float64=0.2,
                          K_max::Union{Int, Nothing}=nothing,
                          seed::Union{Int, Nothing}=nothing)
    sort!(lengths)
    n_lengths = length(lengths)

    if K_max === nothing
        K_max = max(10, length(locs))
    end

    kl_divs = Vector{Float64}(undef, n_lengths)
    tv_dists = Vector{Float64}(undef, n_lengths)
    ess_values = Vector{Float64}(undef, n_lengths)
    rhat_values = Vector{Float64}(undef, n_lengths)

    # Reference: empirical K distribution from the longest chain
    ref_hist = nothing

    for (li, len) in enumerate(lengths)
        burn_in = max(1, round(Int, len * burn_fraction))
        all_traces = Vector{Vector{Int}}()

        for c in 1:n_chains
            if seed !== nothing
                Random.seed!(seed + (li - 1) * n_chains + c - 1)
            end
            diag_acc = ChainDiagnosticAccumulator(; thin=1)
            run_collapsed_chain(locs;
                n_iterations=len,
                burn_in=burn_in,
                shape=shape,
                learn_distribution=false,
                accumulators=AbstractAccumulator[diag_acc],
                verbose=false)
            push!(all_traces, accumulator_result(diag_acc).k_trace)
        end

        # Combined K histogram
        hist = zeros(K_max + 1)
        total_samples = 0
        for trace in all_traces
            for k in trace
                idx = clamp(k, 0, K_max) + 1
                hist[idx] += 1.0
                total_samples += 1
            end
        end
        hist ./= total_samples

        if li == n_lengths
            ref_hist = hist
        end

        ess_values[li] = effective_sample_size(all_traces[1])
        rhat_values[li] = n_chains >= 2 ? gelman_rubin(all_traces) : NaN
    end

    # Compute divergences relative to reference (longest chain)
    for (li, _) in enumerate(lengths)
        burn_in = max(1, round(Int, lengths[li] * burn_fraction))
        # Re-run to get histogram (or cache from above — simplify by recomputing)
        all_traces = Vector{Vector{Int}}()
        for c in 1:n_chains
            if seed !== nothing
                Random.seed!(seed + (li - 1) * n_chains + c - 1)
            end
            diag_acc = ChainDiagnosticAccumulator(; thin=1)
            run_collapsed_chain(locs;
                n_iterations=lengths[li],
                burn_in=burn_in,
                shape=shape,
                learn_distribution=false,
                accumulators=AbstractAccumulator[diag_acc],
                verbose=false)
            push!(all_traces, accumulator_result(diag_acc).k_trace)
        end

        hist = zeros(K_max + 1)
        for trace in all_traces
            for k in trace
                idx = clamp(k, 0, K_max) + 1
                hist[idx] += 1.0
            end
        end
        s = sum(hist)
        s > 0 && (hist ./= s)

        # KL(ref || hist) and TV
        kl = 0.0
        tv = 0.0
        for i in eachindex(hist)
            p = ref_hist[i]
            q = hist[i]
            tv += abs(p - q)
            if p > 0 && q > 0
                kl += p * log(p / q)
            elseif p > 0 && q == 0
                kl += p * log(p / 1e-10)  # smoothed
            end
        end
        kl_divs[li] = kl
        tv_dists[li] = 0.5 * tv
    end

    # Verdict: does bias decrease monotonically?
    # Compare first half to second half of KL divergences
    mid = n_lengths ÷ 2
    early_kl = mean(kl_divs[1:mid])
    late_kl = mean(kl_divs[mid+1:end])
    verdict = late_kl < early_kl * 0.5 ? :mixing : :systematic

    return (
        lengths=lengths,
        kl_divs=kl_divs,
        tv_dists=tv_dists,
        ess_values=ess_values,
        rhat_values=rhat_values,
        verdict=verdict,
    )
end

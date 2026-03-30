# Chain mixing diagnostics
#
# ESS (initial positive sequence), autocorrelation, split R-hat,
# indicator ESS, and multi-chain mixing test.
# These assess efficiency, not correctness.

# ============================================================================
# Chain diagnostic accumulator
# ============================================================================

"""
    ChainDiagnosticAccumulator <: AbstractAccumulator

Records K trace during the MCMC chain for post-hoc mixing diagnostics.
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

accumulator_result(acc::ChainDiagnosticAccumulator) = (k_trace=acc.k_trace,)

function accumulator_merge!(target::ChainDiagnosticAccumulator,
                             source::ChainDiagnosticAccumulator)
end

# ============================================================================
# Effective sample size (initial positive sequence only)
# ============================================================================

"""
    effective_sample_size(trace; method=:initial_sequence) -> Float64

Geyer (1992) initial positive sequence ESS estimator.
"""
function effective_sample_size(trace::AbstractVector{<:Real};
                                method::Symbol=:initial_sequence)
    n = length(trace)
    n < 4 && return Float64(n)
    method != :initial_sequence &&
        throw(ArgumentError("Only :initial_sequence method is supported"))
    return _ess_initial_sequence(trace)
end

function _ess_initial_sequence(trace::AbstractVector{<:Real})
    n = length(trace)
    acf = autocorrelation(trace; max_lag=n ÷ 2)
    tau = 1.0
    max_lag = length(acf) - 1
    lag = 1
    while lag < max_lag
        pair_sum = acf[lag + 1]
        if lag + 1 <= max_lag
            pair_sum += acf[lag + 2]
        end
        pair_sum < 0 && break
        tau += 2.0 * pair_sum
        lag += 2
    end
    return clamp(n / tau, 1.0, Float64(n))
end

# ============================================================================
# Autocorrelation (biased, PSD-guaranteed)
# ============================================================================

"""
    autocorrelation(trace; max_lag=nothing) -> Vector{Float64}

Normalized ACF. Denominator n·σ² (not (n-lag)·σ²) for PSD guarantee.
"""
function autocorrelation(trace::AbstractVector{<:Real};
                          max_lag::Union{Int, Nothing}=nothing)
    n = length(trace)
    max_lag === nothing && (max_lag = min(n ÷ 4, 500))
    max_lag = min(max_lag, n - 1)
    μ = mean(trace)
    σ² = var(trace; corrected=false)
    σ² == 0 && return ones(max_lag + 1)
    acf = Vector{Float64}(undef, max_lag + 1)
    acf[1] = 1.0
    for lag in 1:max_lag
        c = 0.0
        @inbounds for t in 1:(n - lag)
            c += (trace[t] - μ) * (trace[t + lag] - μ)
        end
        acf[lag + 1] = c / (n * σ²)
    end
    return acf
end

# ============================================================================
# Split R-hat
# ============================================================================

"""
    split_gelman_rubin(chains) -> Float64

Split each chain in half, compute R-hat on 2m half-chains.
"""
function split_gelman_rubin(chains::AbstractVector{<:AbstractVector{<:Real}})
    split_chains = AbstractVector{<:Real}[]
    for c in chains
        n = length(c)
        mid = n ÷ 2
        mid < 2 && continue
        push!(split_chains, @view c[1:mid])
        push!(split_chains, @view c[(mid+1):(2*mid)])
    end
    length(split_chains) < 2 && return NaN
    return _gelman_rubin(split_chains)
end

function _gelman_rubin(chains::AbstractVector{<:AbstractVector{<:Real}})
    m = length(chains)
    n = minimum(length(c) for c in chains)
    n < 2 && return NaN
    chain_means = [mean(c[1:n]) for c in chains]
    chain_vars = [var(c[1:n]) for c in chains]
    B = n * var(chain_means)
    W = mean(chain_vars)
    W <= 0 && return (W == 0 && B == 0 ? 1.0 : Inf)
    V_hat = ((n - 1) / n) * W + (1 / n) * B
    return sqrt(V_hat / W)
end

# ============================================================================
# Indicator ESS for discrete K
# ============================================================================

"""
    indicator_ess(trace; k_values=nothing) -> Dict{Int, Float64}

ESS on 1[K=k] for MAP-K ±2 neighbors. Catches P(K=k) estimation issues
that K-trace ESS misses.
"""
function indicator_ess(trace::AbstractVector{<:Integer};
                        k_values::Union{Nothing, Vector{Int}}=nothing)
    n = length(trace)
    n == 0 && return Dict{Int, Float64}()
    if k_values === nothing
        counts = Dict{Int, Int}()
        for k in trace
            counts[k] = get(counts, k, 0) + 1
        end
        mode_k = first(sort(collect(counts); by=x -> -x[2]))[1]
        k_values = [k for k in (mode_k - 2):(mode_k + 2) if k >= 0]
    end
    result = Dict{Int, Float64}()
    for k in k_values
        indicator = Float64[x == k ? 1.0 : 0.0 for x in trace]
        result[k] = var(indicator) > 0 ?
            effective_sample_size(indicator) : Float64(n)
    end
    return result
end

# ============================================================================
# Multi-chain mixing test
# ============================================================================

"""
    run_mixing_test(locs; n_chains=4, n_iterations=10000, burn_in=2000,
                    shape=2.0, seed=nothing, kwargs...) -> NamedTuple

Run multiple independent chains and assess mixing quality.
Returns ESS, split R-hat, indicator ESS, chain means/vars, acceptance rates.
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
        seed !== nothing && Random.seed!(seed + c - 1)
        diag_acc = ChainDiagnosticAccumulator(; thin=1)
        result = run_collapsed_chain(locs;
            n_iterations=n_iterations, burn_in=burn_in,
            shape=shape, learn_distribution=learn_distribution,
            μ_prior_shape=μ_prior_shape, μ_prior_scale=μ_prior_scale,
            accumulators=AbstractAccumulator[diag_acc],
            verbose=false, kwargs...)
        k_traces[c] = accumulator_result(diag_acc).k_trace
        rates = Dict{Symbol, Float64}()
        for (move, (acc, tot)) in result.acceptance
            rates[move] = tot > 0 ? acc / tot : 0.0
        end
        acceptance_rates[c] = rates
    end

    ess = effective_sample_size(k_traces[1])
    rhat_split = split_gelman_rubin(k_traces)
    chain_means = [mean(Float64.(t)) for t in k_traces]
    chain_vars = [var(Float64.(t)) for t in k_traces]
    ind_ess = indicator_ess(k_traces[1])

    return (ess=ess, rhat_split=rhat_split,
            k_traces=k_traces, acceptance_rates=acceptance_rates,
            chain_means=chain_means, chain_vars=chain_vars,
            indicator_ess=ind_ess)
end

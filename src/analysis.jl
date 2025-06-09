# RJMCMC Chain Analysis and Diagnostics
using Statistics
using StatsBase
using Distributions
using LinearAlgebra

"""
    ChainStatistics{T}

Container for chain diagnostics and statistics.
"""
struct ChainStatistics{T<:AbstractFloat}
    # Basic statistics
    chain_length::Int
    mean_emitters::T
    std_emitters::T
    
    # Acceptance rates by move type
    move_acceptance::T
    add_acceptance::T
    remove_acceptance::T
    split_acceptance::T
    merge_acceptance::T
    reallocate_acceptance::T
    overall_acceptance::T
    
    # Convergence diagnostics
    effective_sample_size::T
    autocorr_time::T
    geweke_z_score::T
    
    # Log probability statistics
    mean_log_prob::T
    std_log_prob::T
    log_prob_trend::T  # Linear trend in log probability
    
    # Mixing diagnostics
    acceptance_rate_stability::T  # CV of acceptance rates across chain
    state_transition_rate::T      # Rate of meaningful state changes
end

"""
    statistics(chain::BaGoLChain{T,O}) -> ChainStatistics{T}

Compute comprehensive statistics for an RJMCMC chain.

Returns detailed diagnostics including acceptance rates by move type,
convergence diagnostics, and mixing statistics.

# Examples
```julia
result = bagol(emitters)
stats = statistics(result.chains[1])
println("Overall acceptance rate: ", stats.overall_acceptance)
println("Effective sample size: ", stats.effective_sample_size)
```
"""
function statistics(chain::BaGoLChain{T,O}) where {T,O}
    n_samples = length(chain.states)
    n_samples == 0 && error("Empty chain")
    
    # Extract number of emitters over time
    n_emitters = T[length(state) for state in chain.states]
    
    # Basic statistics
    mean_emitters = mean(n_emitters)
    std_emitters = std(n_emitters)
    
    # Acceptance rate calculation (requires tracking accepted moves)
    # For now, estimate from log probability changes
    acceptance_rates = estimate_acceptance_rates(chain)
    
    # Convergence diagnostics
    ess = effective_sample_size(chain.log_probs)
    autocorr = autocorrelation_time(chain.log_probs)
    geweke = geweke_diagnostic(chain.log_probs)
    
    # Log probability statistics
    mean_log_prob = mean(chain.log_probs)
    std_log_prob = std(chain.log_probs)
    log_prob_trend = linear_trend(chain.log_probs)
    
    # Mixing diagnostics
    acc_stability = coefficient_of_variation(acceptance_rates)
    transition_rate = state_transition_rate(chain.states)
    
    return ChainStatistics{T}(
        n_samples, mean_emitters, std_emitters,
        acceptance_rates.move, acceptance_rates.add, acceptance_rates.remove,
        acceptance_rates.split, acceptance_rates.merge, acceptance_rates.reallocate,
        acceptance_rates.overall,
        ess, autocorr, geweke,
        mean_log_prob, std_log_prob, log_prob_trend,
        acc_stability, transition_rate
    )
end

"""
    statistics(result::BaGoLResult{T,O}) -> Vector{ChainStatistics{T}}

Compute statistics for all chains in a BaGoL result.
"""
function statistics(result::BaGoLResult{T,O}) where {T,O}
    return [statistics(chain) for chain in result.chains]
end

"""
Container for acceptance rate estimates by move type.
"""
struct AcceptanceRates{T<:AbstractFloat}
    move::T
    add::T
    remove::T
    split::T
    merge::T
    reallocate::T
    overall::T
end

"""
    estimate_acceptance_rates(chain::BaGoLChain) -> AcceptanceRates

Estimate acceptance rates by analyzing state changes and log probability changes.
"""
function estimate_acceptance_rates(chain::BaGoLChain{T,O}) where {T,O}
    n_samples = length(chain.states)
    n_samples < 2 && return AcceptanceRates{T}(0,0,0,0,0,0,0)
    
    # Track different types of changes
    move_changes = 0      # Position changes without count change
    add_changes = 0       # Increase in emitter count
    remove_changes = 0    # Decrease in emitter count
    split_changes = 0     # Split detected (increase + position spread)
    merge_changes = 0     # Merge detected (decrease + position convergence)
    reallocate_changes = 0 # Allocation changes without state change
    total_changes = 0
    
    for i in 2:n_samples
        prev_state = chain.states[i-1]
        curr_state = chain.states[i]
        prev_alloc = chain.allocations[i-1]
        curr_alloc = chain.allocations[i]
        
        n_prev = length(prev_state)
        n_curr = length(curr_state)
        
        # Detect change type
        if n_curr > n_prev
            add_changes += 1
            # Check if this might be a split (multiple emitters near each other)
            if detect_split_pattern(prev_state, curr_state)
                split_changes += 1
            end
        elseif n_curr < n_prev
            remove_changes += 1
            # Check if this might be a merge
            if detect_merge_pattern(prev_state, curr_state)
                merge_changes += 1
            end
        elseif n_curr == n_prev && n_curr > 0
            # Same number of emitters - check for position changes
            if states_different(prev_state, curr_state)
                move_changes += 1
            elseif allocations_different(prev_alloc, curr_alloc)
                reallocate_changes += 1
            end
        end
        
        # Overall acceptance estimated from log probability changes
        if i < n_samples && chain.log_probs[i] != chain.log_probs[i-1]
            total_changes += 1
        end
    end
    
    # Convert to rates
    n_steps = n_samples - 1
    overall_rate = T(total_changes) / T(n_steps)
    
    # Estimate individual rates (these are rough estimates)
    move_rate = T(move_changes) / T(n_steps)
    add_rate = T(add_changes) / T(n_steps)
    remove_rate = T(remove_changes) / T(n_steps)
    split_rate = T(split_changes) / T(n_steps)
    merge_rate = T(merge_changes) / T(n_steps)
    reallocate_rate = T(reallocate_changes) / T(n_steps)
    
    return AcceptanceRates{T}(
        move_rate, add_rate, remove_rate, split_rate, merge_rate, reallocate_rate, overall_rate
    )
end

"""
    effective_sample_size(x::Vector{T}) -> T

Compute effective sample size using autocorrelation.
"""
function effective_sample_size(x::Vector{T}) where T
    n = length(x)
    n < 4 && return T(n)
    
    # Compute autocorrelation
    τ = autocorrelation_time(x)
    τ <= 0 && return T(n)
    
    return T(n) / (one(T) + 2τ)
end

"""
    autocorrelation_time(x::Vector{T}) -> T

Compute integrated autocorrelation time.
"""
function autocorrelation_time(x::Vector{T}) where T
    n = length(x)
    n < 4 && return zero(T)
    
    # Simple autocorrelation without FFT
    x_centered = x .- mean(x)
    var_x = var(x_centered)
    var_x <= 0 && return zero(T)
    
    τ_int = zero(T)
    max_lag = min(n÷4, 50)  # Reasonable maximum lag
    
    for lag in 1:max_lag
        if lag >= n
            break
        end
        
        # Compute autocorrelation at this lag
        sum_prod = zero(T)
        count = 0
        for i in 1:(n-lag)
            sum_prod += x_centered[i] * x_centered[i+lag]
            count += 1
        end
        
        autocorr_lag = sum_prod / (count * var_x)
        
        if autocorr_lag <= 0.1  # Stop when becomes small
            break
        end
        
        τ_int += autocorr_lag
    end
    
    return τ_int
end

"""
    geweke_diagnostic(x::Vector{T}) -> T

Compute Geweke convergence diagnostic (Z-score).
"""
function geweke_diagnostic(x::Vector{T}) where T
    n = length(x)
    n < 10 && return zero(T)
    
    # First 10% and last 50%
    n_first = max(1, n ÷ 10)
    n_last = max(1, n ÷ 2)
    
    first_part = x[1:n_first]
    last_part = x[(n-n_last+1):n]
    
    mean_first = mean(first_part)
    mean_last = mean(last_part)
    
    # Spectral density estimates (simplified)
    var_first = var(first_part) / n_first
    var_last = var(last_part) / n_last
    
    combined_var = var_first + var_last
    combined_var <= 0 && return zero(T)
    
    z_score = (mean_first - mean_last) / sqrt(combined_var)
    
    return z_score
end

"""
    linear_trend(x::Vector{T}) -> T

Compute linear trend (slope) in time series.
"""
function linear_trend(x::Vector{T}) where T
    n = length(x)
    n < 2 && return zero(T)
    
    t = collect(1:n)
    
    # Simple linear regression: slope = Σ(t-t̄)(x-x̄) / Σ(t-t̄)²
    t_mean = mean(t)
    x_mean = mean(x)
    
    numerator = sum((t[i] - t_mean) * (x[i] - x_mean) for i in 1:n)
    denominator = sum((t[i] - t_mean)^2 for i in 1:n)
    
    denominator <= 0 && return zero(T)
    
    return numerator / denominator
end

"""
    coefficient_of_variation(rates::AcceptanceRates{T}) -> T

Compute coefficient of variation of acceptance rates.
"""
function coefficient_of_variation(rates::AcceptanceRates{T}) where T
    values = T[rates.move, rates.add, rates.remove, rates.split, rates.merge, rates.reallocate]
    μ = mean(values)
    μ <= 0 && return T(Inf)
    return std(values) / μ
end

"""
    state_transition_rate(states::Vector{Vector{Emitter2D{T}}}) -> T

Measure how frequently the chain transitions between meaningfully different states.
"""
function state_transition_rate(states::Vector{Vector{Emitter2D{T}}}) where T
    n = length(states)
    n < 2 && return zero(T)
    
    transitions = 0
    for i in 2:n
        if !states_similar(states[i-1], states[i])
            transitions += 1
        end
    end
    
    return T(transitions) / T(n-1)
end

# Helper functions for detecting different types of moves

function states_different(s1::Vector{Emitter2D{T}}, s2::Vector{Emitter2D{T}}) where T
    length(s1) != length(s2) && return true
    isempty(s1) && return false
    
    # Check if positions are meaningfully different
    for i in eachindex(s1)
        if abs(s1[i].x - s2[i].x) > 1e-10 || abs(s1[i].y - s2[i].y) > 1e-10
            return true
        end
    end
    return false
end

function states_similar(s1::Vector{Emitter2D{T}}, s2::Vector{Emitter2D{T}}, tol::T = T(0.01)) where T
    length(s1) != length(s2) && return false
    isempty(s1) && return true
    
    # Simple distance-based similarity
    for i in eachindex(s1)
        dist = sqrt((s1[i].x - s2[i].x)^2 + (s1[i].y - s2[i].y)^2)
        if dist > tol
            return false
        end
    end
    return true
end

function allocations_different(a1::Vector{Int}, a2::Vector{Int})
    length(a1) != length(a2) && return true
    return a1 != a2
end

function detect_split_pattern(prev::Vector{Emitter2D{T}}, curr::Vector{Emitter2D{T}}) where T
    # Heuristic: if we added an emitter and two emitters are very close, likely a split
    length(curr) != length(prev) + 1 && return false
    isempty(prev) && return false
    
    # Find the closest pair in current state
    min_dist = T(Inf)
    for i in 1:length(curr)
        for j in (i+1):length(curr)
            dist = sqrt((curr[i].x - curr[j].x)^2 + (curr[i].y - curr[j].y)^2)
            min_dist = min(min_dist, dist)
        end
    end
    
    # If minimum distance is small, probably a split
    return min_dist < T(0.1)
end

function detect_merge_pattern(prev::Vector{Emitter2D{T}}, curr::Vector{Emitter2D{T}}) where T
    # Heuristic: if we removed an emitter, it was likely a merge
    return length(curr) == length(prev) - 1 && !isempty(curr)
end

"""
    summary(stats::ChainStatistics)

Print a formatted summary of chain statistics.
"""
function Base.summary(stats::ChainStatistics{T}) where T
    println("RJMCMC Chain Statistics")
    println("=" ^ 50)
    println("Chain length: ", stats.chain_length)
    println("Mean emitters: ", round(stats.mean_emitters, digits=2), " ± ", round(stats.std_emitters, digits=2))
    println()
    
    println("Acceptance Rates:")
    println("  Overall: ", round(stats.overall_acceptance * 100, digits=1), "%")
    println("  Move: ", round(stats.move_acceptance * 100, digits=1), "%")
    println("  Add: ", round(stats.add_acceptance * 100, digits=1), "%")
    println("  Remove: ", round(stats.remove_acceptance * 100, digits=1), "%")
    println("  Split: ", round(stats.split_acceptance * 100, digits=1), "%")
    println("  Merge: ", round(stats.merge_acceptance * 100, digits=1), "%")
    println("  Reallocate: ", round(stats.reallocate_acceptance * 100, digits=1), "%")
    println()
    
    println("Convergence Diagnostics:")
    println("  Effective sample size: ", round(stats.effective_sample_size, digits=1))
    println("  Autocorr. time: ", round(stats.autocorr_time, digits=2))
    println("  Geweke Z-score: ", round(stats.geweke_z_score, digits=2))
    println()
    
    println("Log Probability:")
    println("  Mean: ", round(stats.mean_log_prob, digits=2))
    println("  Std: ", round(stats.std_log_prob, digits=2))
    println("  Trend: ", round(stats.log_prob_trend, digits=6))
    println()
    
    println("Mixing Quality:")
    println("  Acceptance stability: ", round(stats.acceptance_rate_stability, digits=3))
    println("  State transition rate: ", round(stats.state_transition_rate * 100, digits=1), "%")
end

"""
    summary(chain_stats::Vector{ChainStatistics{T}})

Print summary for multiple chains.
"""
function Base.summary(chain_stats::Vector{ChainStatistics{T}}) where T
    println("RJMCMC Multi-Chain Statistics")
    println("=" ^ 50)
    println("Number of chains: ", length(chain_stats))
    println()
    
    # Aggregate statistics
    overall_acc = mean(s.overall_acceptance for s in chain_stats)
    mean_ess = mean(s.effective_sample_size for s in chain_stats)
    mean_autocorr = mean(s.autocorr_time for s in chain_stats)
    
    println("Aggregated Metrics:")
    println("  Mean overall acceptance: ", round(overall_acc * 100, digits=1), "%")
    println("  Mean effective sample size: ", round(mean_ess, digits=1))
    println("  Mean autocorr. time: ", round(mean_autocorr, digits=2))
    println()
    
    println("Per-chain breakdown:")
    for (i, stats) in enumerate(chain_stats)
        println("  Chain $i: ", round(stats.overall_acceptance * 100, digits=1), "% acceptance, ",
                round(stats.effective_sample_size, digits=1), " ESS")
    end
end

"""
    convergence_summary(result::BaGoLResult)

Quick convergence assessment for all chains.
"""
function convergence_summary(result::BaGoLResult{T,O}) where {T,O}
    chain_stats = statistics(result)
    
    println("Convergence Assessment")
    println("=" ^ 30)
    
    # Check convergence criteria
    good_acceptance = count(s -> 0.2 <= s.overall_acceptance <= 0.7, chain_stats)
    good_ess = count(s -> s.effective_sample_size >= 100, chain_stats)
    converged_geweke = count(s -> abs(s.geweke_z_score) < 2.0, chain_stats)
    
    total_chains = length(chain_stats)
    
    println("Chains with good acceptance (20-70%): $good_acceptance/$total_chains")
    println("Chains with ESS ≥ 100: $good_ess/$total_chains")
    println("Chains passing Geweke test (|Z| < 2): $converged_geweke/$total_chains")
    
    if good_acceptance == total_chains && good_ess == total_chains && converged_geweke == total_chains
        println("\n✓ All chains show good convergence!")
    elseif good_acceptance + good_ess + converged_geweke >= 2 * total_chains
        println("\n⚠ Most chains converged, some may need longer runs")
    else
        println("\n⚠ Poor convergence detected - consider longer runs or different settings")
    end
end

# Plotting functions (require Plots.jl to be loaded separately)

"""
    plot_trace(chain::BaGoLChain{T,O}; kwargs...) -> Plot

Plot trace plots for chain diagnostics. Shows number of emitters and log probability over time.

Requires Plots.jl to be imported in your session.

# Examples
```julia
using Plots
result = bagol(emitters)
plot_trace(result.chains[1])
```
"""
function plot_trace end  # Will be extended if Plots is available

"""
    plot_acceptance_rates(stats::ChainStatistics{T}; kwargs...) -> Plot

Plot acceptance rates by move type as a bar chart.
"""
function plot_acceptance_rates end  # Will be extended if Plots is available

"""
    plot_autocorrelation(chain::BaGoLChain{T,O}; max_lag::Int=50, kwargs...) -> Plot

Plot autocorrelation function for log probabilities.
"""
function plot_autocorrelation end  # Will be extended if Plots is available

# Extension methods that will be loaded if Plots.jl is available
function __init__()
    @static if VERSION >= v"1.9"
        # Use package extensions for Julia 1.9+
        return
    else
        # Fallback for older Julia versions
        if isdefined(Main, :Plots)
            _define_plotting_methods()
        end
    end
end

# Define plotting methods (will be moved to extension in Julia 1.9+)
function _define_plotting_methods()
    if isdefined(Main, :Plots)
        Plots = Main.Plots
        
        # Extend plot_trace
        function plot_trace(chain::BaGoLChain{T,O}; kwargs...) where {T,O}
            n_samples = length(chain.states)
            n_emitters = [length(state) for state in chain.states]
            
            p1 = Plots.plot(1:n_samples, n_emitters, 
                title="Number of Emitters", xlabel="MCMC Step", ylabel="Count",
                label="N emitters", lw=1.5; kwargs...)
                
            p2 = Plots.plot(1:n_samples, chain.log_probs,
                title="Log Probability", xlabel="MCMC Step", ylabel="Log P",
                label="Log prob", lw=1.5, color=:red; kwargs...)
                
            return Plots.plot(p1, p2, layout=(2,1), size=(600,400))
        end
        
        # Extend plot_acceptance_rates
        function plot_acceptance_rates(stats::ChainStatistics{T}; kwargs...) where T
            move_types = ["Move", "Add", "Remove", "Split", "Merge", "Reallocate"]
            rates = [stats.move_acceptance, stats.add_acceptance, stats.remove_acceptance,
                    stats.split_acceptance, stats.merge_acceptance, stats.reallocate_acceptance]
            
            return Plots.bar(move_types, rates .* 100,
                title="Acceptance Rates by Move Type",
                xlabel="Move Type", ylabel="Acceptance Rate (%)",
                legend=false, color=:lightblue; kwargs...)
        end
        
        # Extend plot_autocorrelation  
        function plot_autocorrelation(chain::BaGoLChain{T,O}; max_lag::Int=50, kwargs...) where {T,O}
            x = chain.log_probs
            n = length(x)
            max_lag = min(max_lag, n÷4)
            
            x_centered = x .- mean(x)
            var_x = var(x_centered)
            
            autocorr = zeros(T, max_lag+1)
            autocorr[1] = one(T)  # lag 0
            
            for lag in 1:max_lag
                if lag < n
                    sum_prod = zero(T)
                    count = 0
                    for i in 1:(n-lag)
                        sum_prod += x_centered[i] * x_centered[i+lag]
                        count += 1
                    end
                    autocorr[lag+1] = sum_prod / (count * var_x)
                end
            end
            
            return Plots.plot(0:max_lag, autocorr,
                title="Autocorrelation Function", xlabel="Lag", ylabel="Autocorrelation",
                label="ACF", lw=2, marker=:circle; kwargs...)
        end
        
        # Multi-chain plotting
        function plot_convergence(result::BaGoLResult{T,O}; kwargs...) where {T,O}
            n_chains = length(result.chains)
            n_chains == 0 && error("No chains to plot")
            
            # Plot all chain traces
            p = Plots.plot()
            for (i, chain) in enumerate(result.chains)
                n_samples = length(chain.log_probs)
                Plots.plot!(p, 1:n_samples, chain.log_probs, 
                    label="Chain $i", alpha=0.7, lw=1.5; kwargs...)
            end
            
            Plots.title!(p, "Multi-Chain Convergence")
            Plots.xlabel!(p, "MCMC Step")
            Plots.ylabel!(p, "Log Probability")
            
            return p
        end
        
        function plot_chain_comparison(result::BaGoLResult{T,O}; kwargs...) where {T,O}
            stats = statistics(result)
            n_chains = length(stats)
            
            # Acceptance rates comparison
            move_types = ["Move", "Add", "Remove", "Split", "Merge", "Reallocate", "Overall"]
            
            p1 = Plots.plot()
            for (i, stat) in enumerate(stats)
                rates = [stat.move_acceptance, stat.add_acceptance, stat.remove_acceptance,
                        stat.split_acceptance, stat.merge_acceptance, stat.reallocate_acceptance,
                        stat.overall_acceptance] .* 100
                Plots.plot!(p1, move_types, rates, label="Chain $i", marker=:circle, lw=2)
            end
            Plots.title!(p1, "Acceptance Rates by Chain")
            Plots.ylabel!(p1, "Acceptance Rate (%)")
            Plots.xticks!(p1, rotation=45)
            
            # ESS comparison
            ess_values = [stat.effective_sample_size for stat in stats]
            p2 = Plots.bar(1:n_chains, ess_values,
                title="Effective Sample Size", xlabel="Chain", ylabel="ESS",
                legend=false, color=:lightgreen)
            
            return Plots.plot(p1, p2, layout=(2,1), size=(800,600))
        end
    end
end
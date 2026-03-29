# Detailed balance verification
#
# For small systems, explicitly compute π(z)·Q(z→z')·α(z→z') for both
# directions of specific transitions. If detailed balance holds, the
# products match to machine precision.

# ============================================================================
# Result type
# ============================================================================

"""
    DetailedBalanceResult

Result of checking detailed balance for a single (z_from, z_to) transition.

Fields:
- `z_from`, `z_to`: The two states being compared
- `log_pi_from`, `log_pi_to`: Log target densities
- `log_q_fwd`, `log_q_rev`: Log proposal densities (forward and reverse)
- `log_ratio_direct`: log[π(to)q(rev)] - log[π(from)q(fwd)] — the "true" MH ratio
- `log_ratio_code`: Code's computed acceptance log-ratio (if provided)
- `difference`: |log_ratio_direct - log_ratio_code| (if code ratio provided)
- `pass`: Whether |difference| < tolerance
"""
struct DetailedBalanceResult
    z_from::Vector{Int}
    z_to::Vector{Int}
    log_pi_from::Float64
    log_pi_to::Float64
    log_q_fwd::Float64
    log_q_rev::Float64
    log_ratio_direct::Float64
    log_ratio_code::Union{Float64, Nothing}
    difference::Float64
    pass::Bool
end

# ============================================================================
# Core checking function
# ============================================================================

"""
    check_detailed_balance(z_from, z_to, locs, td;
                           μ, shape, log_q_fwd, log_q_rev,
                           log_ratio_code=nothing, tol=1e-8) -> DetailedBalanceResult

Verify detailed balance for a specific (z_from, z_to) transition pair.

Computes the direct log-ratio:
    Δ_direct = log[π(z_to) × q(z_to → z_from)] - log[π(z_from) × q(z_from → z_to)]

If `log_ratio_code` is provided (the code's computed MH log-ratio),
checks that it matches Δ_direct to within `tol`.

If `log_ratio_code` is not provided, checks pass trivially (we're just
computing the ratio, not verifying code against it).

Arguments:
- `z_from`, `z_to`: Assignment vectors (canonical labeling)
- `locs`: Localizations
- `td`: Target density (DecoupledTarget, MFMTarget, etc.)
- `μ`, `shape`: Count model parameters
- `log_q_fwd`: log q(z_from → z_to) — forward proposal log-density
- `log_q_rev`: log q(z_to → z_from) — reverse proposal log-density
- `log_ratio_code`: Code's acceptance log-ratio (optional)
- `tol`: Tolerance for matching (default 1e-8)
"""
function check_detailed_balance(z_from::AbstractVector{<:Integer},
                                 z_to::AbstractVector{<:Integer},
                                 locs::Vector{<:SMLMData.AbstractEmitter},
                                 td::AbstractTargetDensity;
                                 μ::Float64, shape::Float64,
                                 log_q_fwd::Float64,
                                 log_q_rev::Float64,
                                 log_ratio_code::Union{Float64, Nothing}=nothing,
                                 tol::Float64=1e-8)
    loc_precs = precompute_loc_precisions(locs)
    grid = build_locmix_grid(loc_precs)

    log_pi_from = log_target(td, z_from, loc_precs, grid, μ, shape)
    log_pi_to = log_target(td, z_to, loc_precs, grid, μ, shape)

    # Direct ratio: log[π(to)q(rev) / π(from)q(fwd)]
    log_ratio_direct = (log_pi_to + log_q_rev) - (log_pi_from + log_q_fwd)

    # Compare to code's ratio
    if log_ratio_code !== nothing
        diff = abs(log_ratio_direct - log_ratio_code)
        pass = diff < tol
    else
        diff = 0.0
        pass = true  # No code ratio to compare against
    end

    return DetailedBalanceResult(
        Vector{Int}(z_from),
        Vector{Int}(z_to),
        log_pi_from, log_pi_to,
        log_q_fwd, log_q_rev,
        log_ratio_direct,
        log_ratio_code,
        diff, pass
    )
end

"""
    check_detailed_balance(z_from, z_to, loc_precs, grid, td;
                           μ, shape, log_q_fwd, log_q_rev,
                           log_ratio_code=nothing, tol=1e-8) -> DetailedBalanceResult

Precomputed-arguments version: uses pre-built loc_precs and grid for efficiency
when checking multiple transitions on the same dataset.
"""
function check_detailed_balance(z_from::AbstractVector{<:Integer},
                                 z_to::AbstractVector{<:Integer},
                                 loc_precs::Vector{LocPrecision},
                                 grid::LocmixGrid,
                                 td::AbstractTargetDensity;
                                 μ::Float64, shape::Float64,
                                 log_q_fwd::Float64,
                                 log_q_rev::Float64,
                                 log_ratio_code::Union{Float64, Nothing}=nothing,
                                 tol::Float64=1e-8)
    log_pi_from = log_target(td, z_from, loc_precs, grid, μ, shape)
    log_pi_to = log_target(td, z_to, loc_precs, grid, μ, shape)

    log_ratio_direct = (log_pi_to + log_q_rev) - (log_pi_from + log_q_fwd)

    if log_ratio_code !== nothing
        diff = abs(log_ratio_direct - log_ratio_code)
        pass = diff < tol
    else
        diff = 0.0
        pass = true
    end

    return DetailedBalanceResult(
        Vector{Int}(z_from),
        Vector{Int}(z_to),
        log_pi_from, log_pi_to,
        log_q_fwd, log_q_rev,
        log_ratio_direct,
        log_ratio_code,
        diff, pass
    )
end

# ============================================================================
# Automated suite
# ============================================================================

"""
    generate_split_transitions(N, K_max) -> Vector{Tuple{Vector{Int}, Vector{Int}}}

Auto-generate all K → K+1 split transitions for small N.
For each canonical partition with K clusters, generates all possible
canonical splits (one cluster splits into two).

Returns pairs (z_from, z_to) where z_to has one more cluster than z_from.
"""
function generate_split_transitions(N::Int, K_max::Int)
    partitions = enumerate_canonical_partitions(N, K_max)
    transitions = Vector{Tuple{Vector{Int}, Vector{Int}}}()

    for z_from in partitions
        K = _count_clusters(z_from)
        K >= K_max && continue  # Can't split further

        # For each cluster, try splitting it
        for split_cluster in 1:K
            members = findall(x -> x == split_cluster, z_from)
            length(members) < 2 && continue  # Need at least 2 to split

            # Generate all binary splits of members into (A, B)
            # where A gets at least one member and B gets at least one
            for mask in 1:(2^length(members) - 2)
                z_to = copy(z_from)
                new_label = K + 1
                for (bit_idx, mem_idx) in enumerate(members)
                    if (mask >> (bit_idx - 1)) & 1 == 1
                        z_to[mem_idx] = new_label
                    end
                end
                z_canon = canonicalize(z_to)

                # Check if this transition is already in the list
                already = false
                for (_, zt) in transitions
                    if zt == z_canon
                        already = true
                        break
                    end
                end
                if !already
                    push!(transitions, (z_from, z_canon))
                end
            end
        end
    end

    return transitions
end

"""
    run_detailed_balance_suite(locs, td; μ, shape, K_max=3,
                               transitions=nothing, verbose=false)
        -> Vector{DetailedBalanceResult}

Run detailed balance checks for a set of transitions.

If `transitions` is not specified, auto-generates all K±1 split transitions
for the given N and K_max. For each transition, computes the target density
ratio. Proposal densities default to 0 (symmetric proposal) — override
by providing explicit transitions with proposal densities.

This checks the **target density** side of detailed balance. The proposal
density side must be verified separately (it depends on the specific move
implementation).

Note: With log_q_fwd = log_q_rev = 0 (default), this verifies that the
target density ratio log[π(z_to)/π(z_from)] is correct. To verify full
DB including proposal asymmetry, provide actual proposal log-densities.
"""
function run_detailed_balance_suite(locs::Vector{<:SMLMData.AbstractEmitter},
                                     td::AbstractTargetDensity;
                                     μ::Float64, shape::Float64,
                                     K_max::Int=3,
                                     transitions::Union{Nothing, Vector}=nothing,
                                     verbose::Bool=false)
    N = length(locs)

    # Precompute shared structures
    loc_precs = precompute_loc_precisions(locs)
    grid = build_locmix_grid(loc_precs)

    if transitions === nothing
        verbose && println("Auto-generating split transitions (N=$N, K_max=$K_max)...")
        trans_pairs = generate_split_transitions(N, K_max)
        verbose && println("  Found $(length(trans_pairs)) unique transitions")
    else
        trans_pairs = transitions
    end

    results = DetailedBalanceResult[]

    for (i, (z_from, z_to)) in enumerate(trans_pairs)
        # Default: symmetric proposal (log_q = 0 both directions)
        # This checks target density ratio only
        r = check_detailed_balance(z_from, z_to, loc_precs, grid, td;
            μ=μ, shape=shape,
            log_q_fwd=0.0, log_q_rev=0.0)

        push!(results, r)

        verbose && println(
            "  [$i] K=$(length(unique(z_from)))→$(length(unique(z_to))): " *
            "Δ_target = $(round(r.log_ratio_direct; digits=4)), " *
            "π(from)=$(round(r.log_pi_from; digits=2)), " *
            "π(to)=$(round(r.log_pi_to; digits=2))"
        )
    end

    return results
end

"""
    print_db_summary(results::Vector{DetailedBalanceResult})

Print a human-readable summary of detailed balance check results.
"""
function print_db_summary(results::Vector{DetailedBalanceResult})
    n_pass = count(r -> r.pass, results)
    n_total = length(results)
    println("Detailed Balance Summary: $n_pass / $n_total passed")
    println()

    for (i, r) in enumerate(results)
        K_from = length(unique(r.z_from))
        K_to = length(unique(r.z_to))
        status = r.pass ? "PASS" : "FAIL"
        println("[$i] $status  K=$K_from→$K_to")
        println("     z_from = $(r.z_from)")
        println("     z_to   = $(r.z_to)")
        println("     log π(from) = $(round(r.log_pi_from; digits=6))")
        println("     log π(to)   = $(round(r.log_pi_to; digits=6))")
        println("     log q(fwd)  = $(round(r.log_q_fwd; digits=6))")
        println("     log q(rev)  = $(round(r.log_q_rev; digits=6))")
        println("     Δ_direct    = $(round(r.log_ratio_direct; digits=6))")
        if r.log_ratio_code !== nothing
            println("     Δ_code      = $(round(r.log_ratio_code; digits=6))")
            println("     |diff|      = $(round(r.difference; sigdigits=3))")
        end
        println()
    end
end

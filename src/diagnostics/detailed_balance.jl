# Detailed balance verification
#
# For samplers with tractable proposal densities, verify that
# π(z)Q(z→z')α(z→z') = π(z')Q(z'→z)α(z'→z) for specific transitions.

"""
    DetailedBalanceResult

Result of checking detailed balance for a single (z_from, z_to) transition.
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

"""
    check_detailed_balance(z_from, z_to, locs, td;
                           μ, shape, ρ, log_q_fwd, log_q_rev,
                           log_ratio_code=nothing, tol=1e-8) -> DetailedBalanceResult

Verify detailed balance for a specific transition pair.

Computes: Δ = [log π(z') + log q(z'→z)] - [log π(z) + log q(z→z')]

If `log_ratio_code` is provided, checks |Δ - Δ_code| < tol.
"""
function check_detailed_balance(z_from::AbstractVector{<:Integer},
                                 z_to::AbstractVector{<:Integer},
                                 locs::Vector{<:SMLMData.AbstractEmitter},
                                 td::AbstractTargetDensity;
                                 μ::Float64, shape::Float64, ρ::Float64=2.0,
                                 log_q_fwd::Float64, log_q_rev::Float64,
                                 log_ratio_code::Union{Float64, Nothing}=nothing,
                                 tol::Float64=1e-8)
    loc_precs = precompute_loc_precisions(locs)
    spatial_prior = UniformSpatialPrior(locs)
    log_area = log(area(spatial_prior))
    return check_detailed_balance(z_from, z_to, loc_precs, log_area, td;
        μ=μ, shape=shape, ρ=ρ, log_q_fwd=log_q_fwd, log_q_rev=log_q_rev,
        log_ratio_code=log_ratio_code, tol=tol)
end

"""
    check_detailed_balance(z_from, z_to, loc_precs, log_area, td; ...) -> DetailedBalanceResult

Precomputed-arguments version for checking multiple transitions efficiently.
"""
function check_detailed_balance(z_from::AbstractVector{<:Integer},
                                 z_to::AbstractVector{<:Integer},
                                 loc_precs::Vector{LocPrecision},
                                 log_area::Float64,
                                 td::AbstractTargetDensity;
                                 μ::Float64, shape::Float64, ρ::Float64=2.0,
                                 log_q_fwd::Float64, log_q_rev::Float64,
                                 log_ratio_code::Union{Float64, Nothing}=nothing,
                                 tol::Float64=1e-8)
    sp = _target_spatial(td, loc_precs, log_area)
    log_pi_from = log_target(td, z_from, loc_precs, sp, μ, shape, ρ)
    log_pi_to = log_target(td, z_to, loc_precs, sp, μ, shape, ρ)
    log_ratio_direct = (log_pi_to + log_q_rev) - (log_pi_from + log_q_fwd)

    if log_ratio_code !== nothing
        diff = abs(log_ratio_direct - log_ratio_code)
        pass = diff < tol
    else
        diff = 0.0
        pass = true
    end

    return DetailedBalanceResult(
        Vector{Int}(z_from), Vector{Int}(z_to),
        log_pi_from, log_pi_to, log_q_fwd, log_q_rev,
        log_ratio_direct, log_ratio_code, diff, pass)
end

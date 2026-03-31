# Thermodynamic Integration for P(K|data)
#
# Estimates the marginal Z(K) = Σ_z P_DM(z|K) × exp(ML(z)) for each K
# by tempering the spatial likelihood: π_β(z) ∝ P_DM(z) × exp(β × ML(z))
# and integrating: log Z(K) = ∫₀¹ E_β[ML(z)] dβ
#
# This gives the EXACT marginal P(K|data) ∝ P(N|K) × Z(K), not just
# per-allocation scores.
#
# Usage: julia --project=dev dev/thermodynamic_integration.jl

using SMLMBaGoL
using SMLMData
using Random
using Printf
using SpecialFunctions: loggamma
using Distributions
using Statistics

# ============================================================================
# Setup
# ============================================================================

function make_octamer_locs(; NN_over_sigma=1.9, σ=0.007, n_per=5, seed=42)
    NN = NN_over_sigma * σ
    R = NN / (2 * sin(π / 8))
    positions = [(R * cos(2π * k / 8), R * sin(2π * k / 8)) for k in 0:7]
    rng = MersenneTwister(seed)
    locs = SMLMData.Emitter2DFit[]
    true_z = Int[]
    for (eid, (ex, ey)) in enumerate(positions)
        for _ in 1:n_per
            x = ex + σ * randn(rng)
            y = ey + σ * randn(rng)
            push!(locs, SMLMData.Emitter2DFit(
                x, y, 1000.0, 0.0, σ, σ, 0.0, 0.0, 0.0,
                1, 1, eid, length(locs) + 1))
            push!(true_z, eid)
        end
    end
    return locs, true_z, positions
end

# ============================================================================
# β-tempered Gibbs sweep: P(z_i=k|rest) ∝ (n_{-i,k}+γ) × exp(β × predictive)
# ============================================================================

function tempered_gibbs_sweep!(assignments::Vector{Int16}, K::Int, N::Int,
                                loc_precs::Vector{SMLMBaGoL.LocPrecision},
                                grid::SMLMBaGoL.LocmixGrid,
                                γ::Float64, β::Float64)
    # Build cluster stats
    clusters = [SMLMBaGoL.ClusterStats() for _ in 1:K]
    for i in 1:N
        k = Int(assignments[i])
        clusters[k] = SMLMBaGoL.add_loc(clusters[k], loc_precs[i])
    end

    perm = randperm(N)
    log_probs = Vector{Float64}(undef, K)

    for loc_pos in 1:N
        i = perm[loc_pos]
        lp = loc_precs[i]
        old_k = Int(assignments[i])

        # Skip sole occupants
        if clusters[old_k].n <= 1
            continue
        end

        # Remove from current cluster
        clusters[old_k] = SMLMBaGoL.remove_loc(clusters[old_k], lp)

        # Compute weights for each cluster
        for k in 1:K
            cs = clusters[k]
            log_dm = log(Float64(cs.n) + γ)
            if β > 0
                log_pred = SMLMBaGoL.log_predictive_locmix(cs, lp, grid)
                log_probs[k] = log_dm + β * log_pred
            else
                log_probs[k] = log_dm
            end
        end

        # Sample from categorical
        max_lp = maximum(@view log_probs[1:K])
        total = 0.0
        for k in 1:K
            log_probs[k] = exp(log_probs[k] - max_lp)
            total += log_probs[k]
        end

        u = rand() * total
        cumsum = 0.0
        chosen = K
        for k in 1:K
            cumsum += log_probs[k]
            if u < cumsum
                chosen = k
                break
            end
        end

        clusters[chosen] = SMLMBaGoL.add_loc(clusters[chosen], lp)
        assignments[i] = Int16(chosen)
    end

    return clusters
end

# Compute total spatial ML for an allocation
function total_ml(assignments::Vector{Int16}, K::Int, N::Int,
                  loc_precs::Vector{SMLMBaGoL.LocPrecision},
                  grid::SMLMBaGoL.LocmixGrid)
    clusters = [SMLMBaGoL.ClusterStats() for _ in 1:K]
    for i in 1:N
        k = Int(assignments[i])
        clusters[k] = SMLMBaGoL.add_loc(clusters[k], loc_precs[i])
    end
    ml = 0.0
    for k in 1:K
        ml += SMLMBaGoL.log_marginal_likelihood_locmix(clusters[k], grid)
    end
    return ml
end

# ============================================================================
# Thermodynamic integration at fixed K
# ============================================================================

function thermodynamic_integration(K::Int, N::Int,
                                    loc_precs::Vector{SMLMBaGoL.LocPrecision},
                                    grid::SMLMBaGoL.LocmixGrid,
                                    γ::Float64;
                                    n_beta::Int=21,
                                    n_burn::Int=2000,
                                    n_samples::Int=5000)
    betas = range(0.0, 1.0, length=n_beta)
    E_ML = Float64[]  # E_β[ML(z)] for each β

    for (bi, β) in enumerate(betas)
        # Initialize with random balanced allocation
        assignments = Int16.(repeat(1:K, cld(N, K))[1:N])
        # Shuffle
        shuffle!(assignments)

        # Burn-in
        for _ in 1:n_burn
            tempered_gibbs_sweep!(assignments, K, N, loc_precs, grid, γ, β)
        end

        # Collect samples
        ml_samples = Float64[]
        for s in 1:n_samples
            tempered_gibbs_sweep!(assignments, K, N, loc_precs, grid, γ, β)
            push!(ml_samples, total_ml(assignments, K, N, loc_precs, grid))
        end

        push!(E_ML, mean(ml_samples))
        @printf("    β=%.2f: E[ML]=%.2f (std=%.2f)\n", β, mean(ml_samples), std(ml_samples))
    end

    # Trapezoidal integration: log Z = ∫₀¹ E_β[ML] dβ
    log_Z = 0.0
    for i in 1:(n_beta-1)
        dβ = betas[i+1] - betas[i]
        log_Z += dβ * (E_ML[i] + E_ML[i+1]) / 2
    end

    return log_Z, betas, E_ML
end

# ============================================================================
# Main
# ============================================================================

function main()
    for (label, nn_sigma) in [("CO-LOCATED (d=0)", 0.0), ("OCTAMER (NN=1.9σ)", 1.9)]
        println("="^70)
        println("THERMODYNAMIC INTEGRATION: $label")
        println("="^70)

        σ = 0.007; μ = 5.0; shape = 2.0; n_per = 5
        locs, true_z, _ = make_octamer_locs(; NN_over_sigma=nn_sigma, σ=σ, n_per=n_per, seed=42)
        N = length(locs)
        loc_precs = SMLMBaGoL.precompute_loc_precisions(locs)
        grid = SMLMBaGoL.build_locmix_grid(loc_precs)
        γ = shape
        p = shape / (shape + μ)

        println("\n  N=$N, μ=$μ, shape=$shape, γ=$γ")
        println("  Q-PAINT MAP K = $(argmax([logpdf(NegativeBinomial(K*shape, p), N) for K in 1:12]))")

        log_Zs = Float64[]
        K_range = 1:10

        for K in K_range
            K > N && break
            println("\n  --- K=$K ---")
            Random.seed!(42 + K)
            log_Z, betas, E_ML = thermodynamic_integration(K, N, loc_precs, grid, γ;
                n_beta=21, n_burn=2000, n_samples=5000)
            push!(log_Zs, log_Z)
            @printf("    log Z(%d) = %.4f\n", K, log_Z)
        end

        # Compute P(K) ∝ P(N|K) × Z(K)
        println("\n  " * "="^60)
        println("  MARGINAL P(K|data) via Thermodynamic Integration")
        println("  " * "="^60)
        @printf("  %3s  %8s  %10s  %10s  %10s\n", "K", "count", "log Z(K)", "log P(K)", "P/P(max)")

        log_PK = Float64[]
        for (i, K) in enumerate(K_range)
            K > N && break
            lc = logpdf(NegativeBinomial(K * shape, p), N)
            lpk = lc + log_Zs[i]
            push!(log_PK, lpk)
        end

        max_lpk = maximum(log_PK)
        println("  " * "-"^50)
        for (i, K) in enumerate(K_range)
            K > N && break
            lc = logpdf(NegativeBinomial(K * shape, p), N)
            @printf("  %3d  %+8.2f  %+10.2f  %+10.2f  %10.4f\n",
                    K, lc, log_Zs[i], log_PK[i], exp(log_PK[i] - max_lpk))
        end

        map_K = K_range[argmax(log_PK)]
        println("\n  TI MAP K = $map_K")
        println("  Q-PAINT MAP K = $(argmax([logpdf(NegativeBinomial(K*shape, p), N) for K in 1:12]))")
    end

    println("\nDone.")
end

main()

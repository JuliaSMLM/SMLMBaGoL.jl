using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions
import LinearAlgebra
import SMLMSim

# ================================================================
# Diagnostics module tests
# ================================================================
@testset "Diagnostics" begin

    @testset "Finite State" begin
        # Bell number B₃ = 5
        parts_3 = enumerate_canonical_partitions(3, 3)
        @test length(parts_3) == 5

        # S(4,1) + S(4,2) = 1 + 7 = 8
        parts_4_2 = enumerate_canonical_partitions(4, 2)
        @test length(parts_4_2) == 8

        # Canonical partitions
        for z in parts_3
            @test z == canonicalize(z)
        end
        @test canonicalize([3, 3, 1, 1]) == [1, 1, 2, 2]
        @test canonicalize([2, 1, 2, 1]) == [1, 2, 1, 2]
        @test canonicalize([1, 1, 1]) == [1, 1, 1]
        @test canonicalize([5, 5, 3, 3, 5]) == [1, 1, 2, 2, 1]
    end

    @testset "Mixing" begin
        Random.seed!(42)
        iid = randn(1000)
        ess_iid = effective_sample_size(iid)
        @test ess_iid > 500

        constant = fill(5.0, 100)
        ess_const = effective_sample_size(constant)
        @test ess_const ≤ 100

        acf = autocorrelation(iid; max_lag=10)
        @test acf[1] ≈ 1.0
        @test all(abs.(acf[2:end]) .< 0.1)

        # Split R-hat of identical chains ≈ 1.0
        chain1 = randn(500)
        rhat_val = split_gelman_rubin([chain1, copy(chain1)])
        @test rhat_val ≈ 1.0 atol=0.05

        # Split R-hat of very different chains > 1
        chain_a = randn(500)
        chain_b = randn(500) .+ 10.0
        rhat_diff = split_gelman_rubin([chain_a, chain_b])
        @test rhat_diff > 1.5
    end

    @testset "Target Density" begin
        σ = 0.005
        locs = [
            SMLMData.Emitter2DFit(0.1, 0.1, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, 1),
            SMLMData.Emitter2DFit(0.12, 0.1, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, 2),
            SMLMData.Emitter2DFit(0.1, 0.12, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, 3),
            SMLMData.Emitter2DFit(0.12, 0.12, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1, 0, 4),
        ]

        z_all_one = [1, 1, 1, 1]
        z_two = [1, 1, 2, 2]

        td = DecoupledTarget()
        lt1 = evaluate_target(td, z_all_one, locs; μ=10.0, shape=2.0)
        lt2 = evaluate_target(td, z_two, locs; μ=10.0, shape=2.0)
        @test isfinite(lt1)
        @test isfinite(lt2)

        # Relabeling should not change target density
        z_a = [1, 2, 1, 2]
        z_b = [2, 1, 2, 1]
        lt_a = evaluate_target(td, canonicalize(z_a), locs; μ=10.0, shape=2.0)
        lt_b = evaluate_target(td, canonicalize(z_b), locs; μ=10.0, shape=2.0)
        @test lt_a ≈ lt_b atol=1e-10

        # Exact posterior should sum to 1
        parts = enumerate_canonical_partitions(4, 3)
        probs, log_targets = exact_posterior(parts, locs, td; μ=10.0, shape=2.0)
        @test sum(probs) ≈ 1.0 atol=1e-10
        @test all(isfinite, log_targets)

        probs_none, log_targets_none = exact_posterior(
            parts, locs, td; μ=10.0, shape=2.0, label_multiplicity=:none)
        @test sum(probs_none) ≈ 1.0 atol=1e-10
        for (i, z) in enumerate(parts)
            K = maximum(z)
            @test log_targets[i] ≈ log_targets_none[i] + SMLMBaGoL.logfactorial(K) atol=1e-10
        end

        # The collapsed DM form must match the direct NegBin-assignment form
        # exactly for every labeled partition.
        td_dm = DMFlatTarget()
        td_direct = DirectNegBinFlatTarget()
        for z in parts
            lt_dm = evaluate_target(td_dm, z, locs; μ=10.0, shape=2.0, ρ=2.0)
            lt_direct = evaluate_target(td_direct, z, locs; μ=10.0, shape=2.0, ρ=2.0)
            @test lt_dm ≈ lt_direct atol=1e-10
        end

        probs_dm, _ = exact_posterior(parts, locs, td_dm; μ=10.0, shape=2.0, ρ=2.0)
        probs_direct, _ = exact_posterior(parts, locs, td_direct; μ=10.0, shape=2.0, ρ=2.0)
        @test all(isapprox.(probs_dm, probs_direct; atol=1e-12))

        # Regression: the flat + Poisson(ρA) K target must be AREA-INVARIANT.
        # log_prior_k_poisson's A^K cancels the flat ML's per-cluster A^{-K}, so the
        # normalized P(K) is independent of region area A (ρ held fixed). Before the
        # v0.4 fix the K prior lacked the +K·logA term and P(K) drifted with A.
        let lps_area = [SMLMBaGoL._loc_precision(l) for l in locs]
            pk_at = A -> begin
                sp = SMLMBaGoL.FlatSpatial(log(A))
                lts = [SMLMBaGoL.log_target(td_dm, z, lps_area, sp, 10.0, 2.0, 2.0) for z in parts]
                m = maximum(lts); w = exp.(lts .- m); w ./= sum(w)
                p = zeros(length(locs))
                for (z, wi) in zip(parts, w); p[length(unique(z))] += wi; end
                p
            end
            @test maximum(abs.(pk_at(1.0) .- pk_at(10.0))) < 1e-10
            @test maximum(abs.(pk_at(1.0) .- pk_at(0.05))) < 1e-10
        end

        # Locmix targets should also normalize and the DM decomposition should
        # match the direct NegBin assignment form, without a Poisson K prior.
        td_locmix = DMLocmixTarget()
        td_direct_locmix = DirectNegBinLocmixTarget()
        for z in parts
            lt_dm = evaluate_target(td_locmix, z, locs; μ=10.0, shape=2.0, ρ=2.0)
            lt_direct = evaluate_target(td_direct_locmix, z, locs; μ=10.0, shape=2.0, ρ=2.0)
            @test isfinite(lt_dm)
            @test lt_dm ≈ lt_direct atol=1e-10
        end

        probs_locmix, _ = exact_posterior(parts, locs, td_locmix; μ=10.0, shape=2.0, ρ=2.0)
        probs_direct_locmix, _ = exact_posterior(parts, locs, td_direct_locmix; μ=10.0, shape=2.0, ρ=2.0)
        @test sum(probs_locmix) ≈ 1.0 atol=1e-10
        @test all(isapprox.(probs_locmix, probs_direct_locmix; atol=1e-12))

        @test SMLMBaGoL._diagnostic_sampler_kwargs(td_dm).spatial_model == :flat
        @test SMLMBaGoL._diagnostic_sampler_kwargs(td_locmix).spatial_model == :locmix
        @test SMLMBaGoL._diagnostic_sampler_kwargs(DecoupledTarget()).allocation_model == :decoupled
        @test SMLMBaGoL._diagnostic_sampler_kwargs(DecoupledLocmixTarget()).allocation_model == :decoupled

        # FazelFlatTarget: count × K^-N × ∏ ML_flat, NO K prior
        td_fazel = FazelFlatTarget()
        for z in parts
            lt_fazel = evaluate_target(td_fazel, z, locs; μ=10.0, shape=2.0, ρ=2.0)
            @test isfinite(lt_fazel)
            K = maximum(z); N = length(z)
            # Verify the log target differs from DMFlatTarget by exactly
            # (-N log K) - log_dm_term - log_poisson_k_prior, which the
            # helper algebra proves indirectly. Here we just check
            # ρ-invariance: no Poisson(ρA) dependence on ρ.
            lt_fazel_ρ5 = evaluate_target(td_fazel, z, locs; μ=10.0, shape=2.0, ρ=5.0)
            @test lt_fazel ≈ lt_fazel_ρ5 atol=1e-10
        end
        probs_fazel, _ = exact_posterior(parts, locs, td_fazel; μ=10.0, shape=2.0, ρ=2.0)
        @test sum(probs_fazel) ≈ 1.0 atol=1e-10
        kw_fazel = SMLMBaGoL._diagnostic_sampler_kwargs(td_fazel)
        @test kw_fazel.spatial_model == :flat
        @test kw_fazel.allocation_model == :categorical
        @test kw_fazel.k_prior == :none
    end

end

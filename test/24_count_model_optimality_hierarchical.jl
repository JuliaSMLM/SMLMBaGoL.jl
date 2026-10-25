using Test
using SMLMBaGoL
using SMLMData
using Random
using Distributions

# ================================================================
# Count-Model Optimality with Hierarchical Learning
# ================================================================
# Spatially separated emitters with hierarchical μ/shape learning.
# Hier must learn (μ, α) from cluster counts during the chain.
# Spatial separation resolves K; hier calibrates the count model.
# Pass: observed MAP-K accuracy ≥ 80% of oracle (known μ, α), scored on the Dahl
# consensus MAP-N (production estimator), which is robust to the locmix K-marginal's
# diffuse upward bias that the peak-K estimator conflates with recovery error.
@testset "Count-Model Optimality (Hierarchical)" begin
    Random.seed!(2024)

    μ_true = 20.0
    α_true = 5.0
    σ = 0.005  # Localization precision (μm)
    separation = 0.05  # 10σ between emitters — well resolved

    # Oracle MAP-K accuracy with known (μ, α) — pure count model, no K prior
    function oracle_accuracy(K_true; n_mc=20000)
        p = α_true / (α_true + μ_true)
        dist_N = NegativeBinomial(K_true * α_true, p)
        n_correct = 0
        n_valid = 0
        K_max = max(3 * K_true, 15)
        for _ in 1:n_mc
            N = rand(dist_N)
            N < K_true && continue
            n_valid += 1
            best_K = 1
            best_lp = -Inf
            for k in 1:K_max
                lp = SMLMBaGoL._log_count_posterior(k, N, α_true, μ_true)
                if lp > best_lp
                    best_lp = lp
                    best_K = k
                end
            end
            n_correct += (best_K == K_true)
        end
        return n_correct / n_valid
    end

    for K_true in [1, 2, 4]
        @testset "K=$K_true separated, hier on" begin
            p_theory = oracle_accuracy(K_true)

            n_trials = 25
            n_correct = 0

            for _ in 1:n_trials
                # Generate well-separated emitters on a line
                locs = SMLMData.Emitter2DFit[]
                p = α_true / (α_true + μ_true)
                id = 0
                for k in 1:K_true
                    cx = 0.1 + (k - 1) * separation
                    cy = 0.1
                    n_k = max(rand(NegativeBinomial(α_true, p)), 1)
                    for _ in 1:n_k
                        id += 1
                        x = cx + randn() * σ
                        y = cy + randn() * σ
                        push!(locs, SMLMData.Emitter2DFit(
                            x, y, 1000.0, 10.0, σ, σ,
                            0.0, 0.0, 0.0, 1, 1, 0, id))
                    end
                end

                ps_acc = PartitionSamples(thin=5)
                psm_acc = PSMAccumulator()
                result = run_collapsed_chain(locs;
                    n_iterations=4000, burn_in=800,
                    shape=2.0, learn_distribution=true,
                    accumulators=AbstractAccumulator[ps_acc, psm_acc],
                    verbose=false)

                # Score the PRODUCTION estimator (Dahl consensus), NOT peak-K.
                # peak-K (estimate_mapn_collapsed = histogram-mode of per-iteration K) reads
                # the locmix marginal's diffuse upward K-bias directly and regresses
                # ~15-20pt
                # once Fix A removes the compensating over-merge; Dahl reads a PSM
                # consensus,
                # is robust (mean-K stays at truth), and is what production/render ship.
                # See dev/KNOWLEDGE_BASE.md (peak-K vs Dahl under Fix A).
                samples = SMLMBaGoL.accumulator_result(ps_acc)
                psm = SMLMBaGoL.accumulator_result(psm_acc).psm
                emitters, _, _, _ = estimate_dahl(samples, locs, psm)
                n_correct += (length(emitters) == K_true)
            end

            observed = n_correct / n_trials
            threshold = 0.80 * p_theory
            @test observed >= threshold
            println("  K=$K_true: $(n_correct)/$(n_trials) = " *
                    "$(round(observed*100, digits=1))% " *
                    "(theory=$(round(p_theory*100, digits=1))%, " *
                    "threshold=$(round(threshold*100, digits=1))%)")
        end
    end
end

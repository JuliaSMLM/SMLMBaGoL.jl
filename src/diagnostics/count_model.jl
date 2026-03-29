# Count model analysis utilities
#
# Tools for analyzing the NegBin count model in isolation:
# posterior P(K|N), theoretical recovery rates, confusion matrices.
# These establish the Q-PAINT baseline — the best K-estimation possible
# without spatial information.

"""
    count_model_posterior(N, μ, shape; K_max=nothing) -> (ks, probs)

Full discrete posterior P(K | N, μ, α) for K = 1..K_max.

    P(K | N) ∝ NegBin(N; K×α, α/(α+μ))

Returns vectors of K values and normalized probabilities.
If `K_max` is nothing, automatically determines a reasonable upper bound.
"""
function count_model_posterior(N::Int, μ::Float64, shape::Float64;
                               K_max::Union{Int, Nothing}=nothing)
    if K_max === nothing
        K_max = max(10, ceil(Int, 3 * N / μ))
    end
    K_max = max(K_max, 1)

    ks = collect(1:K_max)
    log_probs = Vector{Float64}(undef, K_max)

    p = shape / (shape + μ)
    @inbounds for k in 1:K_max
        log_probs[k] = logpdf(NegativeBinomial(k * shape, p), N)
    end

    # Normalize via logsumexp
    max_lp = maximum(log_probs)
    probs = exp.(log_probs .- max_lp)
    total = sum(probs)
    probs ./= total

    return ks, probs
end

"""
    qpaint_recovery_rate(K_true, μ, shape; n_mc=100_000) -> Float64

Monte Carlo estimate of P(MAP_K = K_true) under the count model.

Simulates N ~ NegBin(K_true × α, p) and computes MAP_K for each sample.
This is the theoretical ceiling for K-recovery when emitters are co-located
(no spatial information available). BaGoL should match this at d/σ = 0
and beat it at d/σ > 0.
"""
function qpaint_recovery_rate(K_true::Int, μ::Float64, shape::Float64;
                               n_mc::Int=100_000)
    K_true < 1 && return 0.0
    p = shape / (shape + μ)
    dist = NegativeBinomial(K_true * shape, p)

    n_correct = 0
    for _ in 1:n_mc
        N = rand(dist)
        N < 1 && continue  # skip zero-count draws
        map_k = count_model_map_k(N, μ, shape)
        if map_k == K_true
            n_correct += 1
        end
    end
    return n_correct / n_mc
end

"""
    count_model_confusion_matrix(K_max, μ, shape; n_mc=100_000) -> Matrix{Float64}

P(MAP_K = j | K_true = i) for i, j ∈ 1..K_max.

Rows = K_true, columns = K_estimated. Diagonal entries are the
per-K recovery rates. Off-diagonal shows systematic over/under-estimation.
"""
function count_model_confusion_matrix(K_max::Int, μ::Float64, shape::Float64;
                                       n_mc::Int=100_000)
    C = zeros(Float64, K_max, K_max)
    p = shape / (shape + μ)

    for k_true in 1:K_max
        dist = NegativeBinomial(k_true * shape, p)
        for _ in 1:n_mc
            N = rand(dist)
            N < 1 && continue
            map_k = count_model_map_k(N, μ, shape)
            j = clamp(map_k, 1, K_max)
            C[k_true, j] += 1.0
        end
        # Normalize row
        row_sum = sum(C[k_true, :])
        if row_sum > 0
            C[k_true, :] ./= row_sum
        end
    end
    return C
end

"""
    theoretical_accuracy_bound(K_true, μ, shape) -> Float64

Closed-form approximation of Q-PAINT recovery rate from NegBin variance.

    P(correct) ≈ erf(μ / (2σ_N √2))

where σ_N = √(K × μ × (1 + μ/α)) is the std dev of total count N.
This is the Gaussian approximation to the NegBin MAP accuracy.

The bound tightens with increasing μ (more counts → more information).
"""
function theoretical_accuracy_bound(K_true::Int, μ::Float64, shape::Float64)
    K_true < 1 && return 0.0
    σ_N = sqrt(K_true * μ * (1.0 + μ / shape))
    # Half-width of the MAP-K decision region is approximately μ/2
    # P(correct) ≈ P(|N - K*μ| < μ/2) ≈ erf(μ/(2σ√2))
    z = μ / (2.0 * σ_N * sqrt(2.0))
    # erf via the normal CDF: erf(z) = 2Φ(z√2) - 1
    return 2.0 * cdf(Normal(), z * sqrt(2.0)) - 1.0
end

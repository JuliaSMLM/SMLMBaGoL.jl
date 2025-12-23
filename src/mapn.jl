# MAP-N estimation for BaGoL

"""
Result of MAP-N estimation.
"""
struct MAPNResult
    n_emitters::Int
    emitters::Vector{Tuple{Float64, Float64}}  # (x, y) positions
    uncertainties::Vector{Tuple{Float64, Float64}}  # (σ_x, σ_y) uncertainties
    posterior_k::Vector{Int}  # Histogram of K across samples
end

"""
Estimate the MAP (Maximum A Posteriori) number of emitters and their positions.

Uses Hungarian algorithm to match emitters across samples, then computes
mean position and uncertainty for each matched emitter.
"""
function estimate_mapn(chain::RJMCMCChain)
    samples = chain.samples

    if isempty(samples)
        error("No samples in chain - run more iterations or reduce burn_in")
    end

    # Build histogram of K
    ks = [length(s.emitters) for s in samples]
    k_max = maximum(ks)
    posterior_k = zeros(Int, k_max + 1)
    for k in ks
        posterior_k[k + 1] += 1
    end

    # Find MAP-N (most common K)
    map_n = argmax(posterior_k) - 1

    if map_n == 0
        return MAPNResult(0, Tuple{Float64, Float64}[], Tuple{Float64, Float64}[], posterior_k)
    end

    # Get samples with MAP-N emitters
    map_samples = filter(s -> length(s.emitters) == map_n, samples)

    if isempty(map_samples)
        return MAPNResult(map_n, Tuple{Float64, Float64}[], Tuple{Float64, Float64}[], posterior_k)
    end

    # Use first sample as reference for matching
    reference = map_samples[1]

    # Collect matched positions
    matched_positions = [Vector{Tuple{Float64, Float64}}() for _ in 1:map_n]

    for sample in map_samples
        # Build cost matrix for Hungarian matching
        cost = zeros(map_n, map_n)
        for i in 1:map_n
            for j in 1:map_n
                ref_e = reference.emitters[i]
                samp_e = sample.emitters[j]
                cost[i, j] = (ref_e.x - samp_e.x)^2 + (ref_e.y - samp_e.y)^2
            end
        end

        # Hungarian matching
        assignment, _ = Hungarian.hungarian(cost)

        # Record matched positions
        for (i, j) in enumerate(assignment)
            if j <= length(sample.emitters)
                e = sample.emitters[j]
                push!(matched_positions[i], (e.x, e.y))
            end
        end
    end

    # Compute mean and uncertainty for each emitter
    emitters = Tuple{Float64, Float64}[]
    uncertainties = Tuple{Float64, Float64}[]

    for positions in matched_positions
        if isempty(positions)
            continue
        end
        xs = [p[1] for p in positions]
        ys = [p[2] for p in positions]

        mean_x = mean(xs)
        mean_y = mean(ys)
        σ_x = length(xs) > 1 ? std(xs) : 0.0
        σ_y = length(ys) > 1 ? std(ys) : 0.0

        push!(emitters, (mean_x, mean_y))
        push!(uncertainties, (σ_x, σ_y))
    end

    return MAPNResult(map_n, emitters, uncertainties, posterior_k)
end

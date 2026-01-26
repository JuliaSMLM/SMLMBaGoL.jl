# MAP-N estimation for BaGoL

"""
    estimate_mapn(chain::RJMCMCChain) -> (Vector{Emitter2DFit}, Vector{Int})

Estimate the MAP (Maximum A Posteriori) number of emitters and their positions.

Uses Hungarian algorithm to match emitters across samples, then computes
mean position and uncertainty for each matched emitter.

Returns (emitters, posterior_k) where:
- emitters: Vector of Emitter2DFit with position and covariance uncertainties
- posterior_k: Histogram of K across samples
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
        return SMLMData.Emitter2DFit[], posterior_k
    end

    # Get samples with MAP-N emitters
    map_samples = filter(s -> length(s.emitters) == map_n, samples)

    if isempty(map_samples)
        return SMLMData.Emitter2DFit[], posterior_k
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
                push!(matched_positions[i], (Float64(e.x), Float64(e.y)))
            end
        end
    end

    # Compute mean and uncertainty for each emitter, create Emitter2DFit
    result_emitters = SMLMData.Emitter2DFit[]

    for (id, positions) in enumerate(matched_positions)
        if isempty(positions)
            continue
        end
        xs = [p[1] for p in positions]
        ys = [p[2] for p in positions]

        mean_x = mean(xs)
        mean_y = mean(ys)
        n = length(xs)
        if n > 1
            σ_x = std(xs)
            σ_y = std(ys)
            # Compute covariance: cov(x,y) = E[(x-μx)(y-μy)]
            σ_xy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in 1:n) / (n - 1)
        else
            σ_x = 0.0
            σ_y = 0.0
            σ_xy = 0.0
        end

        push!(result_emitters, SMLMData.Emitter2DFit(
            mean_x, mean_y,  # position
            0.0, 0.0,        # photons, bg (not applicable for grouped result)
            σ_x, σ_y, σ_xy,  # position uncertainties with covariance
            0.0, 0.0,        # σ_photons, σ_bg
            1, 1, 0, id      # frame, dataset, track_id, id
        ))
    end

    return result_emitters, posterior_k
end

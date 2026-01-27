# BaGoL Evaluation Metrics
# ========================
# Quantitative metrics for comparing estimated emitter positions to ground truth.
#
# Usage:
#   include("viz_metrics.jl")
#   ji = jaccard_index(emitters, true_positions; threshold=0.020)
#   pr, rc, f1 = precision_recall(emitters, true_positions; threshold=0.020)

using Hungarian
using Statistics
using SMLMData

"""
Compute Jaccard Index between estimated and true emitter positions.

Uses Hungarian matching to find optimal assignment between estimated
and true positions, then counts matches within threshold distance.

JI = |matched| / (|estimated| + |true| - |matched|)

# Arguments
- `estimated`: Vector of Emitter2DFit from BaGoL
- `true_positions`: Vector of (x, y) tuples for ground truth
- `threshold`: Maximum distance (μm) to consider a match (default 0.020 = 20 nm)

# Returns
- Jaccard index in [0, 1] where 1 is perfect match
"""
function jaccard_index(
    estimated::Vector{<:SMLMData.AbstractEmitter},
    true_positions::Vector{Tuple{Float64, Float64}};
    threshold::Float64 = 0.020
)
    if isempty(estimated) && isempty(true_positions)
        return 1.0  # Both empty = perfect match
    elseif isempty(estimated) || isempty(true_positions)
        return 0.0  # One empty = no match
    end

    n_matched = count_matches(estimated, true_positions, threshold)

    n_est = length(estimated)
    n_true = length(true_positions)

    return n_matched / (n_est + n_true - n_matched)
end

"""
Compute precision, recall, and F1 score.

Precision = |matched| / |estimated|  (what fraction of estimates are correct)
Recall = |matched| / |true|  (what fraction of true were found)
F1 = 2 * precision * recall / (precision + recall)

# Arguments
- `estimated`: Vector of Emitter2DFit from BaGoL
- `true_positions`: Vector of (x, y) tuples for ground truth
- `threshold`: Maximum distance (μm) to consider a match (default 0.020 = 20 nm)

# Returns
- (precision, recall, f1) tuple
"""
function precision_recall(
    estimated::Vector{<:SMLMData.AbstractEmitter},
    true_positions::Vector{Tuple{Float64, Float64}};
    threshold::Float64 = 0.020
)
    if isempty(estimated) && isempty(true_positions)
        return (1.0, 1.0, 1.0)
    elseif isempty(estimated)
        return (0.0, 0.0, 0.0)
    elseif isempty(true_positions)
        return (0.0, 0.0, 0.0)
    end

    n_matched = count_matches(estimated, true_positions, threshold)

    precision = n_matched / length(estimated)
    recall = n_matched / length(true_positions)

    if precision + recall > 0
        f1 = 2 * precision * recall / (precision + recall)
    else
        f1 = 0.0
    end

    return (precision, recall, f1)
end

"""
Compute RMSE between matched estimated and true positions.

Only considers matched pairs within threshold distance.

# Arguments
- `estimated`: Vector of Emitter2DFit from BaGoL
- `true_positions`: Vector of (x, y) tuples for ground truth
- `threshold`: Maximum distance (μm) to consider a match (default 0.020 = 20 nm)

# Returns
- (rmse, n_matched) tuple where rmse is in micrometers
"""
function rmse_matched(
    estimated::Vector{<:SMLMData.AbstractEmitter},
    true_positions::Vector{Tuple{Float64, Float64}};
    threshold::Float64 = 0.020
)
    if isempty(estimated) || isempty(true_positions)
        return (NaN, 0)
    end

    _, _, matched_distances = match_positions(estimated, true_positions, threshold)

    if isempty(matched_distances)
        return (NaN, 0)
    end

    rmse = sqrt(mean(matched_distances .^ 2))
    return (rmse, length(matched_distances))
end

"""
Count number of matches within threshold using Hungarian matching.
"""
function count_matches(
    estimated::Vector{<:SMLMData.AbstractEmitter},
    true_positions::Vector{Tuple{Float64, Float64}},
    threshold::Float64
)
    _, _, matched_distances = match_positions(estimated, true_positions, threshold)
    return length(matched_distances)
end

"""
Match estimated to true positions using Hungarian algorithm.

Returns (assignments, costs, matched_distances) where:
- assignments: Vector mapping estimated indices to true indices (0 if unmatched)
- costs: Cost matrix used for matching
- matched_distances: Distances for matched pairs within threshold
"""
function match_positions(
    estimated::Vector{<:SMLMData.AbstractEmitter},
    true_positions::Vector{Tuple{Float64, Float64}},
    threshold::Float64
)
    n_est = length(estimated)
    n_true = length(true_positions)

    # Build cost matrix
    cost = zeros(n_est, n_true)
    for i in 1:n_est
        for j in 1:n_true
            dx = estimated[i].x - true_positions[j][1]
            dy = estimated[i].y - true_positions[j][2]
            cost[i, j] = sqrt(dx^2 + dy^2)
        end
    end

    # Hungarian matching
    assignment, _ = Hungarian.hungarian(cost)

    # Extract matched distances within threshold
    assignments = zeros(Int, n_est)
    matched_distances = Float64[]

    for (i, j) in enumerate(assignment)
        if j <= n_true
            d = cost[i, j]
            if d <= threshold
                assignments[i] = j
                push!(matched_distances, d)
            end
        end
    end

    return assignments, cost, matched_distances
end

"""
Get detailed match statistics for analysis.

# Returns NamedTuple with:
- n_estimated: Number of estimated emitters
- n_true: Number of true emitters
- n_matched: Number of matched pairs within threshold
- jaccard: Jaccard index
- precision: Precision
- recall: Recall
- f1: F1 score
- rmse: RMSE of matched positions (μm)
- mean_distance: Mean distance of matched pairs (μm)
- max_distance: Maximum distance of matched pairs (μm)
"""
function compute_all_metrics(
    estimated::Vector{<:SMLMData.AbstractEmitter},
    true_positions::Vector{Tuple{Float64, Float64}};
    threshold::Float64 = 0.020
)
    ji = jaccard_index(estimated, true_positions; threshold)
    prec, rec, f1 = precision_recall(estimated, true_positions; threshold)
    rmse, n_matched = rmse_matched(estimated, true_positions; threshold)

    _, _, matched_distances = match_positions(estimated, true_positions, threshold)

    mean_dist = isempty(matched_distances) ? NaN : mean(matched_distances)
    max_dist = isempty(matched_distances) ? NaN : maximum(matched_distances)

    return (
        n_estimated = length(estimated),
        n_true = length(true_positions),
        n_matched = n_matched,
        jaccard = ji,
        precision = prec,
        recall = rec,
        f1 = f1,
        rmse = rmse,
        mean_distance = mean_dist,
        max_distance = max_dist
    )
end

"""
Print a summary of all metrics.
"""
function print_metrics(
    estimated::Vector{<:SMLMData.AbstractEmitter},
    true_positions::Vector{Tuple{Float64, Float64}};
    threshold::Float64 = 0.020
)
    m = compute_all_metrics(estimated, true_positions; threshold)

    println("="^50)
    println("BaGoL Evaluation Metrics (threshold = $(threshold * 1000) nm)")
    println("="^50)
    println("Counts:")
    println("  Estimated emitters: $(m.n_estimated)")
    println("  True emitters:      $(m.n_true)")
    println("  Matched pairs:      $(m.n_matched)")
    println()
    println("Detection metrics:")
    println("  Jaccard Index: $(round(m.jaccard, digits=3))")
    println("  Precision:     $(round(m.precision, digits=3))")
    println("  Recall:        $(round(m.recall, digits=3))")
    println("  F1 Score:      $(round(m.f1, digits=3))")
    println()
    println("Localization accuracy:")
    println("  RMSE:          $(round(m.rmse * 1000, digits=1)) nm")
    println("  Mean distance: $(round(m.mean_distance * 1000, digits=1)) nm")
    println("  Max distance:  $(round(m.max_distance * 1000, digits=1)) nm")
    println("="^50)

    return m
end

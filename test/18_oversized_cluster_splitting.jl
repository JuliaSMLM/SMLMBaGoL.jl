using Test
using SMLMBaGoL
using SMLMData
using Random

@testset "Oversized Cluster Splitting" begin
    Random.seed!(789)

    locs = SMLMData.Emitter2DFit[]
    σ = 0.005

    for i in 1:50
        x = 0.1 + randn() * 0.02
        y = 0.1 + randn() * 0.02
        push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 0.0, 0.0, 0.0, 1, 1,
            0, i))
    end

    partitions, skipped = partition_locs(locs; partition_sigma=6.0, min_size=3, max_size=20)

    @test isempty(skipped)
    @test length(partitions) >= 2

    for p in partitions
        # Core locs (non-overlap) should be near max_size.
        # With overlap wider than the cluster, cores may slightly exceed max_size
        # because recursion stops when bisection can't reduce core further.
        n_core = count(.!p.is_overlap)
        @test n_core <= 30  # allow some slack for overlap edge cases
    end

    # With overlap, locs may appear in multiple partitions.
    # But unique original indices must cover all 50.
    all_indices = reduce(vcat, [p.original_indices for p in partitions])
    @test length(unique(all_indices)) == 50

    partitions_skip, skipped_skip = partition_locs(locs; partition_sigma=6.0,
        min_size=3, max_size=20, skip_size=20)
    @test length(skipped_skip) >= 1
end

# Reference split with the original linear overlap membership check.
function reference_split_partition(
    partition::Partition{E},
    max_size::Int,
    boundary_margin::Float64,
    start_id::Int;
    overlap::Float64=0.0,
    k_neighbors::Int=15
) where E
    locs = partition.locs
    n = length(locs)

    if n <= max_size
        return [Partition{E}(start_id, locs, partition.original_indices,
                            partition.is_boundary, partition.id,
                            falses(n))]
    end

    # --- Build symmetric kNN graph with precision-weighted affinity ---
    coords = SMLMBaGoL.coords_matrix(locs)
    tree = SMLMBaGoL.KDTree(coords)
    k = min(k_neighbors, n - 1)

    # Collect edges: precision-weighted distance per pair, keep closest
    edge_dict = Dict{Tuple{Int,Int}, Float64}()
    for i in 1:n
        idxs, _ = SMLMBaGoL.knn(tree, coords[:, i], k + 1)
        for j in idxs
            j == i && continue
            d = SMLMBaGoL.precision_weighted_distance(locs[i], locs[j])
            key = i < j ? (i, j) : (j, i)
            if !haskey(edge_dict, key) || d < edge_dict[key]
                edge_dict[key] = d
            end
        end
    end

    # --- METIS k-way partition ---
    # Convert distances to integer affinities (closer = heavier = don't cut)
    max_d = maximum(values(edge_dict))
    I = Int[]; J = Int[]; W = Int[]
    for ((i, j), d) in edge_dict
        w = max(1, round(Int, 1000.0 * (max_d / max(d, 1e-10))))
        push!(I, i); push!(J, j); push!(W, w)
        push!(I, j); push!(J, i); push!(W, w)
    end
    adj = SMLMBaGoL.sparse(I, J, W, n, n)

    # Target ~90% of max_size per partition (leave room for halo)
    target_core = max(div(max_size * 9, 10), 1)
    nparts = max(2, cld(n, target_core))

    g = SMLMBaGoL.Metis.graph(adj)
    partition_vec = SMLMBaGoL.Metis.partition(g, nparts)

    # Extract components
    components = [Int[] for _ in 1:nparts]
    for i in 1:n
        push!(components[partition_vec[i]], i)
    end
    filter!(!isempty, components)

    # --- Build sub-partitions with pointwise sibling overlap ---
    # For each component, find locs in OTHER components within overlap distance
    sub_partitions = Partition{E}[]
    current_id = start_id

    # Component membership lookup
    loc_component = zeros(Int, n)
    for (ci, comp) in enumerate(components)
        for idx in comp
            loc_component[idx] = ci
        end
    end

    for (ci, core_indices) in enumerate(components)
        if overlap > 0.0
            # Find overlap locs: locs in sibling components within proximity
            overlap_indices = Int[]
            for core_idx in core_indices
                loc_i = locs[core_idx]
                σ_i = SMLMBaGoL.mean_sigma(loc_i)
                # Query nearby locs
                radius = overlap > 0 ? overlap : 3 * σ_i
                nearby = SMLMBaGoL.inrange(tree, coords[:, core_idx], radius)
                for j in nearby
                    if loc_component[j] != ci && j ∉ overlap_indices
                        push!(overlap_indices, j)
                    end
                end
            end
            unique!(overlap_indices)

            # Combine core + overlap
            all_indices = vcat(core_indices, overlap_indices)
            is_overlap = BitVector(vcat(falses(length(core_indices)),
                                         trues(length(overlap_indices))))
        else
            all_indices = core_indices
            is_overlap = falses(length(core_indices))
        end

        sub_locs = locs[all_indices]
        sub_orig = partition.original_indices[all_indices]
        sub_boundary = BitVector([partition.is_boundary[idx] for idx in all_indices])

        push!(sub_partitions, Partition{E}(current_id, sub_locs, sub_orig,
                                            sub_boundary, partition.id, is_overlap))
        current_id += 1
    end

    return sub_partitions
end

@testset "Dense split overlap preserves partitions and order" begin
    rng = MersenneTwister(712)
    locs = [SMLMData.Emitter2DFit(0.001i + 0.0001rand(rng), 0.001j + 0.0001rand(rng),
                1000.0, 0.0, 0.002 + 0.001rand(rng), 0.002 + 0.001rand(rng),
                0.0, 0.0, 0.0, 1, 12(i - 1) + j, 0, 1)
            for i in 1:20 for j in 1:12]
    n = length(locs)
    original_indices = 1000 .+ 3 .* randperm(rng, n)
    boundary = BitVector(isodd(i) for i in 1:n)
    parent = Partition{eltype(locs)}(9, locs, original_indices, boundary, 0, falses(n))

    for (max_size, overlap) in ((40, 0.0), (40, 0.004), (40, 0.04), (1000, 0.004))
        expected = reference_split_partition(parent, max_size, 0.002, 17; overlap)
        got = SMLMBaGoL.split_partition(parent, max_size, 0.002, 17; overlap)
        @test length(got) == length(expected)
        @test max_size < n ? length(got) >= 3 : length(got) == 1
        if max_size < n && overlap > 0.0
            @test all(p -> any(p.is_overlap), got)
        end
        for (p, ref) in zip(got, expected)
            @test p.id == ref.id
            @test p.parent_id == ref.parent_id
            @test p.locs == ref.locs
            @test p.original_indices == ref.original_indices
            @test p.is_boundary == ref.is_boundary
            @test p.is_overlap == ref.is_overlap
            @test length(unique(p.original_indices)) == length(p.original_indices)
        end
        core_indices = reduce(vcat, [p.original_indices[.!p.is_overlap] for p in got])
        @test sort(core_indices) == sort(original_indices)
    end
end

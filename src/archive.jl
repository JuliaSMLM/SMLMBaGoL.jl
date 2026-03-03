# Mmap-based chain archive for collapsed Gibbs sampler
#
# Binary format per partition:
#   Header (64 bytes): magic, n_locs, n_samples, reserved
#   Per sample: Int16[n_locs] assignments + Float64 μ + Float64 shape
#
# For N=50 locs, 8k samples: ~864 KB per partition.
# Post-hoc analysis loads lazily via Mmap.mmap.

const ARCHIVE_MAGIC = UInt64(0x4241474F4C415243)  # "BAGOLARC"
const ARCHIVE_HEADER_SIZE = 64

"""
    BaGoLArchive

Writer/reader for binary assignment chain archives.

Write mode: created with `BaGoLArchive(path, n_partitions, partition_sizes)`
Read mode: created with `BaGoLArchive(path)`
"""
mutable struct BaGoLArchive
    path::String
    n_partitions::Int
    partition_sizes::Vector{Int}   # n_locs per partition
    ios::Vector{Union{IOStream, Nothing}}  # Write handles (nothing if read-only)
    sample_counts::Vector{Int}     # Number of samples written per partition
    mode::Symbol                   # :write or :read
end

"""
    BaGoLArchive(path, n_partitions, partition_sizes)

Create a new archive for writing. Creates directory and per-partition files.
"""
function BaGoLArchive(path::String, n_partitions::Int, partition_sizes::Vector{Int})
    mkpath(path)
    ios = Vector{Union{IOStream, Nothing}}(undef, n_partitions)
    sample_counts = zeros(Int, n_partitions)

    for i in 1:n_partitions
        filepath = joinpath(path, "partition_$(lpad(i, 5, '0')).bin")
        io = open(filepath, "w")

        # Write header
        write(io, ARCHIVE_MAGIC)
        write(io, Int64(partition_sizes[i]))  # n_locs
        write(io, Int64(0))                   # n_samples (updated on close)
        # Padding to 64 bytes
        for _ in 1:(ARCHIVE_HEADER_SIZE - 24)
            write(io, UInt8(0))
        end

        ios[i] = io
    end

    return BaGoLArchive(path, n_partitions, partition_sizes, ios, sample_counts, :write)
end

"""
    BaGoLArchive(path)

Open an existing archive for reading.
"""
function BaGoLArchive(path::String)
    # Count partition files
    files = sort(filter(f -> endswith(f, ".bin"), readdir(path)))
    n_partitions = length(files)

    partition_sizes = zeros(Int, n_partitions)
    sample_counts = zeros(Int, n_partitions)

    for (i, file) in enumerate(files)
        filepath = joinpath(path, file)
        open(filepath, "r") do io
            magic = read(io, UInt64)
            magic == ARCHIVE_MAGIC || error("Invalid archive file: $filepath")
            partition_sizes[i] = read(io, Int64)
            sample_counts[i] = read(io, Int64)
        end
    end

    ios = fill(nothing, n_partitions)
    return BaGoLArchive(path, n_partitions, partition_sizes, ios, sample_counts, :read)
end

"""
    write_sample!(archive, partition_id, state, μ, shape)

Write one sample (assignments + μ + shape) to the archive.
"""
function write_sample!(archive::BaGoLArchive, partition_id::Int,
                        state::CollapsedState, μ::Float64, shape::Float64)
    archive.mode == :write || error("Archive opened in read mode")
    io = archive.ios[partition_id]
    io === nothing && error("Archive partition $partition_id not open")

    # Write assignments as Int16 array
    write(io, state.assignments)
    write(io, μ)
    write(io, shape)

    archive.sample_counts[partition_id] += 1
end

"""
    close(archive)

Close all open file handles and update headers with sample counts.
"""
function Base.close(archive::BaGoLArchive)
    if archive.mode == :write
        for i in 1:archive.n_partitions
            io = archive.ios[i]
            io === nothing && continue

            # Update n_samples in header
            seek(io, 16)  # offset of n_samples field
            write(io, Int64(archive.sample_counts[i]))

            close(io)
            archive.ios[i] = nothing
        end
    end
end

"""
    _sample_size(n_locs)

Size in bytes of one sample: Int16[n_locs] + Float64 μ + Float64 shape.
"""
_sample_size(n_locs::Int) = n_locs * sizeof(Int16) + 2 * sizeof(Float64)

"""
    read_partition(archive, partition_id) -> (assignments_matrix, μs, shapes)

Read all samples from a partition via mmap.

Returns:
- `assignments`: Matrix{Int16} of size (n_locs, n_samples)
- `μs`: Vector{Float64}
- `shapes`: Vector{Float64}
"""
function read_partition(archive::BaGoLArchive, partition_id::Int)
    archive.mode == :read || error("Archive opened in write mode")

    filepath = joinpath(archive.path,
                        "partition_$(lpad(partition_id, 5, '0')).bin")

    n_locs = archive.partition_sizes[partition_id]
    n_samples = archive.sample_counts[partition_id]

    if n_samples == 0
        return zeros(Int16, n_locs, 0), Float64[], Float64[]
    end

    assignments = zeros(Int16, n_locs, n_samples)
    μs = zeros(Float64, n_samples)
    shapes = zeros(Float64, n_samples)

    open(filepath, "r") do io
        seek(io, ARCHIVE_HEADER_SIZE)

        sample_bytes = _sample_size(n_locs)
        for s in 1:n_samples
            for l in 1:n_locs
                assignments[l, s] = read(io, Int16)
            end
            μs[s] = read(io, Float64)
            shapes[s] = read(io, Float64)
        end
    end

    return assignments, μs, shapes
end

"""
    compute_from_archive(acc_type, archive, partition_id, locs; kwargs...)

Run an accumulator on archived samples from a partition.
Enables post-hoc analysis with any accumulator type.

# Example
```julia
archive = BaGoLArchive("results/archive/")
nn = compute_from_archive(NNDistHist, archive, 1, partition_locs;
                           max_dist=0.1)
```
"""
function compute_from_archive(::Type{T}, archive::BaGoLArchive,
                               partition_id::Int,
                               locs::Vector{<:SMLMData.AbstractEmitter};
                               kwargs...) where T<:AbstractAccumulator
    acc = T(; kwargs...)

    assignments, μs, shapes = read_partition(archive, partition_id)
    n_locs = size(assignments, 1)
    n_samples = size(assignments, 2)

    # Reconstruct state for each sample and update accumulator
    spatial_prior = UniformSpatialPrior(locs)
    log_area = log(area(spatial_prior))

    for s in 1:n_samples
        # Reconstruct CollapsedState from assignments
        state = _reconstruct_state(assignments[:, s], locs, log_area)
        accumulator_update!(acc, state, locs, μs[s], shapes[s], s)
    end

    return accumulator_result(acc)
end

"""
    _reconstruct_state(assignments, locs, log_area) -> CollapsedState

Reconstruct a CollapsedState from an assignment vector.
"""
function _reconstruct_state(assignments::Vector{Int16},
                             locs::Vector{<:SMLMData.AbstractEmitter},
                             log_area::Float64)
    max_cluster = maximum(assignments)
    clusters = [ClusterStats() for _ in 1:max_cluster]
    active = falses(max_cluster)

    for (i, a) in enumerate(assignments)
        if a > 0
            clusters[a] = add_loc(clusters[a], locs[i])
            active[a] = true
        end
    end

    n_active = count(active)
    return CollapsedState(assignments, clusters, active, n_active, log_area)
end

using Test
using SMLMBaGoL
using SMLMData
using Random

# ================================================================
# Accumulators
# ================================================================
@testset "Accumulators" begin
    # EmitterCountHist merge
    h1 = EmitterCountHist()
    h1.counts = [10, 20, 5]
    h2 = EmitterCountHist()
    h2.counts = [5, 15, 10, 3]

    SMLMBaGoL.accumulator_merge!(h1, h2)
    @test h1.counts == [15, 35, 15, 3]

    # NNDistHist basic
    nn = NNDistHist(max_dist=0.1, n_bins=50)
    @test length(nn.counts) == 50
end

# Dense UInt32 reference from before the packed-counts change.
mutable struct DensePSMReference
    counts::Matrix{UInt32}
    n_samples::Int
    n_locs::Int
    _members::Vector{Int}
end
DensePSMReference() = DensePSMReference(zeros(UInt32, 0, 0), 0, 0, Int[])

function update_reference!(acc::DensePSMReference, state, locs)
    n = length(locs)
    if acc.n_locs == 0
        acc.n_locs = n
        acc.counts = zeros(UInt32, n, n)
        sizehint!(acc._members, n)
    end
    @inbounds for (j, cs) in enumerate(state.clusters)
        state.active[j] || continue
        cs.n == 0 && continue
        empty!(acc._members)
        for i in 1:n
            if state.assignments[i] == j
                push!(acc._members, i)
            end
        end
        nm = length(acc._members)
        for a in 1:nm
            ia = acc._members[a]
            acc.counts[ia, ia] += UInt32(1)
            for b in (a+1):nm
                ib = acc._members[b]
                acc.counts[ia, ib] += UInt32(1)
            end
        end
    end
    acc.n_samples += 1
end

function reference_result(acc::DensePSMReference)
    n = acc.n_locs
    if n == 0 || acc.n_samples == 0
        return (psm=zeros(0, 0), n_samples=0)
    end
    psm = zeros(Float64, n, n)
    inv_t = 1.0 / acc.n_samples
    @inbounds for i in 1:n
        psm[i, i] = 1.0
        for j in (i+1):n
            v = Float64(acc.counts[i, j]) * inv_t
            psm[i, j] = v
            psm[j, i] = v
        end
    end
    (psm=psm, n_samples=acc.n_samples)
end

function check_psm(acc, reference)
    got = SMLMBaGoL.accumulator_result(acc)
    expected = reference_result(reference)
    @test got == expected
    @test reinterpret(UInt64, vec(got.psm)) == reinterpret(UInt64, vec(expected.psm))
    @test acc.counts == [reference.counts[i, j] for j in 1:acc.n_locs for i in 1:j]
end

function psm_locs(n)
    [SMLMData.Emitter2DFit(0.01i, 0.002i, 1000.0, 0.0,
         0.01, 0.01, 0.0, 0.0, 0.0, 1, i, 0, 1) for i in 1:n]
end

@testset "Packed PSM exact identity" begin
    rng = MersenneTwister(437)
    for n in (1, 12, 24)
        locs = psm_locs(n)
        acc = PSMAccumulator(); reference = DensePSMReference()
        members = acc._members
        check_psm(acc, reference)
        for iter in 1:30
            assignments = rand(rng, [1, 3, 5], n)
            assignments[1] = 8                     # singleton, with empty slots below it
            state = initialize_from_assignments(assignments, locs)
            state.active[7] = true                  # active-but-empty cluster guard
            state.n_active += 1
            SMLMBaGoL.accumulator_update!(acc, state, locs, 1.0, 2.0, iter)
            update_reference!(reference, state, locs)
            check_psm(acc, reference)
            @test acc.counts isa Vector{UInt16}
            @test acc._members === members
        end
    end

    locs = psm_locs(0)
    state = SMLMBaGoL.initialize_collapsed_state(locs, SMLMBaGoL.FlatSpatial(0.0))
    acc = PSMAccumulator(); reference = DensePSMReference()
    for iter in 1:3
        SMLMBaGoL.accumulator_update!(acc, state, locs, 1.0, 2.0, iter)
        update_reference!(reference, state, locs)
        check_psm(acc, reference)
    end
end

@testset "Packed PSM widens once before overflow" begin
    locs = psm_locs(3)
    states = [initialize_from_assignments(z, locs) for z in ([1, 1, 1], [1, 2, 2], [3, 1, 3])]
    acc = PSMAccumulator(); reference = DensePSMReference()
    for iter in 1:65535
        state = states[mod1(iter, length(states))]
        SMLMBaGoL.accumulator_update!(acc, state, locs, 1.0, 2.0, iter)
        update_reference!(reference, state, locs)
    end
    @test acc.counts isa Vector{UInt16}
    @test maximum(acc.counts) == typemax(UInt16)
    check_psm(acc, reference)

    state = first(states)
    SMLMBaGoL.accumulator_update!(acc, state, locs, 1.0, 2.0, 65536)
    update_reference!(reference, state, locs)
    @test acc.counts isa Vector{UInt32}
    @test maximum(acc.counts) == 65536
    check_psm(acc, reference)
    wide_counts = acc.counts
    for iter in 65537:65540
        state = states[mod1(iter, length(states))]
        SMLMBaGoL.accumulator_update!(acc, state, locs, 1.0, 2.0, iter)
        update_reference!(reference, state, locs)
    end
    @test acc.counts === wide_counts
    check_psm(acc, reference)
end

@testset "Packed PSM storage" begin
    n = 1000
    locs = psm_locs(n)
    state = initialize_from_assignments(ones(Int, n), locs)
    acc = PSMAccumulator()
    SMLMBaGoL.accumulator_update!(acc, state, locs, 1.0, 2.0, 1)
    @test acc.counts isa Vector{UInt16}
    @test length(acc.counts) == n * (n + 1) ÷ 2
    @test sizeof(acc.counts) == 1_001_000
    # Including the diagonal adds n bytes to a quarter of the old 4n²-byte matrix.
    @test sizeof(acc.counts) <= 4n^2 ÷ 4 + n
end

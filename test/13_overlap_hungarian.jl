using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics
using Distributions
import LinearAlgebra
import SMLMSim

@testset "overlap_hungarian" begin
    # Identity match: same assignments
    ref = Int16[1, 1, 2, 2, 3, 3]
    sample = Int16[1, 1, 2, 2, 3, 3]
    ref_labels = Int16[1, 2, 3]
    sample_labels = Int16[1, 2, 3]
    assignment, total, coverlaps = SMLMBaGoL.overlap_hungarian(ref, sample, ref_labels, sample_labels)
    @test total == 6
    @test coverlaps == [2, 2, 2]
    # Each ref cluster should map to itself
    @test assignment == [1, 2, 3]

    # Permuted labels: ref 1↔sample 3, ref 2↔sample 1, ref 3↔sample 2
    sample_perm = Int16[3, 3, 1, 1, 2, 2]
    sample_labels_perm = Int16[1, 2, 3]
    assignment_p, total_p, coverlaps_p = SMLMBaGoL.overlap_hungarian(ref, sample_perm, ref_labels, sample_labels_perm)
    @test total_p == 6
    @test coverlaps_p == [2, 2, 2]
    # ref cluster 1 (indices 1,2 which have sample label 3) → sample cluster 3
    @test assignment_p[1] == 3
    @test assignment_p[2] == 1
    @test assignment_p[3] == 2

    # K=1 short-circuit
    ref1 = Int16[1, 1, 1]
    sample1 = Int16[1, 1, 1]
    a1, t1, c1 = SMLMBaGoL.overlap_hungarian(ref1, sample1, Int16[1], Int16[1])
    @test a1 == [1]
    @test t1 == 3
    @test c1 == [3]
end

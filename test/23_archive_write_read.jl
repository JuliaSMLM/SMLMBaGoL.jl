using Test
using SMLMBaGoL
using SMLMData
using Random

# ================================================================
# Archive
# ================================================================
@testset "Archive Write/Read" begin
    Random.seed!(999)

    sim = simulate_localizations([(0.1, 0.1)];
        fixed_sigma=0.005, mean_count=10.0, count_model=:fixed,
        field_size=6.4, pixel_size=0.1)
    smld = sim.smld

    archive_dir = mktempdir()

    # Run with archive
    result_smld, diagnostics = run_bagol(smld;
        n_iterations=500,
        burn_in=100,
        archive_path=archive_dir,
        verbose=false
    )

    @test result_smld isa SMLMData.BasicSMLD

    # Read archive back
    archive = BaGoLArchive(archive_dir)
    @test archive.n_partitions >= 1
    @test archive.sample_counts[1] > 0

    assignments, μs, shapes = SMLMBaGoL.read_partition(archive, 1)
    @test size(assignments, 2) == archive.sample_counts[1]
    @test length(μs) == archive.sample_counts[1]
    @test all(isfinite, μs)
    @test all(isfinite, shapes)
end

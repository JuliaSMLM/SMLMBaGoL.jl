using Test
using SMLMBaGoL
using SMLMData
using Random
using Statistics
import SMLMSim

# A field cut into METIS pieces with overlap copies: the posterior image holds one copy
# of each emitter (image mass per recorded iteration = the field's mean total K, near the
# number of emitters reported), and diagnostics.posterior_k is the posterior over the
# field's total K (one entry per recorded iteration), not the pooled per-partition counts.

const NITER, BURN = 600, 300
const NREC = NITER - BURN

function sim_field(starting; seed, npix = 16)
    Random.seed!(seed)
    cam = SMLMData.IdealCamera(1:npix, 1:npix, 0.1)
    params = SMLMSim.StaticSMLMConfig(; density = 1.0, nframes = 1000, framerate = 50.0,
        ndatasets = 1, σ_psf = 0.13, minphotons = 50)
    smld, _ = SMLMSim.simulate(params; starting_conditions = starting,
        molecule = SMLMSim.GenericFluor(photons = 1e4, k_off = 50.0, k_on = 0.5), camera = cam)
    return smld
end
site(x, y, i) = SMLMData.Emitter2DFit{Float64}(x, y, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0; track_id = i)

@testset "METIS-split field: image mass and posterior_k count each emitter once" begin
    # 15 x 15 grid at 50 nm pitch: one DBSCAN component, cut into METIS pieces.
    sites = [site(0.5 + 0.05 * i, 0.5 + 0.05 * j, 15i + j + 1) for i in 0:14 for j in 0:14]
    smld = sim_field(sites; seed = 7)
    parts, _ = SMLMBaGoL.partition_locs(smld.emitters; partition_sigma = 3.0, max_size = 300)
    @test length(parts) > 1
    @test sum(count(p.is_overlap) for p in parts) > 0          # the fixture has overlap copies

    Random.seed!(1)
    _, d = run_bagol(smld; n_iterations = NITER, burn_in = BURN, max_partition_size = 300,
                     posterior_pixel_size = 0.002, verbose = false)
    @test d.n_partitions == length(parts)
    # partition_ids name the partition in which each localization is core, not the last
    # neighbor holding an overlap copy of it
    core = zeros(Int, length(smld.emitters))
    for (pid, p) in enumerate(parts), (j, idx) in enumerate(p.original_indices)
        p.is_overlap[j] || (core[idx] = pid)
    end
    @test all(>(0), core)
    @test d.partition_ids == core
    pk = d.posterior_k
    ks = 0:length(pk) - 1
    meanK = sum(ks .* pk) / sum(pk)
    mass = sum(d.posterior_image.image) / NREC
    @test sum(pk) == NREC                                      # one field total per iteration
    @test abs(mass - meanK) <= 0.6                             # image holds the same emitters
    @test abs(mass / d.n_emitters - 1) < 0.10                  # no extra copies in the image
    @test abs(meanK / d.n_emitters - 1) < 0.10
    @test d.posterior_image.image isa Matrix{Float64}
    @test all(abs.(d.convergence_trace.K[d.convergence_trace.iters .> BURN] ./ meanK .- 1) .< 0.10)
end

@testset "posterior_k of two separate groups is the posterior of their total" begin
    # Two DBSCAN components far apart: 3 emitters and 4 emitters, 60 nm apart within each.
    a = [site(0.6 + 0.06 * k, 0.6, k) for k in 0:2]
    b = [site(1.0 + 0.06 * (k % 2), 1.0 + 0.06 * (k ÷ 2), 3 + k) for k in 0:3]
    smld = sim_field(vcat(a, b); seed = 11)
    Random.seed!(2)
    _, d = run_bagol(smld; n_iterations = NITER, burn_in = BURN, verbose = false,
                     posterior_pixel_size = 0.0)
    @test d.n_partitions == 2
    pk = d.posterior_k
    @test sum(pk) == NREC
    # Every recorded K is a total over both groups (pooled per-group counts would put
    # mass at 3 and 4), with the mode at or above 3 + 4.
    @test findfirst(>(0), pk) - 1 >= 6
    @test argmax(pk) - 1 >= 7
end

@testset "A run that never syncs (sync_interval = typemax(Int)) still records every iteration" begin
    a = [site(0.6 + 0.06 * k, 0.6, k) for k in 0:2]
    smld = sim_field(a; seed = 5)
    Random.seed!(3)
    _, d = run_bagol(smld; n_iterations = 200, burn_in = 50, sync_interval = typemax(Int),
                     verbose = false, posterior_pixel_size = 0.002)
    @test sum(d.posterior_k) == 150
    ks = 0:length(d.posterior_k) - 1
    @test sum(d.posterior_image.image) / 150 ≈ sum(ks .* d.posterior_k) / 150 atol = 0.6
end

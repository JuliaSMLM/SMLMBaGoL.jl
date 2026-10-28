using Test
using SMLMBaGoL
using SMLMData
using Random
import SMLMSim

@testset "τ finder: starts above the root agree" begin
    # The answer does not depend on g_start when every start is above the root
    # (the finder premise). Use a sparse field at 20 emitters/μm², σ ≈ 11 nm.
    # At density 250 emitters/μm² and the same σ, crowding puts the root near
    # 16 nm despite simulated τ = 0, so starts 3 and 12 nm are below the root.
    # The old dense-field test passed only through clipping at the 14 nm grid top.
    Random.seed!(1)
    cam = SMLMData.IdealCamera(1:40, 1:40, 0.1)
    params = SMLMSim.StaticSMLMConfig(; density = 20, nframes = 300, framerate = 50.0,
        ndatasets = 1, σ_psf = 0.13, minphotons = 30)
    smld, _ = SMLMSim.simulate(params; pattern = SMLMSim.Nmer2D(n = 1, d = 0.0),
        molecule = SMLMSim.GenericFluor(photons = 12000.0, k_off = 50.0, k_on = 0.5), camera = cam)
    f(g) = estimate_se_adjust(smld; g_start_um = g, n_iterations = 1000, burn_in = 500,
                              n_boot = 20, max_steps = 12)
    lo = f(0.003)
    hi = f(0.012)
    @test lo.start_overmerged
    @test hi.start_overmerged
    @test lo.converged
    @test hi.converged
    @test abs(lo.tau_hat_um - hi.tau_hat_um) ≤ 0.25e-3
end

using Test
using SMLMBaGoL
using SMLMData
using Random
import SMLMSim

@testset "τ finder: the answer does not depend on g_start" begin
    # A small SMLMSim field of single emitters (σ ≈ 11 nm); starts 3 nm and 12 nm
    # bracket the root and must give the same τ̂.
    Random.seed!(1)
    cam = SMLMData.IdealCamera(1:11, 1:11, 0.1)
    params = SMLMSim.StaticSMLMConfig(; density = 250, nframes = 300, framerate = 50.0,
        ndatasets = 1, σ_psf = 0.13, minphotons = 30)
    smld, _ = SMLMSim.simulate(params; pattern = SMLMSim.Nmer2D(n = 1, d = 0.0),
        molecule = SMLMSim.GenericFluor(photons = 12000.0, k_off = 50.0, k_on = 0.5), camera = cam)
    f(g) = estimate_se_adjust(smld; g_start_um = g, n_iterations = 1000, burn_in = 500,
                              n_boot = 20, max_steps = 12)
    lo = f(0.003)
    hi = f(0.012)
    @test abs(lo.tau_hat_um - hi.tau_hat_um) ≤ 0.25e-3
end

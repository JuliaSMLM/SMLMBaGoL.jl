# Test direct use of emitters from SMLMSim
using Pkg
Pkg.activate(".")
using SMLMBaGoL
BGL = SMLMBaGoL
using SMLMData
using SMLMSim
using Distributions

println("Testing direct emitter usage...")

# Small simulation for quick test
smld_true, smld_model, smld_noisy = SMLMSim.simulate(
    SMLMSim.StaticSMLMParams(
        density=3.0,
        σ_psf=0.13,
        minphotons=100,
        ndatasets=1,
        nframes=50,
        framerate=50.0
    );
    pattern=SMLMSim.Nmer2D(d=0.05),
    molecule=SMLMSim.GenericFluor(photons=1e5, k_off=50.0, k_on=0.05),
    camera=SMLMData.IdealCamera(16, 16, 0.1)
)

println("Generated $(length(smld_noisy.emitters)) noisy localizations")
println("Emitter type: $(typeof(smld_noisy.emitters[1]))")

# Test direct visualization
try
    fig_circles = BGL.plot_circles(smld_noisy.emitters; 
                                   true_emitters=smld_true.emitters, 
                                   title="Direct Test")
    println("✅ plot_circles works with direct emitters")
    
    fig_sr = BGL.plot_sr(smld_noisy.emitters; pixelsize=0.01, title="SR")
    println("✅ plot_sr works with direct emitters")
    
    fig_posterior = BGL.plot_posterior(smld_noisy.emitters; pixelsize=1.0, title="Posterior")
    println("✅ plot_posterior works with direct emitters")
catch e
    println("❌ Visualization error: $e")
end

# Test direct BaGoL
try
    μ_λ = length(smld_noisy.emitters) / length(smld_true.emitters)
    prior_λ = Gamma(μ_λ, 1.0)
    
    println("Running BaGoL with direct SMLD...")
    srs, post = BGL.bagol(smld_noisy; prior_λ=prior_λ, posterior_pixel_size=0.005)
    println("✅ BaGoL works directly with SMLD data")
    println("   $(length(srs)) subregions processed")
catch e
    println("❌ BaGoL error: $e")
end

println("\n🎉 Direct emitter usage test complete!")
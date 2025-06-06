# Quick test of smlmsim_bagol_new migration to verify types work
using Pkg
Pkg.activate(".")
using SMLMBaGoL
BGL = SMLMBaGoL
using SMLMData
using SMLMSim
using Distributions

println("Testing migration with smaller parameters...")

# Small simulation for quick test
smld_true, smld_model, smld_noisy = SMLMSim.simulate(
    SMLMSim.StaticSMLMParams(
        density=3.0,
        σ_psf=0.13,
        minphotons=100,
        ndatasets=2,
        nframes=50,
        framerate=50.0
    );
    pattern=SMLMSim.Nmer2D(d=0.05),
    molecule=SMLMSim.GenericFluor(photons=1e5, k_off=50.0, k_on=0.05),
    camera=SMLMData.IdealCamera(16, 16, 0.1)
)

println("Generated $(length(smld_noisy.emitters)) noisy localizations")

# Test direct emitter usage (no conversion needed)
println("Created direct emitters: $(length(smld_noisy.emitters)) entries")
println("Emitter type: $(typeof(smld_noisy.emitters[1]))")

# Test visualizations work with direct emitters
try
    fig_circles = BGL.plot_circles(smld_noisy.emitters; true_emitters=smld_true.emitters, title="Test")
    println("✅ plot_circles works with direct emitters")
    
    fig_sr = BGL.plot_sr(smld_noisy.emitters; pixelsize=0.01, title="Test SR")
    println("✅ plot_sr works with direct emitters")
    
    fig_posterior = BGL.plot_posterior(smld_noisy.emitters; pixelsize=1.0, title="Test Posterior")
    println("✅ plot_posterior works with direct emitters")
catch e
    println("❌ Visualization error: $e")
end

# Quick BaGoL test
try
    μ_λ = length(smld_noisy.emitters) / length(smld_true.emitters)
    prior_λ = Gamma(μ_λ, 1.0)
    
    println("Running quick BaGoL test...")
    srs, post = BGL.bagol(smld_noisy; prior_λ=prior_λ, posterior_pixel_size=0.005)
    println("✅ BaGoL runs with $(length(srs)) subregions")
    
    # Test MAP-N extraction
    all_mapn_coords = []
    for sr in srs
        if length(sr.chains) > 0 && length(sr.chains[1].states) > 0
            chain_mapn = BGL.RJMCMC.extract_mapn_chain(sr.chains[1])
            if length(chain_mapn.states) > 0
                BGL.RJMCMC.sort_mapn_chain!(chain_mapn)
                mapn_coords = BGL.RJMCMC.get_mapn_emitters(chain_mapn, sr.obs)
                append!(all_mapn_coords, mapn_coords)
            end
        end
    end
    
    if !isempty(all_mapn_coords)
        println("✅ MAP-N extraction works: $(length(all_mapn_coords)) emitters")
        println("MAP-N type: $(typeof(all_mapn_coords[1]))")
        
        # Test MAP-N visualization
        fig_mapn = BGL.plot_mapn(all_mapn_coords; pixelsize=0.01, title="Test MAP-N")
        println("✅ plot_mapn works")
        
        fig_combined = BGL.plot_circles(smld_noisy.emitters; mapn_emitters=all_mapn_coords, 
                                        true_emitters=smld_true.emitters, title="Combined")
        println("✅ Combined plot works")
    end
    
catch e
    println("❌ BaGoL error: $e")
end

println("\n🎉 Migration test complete!")
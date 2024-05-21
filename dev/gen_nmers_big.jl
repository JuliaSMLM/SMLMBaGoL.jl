include("setup_dev.jl")

using SMLMData
using SMLMSim
using CairoMakie

k_on = 5e-2
n_datasets = 10
n_frames = 1000
framerate = 50.0
μ = n_datasets * n_frames / framerate * k_on

smld_true, smld_model, smld_noisy = SMLMSim.sim(;
    ρ=50.0,
    σ_PSF=0.13, 
    minphotons=50,
    ndatasets=n_datasets,
    nframes=n_frames,
    framerate=framerate, # 1/s
    pattern=SMLMSim.Nmer2D(d = .05),
    molecule=SMLMSim.GenericFluor(; q=[0 50; k_on 0]), #1/s 
    camera=SMLMSim.IdealCamera(; ypixels=128, xpixels=128, pixelsize=0.1) #pixelsize is microns
)




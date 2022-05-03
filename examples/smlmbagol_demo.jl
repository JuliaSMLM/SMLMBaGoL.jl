# This script demonstrates usage of the SMLMBaGoL package.
using Revise
using SMLMBaGoL
using SMLMData
using SMLMSim
using Plots
using ImageView
using StatsBase

# Simulate some data.
smld_true, smld_model, smld = SMLMSim.sim(;
    ρ=0.1, # patterns / pixel^2
    σ_PSF=1.3, # pixels
    minphotons=50,
    ndatasets=10,
    nframes=1000,
    framerate=1.0, # set to 1.0 to keep things in camera units
    pattern=SMLMSim.Nmer2D(; n=8, d=1.0),
    molecule=SMLMSim.GenericFluor(; q=[0 0.5; 1e-3 0]), # 1 / frame 
    camera=SMLMSim.IdealCamera(; xpixels=16, ypixels=16, pixelsize=1.0) # set pixelsize to 1.0 to keep things in camera units
)

# Perform BaGoL.
params = SMLMBaGoL.BaGoLParams2D()
params.subregion.on = true
params.subregion.roisize = 2.0
params.subregion.roioverlap = 0.5
params.prethresholds.maxsigmadev_photons = 1.0
params.prethresholds.n_min = 1
params.prethresholds.r = 10.0
params.preclustering.on = true
params.preclustering.maxdist = 0.15
params.mcparams.σ_a = 0.0
# params.mcparams.η = 5.0
# params.mcparams.γ = 10.0
params.mcparams.η = 2.0
params.mcparams.γ = (length(smld_model) / length(smld_true)) / params.mcparams.η
params.mcparams.srmag = 10.0
params.mcparams.nsigma = 5.0
params.mcparams.p_jump = [1.0; 1.0; 1.0; 1.0]
params.mcparams.p_jump = params.mcparams.p_jump / sum(params.mcparams.p_jump)
params.mcparams.n_burnin = 2000
params.mcparams.n_chain = 3000
chain, validchain, mapnout, rois, roioverlap = SMLMBaGoL.runbagol(smld; 
    subregion_params = params.subregion, 
    prethresholds = params.prethresholds, 
    preclustering_params = params.preclustering, 
    mcparams = params.mcparams);
# chain, validchain, mapnout, λchain, rois, roioverlap = SMLMBaGoL.runbagol(smld, params.hbparams;
#     subregion_params=params.subregion,
#     prethresholds=params.prethresholds,
#     preclustering_params=params.preclustering,
#     mcparams=params.mcparams);

# Plot some results.
Plots.histogram(mean.(mapnout[5])) # localizations per emitter
Plots.histogram(mapnout[6])
srmag = 100.0
circleim_mapn = SMLMData.makecircleim(mapnout[1], vec(mean(mapnout[2], dims=2)), smld.datasize, srmag)
coords = [smld.y smld.x]
σ_coords = [smld.σ_y smld.σ_x]
circleim = SMLMData.makecircleim(coords, vec(mean(σ_coords, dims=2)), smld.datasize, srmag)
circleim ./= maximum(circleim)
circleim_mapn ./= maximum(circleim_mapn)
testim = RGB.(circleim, circleim_mapn, circleim)
ImageView.imshow(testim)

binimgt = SMLMData.makebinim(smld_true, srmag)
binimgt ./= maximum(binimgt)
testim = RGB.(binimgt, circleim_mapn, binimgt)
ImageView.imshow(testim)

circleimGT = SMLMData.makecircleim(smld_true, srmag)
circleimGT ./= maximum(circleimGT)
testim = RGB.(circleim * 0, circleim_mapn, circleimGT)
ImageView.imshow(testim)
# This script demonstrates usage of the SMLMBaGoL package.
using Revise
using SMLMBaGoL
using SMLMData
using SMLMSim
using DataFrames
using CSV
using Plots
using SpecialFunctions
using StatsBase
using ImageView

# Simulate some data.
γ = 1e5 # Fluorophore emission rate
q = [0 1
    1e-3 0] # Fluorophore blinking rates
n = 6 # Nmer rank
d = 0.5 # Nmer diameter
ρ = 0.1 # density of Nmers 
xsize = 32.0 # image size
ysize = 32.0
nframes = 10000 # number of frames
ndatasets = 1
framerate = 1.0 # set to 1.0 to keep things in physical units
σ_psf = 1.3 # psf sigma used for uncertainty calcs
minphotons = 500 # minimum number of photons per frame accepted
f = SMLMSim.GenericFluor(γ, q)
pattern = SMLMSim.Nmer2D(n, d)
smld_true = SMLMSim.uniform2D(ρ, pattern, xsize, ysize)
smld_true.datasize = [ysize; xsize]
smld_model = SMLMSim.kineticmodel(smld_true, f, nframes, framerate; ndatasets = ndatasets, minphotons = minphotons)
smld_model.datasize = [ysize; xsize]
smld = SMLMSim.noise(smld_model, σ_psf)
smld.datasize = [ysize; xsize]

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
params.mcparams.η = 5.0
params.mcparams.γ = 10.0
# params.mcparams.η = 2.0
# params.mcparams.γ = (length(smld_model) / length(smld_true)) * params.mcparams.η
params.mcparams.srmag = 10.0
params.mcparams.nsigma = 5.0
params.mcparams.p_jump = [1.0; 1.0; 1.0; 1.0]
params.mcparams.p_jump = params.mcparams.p_jump / sum(params.mcparams.p_jump)
params.mcparams.n_burnin = 2000
params.mcparams.n_chain = 3000
# chain, rois, roioverlap = SMLMBaGoL.runbagol(smld; 
#     subregion_params = params.subregion, 
#     prethresholds = params.prethresholds, 
#     preclustering_params = params.preclustering, 
#     mcparams = params.mcparams);
chain, λchain, rois, roioverlap = SMLMBaGoL.runbagol(smld, params.hbparams;
    subregion_params = params.subregion,
    prethresholds = params.prethresholds,
    preclustering_params = params.preclustering,
    mcparams = params.mcparams);
validchain = SMLMBaGoL.removeoverlap(chain, smld.datasize, rois, roioverlap)
mapnout = SMLMBaGoL.mapn(validchain)

# Plot some results.
Plots.histogram(mean.(mapnout[5]))
Plots.histogram(mapnout[6])
srmag = 100.0
gaussim = SMLMData.makegaussim(mapnout[1], mapnout[2], smld.datasize)
circleim_mapn = SMLMData.makecircleim(mapnout[1], vec(mean(mapnout[2], dims = 2)), smld.datasize, srmag)
coords = [smld.y smld.x]
σ_coords = [smld.σ_y smld.σ_x]
circleim = SMLMData.makecircleim(coords, vec(mean(σ_coords, dims = 2)), smld.datasize, srmag)
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
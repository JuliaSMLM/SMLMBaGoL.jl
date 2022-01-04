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

# ## Load some example data and modify for the demonstration.
# filepath = "C:\\Users\\David\\Documents\\work_stuff\\bagol"
# filename = "uniformSMD.mat"
# filenameGT = "uniformSMDGT.mat"
# smld = SMLMData.SMLD2D(SMLMData.SMITEsmd(filepath, filename))
# smldgt = SMLMData.SMLD2D(SMLMData.SMITEsmd(filepath, filenameGT))

# Simulate some data.
γ = 1e5 # Fluorophore emission rate
q = [0 1
    1e-3 0] # Fluorophore blinking rates
n = 6 # Nmer rank
d = 0.3 # Nmer diameter
ρ = 0.1 # density of Nmers 
xsize = 25.6 # image size
ysize = 25.6
nframes = 50000 # number of frames
ndatasets = 1
framerate = 1.0 # set to 1.0 to keep things in physical units
σ_psf = 1.3 # psf sigma used for uncertainty calcs
minphotons = 500 # minimum number of photons per frame accepted

# Simulation sequence
f = SMLMSim.GenericFluor(γ, q)
pattern = SMLMSim.Nmer2D(n, d)
smld_true = SMLMSim.uniform2D(ρ, pattern, xsize, ysize)
smld_model = SMLMSim.kineticmodel(smld_true, f, nframes, framerate; ndatasets = ndatasets, minphotons = minphotons)
smld = SMLMSim.noise(smld_model, σ_psf)

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
params.mcparams.α = length(smld_model) / length(smld_true)
params.mcparams.β = 1.0
# params.mcparams.α = 1.0
# params.mcparams.β = length(smld_true) / length(smld_model)
params.mcparams.srmag = 10.0
params.mcparams.nsigma = 5.0
params.mcparams.p_jump = [1.0; 1.0; 1.0; 1.0]
params.mcparams.p_jump = params.mcparams.p_jump / sum(params.mcparams.p_jump)
params.mcparams.n_burnin = 2000
params.mcparams.n_chain = 3000
chain, rois, roioverlap = SMLMBaGoL.runbagol(smld, params);
validchain = SMLMBaGoL.removeoverlap(chain, smld.datasize, rois, roioverlap)
mapnout = SMLMBaGoL.mapn(validchain)
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
plt = plot(testim)
ImageView.imshow(testim)

binimgt = SMLMData.makebinim(smld_true, srmag)
binimgt ./= maximum(binimgt)
testim = RGB.(binimgt, circleim_mapn, binimgt)
plot(testim)

testim = RGB.(binimgt, circleim, binimgt)
plot(testim)

testim = RGB.(circleim_mapn, circleim, circleim_mapn .+ binimgt)
plot(testim)

circleimGT = SMLMData.makecircleim(smld_true, srmag)
circleimGT ./= maximum(circleimGT)
testim = RGB.(circleim * 0, circleim_mapn, circleimGT)
plot(testim)

# smld_subregions, rois, _ = SMLMBaGoL.gensubregions(smld, 
#     params.subregion.roisize, params.subregion.roioverlap)
# SMLMBaGoL.removeoutliers!(smld_subregions, params.prethresholds)
# smld_preclustered = SMLMBaGoL.precluster_hierarchical.(smld_subregions, 
#     params.preclustering.maxdist)
# smldclusters, _ = SMLMData.isolateconnected(smld_preclustered[1])
# smld = deepcopy(smldclusters[4])
# chain = SMLMBaGoL.runRJMCMC(smld, rois[1], params.mcparams)
# μ = SMLMBaGoL.catfields(chain)

# chain2 = SMLMBaGoL.runRJMCMC(smldclusters, rois[1], params.mcparams)
# μ2 = SMLMBaGoL.catfields(chain2)

# chain3 = SMLMBaGoL.runRJMCMC(smld_preclustered, rois, params.mcparams)
# μ3 = SMLMBaGoL.catfields(chain)
# This script demonstrates usage of the SMLMBaGoL package.
using Revise
using SMLMBaGoL
using SMLMData
using FrameConnection
using DataFrames
using CSV
using Plots
using SpecialFunctions

## Load some example data and modify for the demonstration.
data = DataFrames.DataFrame(CSV.File("C:\\Users\\David\\Documents\\GitHub\\example_data.csv"))
smld = SMLMData.SMLD2D(data)
smld.x = smld.x .- minimum(smld.x)
smld.x = smld.x ./ maximum(smld.x)
smld.y = smld.y .- minimum(smld.y)
smld.y = smld.y ./ maximum(smld.y)
smld.x = 32.0*smld.x .+ 0.5
smld.y = 32.0*smld.y .+ 0.5
smld.datasize = [32; 32]

# ## Define the parameter structure.
# params = SMLMBaGoL.BaGoLParams()

# ## Split the data into subregions.
# params.subregion.roisize = 5
# params.subregion.roioverlap = 1
# smld_subregions, rois, connectID = SMLMBaGoL.gensubregions(smld, 
#     params.subregion.roisize, params.subregion.roioverlap)

# ## Remove outlier localizations.
# params.prethresholds.maxsigmadev_photons = 1.0
# params.prethresholds.n_min = 1
# params.prethresholds.r = 10.0
# SMLMBaGoL.removeoutliers!(smld_subregions, params.prethresholds)

# ## Perform hierarchical clustering.
# params.preclustering.maxdist = 0.15 # pixels
# smld_preclustered = SMLMBaGoL.precluster_hierarchical.(smld_subregions, 
#     params.preclustering.maxdist)

# ##
# nloc = 2
# smld_test = SMLMData.isolatesmld(smld, 1:nloc)
# kinit = 2
# zinit = collect(1:kinit)
# zprime = SMLMBaGoL.allocatelocs(smld_test, [smld.x[zinit] smld.y[zinit]])
# muprime = SMLMBaGoL.moveemitters(smld_test, zprime, nloc, kinit)
# # muprime, _ = SMLMBaGoL.moveemitters(smld_test, zprime, 1e-5, nloc, kinit)

# ##
# smld_test = SMLMData.SMLD2D()
# smld_test.framenum = [100; 700; 200; 500]
# smld_test.x = [1.1; 1.3; 5.6; 5.5]
# smld_test.y = [2.1; 2.3; 8.6; 8.5]
# smld_test.σ_x = [0.11; 0.1; 0.12; 0.13]
# smld_test.σ_y = [0.1; 0.13; 0.11; 0.11]
# μ = [1.2 2.2; 5.55 8.55]
# a = [0.0 0.0; 0.0 0.0]
# w = ones(size(μ, 1)) / size(μ, 1)
# z = SMLMBaGoL.allocatelocs(smld_test, μ)
# logLy = SMLMBaGoL.emitterlogL1D(smld_test.y, smld_test.σ_y, Float64.(smld_test.framenum), μ[:, 2], a[:, 2], z)
# logLx = SMLMBaGoL.emitterlogL1D(smld_test.x, smld_test.σ_x, Float64.(smld_test.framenum), μ[:, 1], a[:, 1], z)
# logL = SMLMBaGoL.emitterlogL2D(smld_test, μ, a, z)
# logL1 = SMLMBaGoL.emitterlogL2D([smld_test.x[1:2] smld_test.y[1:2]], [smld_test.σ_x[1:2] smld_test.σ_y[1:2]], Float64.(smld_test.framenum[1:2]), μ[1, :], a[1, :])
# logL2 = SMLMBaGoL.emitterlogL2D([smld_test.x[3:4] smld_test.y[3:4]], [smld_test.σ_x[3:4] smld_test.σ_y[3:4]], Float64.(smld_test.framenum[3:4]), μ[2, :], a[2, :])
# logLtest = log(SMLMBaGoL.emitterlikelihood1D(smld_test.y, smld_test.σ_y, Float64.(smld_test.framenum), μ[:, 2], a[:, 2], z))

# palloc = SMLMBaGoL.palloc_kernel([smld_test.x smld_test.y], [smld_test.σ_x smld_test.σ_y], Float64.(smld_test.framenum), μ[1, :], a[1, :], w[1])

# palloc = SMLMBaGoL.palloc_kernel([smld_test.x smld_test.y], [smld_test.σ_x smld_test.σ_y], Float64.(smld_test.framenum), μ, a, w)

# logLalloc = SMLMBaGoL.logLalloc_kernel([smld_test.x smld_test.y], [smld_test.σ_x smld_test.σ_y], Float64.(smld_test.framenum), μ, a, w)

##
# smld_test = SMLMData.SMLD2D()
# smld_test.framenum = [100; 700; 200; 500]
# smld_test.x = [1.1; 1.3; 5.6; 5.5]
# smld_test.y = [2.1; 2.3; 8.4; 8.5]
# smld_test.σ_x = [0.11; 0.1; 0.12; 0.13]
# smld_test.σ_y = [0.1; 0.13; 0.11; 0.11]
# smld_test.datasize = [8.0; 8.0]

params = SMLMBaGoL.BaGoLParams2D()
params.subregion.roisize = 5.0
params.subregion.roioverlap = 1.0
params.prethresholds.maxsigmadev_photons = 1.0
params.prethresholds.n_min = 1
params.prethresholds.r = 10.0
params.preclustering.maxdist = 0.15
params.mcparams.n_burnin
params.mcparams.σ_a = 0.0
params.mcparams.α = 1.0
params.mcparams.β = 0.1
params.mcparams.srmag = 10.0
params.mcparams.nsigma = 5.0
params.mcparams.p_jump = [1.0; 1.0; 1.0; 1.0]
params.mcparams.p_jump = params.mcparams.p_jump / sum(params.mcparams.p_jump)
params.mcparams.n_burnin = 100
params.mcparams.n_chain = 100
chain = SMLMBaGoL.runbagol!(smld, params)

smld_subregions, rois, connectID = SMLMBaGoL.gensubregions(smld, 
    params.subregion.roisize, params.subregion.roioverlap)
smld_preclustered = SMLMBaGoL.precluster_hierarchical.(smld_subregions, 
    params.preclustering.maxdist)
smldclusters, _ = SMLMData.isolateconnected(smld_preclustered[1,1])
params.mcparams.roi = [1; 1; 5; 5]
testchain = SMLMBaGoL.runRJMCMC!(smldclusters[1], params.mcparams)
SMLMBaGoL.removeoverlap!(testchain, smld.datasize, params.mcparams.roi, params.subregion.roioverlap)

# smld_preclustered = FrameConnection.precluster(smld) # not meaningful, just to test!
# alpha, beta = SMLMBaGoL.constructprior_lambda(smld_preclustered, false)

# clusterdata = FrameConnection.organizeclusters(smld_preclustered)
# _, nobservations = FrameConnection.computeclusterinfo(clusterdata)
# histogram(nobservations, normalize=:probability)
# x = LinRange(1.0, 6.0, Int64(1e3))
# gammadist(x) = (beta^alpha / SpecialFunctions.gamma(alpha)) *
#     x.^(alpha-1.0) .* exp.(-beta*x)
# plot!(x, gammadist(x))
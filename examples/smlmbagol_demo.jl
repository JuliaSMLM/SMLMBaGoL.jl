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

## Define the parameter structure.
params = SMLMBaGoL.BaGoLParams()

## Split the data into subregions.
params.subregion.roisize = 1.1
params.subregion.roioverlap = 0.15
smld_subregions, rois, connectID = SMLMBaGoL.gensubregions(smld, 
    params.subregion.roisize, params.subregion.roioverlap)

## Remove outlier localizations.
params.prethresholds.maxsigmadev_photons = 1.0
params.prethresholds.n_min = 1
params.prethresholds.r = 10.0
SMLMBaGoL.removeoutliers!(smld_subregions, params.prethresholds)

## Perform hierarchical clustering.
params.preclustering.maxdist = 0.15 # pixels
smld_preclustered = SMLMBaGoL.precluster_hierarchical.(smld_subregions, 
    params.preclustering.maxdist)

# smld_preclustered = FrameConnection.precluster(smld) # not meaningful, just to test!
# alpha, beta = SMLMBaGoL.constructprior_lambda(smld_preclustered, false)

# clusterdata = FrameConnection.organizeclusters(smld_preclustered)
# _, nobservations = FrameConnection.computeclusterinfo(clusterdata)
# histogram(nobservations, normalize=:probability)
# x = LinRange(1.0, 6.0, Int64(1e3))
# gammadist(x) = (beta^alpha / SpecialFunctions.gamma(alpha)) *
#     x.^(alpha-1.0) .* exp.(-beta*x)
# plot!(x, gammadist(x))
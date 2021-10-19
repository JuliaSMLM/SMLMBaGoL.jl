# This script demonstrates usage of the SMLMBaGoL package.
using Revise
using SMLMBaGoL
using SMLMData
using FrameConnection
using DataFrames
using CSV
using Plots
using SpecialFunctions

## Load some example data.
data = DataFrames.DataFrame(CSV.File("C:\\Users\\David\\Documents\\GitHub\\example_data.csv"))
smld = SMLMData.SMLD2D(data)
smld.datasize = [32; 32]

## Split the data into subregions.
roisize = 5
roioverlap = 1
smld_subregions = SMLMBaGoL.gensubregions(smld, roisize, roioverlap)

# smld_preclustered = FrameConnection.precluster(smld) # not meaningful, just to test!
# alpha, beta = SMLMBaGoL.constructprior_lambda(smld_preclustered, false)

# clusterdata = FrameConnection.organizeclusters(smld_preclustered)
# _, nobservations = FrameConnection.computeclusterinfo(clusterdata)
# histogram(nobservations, normalize=:probability)
# x = LinRange(1.0, 6.0, Int64(1e3))
# gammadist(x) = (beta^alpha / SpecialFunctions.gamma(alpha)) *
#     x.^(alpha-1.0) .* exp.(-beta*x)
# plot!(x, gammadist(x))
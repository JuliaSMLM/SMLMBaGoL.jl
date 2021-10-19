# This script demonstrates usage of the SMLMBaGoL package.
using SMLMBaGoL
using SMLMData
using FrameConnection
using DataFrames
using CSV
using Plots
using SpecialFunctions

## Load some data and compute some info. from pre-clusters.
data = DataFrames.DataFrame(CSV.File(pwd() * "\\example_data.csv"))
smld = SMLMData.SMLD(data)
smld_preclustered = FrameConnection.precluster(smld)
alpha, beta = SMLMBaGoL.constructprior_lambda(smld_preclustered, false)
alpha = 1.0

clusterdata = FrameConnection.organizeclusters(smld_preclustered)
_, nobservations = FrameConnection.computeclusterinfo(clusterdata)
histogram(nobservations, normalize=:probability)
x = LinRange(1.0, 6.0, Int64(1e3))
gammadist(x) = (beta^alpha / SpecialFunctions.gamma(alpha)) *
    x.^(alpha-1.0) .* exp.(-beta*x)
plot!(x, gammadist(x))
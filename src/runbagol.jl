using SMLMData

"""
    chain = runbagol(smld::SMLMData.SMLD2D, params::SMLMBaGoL.BaGoLParams2D)

Perform a typical BaGoL analysis based on localizations in `smld`.

# Description
This function is intended to be the main entry for users of the SMLMBaGoL.jl
package, meaning that it will run the standard analysis workflow for Bayesian 
Grouping of Localizations (BaGoL).

# Inputs
-`smld`: SMLD2D structure containing localizations.
-`params`: Structure of parameter structures used/updated throughout the BaGoL
           analysis.

# Outputs
-`chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
"""
function runbagol(smld::SMLMData.SMLD2D, params::SMLMBaGoL.BaGoLParams2D)
    # Split the data into subregions.
    smld_subregions, rois, _ = SMLMBaGoL.gensubregions(smld, 
        params.subregion.roisize, params.subregion.roioverlap)

    # Remove outlier localizations.
    SMLMBaGoL.removeoutliers!(smld_subregions, params.prethresholds)

    # Perform hierarchical clustering on the subregions.
    smld_preclustered = SMLMBaGoL.precluster_hierarchical.(smld_subregions, 
        params.preclustering.maxdist)

    # Perform RJMCMC on each precluster.
    chain = SMLMBaGoL.runRJMCMC(smld_preclustered, rois, params.mcparams)

    return chain
end